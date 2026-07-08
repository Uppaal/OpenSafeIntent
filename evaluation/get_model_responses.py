import sys
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_generation.utils import load_json, save_json
from llm_calls.api_models import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_SECONDS,
    DEFAULT_RETRY_MAX_SECONDS,
    DEFAULT_VERTEX_PROJECT_ID,
    SUPPORTED_MODEL_NAMES,
    get_api_responses_batch,
)
from OpenSafeIntent.project_config import (
    MAX_COMPLETION_TOKENS as CONFIG_MAX_COMPLETION_TOKENS,
    RESPONSE_OUTPUT_DIR,
)


FINAL_DATASET_PATH = REPO_ROOT / "dataset.json"
MODEL_RESPONSES_OUTPUT_DIR = RESPONSE_OUTPUT_DIR
DEFAULT_MAX_COMPLETION_TOKENS = CONFIG_MAX_COMPLETION_TOKENS
PROMPT_FIELDS = ("benign_prompt", "dual_use_prompt", "malicious_prompt")
PARAPHRASE_FIELD = "dual_use_paraphrases"


def load_final_dataset(input_path=FINAL_DATASET_PATH, limit=None):
    rows = load_json(input_path)
    if not isinstance(rows, list):
        raise ValueError("Evaluation dataset must be a list of datapoints.")

    return rows[:limit] if limit is not None else rows


def get_required_generated_prompt(row, prompt_field):
    generated_prompts = row.get("generated_prompts", {})
    if not isinstance(generated_prompts, dict):
        raise ValueError("Datapoint generated_prompts was not an object.")

    prompt = generated_prompts.get(prompt_field)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Datapoint is missing generated_prompts.{prompt_field}.")

    return prompt


def get_required_paraphrases(row):
    paraphrases = row.get(PARAPHRASE_FIELD)
    if not isinstance(paraphrases, list):
        raise ValueError(f"Datapoint {PARAPHRASE_FIELD} was not a list.")

    for index, prompt in enumerate(paraphrases):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Datapoint {PARAPHRASE_FIELD}[{index}] contained no prompt text."
            )

    return paraphrases


def generate_datapoint_responses(
    row,
    model_name,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_base_seconds=DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds=DEFAULT_RETRY_MAX_SECONDS,
):
    prompts = [
        (prompt_field, get_required_generated_prompt(row, prompt_field))
        for prompt_field in PROMPT_FIELDS
    ]
    paraphrases = get_required_paraphrases(row)
    prompts.extend((PARAPHRASE_FIELD, prompt) for prompt in paraphrases)
    batch_results = get_api_responses_batch(
        [prompt for _prompt_field, prompt in prompts],
        model_name=model_name,
        vertex_project_id=vertex_project_id,
        max_completion_tokens=max_completion_tokens,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
        raise_on_error=True,
    )
    model_responses = {}
    model_responses[PARAPHRASE_FIELD] = []
    stats = Counter()

    for (prompt_field, _prompt), result in zip(prompts, batch_results):
        response = result["response"]
        if prompt_field == PARAPHRASE_FIELD:
            model_responses[PARAPHRASE_FIELD].append(response)
        else:
            model_responses[prompt_field] = response

        stats["api_calls"] += result["api_calls"]
        stats["retry_attempts"] += result["retry_attempts"]
        if result["success"]:
            stats["successful_calls"] += 1
        elif result["filtered"]:
            stats["filtered_calls"] += 1
        else:
            raise RuntimeError(result["error"] or "API model call failed.")

    return model_responses, stats


def add_model_responses(row, model_responses):
    output_row = deepcopy(row)
    output_row["model_responses"] = model_responses
    return output_row


def process_rows(
    rows,
    model_name,
    output_path=None,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_base_seconds=DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds=DEFAULT_RETRY_MAX_SECONDS,
):
    output_rows = []
    stats = Counter()

    for index, row in enumerate(tqdm(rows, desc=f"Evaluating {model_name}")):
        try:
            model_responses, row_stats = generate_datapoint_responses(
                row=row,
                model_name=model_name,
                vertex_project_id=vertex_project_id,
                max_completion_tokens=max_completion_tokens,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
            )
        except Exception as error:
            stats["skipped_datapoints"] += 1
            stats["datapoint_errors"] += 1
            tqdm.write(
                f"Skipping datapoint {index} after evaluation error: {error}"
            )
            if output_path is not None:
                save_json(output_rows, output_path)
            continue

        output_rows.append(add_model_responses(row, model_responses))
        stats["evaluated_rows"] += 1
        stats.update(row_stats)
        if output_path is not None:
            save_json(output_rows, output_path)

    stats["input_rows"] = len(rows)
    stats["output_rows"] = len(output_rows)
    return output_rows, stats


def print_summary(stats, model_name):
    print(f"\n{model_name} Evaluation Summary")
    print("-" * 40)
    for field in (
        "input_rows",
        "evaluated_rows",
        "api_calls",
        "retry_attempts",
        "successful_calls",
        "filtered_calls",
        "skipped_datapoints",
        "datapoint_errors",
        "output_rows",
    ):
        print(f"{field}: {stats.get(field, 0)}")


def generate_model_responses(
    model_name,
    input_path=FINAL_DATASET_PATH,
    output_path=None,
    limit=None,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_base_seconds=DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds=DEFAULT_RETRY_MAX_SECONDS,
):
    if output_path is None:
        output_path = MODEL_RESPONSES_OUTPUT_DIR / f"{model_name}.json"

    rows = load_final_dataset(input_path=input_path, limit=limit)
    output_rows, stats = process_rows(
        rows,
        model_name=model_name,
        output_path=output_path,
        vertex_project_id=vertex_project_id,
        max_completion_tokens=max_completion_tokens,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
    )
    save_json(output_rows, output_path)
    print(f"Saved model responses to: {Path(output_path).resolve()}")
    print_summary(stats, model_name)
    return {"rows": output_rows, "stats": dict(stats)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate model responses for each final evaluation datapoint."
    )
    parser.add_argument(
        "model_name",
        choices=SUPPORTED_MODEL_NAMES,
        help="API-backed model to evaluate.",
    )
    parser.add_argument("--input", type=Path, default=FINAL_DATASET_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path; defaults to evaluation/model_responses/<model_name>.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For debugging, process only the first N datapoints.",
    )
    parser.add_argument(
        "--vertex-project-id",
        default=DEFAULT_VERTEX_PROJECT_ID,
        help="Google Cloud project ID used for Vertex models.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help="Maximum tokens to request for each prompt.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries for transient provider errors such as 429s.",
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=DEFAULT_RETRY_BASE_SECONDS,
        help="Initial retry delay for transient provider errors.",
    )
    parser.add_argument(
        "--retry-max-seconds",
        type=float,
        default=DEFAULT_RETRY_MAX_SECONDS,
        help="Maximum retry delay for transient provider errors.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_model_responses(
        model_name=args.model_name,
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        vertex_project_id=args.vertex_project_id,
        max_completion_tokens=args.max_completion_tokens,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        retry_max_seconds=args.retry_max_seconds,
    )
