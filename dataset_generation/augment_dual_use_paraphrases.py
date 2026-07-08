import sys
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_calls.api_models import get_api_responses_batch
from OpenSafeIntent.project_config import (
    DATASET_OUTPUT_DIR,
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_TEMPERATURE,
)

try:
    from .utils import extract_json_object_with_keys, load_json, load_text, save_json
except ImportError:
    from utils import extract_json_object_with_keys, load_json, load_text, save_json


PROMPT_DIR = REPO_ROOT / "prompts"
PARAPHRASE_PROMPT_PATH = PROMPT_DIR / "metrics" / "paraphrase_generation.txt"
STAGE_5_OUTPUT_DIR = DATASET_OUTPUT_DIR / "stage_5"
STAGE_5_DEDUPLICATED_INPUT_PATH = STAGE_5_OUTPUT_DIR / "stage_5_deduplicated.json"
FINAL_OUTPUT_PATH = DATASET_OUTPUT_DIR / "dataset.json"
DEFAULT_NUM_PARAPHRASES = 4
DEFAULT_MAX_COMPLETION_TOKENS = 1500
REQUIRED_RESPONSE_KEYS = ("paraphrases",)
REQUIRED_SELF_CHECK_KEYS = (
    "same_meaning",
    "same_intent",
    "same_specificity",
    "same_ambiguity",
    "not_near_duplicate",
)


def load_stage_5_deduplicated(input_path=STAGE_5_DEDUPLICATED_INPUT_PATH, limit=None):
    rows = load_json(input_path)
    return rows[:limit] if limit is not None else rows


def load_prompt_template(prompt_path=PARAPHRASE_PROMPT_PATH):
    return load_text(prompt_path)


def get_required_api_response(prompt, max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS):
    result = get_api_responses_batch(
        [prompt],
        model_name=DEFAULT_GENERATOR_MODEL,
        max_completion_tokens=max_completion_tokens,
        temperature=DEFAULT_TEMPERATURE,
        raise_on_error=True,
    )[0]
    if not result["success"]:
        raise RuntimeError(result["error"] or "API model call failed.")
    return result["response"]


def get_required_dual_use_prompt(row):
    generated_prompts = row.get("generated_prompts", {})
    if not isinstance(generated_prompts, dict):
        raise ValueError("Datapoint generated_prompts was not an object.")

    dual_use_prompt = generated_prompts.get("dual_use_prompt")
    if not isinstance(dual_use_prompt, str) or not dual_use_prompt.strip():
        raise ValueError("Datapoint is missing generated_prompts.dual_use_prompt.")

    return dual_use_prompt


def build_paraphrase_prompt(row, num_paraphrases, prompt_template=None):
    if num_paraphrases < 1:
        raise ValueError("num_paraphrases must be at least 1.")

    prompt_template = prompt_template or load_prompt_template()
    return (
        prompt_template.replace(
            "{{DUAL_USE_PROMPT}}",
            get_required_dual_use_prompt(row),
        )
        .replace("{{K}}", str(num_paraphrases))
    )


def parse_paraphrase_response(raw_response, expected_count):
    parsed = extract_json_object_with_keys(
        raw_response,
        required_keys=REQUIRED_RESPONSE_KEYS,
    )
    paraphrase_items = parsed["paraphrases"]
    if not isinstance(paraphrase_items, list):
        raise ValueError("Paraphrase response field 'paraphrases' was not a list.")
    if len(paraphrase_items) != expected_count:
        raise ValueError(
            "Paraphrase response returned "
            f"{len(paraphrase_items)} paraphrases; expected {expected_count}."
        )

    paraphrases = []
    for index, item in enumerate(paraphrase_items):
        if not isinstance(item, dict):
            raise ValueError(f"Paraphrase item {index} was not an object.")

        paraphrase = item.get("paraphrase")
        if not isinstance(paraphrase, str) or not paraphrase.strip():
            raise ValueError(f"Paraphrase item {index} contained no paraphrase text.")

        self_check = item.get("self_check")
        if not isinstance(self_check, dict):
            raise ValueError(f"Paraphrase item {index} self_check was not an object.")
        missing_self_checks = [
            key for key in REQUIRED_SELF_CHECK_KEYS if key not in self_check
        ]
        if missing_self_checks:
            raise ValueError(
                f"Paraphrase item {index} self_check was missing "
                f"{missing_self_checks}."
            )
        failed_self_checks = [
            key for key in REQUIRED_SELF_CHECK_KEYS if self_check[key] is not True
        ]
        if failed_self_checks:
            raise ValueError(
                f"Paraphrase item {index} failed self checks {failed_self_checks}."
            )

        paraphrases.append(paraphrase.strip())

    if len(set(paraphrases)) != len(paraphrases):
        raise ValueError("Paraphrase response contained duplicate paraphrase texts.")

    return paraphrases


def call_llm_for_paraphrases(
    row,
    num_paraphrases=DEFAULT_NUM_PARAPHRASES,
    prompt_template=None,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
):
    prompt = build_paraphrase_prompt(
        row=row,
        num_paraphrases=num_paraphrases,
        prompt_template=prompt_template,
    )
    raw_response = get_required_api_response(
        prompt,
        max_completion_tokens=max_completion_tokens,
    )
    return parse_paraphrase_response(raw_response, expected_count=num_paraphrases)


def add_dual_use_paraphrases(row, paraphrases):
    output_row = deepcopy(row)
    output_row["dual_use_paraphrases"] = list(paraphrases)
    return output_row


def process_rows(
    rows,
    num_paraphrases=DEFAULT_NUM_PARAPHRASES,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
):
    if num_paraphrases < 1:
        raise ValueError("num_paraphrases must be at least 1.")

    prompt_template = load_prompt_template()
    output_rows = []
    stats = Counter()

    for index, row in enumerate(tqdm(rows, desc="Generating dual-use paraphrases")):
        try:
            paraphrases = call_llm_for_paraphrases(
                row=row,
                num_paraphrases=num_paraphrases,
                prompt_template=prompt_template,
                max_completion_tokens=max_completion_tokens,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to generate paraphrases for datapoint {index}: {error}"
            ) from error

        output_rows.append(add_dual_use_paraphrases(row, paraphrases))
        stats["augmented_rows"] += 1
        stats["paraphrases_generated"] += len(paraphrases)

    stats["input_rows"] = len(rows)
    stats["output_rows"] = len(output_rows)
    return output_rows, stats


def print_summary(stats, num_paraphrases):
    print("\nDual-Use Paraphrase Augmentation Summary")
    print("-" * 40)
    print(f"paraphrases_per_datapoint: {num_paraphrases}")
    for field in (
        "input_rows",
        "augmented_rows",
        "paraphrases_generated",
        "output_rows",
    ):
        print(f"{field}: {stats.get(field, 0)}")


def generate_final_dataset(
    input_path=STAGE_5_DEDUPLICATED_INPUT_PATH,
    output_path=FINAL_OUTPUT_PATH,
    limit=None,
    num_paraphrases=DEFAULT_NUM_PARAPHRASES,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
):
    rows = load_stage_5_deduplicated(input_path=input_path, limit=limit)
    output_rows, stats = process_rows(
        rows,
        num_paraphrases=num_paraphrases,
        max_completion_tokens=max_completion_tokens,
    )
    save_json(output_rows, output_path)
    print(f"Saved augmented outputs to: {Path(output_path).resolve()}")
    print_summary(stats, num_paraphrases)
    return {"rows": output_rows, "stats": dict(stats)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paraphrases for each final dual-use prompt."
    )
    parser.add_argument("--input", type=Path, default=STAGE_5_DEDUPLICATED_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=FINAL_OUTPUT_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For debugging, process only the first N datapoints.",
    )
    parser.add_argument(
        "--num-paraphrases",
        type=int,
        default=DEFAULT_NUM_PARAPHRASES,
        help="Number of paraphrases to generate for each dual-use prompt.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help="Maximum tokens to request from the API model for each datapoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_final_dataset(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        num_paraphrases=args.num_paraphrases,
        max_completion_tokens=args.max_completion_tokens,
    )
