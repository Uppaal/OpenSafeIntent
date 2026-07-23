import re
import sys
import time
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_generation.utils import load_json, load_text, save_json
from llm_calls.api_models import get_api_responses_batch
from project_config import (
    DEFAULT_JUDGE_MODEL,
    RESPONSE_OUTPUT_DIR,
    VERTEX_PROJECT_ID,
    DEFAULT_TEMPERATURE as CONFIG_DEFAULT_TEMPERATURE,
)

MODEL_RESPONSES_INPUT_DIR = RESPONSE_OUTPUT_DIR
HELPFULNESS_PROMPT_PATH = REPO_ROOT / "prompts" / "metrics" / "helpfulness.txt"
HARMFULNESS_PROMPT_PATH = REPO_ROOT / "prompts" / "metrics" / "harmfulness.txt"
PROMPT_FIELDS = ("benign_prompt", "dual_use_prompt", "malicious_prompt")
PARAPHRASE_FIELD = "dual_use_paraphrases"
HELPFULNESS_FIELD = "response_scores_helpfulness"
HARMFULNESS_FIELD = "response_scores_harmfulness"
FAILURES_FIELD = "response_score_failures"
DEFAULT_METRIC_MODEL = DEFAULT_JUDGE_MODEL
DEFAULT_VERTEX_PROJECT_ID = VERTEX_PROJECT_ID
DEFAULT_MAX_COMPLETION_TOKENS = 16
DEFAULT_GEMINI_3_THINKING_LEVEL = "LOW"
DEFAULT_TEMPERATURE = CONFIG_DEFAULT_TEMPERATURE
DEFAULT_MAX_RETRIES = 6
DEFAULT_RETRY_BASE_SECONDS = 30
DEFAULT_RETRY_MAX_SECONDS = 300
RETRYABLE_ERROR_MARKERS = (
    "429",
    "resource_exhausted",
    "resource exhausted",
    "rate_limit",
    "rate limit",
    "too many requests",
    "quota exceeded",
    "retryable gemini",
    "overloaded",
    "service unavailable",
    "currently unavailable",
    "temporarily unavailable",
    "try again later",
)


def load_model_response_rows(input_path, limit=None):
    rows = load_json(input_path)
    if not isinstance(rows, list):
        raise ValueError("Model response input must be a list of datapoints.")

    return rows[:limit] if limit is not None else rows


def default_input_path(model_name):
    return MODEL_RESPONSES_INPUT_DIR / f"{model_name}.json"


def default_output_path(model_name):
    return MODEL_RESPONSES_INPUT_DIR / f"{model_name}_scored.json"


def get_required_generated_prompt(row, prompt_field):
    generated_prompts = row.get("generated_prompts", {})
    if not isinstance(generated_prompts, dict):
        raise ValueError("Datapoint generated_prompts was not an object.")

    prompt = generated_prompts.get(prompt_field)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Datapoint is missing generated_prompts.{prompt_field}.")

    return prompt


def get_required_paraphrase_prompts(row):
    paraphrases = row.get(PARAPHRASE_FIELD)
    if not isinstance(paraphrases, list):
        raise ValueError(f"Datapoint {PARAPHRASE_FIELD} was not a list.")

    for index, prompt in enumerate(paraphrases):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Datapoint {PARAPHRASE_FIELD}[{index}] contained no prompt text."
            )

    return paraphrases


def get_required_model_response(row, prompt_field):
    model_responses = row.get("model_responses", {})
    if not isinstance(model_responses, dict):
        raise ValueError("Datapoint model_responses was not an object.")

    response = model_responses.get(prompt_field)
    if not isinstance(response, str):
        raise ValueError(f"Datapoint is missing model_responses.{prompt_field}.")

    return response


def get_required_paraphrase_responses(row, expected_count):
    model_responses = row.get("model_responses", {})
    if not isinstance(model_responses, dict):
        raise ValueError("Datapoint model_responses was not an object.")

    responses = model_responses.get(PARAPHRASE_FIELD)
    if not isinstance(responses, list):
        raise ValueError(f"Datapoint model_responses.{PARAPHRASE_FIELD} was not a list.")
    if len(responses) != expected_count:
        raise ValueError(
            f"Datapoint has {expected_count} paraphrase prompts but "
            f"{len(responses)} paraphrase responses."
        )

    for index, response in enumerate(responses):
        if not isinstance(response, str):
            raise ValueError(
                f"Datapoint model_responses.{PARAPHRASE_FIELD}[{index}] "
                "contained no response text."
            )

    return responses


def get_prompt_response_pairs(row):
    pairs = [
        (
            prompt_field,
            None,
            get_required_generated_prompt(row, prompt_field),
            get_required_model_response(row, prompt_field),
        )
        for prompt_field in PROMPT_FIELDS
    ]
    paraphrase_prompts = get_required_paraphrase_prompts(row)
    paraphrase_responses = get_required_paraphrase_responses(
        row,
        expected_count=len(paraphrase_prompts),
    )
    pairs.extend(
        (PARAPHRASE_FIELD, index, prompt, response)
        for index, (prompt, response) in enumerate(
            zip(paraphrase_prompts, paraphrase_responses)
        )
    )
    return pairs


def empty_score_dict(row):
    paraphrases = row.get(PARAPHRASE_FIELD)
    paraphrase_count = len(paraphrases) if isinstance(paraphrases, list) else 0
    return {
        PROMPT_FIELDS[0]: 0,
        PROMPT_FIELDS[1]: 0,
        PROMPT_FIELDS[2]: 0,
        PARAPHRASE_FIELD: [0] * paraphrase_count,
    }


def assign_score(score_dict, prompt_field, paraphrase_index, score):
    if prompt_field == PARAPHRASE_FIELD:
        score_dict[PARAPHRASE_FIELD][paraphrase_index] = score
    else:
        score_dict[prompt_field] = score


def format_metric_prompt(template, prompt, response):
    return template.format(prompt=prompt, response=response)


def get_error_text(error):
    error_parts = [str(error)]
    response = getattr(error, "response", None)
    response_text = getattr(response, "text", None)
    if response_text:
        error_parts.append(str(response_text))

    return "\n".join(error_parts).lower()


def is_retryable_provider_error(error):
    error_type = error.__class__.__name__
    if error_type == "VertexTransientResponseError":
        return True
    if error_type in {"VertexContentFilteredError", "VertexResponseFormatError"}:
        return False

    error_text = get_error_text(error)
    status_code = getattr(getattr(error, "response", None), "status_code", None)
    return status_code == 429 or any(
        marker in error_text for marker in RETRYABLE_ERROR_MARKERS
    )


def get_retry_delay(attempt, retry_base_seconds, retry_max_seconds):
    return min(retry_max_seconds, retry_base_seconds * (2**attempt))


def summarize_error(error):
    text = str(error)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return repr(error)

    summary = lines[0]
    detail_patterns = (
        ("message", r'"message"\s*:\s*"([^"]+)"'),
        ("finishReason", r'"finishReason"\s*:\s*"([^"]+)"'),
        ("blockReason", r'"blockReason"\s*:\s*"([^"]+)"'),
    )
    for label, pattern in detail_patterns:
        match = re.search(pattern, text)
        if match:
            summary = f"{summary} {label}={match.group(1)}"
            break

    max_length = 500
    if len(summary) > max_length:
        return f"{summary[:max_length - 3]}..."
    return summary


def call_metric_model_for_prompt(
    metric_model,
    prompt,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_base_seconds=DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds=DEFAULT_RETRY_MAX_SECONDS,
):
    thinking_level = (
        DEFAULT_GEMINI_3_THINKING_LEVEL
        if metric_model.startswith("gemini_3")
        else None
    )
    result = get_api_responses_batch(
        [prompt],
        model_name=metric_model,
        vertex_project_id=vertex_project_id,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        thinking_level=thinking_level,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
        raise_on_error=True,
    )[0]
    if not result["success"]:
        raise RuntimeError(result["error"] or "Metric model call failed.")
    return result["response"]


def parse_integer_score(raw_score, min_score, max_score):
    if not isinstance(raw_score, str):
        raise ValueError("Metric model response was not text.")

    stripped = raw_score.strip()
    if re.fullmatch(r"-?\d+", stripped):
        score = int(stripped)
    else:
        match = re.search(r"-?\d+", stripped)
        if match is None:
            raise ValueError(f"Metric response contained no integer: {raw_score!r}")
        score = int(match.group(0))

    if score < min_score or score > max_score:
        raise ValueError(
            f"Metric response score {score} outside allowed range "
            f"{min_score}-{max_score}: {raw_score!r}"
        )

    return score


def score_pair(
    metric_model,
    metric_prompt_template,
    user_prompt,
    assistant_response,
    min_score,
    max_score,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_base_seconds=DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds=DEFAULT_RETRY_MAX_SECONDS,
):
    metric_prompt = format_metric_prompt(
        metric_prompt_template,
        prompt=user_prompt,
        response=assistant_response,
    )
    raw_score = call_metric_model_for_prompt(
        metric_model=metric_model,
        prompt=metric_prompt,
        vertex_project_id=vertex_project_id,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
    )
    return parse_integer_score(raw_score, min_score=min_score, max_score=max_score)


def failure_prompt_name(prompt_field, paraphrase_index):
    if prompt_field == PARAPHRASE_FIELD:
        return f"{PARAPHRASE_FIELD}[{paraphrase_index}]"
    return prompt_field


def record_failure(failures, prompt_field, paraphrase_index, metric_name, error):
    failures.append(
        {
            "prompt": failure_prompt_name(prompt_field, paraphrase_index),
            "metric": metric_name,
            "error": summarize_error(error),
        }
    )


def score_datapoint(
    row,
    metric_model,
    helpfulness_prompt_template,
    harmfulness_prompt_template,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_base_seconds=DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds=DEFAULT_RETRY_MAX_SECONDS,
):
    helpfulness_scores = empty_score_dict(row)
    harmfulness_scores = empty_score_dict(row)
    failures = []
    stats = Counter()

    for prompt_field, paraphrase_index, user_prompt, assistant_response in (
        get_prompt_response_pairs(row)
    ):
        try:
            helpfulness_score = score_pair(
                metric_model=metric_model,
                metric_prompt_template=helpfulness_prompt_template,
                user_prompt=user_prompt,
                assistant_response=assistant_response,
                min_score=1,
                max_score=4,
                vertex_project_id=vertex_project_id,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
            )
            assign_score(
                helpfulness_scores,
                prompt_field,
                paraphrase_index,
                helpfulness_score,
            )
            stats["successful_metric_calls"] += 1
        except Exception as error:
            assign_score(helpfulness_scores, prompt_field, paraphrase_index, 0)
            record_failure(
                failures,
                prompt_field,
                paraphrase_index,
                "helpfulness",
                error,
            )
            stats["failed_metric_calls"] += 1

        try:
            harmfulness_score = score_pair(
                metric_model=metric_model,
                metric_prompt_template=harmfulness_prompt_template,
                user_prompt=user_prompt,
                assistant_response=assistant_response,
                min_score=0,
                max_score=3,
                vertex_project_id=vertex_project_id,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
            )
            assign_score(
                harmfulness_scores,
                prompt_field,
                paraphrase_index,
                harmfulness_score,
            )
            stats["successful_metric_calls"] += 1
        except Exception as error:
            assign_score(harmfulness_scores, prompt_field, paraphrase_index, 0)
            record_failure(
                failures,
                prompt_field,
                paraphrase_index,
                "harmfulness",
                error,
            )
            stats["failed_metric_calls"] += 1

    output_row = deepcopy(row)
    output_row[HELPFULNESS_FIELD] = helpfulness_scores
    output_row[HARMFULNESS_FIELD] = harmfulness_scores
    if failures:
        output_row[FAILURES_FIELD] = failures
        stats["datapoints_with_metric_failures"] += 1

    return output_row, stats


def process_rows(
    rows,
    output_path=None,
    metric_model=DEFAULT_METRIC_MODEL,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_base_seconds=DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds=DEFAULT_RETRY_MAX_SECONDS,
):
    helpfulness_prompt_template = load_text(HELPFULNESS_PROMPT_PATH)
    harmfulness_prompt_template = load_text(HARMFULNESS_PROMPT_PATH)
    output_rows = []
    stats = Counter()

    for index, row in enumerate(tqdm(rows, desc="Scoring model responses")):
        try:
            output_row, row_stats = score_datapoint(
                row=row,
                metric_model=metric_model,
                helpfulness_prompt_template=helpfulness_prompt_template,
                harmfulness_prompt_template=harmfulness_prompt_template,
                vertex_project_id=vertex_project_id,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
            )
        except Exception as error:
            stats["datapoint_errors"] += 1
            tqdm.write(f"Failed datapoint {index}; assigning zero scores: {error}")
            output_row = deepcopy(row)
            output_row[HELPFULNESS_FIELD] = empty_score_dict(row)
            output_row[HARMFULNESS_FIELD] = empty_score_dict(row)
            output_row[FAILURES_FIELD] = [
                {
                    "prompt": "datapoint",
                    "metric": "all",
                    "error": summarize_error(error),
                }
            ]
            row_stats = Counter(failed_metric_calls=0)

        output_rows.append(output_row)
        stats["scored_rows"] += 1
        stats.update(row_stats)
        if output_path is not None:
            save_json(output_rows, output_path)

    stats["input_rows"] = len(rows)
    stats["output_rows"] = len(output_rows)
    return output_rows, stats


def print_summary(stats, model_name):
    print(f"\n{model_name} Metric Scoring Summary")
    print("-" * 40)
    for field in (
        "input_rows",
        "scored_rows",
        "successful_metric_calls",
        "failed_metric_calls",
        "datapoints_with_metric_failures",
        "datapoint_errors",
        "output_rows",
    ):
        print(f"{field}: {stats.get(field, 0)}")


def score_model_responses(
    model_name,
    input_path=None,
    output_path=None,
    limit=None,
    metric_model=DEFAULT_METRIC_MODEL,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_base_seconds=DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds=DEFAULT_RETRY_MAX_SECONDS,
):
    if input_path is None:
        input_path = default_input_path(model_name)
    if output_path is None:
        output_path = default_output_path(model_name)

    rows = load_model_response_rows(input_path=input_path, limit=limit)
    output_rows, stats = process_rows(
        rows=rows,
        output_path=output_path,
        metric_model=metric_model,
        vertex_project_id=vertex_project_id,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
    )
    save_json(output_rows, output_path)
    print(f"Saved scored model responses to: {Path(output_path).resolve()}")
    print_summary(stats, model_name)
    return {"rows": output_rows, "stats": dict(stats)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score model responses for helpfulness and harmfulness."
    )
    parser.add_argument(
        "model_name",
        help=(
            "Model response file stem. Defaults input/output to "
            "evaluation/model_responses/<model_name>.json and "
            "<model_name>_scored.json."
        ),
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For debugging, score only the first N datapoints.",
    )
    parser.add_argument(
        "--metric-model",
        default=DEFAULT_METRIC_MODEL,
        help="Vertex model alias used as the evaluator.",
    )
    parser.add_argument(
        "--vertex-project-id",
        default=DEFAULT_VERTEX_PROJECT_ID,
        help="Google Cloud project ID used for Vertex metric calls.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help="Maximum tokens to request for each metric score.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for metric calls.",
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
    score_model_responses(
        model_name=args.model_name,
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        metric_model=args.metric_model,
        vertex_project_id=args.vertex_project_id,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        retry_max_seconds=args.retry_max_seconds,
    )
