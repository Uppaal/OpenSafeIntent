from __future__ import annotations

import re
import sys
import types
import argparse
from pathlib import Path
from copy import deepcopy
from collections import Counter
from typing import Any, Callable

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable

    tqdm.write = print
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = tqdm
    sys.modules.setdefault("tqdm", tqdm_module)


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENSAFEINTENT_ROOT = REPO_ROOT / "OpenSafeIntent"
for import_root in (REPO_ROOT, OPENSAFEINTENT_ROOT):
    if import_root.exists() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dataset_generation.utils import (  # noqa: E402
    extract_json_object,
    load_json,
    load_text,
    save_json,
)
from evaluation.metrics import (  # noqa: E402
    default_input_path as default_model_response_input_path,
)
from project_config import (  # noqa: E402
    ANALYSIS_OUTPUT_DIR,
    DEFAULT_JUDGE_MODEL as CONFIG_DEFAULT_JUDGE_MODEL,
    DEFAULT_TEMPERATURE as CONFIG_DEFAULT_TEMPERATURE,
    MAX_COMPLETION_TOKENS as CONFIG_MAX_COMPLETION_TOKENS,
    VERTEX_PROJECT_ID,
)


PROMPT_PATH = REPO_ROOT / "prompts" / "metrics" / "assistance_level_classification.txt"
DEFAULT_JUDGE_MODEL = CONFIG_DEFAULT_JUDGE_MODEL
DEFAULT_VERTEX_PROJECT_ID = VERTEX_PROJECT_ID
DEFAULT_MAX_COMPLETION_TOKENS = CONFIG_MAX_COMPLETION_TOKENS
DEFAULT_TEMPERATURE = CONFIG_DEFAULT_TEMPERATURE
DEFAULT_MAX_RETRIES = 6
DEFAULT_RETRY_BASE_SECONDS = 30
DEFAULT_RETRY_MAX_SECONDS = 300
DEFAULT_GEMINI_3_THINKING_LEVEL = "LOW"

ANNOTATED_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "assistance_level"
ASSISTANCE_LEVEL_FIELD = "dual_use_assistance_level"
FAILURES_FIELD = "assistance_level_failures"
HARMFULNESS_FIELD = "response_scores_harmfulness"
DUAL_USE_PROMPT_FIELD = "dual_use_prompt"
A_LEVELS = ("A0", "A1", "A2", "A3", "A4")
VALID_A_LEVELS = set(A_LEVELS)
SCORED_FILENAME_SUFFIX = "_scored"
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


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def default_output_path(input_path: str | Path) -> Path:
    return ANNOTATED_OUTPUT_DIR / Path(input_path).name


def load_rows(input_path: str | Path, limit: int | None = None) -> list[Any]:
    rows = load_json(resolve_path(input_path))
    if not isinstance(rows, list):
        raise ValueError("Input JSON must contain a list of datapoints.")

    return rows[:limit] if limit is not None else rows


def load_json_rows(input_path: str | Path) -> list[Any]:
    rows = load_json(resolve_path(input_path))
    if not isinstance(rows, list):
        raise ValueError(f"{input_path} must contain a JSON list of datapoints.")
    return rows


def get_required_dual_use_pair(row: Any) -> tuple[str, str]:
    generated_prompts = row.get("generated_prompts", {})
    if not isinstance(generated_prompts, dict):
        raise ValueError("Datapoint generated_prompts was not an object.")

    model_responses = row.get("model_responses", {})
    if not isinstance(model_responses, dict):
        raise ValueError("Datapoint model_responses was not an object.")

    prompt = generated_prompts.get(DUAL_USE_PROMPT_FIELD)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            f"Datapoint is missing generated_prompts.{DUAL_USE_PROMPT_FIELD}."
        )

    response = model_responses.get(DUAL_USE_PROMPT_FIELD)
    if not isinstance(response, str):
        raise ValueError(
            f"Datapoint is missing model_responses.{DUAL_USE_PROMPT_FIELD}."
        )

    return prompt, response


def format_judge_prompt(template: str, prompt: str, response: str) -> str:
    if "{{prompt}}" in template or "{{response}}" in template:
        return template.replace("{{prompt}}", prompt).replace("{{response}}", response)

    return template.format(prompt=prompt, response=response)


def get_error_text(error: Exception) -> str:
    error_parts = [str(error)]
    response = getattr(error, "response", None)
    response_text = getattr(response, "text", None)
    if response_text:
        error_parts.append(str(response_text))

    return "\n".join(error_parts).lower()


def is_retryable_provider_error(error: Exception) -> bool:
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


def get_retry_delay(
    attempt: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
) -> float:
    return min(retry_max_seconds, retry_base_seconds * (2**attempt))


def summarize_error(error: Exception) -> str:
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


def call_judge_model(
    judge_model: str,
    prompt: str,
    vertex_project_id: str = DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
) -> str:
    from llm_calls.api_models import get_api_responses_batch

    thinking_level = (
        DEFAULT_GEMINI_3_THINKING_LEVEL
        if judge_model.startswith("gemini_3")
        else None
    )
    result = get_api_responses_batch(
        [prompt],
        model_name=judge_model,
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
        raise RuntimeError(result["error"] or "Judge model call failed.")
    return result["response"]


def parse_assistance_level(raw_response: str) -> dict[str, Any]:
    parsed = extract_json_object(raw_response)
    a_level = parsed.get("a_level")

    if not isinstance(a_level, str):
        raise ValueError(f"Judge response a_level was not text: {raw_response!r}")

    a_level = a_level.strip().upper()
    if a_level not in VALID_A_LEVELS:
        raise ValueError(
            f"Judge response a_level {a_level!r} was not one of "
            f"{sorted(VALID_A_LEVELS)}: {raw_response!r}"
        )

    parsed["a_level"] = a_level
    return parsed


def classify_pair(
    judge_model: str,
    judge_prompt_template: str,
    user_prompt: str,
    assistant_response: str,
    vertex_project_id: str = DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
) -> dict[str, Any]:
    judge_prompt = format_judge_prompt(
        judge_prompt_template,
        prompt=user_prompt,
        response=assistant_response,
    )
    raw_response = call_judge_model(
        judge_model=judge_model,
        prompt=judge_prompt,
        vertex_project_id=vertex_project_id,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
    )
    return parse_assistance_level(raw_response)


def classify_datapoint(
    row: dict[str, Any],
    judge_model: str,
    judge_prompt_template: str,
    vertex_project_id: str = DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
) -> tuple[dict[str, Any], Counter[str]]:
    output_row = deepcopy(row)
    stats: Counter[str] = Counter()

    try:
        user_prompt, assistant_response = get_required_dual_use_pair(row)
        classification = classify_pair(
            judge_model=judge_model,
            judge_prompt_template=judge_prompt_template,
            user_prompt=user_prompt,
            assistant_response=assistant_response,
            vertex_project_id=vertex_project_id,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
        )
        stats["successful_judge_calls"] += 1
    except Exception as error:
        classification = {"a_level": None}
        classification[FAILURES_FIELD] = [
            {
                "prompt": DUAL_USE_PROMPT_FIELD,
                "metric": "assistance_level",
                "error": summarize_error(error),
            }
        ]
        stats["failed_judge_calls"] += 1
        stats["datapoints_with_failures"] += 1

    output_row[ASSISTANCE_LEVEL_FIELD] = classification

    return output_row, stats


def process_rows(
    rows: list[dict[str, Any]],
    output_path: str | Path | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    vertex_project_id: str = DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    judge_prompt_template = load_text(PROMPT_PATH)
    output_rows = []
    stats: Counter[str] = Counter()
    resolved_output_path = resolve_path(output_path) if output_path is not None else None

    for index, row in enumerate(tqdm(rows, desc="Classifying assistance levels")):
        output_row, row_stats = classify_datapoint(
            row=row,
            judge_model=judge_model,
            judge_prompt_template=judge_prompt_template,
            vertex_project_id=vertex_project_id,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
        )
        output_rows.append(output_row)
        stats["classified_rows"] += 1
        stats.update(row_stats)

        if row_stats.get("failed_judge_calls"):
            tqdm.write(f"Failed datapoint {index}; recorded assistance-level failure.")
        if resolved_output_path is not None:
            save_json(output_rows, resolved_output_path)

    stats["input_rows"] = len(rows)
    stats["output_rows"] = len(output_rows)
    return output_rows, stats


def print_classification_summary(stats: Counter[str] | dict[str, int], judge_model: str) -> None:
    print(f"\n{judge_model} Assistance-Level Classification Summary")
    print("-" * 56)
    for field in (
        "input_rows",
        "classified_rows",
        "successful_judge_calls",
        "failed_judge_calls",
        "datapoints_with_failures",
        "output_rows",
    ):
        print(f"{field}: {stats.get(field, 0)}")


def classify_assistance_levels(
    input_path: str | Path,
    output_path: str | Path | None = None,
    limit: int | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    vertex_project_id: str = DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
) -> dict[str, Any]:
    resolved_input_path = resolve_path(input_path)
    resolved_output_path = (
        default_output_path(resolved_input_path)
        if output_path is None
        else resolve_path(output_path)
    )

    rows = load_rows(resolved_input_path, limit=limit)
    output_rows, stats = process_rows(
        rows=rows,
        output_path=resolved_output_path,
        judge_model=judge_model,
        vertex_project_id=vertex_project_id,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
    )
    save_json(output_rows, resolved_output_path)
    print(f"Saved assistance-level classifications to: {resolved_output_path.resolve()}")
    print_classification_summary(stats, judge_model)
    return {"rows": output_rows, "stats": dict(stats), "output_path": resolved_output_path}


def model_name_from_path(input_path: str | Path) -> str:
    model_name = Path(input_path).stem
    if model_name.endswith(SCORED_FILENAME_SUFFIX):
        model_name = model_name[: -len(SCORED_FILENAME_SUFFIX)]
    return model_name.rstrip(".")


def extract_a_level(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None

    classification = row.get(ASSISTANCE_LEVEL_FIELD)
    if not isinstance(classification, dict):
        return None

    a_level = classification.get("a_level")
    if not isinstance(a_level, str):
        return None

    a_level = a_level.strip().upper()
    return a_level if a_level in A_LEVELS else None


def numeric_score(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def dual_use_harmfulness(row: Any) -> float | None:
    if not isinstance(row, dict):
        return None

    harmfulness_scores = row.get(HARMFULNESS_FIELD)
    if not isinstance(harmfulness_scores, dict):
        return None

    return numeric_score(harmfulness_scores.get(DUAL_USE_PROMPT_FIELD))


def keep_harmful_dual_use(row: Any) -> bool:
    harmfulness = dual_use_harmfulness(row)
    return harmfulness is not None and harmfulness > 0


def count_a_levels(
    rows: list[Any],
    row_filter: Callable[[Any], bool] | None = None,
) -> tuple[Counter[str], int, int]:
    counts: Counter[str] = Counter({a_level: 0 for a_level in A_LEVELS})
    considered_rows = 0
    skipped_rows = 0

    for row in rows:
        if row_filter is not None and not row_filter(row):
            continue

        considered_rows += 1
        a_level = extract_a_level(row)
        if a_level is None:
            skipped_rows += 1
            continue

        counts[a_level] += 1

    return counts, considered_rows, skipped_rows


def build_distributions(
    input_paths: list[Path],
    row_filter: Callable[[Any], bool] | None = None,
) -> dict[str, dict[str, Any]]:
    distributions: dict[str, dict[str, Any]] = {}
    for input_path in input_paths:
        rows = load_json_rows(input_path)
        counts, considered_rows, skipped_rows = count_a_levels(rows, row_filter)
        if row_filter is None and rows and distribution_total(counts) == 0:
            raise ValueError(
                f"No valid {ASSISTANCE_LEVEL_FIELD}.a_level values found in "
                f"{input_path}. Expected one of {', '.join(A_LEVELS)}."
            )
        distributions[model_name_from_path(input_path)] = {
            "counts": counts,
            "input_path": input_path,
            "total_rows": len(rows),
            "considered_rows": considered_rows,
            "skipped_rows": skipped_rows,
        }
    return distributions


def distribution_total(counts: Counter[str]) -> int:
    return sum(counts[a_level] for a_level in A_LEVELS)


def distribution_percentages(summary: dict[str, Any]) -> dict[str, float]:
    counts = summary["counts"]
    total = distribution_total(counts)
    if total == 0:
        return {a_level: 0.0 for a_level in A_LEVELS}
    return {a_level: 100.0 * counts[a_level] / total for a_level in A_LEVELS}


def summary_percentages(summary: dict[str, Any]) -> dict[str, float]:
    if "percentages" in summary:
        return summary["percentages"]
    return distribution_percentages(summary)


def build_conditional_unsafe_rate_distributions(
    all_distributions: dict[str, dict[str, Any]],
    unsafe_distributions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    conditional_distributions: dict[str, dict[str, Any]] = {}

    for model_name, all_summary in all_distributions.items():
        unsafe_summary = unsafe_distributions.get(model_name)
        if unsafe_summary is None:
            raise ValueError(f"Missing unsafe distribution for {model_name}.")

        all_counts = all_summary["counts"]
        unsafe_counts = unsafe_summary["counts"]
        conditional_distributions[model_name] = {
            "percentages": {
                a_level: (
                    0.0
                    if all_counts[a_level] == 0
                    else 100.0 * unsafe_counts[a_level] / all_counts[a_level]
                )
                for a_level in A_LEVELS
            }
        }

    return conditional_distributions


def single_model_summary(
    distributions: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if len(distributions) != 1:
        raise ValueError(
            f"Expected exactly one model distribution, got {len(distributions)}."
        )
    return next(iter(distributions.items()))


def print_count_values(
    named_distributions: list[tuple[str, dict[str, dict[str, Any]]]],
) -> None:
    print("\nCount Values")
    print("=" * 100)
    header = (
        f"{'metric':<60} {'rows':>7} {'kept':>7} {'used':>7} {'missing':>8} "
        + " ".join(f"{a_level:>7}" for a_level in A_LEVELS)
    )
    print(header)
    print("-" * 100)

    for title, distributions in named_distributions:
        _, summary = single_model_summary(distributions)
        counts = summary["counts"]
        used_rows = distribution_total(counts)
        print(
            f"{title:<60} "
            f"{summary['total_rows']:>7} "
            f"{summary['considered_rows']:>7} "
            f"{used_rows:>7} "
            f"{summary['skipped_rows']:>8} "
            + " ".join(f"{counts[a_level]:>7}" for a_level in A_LEVELS)
        )


def print_percentage_values(
    named_distributions: list[tuple[str, dict[str, dict[str, Any]]]],
) -> None:
    print("\nPercentage Values")
    print("=" * 100)
    print(
        f"{'metric':<60} "
        + " ".join(f"{a_level + ' (%)':>11}" for a_level in A_LEVELS)
    )
    print("-" * 100)

    for title, distributions in named_distributions:
        _, summary = single_model_summary(distributions)
        percentages = summary_percentages(summary)
        print(
            f"{title:<60} "
            + " ".join(f"{percentages[a_level]:>11.2f}" for a_level in A_LEVELS)
        )


def run_response_mode_reporting(
    annotated_input_path: str | Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    input_path = resolve_path(annotated_input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Annotated JSON does not exist: {input_path}")

    input_paths = [input_path]
    all_distributions = build_distributions(input_paths)
    unsafe_distributions = build_distributions(
        input_paths,
        row_filter=keep_harmful_dual_use,
    )
    conditional_unsafe_rate_distributions = (
        build_conditional_unsafe_rate_distributions(
            all_distributions,
            unsafe_distributions,
        )
    )

    count_distributions = [
        ("Assistance-level distribution", all_distributions),
        ("Conditional unsafe rate", unsafe_distributions),
    ]
    percentage_distributions = [
        ("Assistance-level distribution", all_distributions),
        ("Conditional unsafe rate", conditional_unsafe_rate_distributions),
    ]
    print_count_values(count_distributions)
    print_percentage_values(percentage_distributions)

    return {
        "all": all_distributions,
        "dual_use_harmfulness_gt_0": unsafe_distributions,
        "conditional_unsafe_rate": conditional_unsafe_rate_distributions,
    }


def run_response_mode_analysis(
    model_name: str,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    limit: int | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    vertex_project_id: str = DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
    force_reclassify: bool = False,
) -> dict[str, Any]:
    resolved_input_path = (
        default_model_response_input_path(model_name)
        if input_path is None
        else resolve_path(input_path)
    )
    resolved_output_path = (
        default_output_path(resolved_input_path)
        if output_path is None
        else resolve_path(output_path)
    )

    classification_result = None
    if resolved_output_path.exists() and not resolved_output_path.is_file():
        raise FileExistsError(
            f"Annotated output path exists but is not a file: {resolved_output_path}"
        )

    if resolved_output_path.is_file() and not force_reclassify:
        print(f"Found existing assistance-level annotations: {resolved_output_path.resolve()}")
        if limit is not None:
            print(
                "Existing annotations found; --limit is ignored unless "
                "--force-reclassify is set."
            )
    else:
        if not resolved_input_path.is_file():
            raise FileNotFoundError(f"Input JSON does not exist: {resolved_input_path}")

        if resolved_output_path.is_file() and force_reclassify:
            print(
                "Reclassifying assistance levels and overwriting: "
                f"{resolved_output_path.resolve()}"
            )
        else:
            print(
                "No assistance-level annotations found. Creating: "
                f"{resolved_output_path.resolve()}"
            )

        classification_result = classify_assistance_levels(
            input_path=resolved_input_path,
            output_path=resolved_output_path,
            limit=limit,
            judge_model=judge_model,
            vertex_project_id=vertex_project_id,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
        )

    print(f"\nReporting assistance-level metrics from: {resolved_output_path.resolve()}")
    reporting_result = run_response_mode_reporting(resolved_output_path)

    return {
        "model_name": model_name,
        "input_path": resolved_input_path,
        "annotated_output_path": resolved_output_path,
        "classification": classification_result,
        "reporting": reporting_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate dual-use response assistance levels if needed, then report "
            "response-mode metrics."
        )
    )
    parser.add_argument(
        "model_name",
        help=(
            "Model response file stem. Defaults input to "
            "pilot_dataset/model_responses/<model_name>_scored.json."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Scored model response JSON path. Overrides the model_name default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Annotated JSON output path. Defaults to "
            "analysis/assistance_level/<input filename>."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For debugging, classify only the first N datapoints when annotating.",
    )
    parser.add_argument(
        "--force-reclassify",
        "--force",
        action="store_true",
        dest="force_reclassify",
        help="Rerun judge annotation even if the annotated output already exists.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Vertex model alias used as the assistance-level judge.",
    )
    parser.add_argument(
        "--vertex-project-id",
        default=DEFAULT_VERTEX_PROJECT_ID,
        help="Google Cloud project ID used for Vertex judge calls.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help="Maximum tokens to request for each judge response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for judge calls.",
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


def main() -> None:
    args = parse_args()
    run_response_mode_analysis(
        model_name=args.model_name,
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        judge_model=args.judge_model,
        vertex_project_id=args.vertex_project_id,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        retry_max_seconds=args.retry_max_seconds,
        force_reclassify=args.force_reclassify,
    )


if __name__ == "__main__":
    main()
