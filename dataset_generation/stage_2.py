import sys
import argparse
from tqdm import tqdm
from pathlib import Path
from itertools import product
from collections import Counter


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_calls.api_models import get_api_responses_batch
from OpenSafeIntent.project_config import (
    DATASET_OUTPUT_DIR,
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_COMPLETION_TOKENS,
)

try:
    from .utils import (
        extract_json_object_with_keys,
        get_generation_value,
        load_json,
        load_text,
        normalize_datapoint,
        save_json,
    )
except ImportError:
    from utils import (
        extract_json_object_with_keys,
        get_generation_value,
        load_json,
        load_text,
        normalize_datapoint,
        save_json,
    )


TOTAL_PER_COMBINATION = 100  # 5
DEFAULT_N_SUMMARIES = 10  # 2
DEFAULT_MAX_BACKFILL_CALLS = 10
DEFAULT_MAX_COMPLETION_TOKENS = MAX_COMPLETION_TOKENS
MAX_ROUGE_L_SIMILARITY = 0.7
PROMPT_DIR = REPO_ROOT / "prompts"
DATA_GENERATION_PROMPT_DIR = PROMPT_DIR / "data_generation"
BALANCING_PROMPT_PATH = DATA_GENERATION_PROMPT_DIR / "stage_2.txt"
HARM_DOMAIN_DEFINITIONS_PATH = PROMPT_DIR / "definitions" / "harm_domain.json"
TASK_TYPE_DEFINITIONS_PATH = PROMPT_DIR / "definitions" / "task_type.json"
STAGE_1_OUTPUT_DIR = DATASET_OUTPUT_DIR / "stage_1"
STAGE_1_OUTPUT_PATH = STAGE_1_OUTPUT_DIR / "stage_1_outputs.json"
STAGE_2_OUTPUT_DIR = DATASET_OUTPUT_DIR / "stage_2"
BALANCED_OUTPUT_PATH = STAGE_2_OUTPUT_DIR / "stage_2_outputs.json"

HARM_DOMAINS = tuple(load_json(HARM_DOMAIN_DEFINITIONS_PATH).keys())
TASK_TYPES = tuple(load_json(TASK_TYPE_DEFINITIONS_PATH).keys())


def load_stage_1_outputs(input_path=STAGE_1_OUTPUT_PATH):
    return [normalize_datapoint(row, backfilled=False) for row in load_json(input_path)]


def load_prompt_template(prompt_path=BALANCING_PROMPT_PATH):
    return load_text(prompt_path)


def get_required_api_response(
    prompt,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
):
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


def get_label_pair(row):
    return (
        get_generation_value(row, "task_type"),
        get_generation_value(row, "harm_domain"),
    )


def get_topic_summary(row):
    value = get_generation_value(row, "topic_summary")
    return "" if value == "MISSING" else value


def format_output_datapoint(row):
    return normalize_datapoint(row)


def format_output_datapoints(rows):
    return [format_output_datapoint(row) for row in rows]


def tokenize_for_rouge(text):
    return text.lower().split()


def longest_common_subsequence_length(left_tokens, right_tokens):
    previous = [0] * (len(right_tokens) + 1)

    for left_token in left_tokens:
        current = [0]
        for index, right_token in enumerate(right_tokens, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current

    return previous[-1]


def rouge_l_score(candidate, reference):
    candidate_tokens = tokenize_for_rouge(candidate)
    reference_tokens = tokenize_for_rouge(reference)

    if not candidate_tokens or not reference_tokens:
        return 0.0

    lcs_length = longest_common_subsequence_length(candidate_tokens, reference_tokens)
    precision = lcs_length / len(candidate_tokens)
    recall = lcs_length / len(reference_tokens)

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


def max_rouge_l_score(candidate_summary, existing_summaries):
    if not candidate_summary or not existing_summaries:
        return 0.0

    return max(
        rouge_l_score(candidate_summary, existing_summary)
        for existing_summary in existing_summaries
    )


def get_definition_text(definitions, label):
    definition = definitions[label]
    if isinstance(definition, str):
        return definition

    return definition.get("definition", "")


def format_existing_summaries(existing_summaries):
    if not existing_summaries:
        return "None yet."

    return "\n".join(
        f"{index}. {summary}"
        for index, summary in enumerate(existing_summaries, start=1)
    )


def build_backfill_prompt(
    harm_domain,
    task_type,
    existing_summaries=None,
    n_summaries=DEFAULT_N_SUMMARIES,
    prompt_path=BALANCING_PROMPT_PATH,
    harm_domain_definitions_path=HARM_DOMAIN_DEFINITIONS_PATH,
    task_type_definitions_path=TASK_TYPE_DEFINITIONS_PATH,
):
    prompt_template = load_prompt_template(prompt_path)
    harm_domain_definitions = load_json(harm_domain_definitions_path)
    task_type_definitions = load_json(task_type_definitions_path)

    return (
        prompt_template
        .replace("{{HARM_DOMAIN}}", harm_domain)
        .replace(
            "{{HARM_DOMAIN_DEFINITION}}",
            get_definition_text(harm_domain_definitions, harm_domain),
        )
        .replace("{{TASK_TYPE}}", task_type)
        .replace(
            "{{TASK_TYPE_DEFINITION}}",
            get_definition_text(task_type_definitions, task_type),
        )
        .replace("{{N_SUMMARIES}}", str(n_summaries))
        .replace(
            "{{EXISTING_SUMMARIES}}",
            format_existing_summaries(existing_summaries or []),
        )
    )


def parse_backfill_response(raw_response):
    response_json = extract_json_object_with_keys(raw_response, required_keys=("summaries",))
    summaries = response_json.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError(f"Backfill response did not contain a summaries list: {raw_response!r}")

    return summaries


def summary_to_datapoint(summary, harm_domain, task_type):
    topic_summary = summary.get("topic_summary", "")
    if not topic_summary:
        raise ValueError(f"Generated summary is missing topic_summary: {summary!r}")

    return normalize_datapoint(
        {
            "origin_metadata": {
                "source_dataset": "llm_backfill",
                "source_split": "stage_1_2_balancing",
                "unsafe_prompt": "",
            },
            "generation_metadata": {
                "topic_summary": topic_summary,
                "is_prompt_safe": "False",
                "harm_domain": harm_domain,
                "task_type": task_type,
                "backfilled": True,
            },
        }
    )


def call_llm_for_backfill(
    harm_domain,
    task_type,
    existing_summaries=None,
    n_summaries=DEFAULT_N_SUMMARIES,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
):
    """Call an LLM to generate summary JSON for an underrepresented label pair."""
    prompt = build_backfill_prompt(
        harm_domain=harm_domain,
        task_type=task_type,
        existing_summaries=existing_summaries,
        n_summaries=n_summaries,
    )
    raw_response = get_required_api_response(
        prompt,
        max_completion_tokens=max_completion_tokens,
    )
    return parse_backfill_response(raw_response)


def generate_candidate_backfills(
    harm_domain,
    task_type,
    existing_summaries=None,
    n_summaries=DEFAULT_N_SUMMARIES,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
):
    summaries = call_llm_for_backfill(
        harm_domain=harm_domain,
        task_type=task_type,
        existing_summaries=existing_summaries,
        n_summaries=n_summaries,
        max_completion_tokens=max_completion_tokens,
    )
    return [
        summary_to_datapoint(summary, harm_domain, task_type)
        for summary in summaries
    ]


def filter_by_topic_summary_similarity(
    generated_rows,
    existing_datapoints=None,
    max_rouge_l_similarity=MAX_ROUGE_L_SIMILARITY,
):
    existing_summaries = [
        summary
        for summary in (get_topic_summary(row) for row in existing_datapoints or [])
        if summary
    ]
    filtered_rows = []

    for row in generated_rows:
        topic_summary = get_topic_summary(row)
        if max_rouge_l_score(topic_summary, existing_summaries) > max_rouge_l_similarity:
            continue

        filtered_rows.append(row)
        if topic_summary:
            existing_summaries.append(topic_summary)

    return filtered_rows


def generate_backfill_datapoints(
    harm_domain,
    task_type,
    existing_datapoints=None,
    max_rouge_l_similarity=MAX_ROUGE_L_SIMILARITY,
    n_summaries=DEFAULT_N_SUMMARIES,
    total_per_combination=TOTAL_PER_COMBINATION,
    max_backfill_calls=DEFAULT_MAX_BACKFILL_CALLS,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
):
    label_pair = (task_type, harm_domain)
    existing_pair_rows = [
        row
        for row in existing_datapoints or []
        if get_label_pair(row) == label_pair
    ]
    accepted_rows = []

    if len(existing_pair_rows) >= total_per_combination:
        return accepted_rows

    for attempt in range(max_backfill_calls):
        working_pair_rows = [*existing_pair_rows, *accepted_rows]
        working_datapoints = [*(existing_datapoints or []), *accepted_rows]
        existing_summaries = [
            summary
            for summary in (get_topic_summary(row) for row in working_pair_rows)
            if summary
        ]
        try:
            generated_rows = generate_candidate_backfills(
                harm_domain=harm_domain,
                task_type=task_type,
                existing_summaries=existing_summaries,
                n_summaries=n_summaries,
                max_completion_tokens=max_completion_tokens,
            )
        except Exception as error:
            tqdm.write(
                "Backfill call "
                f"{attempt + 1}/{max_backfill_calls} failed for "
                f"{task_type!r}/{harm_domain!r}: {error}"
            )
            continue

        filtered_rows = filter_by_topic_summary_similarity(
            generated_rows=generated_rows,
            existing_datapoints=working_datapoints,
            max_rouge_l_similarity=max_rouge_l_similarity,
        )
        accepted_rows.extend(filtered_rows)

        current_total = len(existing_pair_rows) + len(accepted_rows)
        if current_total >= total_per_combination:
            needed = total_per_combination - len(existing_pair_rows)
            return accepted_rows[:needed]

    raise ValueError(
        "Backfill generation accepted "
        f"{len(accepted_rows)} new rows for {task_type!r}/{harm_domain!r}; "
        f"{len(existing_pair_rows) + len(accepted_rows)} total rows remain below "
        f"the target of {total_per_combination} after {max_backfill_calls} LLM calls."
    )


def mark_original_datapoints(rows):
    return [normalize_datapoint(row, backfilled=False) for row in rows]


def mark_backfilled_datapoints(rows):
    return [normalize_datapoint(row, backfilled=True) for row in rows]


def calculate_balancing_stats(
    original_counts,
    generated_counts,
    harm_domains=HARM_DOMAINS,
    task_types=TASK_TYPES,
):
    label_pairs = list(product(task_types, harm_domains))
    original_total = sum(original_counts[label_pair] for label_pair in label_pairs)
    generated_total = sum(generated_counts[label_pair] for label_pair in label_pairs)
    total_datapoints = sum(
        original_counts[(task_type, harm_domain)]
        + generated_counts[(task_type, harm_domain)]
        for task_type, harm_domain in label_pairs
    )
    stats = []
    harm_domain_stats = []
    task_type_stats = []

    for task_type, harm_domain in label_pairs:
        original_count = original_counts[(task_type, harm_domain)]
        generated_count = generated_counts[(task_type, harm_domain)]
        total_count = original_count + generated_count
        percentage = (total_count / total_datapoints) * 100 if total_datapoints else 0

        stats.append(
            {
                "task_type": task_type,
                "harm_domain": harm_domain,
                "original_count": original_count,
                "generated_count": generated_count,
                "total_count": total_count,
                "percentage_of_all": round(percentage, 2),
            }
        )

    for harm_domain in harm_domains:
        original_count = sum(
            original_counts[(task_type, harm_domain)]
            for task_type in task_types
        )
        generated_count = sum(
            generated_counts[(task_type, harm_domain)]
            for task_type in task_types
        )
        total_count = original_count + generated_count
        percentage = (total_count / total_datapoints) * 100 if total_datapoints else 0

        harm_domain_stats.append(
            {
                "harm_domain": harm_domain,
                "original_count": original_count,
                "generated_count": generated_count,
                "total_count": total_count,
                "percentage_of_all": round(percentage, 2),
            }
        )

    for task_type in task_types:
        original_count = sum(
            original_counts[(task_type, harm_domain)]
            for harm_domain in harm_domains
        )
        generated_count = sum(
            generated_counts[(task_type, harm_domain)]
            for harm_domain in harm_domains
        )
        total_count = original_count + generated_count
        percentage = (total_count / total_datapoints) * 100 if total_datapoints else 0

        task_type_stats.append(
            {
                "task_type": task_type,
                "original_count": original_count,
                "generated_count": generated_count,
                "total_count": total_count,
                "percentage_of_all": round(percentage, 2),
            }
        )

    return {
        "original_total": original_total,
        "generated_total": generated_total,
        "total_datapoints": total_datapoints,
        "rows": stats,
        "harm_domain_rows": harm_domain_stats,
        "task_type_rows": task_type_stats,
    }


def print_balancing_stats(stats):
    print("\nBalancing Stats")
    print("-" * 120)
    print(
        f"{'Task Type':<30} "
        f"{'Harm Domain':<30} "
        f"{'Original':>10} "
        f"{'Generated':>10} "
        f"{'Total':>10} "
        f"{'% of All':>10}"
    )
    print("-" * 120)

    for row in stats["rows"]:
        print(
            f"{row['task_type']:<30} "
            f"{row['harm_domain']:<30} "
            f"{row['original_count']:>10} "
            f"{row['generated_count']:>10} "
            f"{row['total_count']:>10} "
            f"{row['percentage_of_all']:>9.2f}%"
        )

    print("-" * 120)
    print(
        f"{'TOTAL':<61} "
        f"{stats['original_total']:>10} "
        f"{stats['generated_total']:>10} "
        f"{stats['total_datapoints']:>10} "
        f"{100 if stats['total_datapoints'] else 0:>9.2f}%"
    )

    print("\nCounts by Harm Domain")
    print("-" * 90)
    print(
        f"{'Harm Domain':<30} "
        f"{'Original':>10} "
        f"{'Generated':>10} "
        f"{'Total':>10} "
        f"{'% of All':>10}"
    )
    print("-" * 90)

    for row in stats["harm_domain_rows"]:
        print(
            f"{row['harm_domain']:<30} "
            f"{row['original_count']:>10} "
            f"{row['generated_count']:>10} "
            f"{row['total_count']:>10} "
            f"{row['percentage_of_all']:>9.2f}%"
        )

    print("-" * 90)
    print(
        f"{'TOTAL':<30} "
        f"{stats['original_total']:>10} "
        f"{stats['generated_total']:>10} "
        f"{stats['total_datapoints']:>10} "
        f"{100 if stats['total_datapoints'] else 0:>9.2f}%"
    )

    print("\nCounts by Task Type")
    print("-" * 90)
    print(
        f"{'Task Type':<30} "
        f"{'Original':>10} "
        f"{'Generated':>10} "
        f"{'Total':>10} "
        f"{'% of All':>10}"
    )
    print("-" * 90)

    for row in stats["task_type_rows"]:
        print(
            f"{row['task_type']:<30} "
            f"{row['original_count']:>10} "
            f"{row['generated_count']:>10} "
            f"{row['total_count']:>10} "
            f"{row['percentage_of_all']:>9.2f}%"
        )

    print("-" * 90)
    print(
        f"{'TOTAL':<30} "
        f"{stats['original_total']:>10} "
        f"{stats['generated_total']:>10} "
        f"{stats['total_datapoints']:>10} "
        f"{100 if stats['total_datapoints'] else 0:>9.2f}%"
    )


def report_balancing_stats(
    original_counts,
    generated_counts,
    harm_domains=HARM_DOMAINS,
    task_types=TASK_TYPES,
):
    stats = calculate_balancing_stats(
        original_counts=original_counts,
        generated_counts=generated_counts,
        harm_domains=harm_domains,
        task_types=task_types,
    )
    print_balancing_stats(stats)
    return stats["rows"]


def balance_stage_1_outputs(
    input_path=STAGE_1_OUTPUT_PATH,
    output_path=BALANCED_OUTPUT_PATH,
    total_per_combination=TOTAL_PER_COMBINATION,
    n_summaries=DEFAULT_N_SUMMARIES,
    max_backfill_calls=DEFAULT_MAX_BACKFILL_CALLS,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    strict_backfill=False,
    harm_domains=HARM_DOMAINS,
    task_types=TASK_TYPES,
):
    original_rows = load_stage_1_outputs(input_path)
    balanced_rows = mark_original_datapoints(original_rows)
    original_counts = Counter(get_label_pair(row) for row in original_rows)
    counts = Counter(original_counts)
    generated_counts = Counter()

    label_pairs = list(product(task_types, harm_domains))
    for task_type, harm_domain in tqdm(label_pairs, desc="Balancing label pairs"):
        current_count = counts[(task_type, harm_domain)]
        if current_count >= total_per_combination:
            continue

        try:
            generated_rows = generate_backfill_datapoints(
                harm_domain=harm_domain,
                task_type=task_type,
                existing_datapoints=balanced_rows,
                n_summaries=n_summaries,
                total_per_combination=total_per_combination,
                max_backfill_calls=max_backfill_calls,
                max_completion_tokens=max_completion_tokens,
            )
        except Exception as error:
            if strict_backfill:
                raise
            tqdm.write(
                "Skipping backfill for "
                f"{task_type!r}/{harm_domain!r} after generation error: {error}"
            )
            generated_rows = []
            generated_counts[(task_type, harm_domain)] = 0

        balanced_rows.extend(mark_backfilled_datapoints(generated_rows))
        counts[(task_type, harm_domain)] += len(generated_rows)
        generated_counts[(task_type, harm_domain)] = len(generated_rows)

    output_rows = format_output_datapoints(balanced_rows)
    save_json(output_rows, output_path)
    print(f"Saved balanced stage 2 outputs to: {Path(output_path).resolve()}")
    stats = report_balancing_stats(
        original_counts=original_counts,
        generated_counts=generated_counts,
        harm_domains=harm_domains,
        task_types=task_types,
    )

    return {
        "rows": output_rows,
        "counts": dict(counts),
        "backfill_summary": dict(generated_counts),
        "stats": stats,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Balance stage 1 outputs by label pair.")
    parser.add_argument("--input", type=Path, default=STAGE_1_OUTPUT_PATH)
    parser.add_argument("--output", type=Path, default=BALANCED_OUTPUT_PATH)
    parser.add_argument(
        "--total-per-combination",
        type=int,
        default=TOTAL_PER_COMBINATION,
    )
    parser.add_argument(
        "--n-summaries",
        type=int,
        default=DEFAULT_N_SUMMARIES,
        help="Number of summaries to request from the LLM in each backfill call.",
    )
    parser.add_argument(
        "--max-backfill-calls",
        type=int,
        default=DEFAULT_MAX_BACKFILL_CALLS,
        help="Maximum LLM calls to make for one underfilled label pair.",
    )
    parser.add_argument(
        "--strict-backfill",
        action="store_true",
        help="Abort if a label pair cannot be backfilled after the configured attempts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    balance_stage_1_outputs(
        input_path=args.input,
        output_path=args.output,
        total_per_combination=args.total_per_combination,
        n_summaries=args.n_summaries,
        max_backfill_calls=args.max_backfill_calls,
        strict_backfill=args.strict_backfill,
    )
