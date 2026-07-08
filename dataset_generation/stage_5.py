import sys
import string
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import Counter, defaultdict


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
    from .utils import (
        MISSING_VALUE,
        extract_json_object_with_keys,
        get_generation_value,
        load_json,
        load_text,
        save_json,
    )
except ImportError:
    from utils import (
        MISSING_VALUE,
        extract_json_object_with_keys,
        get_generation_value,
        load_json,
        load_text,
        save_json,
    )


PROMPT_DIR = REPO_ROOT / "prompts"
QUALITY_CHECK_PROMPT_PATH = PROMPT_DIR / "data_generation" / "stage_5.txt"
HARM_DOMAIN_DEFINITIONS_PATH = PROMPT_DIR / "definitions" / "harm_domain.json"
TASK_TYPE_DEFINITIONS_PATH = PROMPT_DIR / "definitions" / "task_type.json"
STAGE_4_OUTPUT_DIR = DATASET_OUTPUT_DIR / "stage_4"
STAGE_4_INPUT_PATH = STAGE_4_OUTPUT_DIR / "stage_4_passed.json"
STAGE_5_OUTPUT_DIR = DATASET_OUTPUT_DIR / "stage_5"
STAGE_5_PASSED_OUTPUT_PATH = STAGE_5_OUTPUT_DIR / "stage_5_passed.json"
STAGE_5_DEDUPLICATED_OUTPUT_PATH = STAGE_5_OUTPUT_DIR / "stage_5_deduplicated.json"
DEFAULT_MAX_COMPLETION_TOKENS = 512
MAX_ROUGE_L_SIMILARITY = 0.7
PROMPT_FIELDS = ("benign_prompt", "dual_use_prompt", "malicious_prompt")
VALID_QUALITY_DECISIONS = {"keep", "drop"}
REQUIRED_QUALITY_RESPONSE_KEYS = ("decision", "failure_reason")
PUNCTUATION_TRANSLATION = str.maketrans("", "", string.punctuation)


def load_stage_4_passed(input_path=STAGE_4_INPUT_PATH, limit=None):
    rows = load_json(input_path)
    return rows[:limit] if limit is not None else rows


def load_prompt_template(prompt_path=QUALITY_CHECK_PROMPT_PATH):
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


def get_definition_text(definitions, label):
    definition = definitions[label]
    if isinstance(definition, str):
        return definition

    return definition.get("definition", "")


def get_required_generation_value(row, field):
    value = get_generation_value(row, field)
    if value == MISSING_VALUE:
        raise ValueError(f"Datapoint is missing generation_metadata.{field}")
    return value


def get_required_generated_prompt(row, prompt_field):
    generated_prompts = row.get("generated_prompts", {})
    if not isinstance(generated_prompts, dict):
        raise ValueError("Datapoint generated_prompts was not an object.")

    prompt = generated_prompts.get(prompt_field)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Datapoint is missing generated_prompts.{prompt_field}")

    return prompt


def build_quality_check_prompt(
    row,
    prompt_template=None,
    harm_domain_definitions=None,
    task_type_definitions=None,
):
    prompt_template = prompt_template or load_prompt_template()
    harm_domain_definitions = harm_domain_definitions or load_json(
        HARM_DOMAIN_DEFINITIONS_PATH
    )
    task_type_definitions = task_type_definitions or load_json(
        TASK_TYPE_DEFINITIONS_PATH
    )

    harm_domain = get_required_generation_value(row, "harm_domain")
    task_type = get_required_generation_value(row, "task_type")
    if harm_domain not in harm_domain_definitions:
        raise ValueError(f"Unknown harm_domain {harm_domain!r}")
    if task_type not in task_type_definitions:
        raise ValueError(f"Unknown task_type {task_type!r}")

    return (
        prompt_template.replace("{{HARM_DOMAIN}}", harm_domain)
        .replace(
            "{{HARM_DOMAIN_DEFINITION}}",
            get_definition_text(harm_domain_definitions, harm_domain),
        )
        .replace("{{TASK_TYPE}}", task_type)
        .replace(
            "{{TASK_TYPE_DEFINITION}}",
            get_definition_text(task_type_definitions, task_type),
        )
        .replace(
            "{{BENIGN_PROMPT}}",
            get_required_generated_prompt(row, "benign_prompt"),
        )
        .replace(
            "{{DUAL_USE_PROMPT}}",
            get_required_generated_prompt(row, "dual_use_prompt"),
        )
        .replace(
            "{{MALICIOUS_PROMPT}}",
            get_required_generated_prompt(row, "malicious_prompt"),
        )
    )


def parse_quality_check_response(raw_response):
    quality_check = extract_json_object_with_keys(
        raw_response,
        required_keys=REQUIRED_QUALITY_RESPONSE_KEYS,
    )
    decision = quality_check["decision"]
    if isinstance(decision, str):
        decision = decision.strip().lower()
    if decision not in VALID_QUALITY_DECISIONS:
        raise ValueError(f"Quality-check response contained unknown decision {decision!r}")

    failure_reason = quality_check["failure_reason"]
    if not isinstance(failure_reason, str) or not failure_reason.strip():
        raise ValueError("Quality-check response contained an empty failure_reason.")

    return {
        "decision": decision,
        "failure_reason": failure_reason.strip(),
    }


def call_llm_for_quality_check(
    row,
    prompt_template=None,
    harm_domain_definitions=None,
    task_type_definitions=None,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
):
    prompt = build_quality_check_prompt(
        row=row,
        prompt_template=prompt_template,
        harm_domain_definitions=harm_domain_definitions,
        task_type_definitions=task_type_definitions,
    )
    raw_response = get_required_api_response(
        prompt,
        max_completion_tokens=max_completion_tokens,
    )
    return parse_quality_check_response(raw_response)


def add_quality_check(row, quality_check):
    output_row = deepcopy(row)
    output_row["additional_checks"] = dict(quality_check)
    return output_row


def process_stage_5_rows(rows, max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS):
    prompt_template = load_prompt_template()
    harm_domain_definitions = load_json(HARM_DOMAIN_DEFINITIONS_PATH)
    task_type_definitions = load_json(TASK_TYPE_DEFINITIONS_PATH)
    passed_rows = []
    stats = Counter()

    for index, row in enumerate(tqdm(rows, desc="Quality-checking prompt triplets")):
        try:
            quality_check = call_llm_for_quality_check(
                row=row,
                prompt_template=prompt_template,
                harm_domain_definitions=harm_domain_definitions,
                task_type_definitions=task_type_definitions,
                max_completion_tokens=max_completion_tokens,
            )
            stats[f"decision:{quality_check['decision']}"] += 1
            if quality_check["decision"] == "keep":
                passed_rows.append(add_quality_check(row, quality_check))
        except Exception as error:
            stats["dropped_quality_check_error"] += 1
            tqdm.write(f"Dropped datapoint {index} after quality-check error: {error}")

    stats["input_rows"] = len(rows)
    stats["passed_rows"] = len(passed_rows)
    return passed_rows, stats


def normalize_prompt_for_similarity(prompt):
    lowered = prompt.lower().translate(PUNCTUATION_TRANSLATION)
    return " ".join(lowered.split())


def tokenize_for_rouge(prompt):
    normalized_prompt = normalize_prompt_for_similarity(prompt)
    return normalized_prompt.split()


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


def get_bucket_key(row):
    return (
        get_required_generation_value(row, "harm_domain"),
        get_required_generation_value(row, "task_type"),
    )


def get_triplet_similarity_text(row):
    return "\n".join(
        get_required_generated_prompt(row, prompt_field)
        for prompt_field in PROMPT_FIELDS
    )


def deduplicate_rows(rows, max_rouge_l_similarity=MAX_ROUGE_L_SIMILARITY):
    retained_rows = []
    retained_texts_by_bucket = defaultdict(list)
    stats = Counter()

    for row in rows:
        bucket_key = get_bucket_key(row)
        candidate_text = get_triplet_similarity_text(row)
        existing_texts = retained_texts_by_bucket[bucket_key]
        max_similarity = max(
            (rouge_l_score(candidate_text, existing_text) for existing_text in existing_texts),
            default=0.0,
        )
        if max_similarity >= max_rouge_l_similarity:
            stats["dropped_as_duplicate"] += 1
            continue

        retained_rows.append(row)
        retained_texts_by_bucket[bucket_key].append(candidate_text)

    stats["input_rows"] = len(rows)
    stats["deduplicated_rows"] = len(retained_rows)
    return retained_rows, stats


def calculate_label_distributions(rows):
    return {
        "harm_domain": Counter(get_generation_value(row, "harm_domain") for row in rows),
        "task_type": Counter(get_generation_value(row, "task_type") for row in rows),
    }


def print_label_distributions(rows, title):
    distributions = calculate_label_distributions(rows)
    total = len(rows)
    print(f"\n{title}")
    print("-" * len(title))
    print(f"total: {total}")

    for field, counts in distributions.items():
        print(f"\n{field} Distribution")
        print("-" * (len(field) + len(" Distribution")))
        for value, count in counts.most_common():
            percentage = (count / total) * 100 if total else 0
            print(f"{value}: {count} ({percentage:.1f}%)")


def print_processing_summary(quality_stats, deduplication_stats):
    print("\nStage 5 Processing Summary")
    print("-" * 40)
    for field in (
        "input_rows",
        "decision:keep",
        "decision:drop",
        "dropped_quality_check_error",
        "passed_rows",
    ):
        print(f"{field}: {quality_stats.get(field, 0)}")
    print(f"dropped_as_duplicate: {deduplication_stats.get('dropped_as_duplicate', 0)}")
    print(f"deduplicated_rows: {deduplication_stats.get('deduplicated_rows', 0)}")


def generate_stage_5_dataset(
    input_path=STAGE_4_INPUT_PATH,
    passed_output_path=STAGE_5_PASSED_OUTPUT_PATH,
    deduplicated_output_path=STAGE_5_DEDUPLICATED_OUTPUT_PATH,
    limit=None,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    max_rouge_l_similarity=MAX_ROUGE_L_SIMILARITY,
):
    rows = load_stage_4_passed(input_path=input_path, limit=limit)
    passed_rows, quality_stats = process_stage_5_rows(
        rows,
        max_completion_tokens=max_completion_tokens,
    )
    save_json(passed_rows, passed_output_path)
    print(f"Saved stage 5 passed outputs to: {Path(passed_output_path).resolve()}")
    print_label_distributions(passed_rows, "Stage 5 Passed Datapoints")

    deduplicated_rows, deduplication_stats = deduplicate_rows(
        passed_rows,
        max_rouge_l_similarity=max_rouge_l_similarity,
    )
    save_json(deduplicated_rows, deduplicated_output_path)
    print(
        "Saved stage 5 deduplicated outputs to: "
        f"{Path(deduplicated_output_path).resolve()}"
    )
    print_label_distributions(deduplicated_rows, "Stage 5 Deduplicated Datapoints")
    print_processing_summary(quality_stats, deduplication_stats)

    return {
        "passed_rows": passed_rows,
        "deduplicated_rows": deduplicated_rows,
        "quality_stats": dict(quality_stats),
        "deduplication_stats": dict(deduplication_stats),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quality-check and deduplicate Stage 4 prompt triplets."
    )
    parser.add_argument("--input", type=Path, default=STAGE_4_INPUT_PATH)
    parser.add_argument(
        "--passed-output",
        type=Path,
        default=STAGE_5_PASSED_OUTPUT_PATH,
    )
    parser.add_argument(
        "--deduplicated-output",
        type=Path,
        default=STAGE_5_DEDUPLICATED_OUTPUT_PATH,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For debugging, process only the first N datapoints.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help="Maximum tokens to request from the API model for each quality check.",
    )
    parser.add_argument(
        "--max-rouge-l-similarity",
        type=float,
        default=MAX_ROUGE_L_SIMILARITY,
        help="Drop later same-bucket triplets at or above this Rouge-L score.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_stage_5_dataset(
        input_path=args.input,
        passed_output_path=args.passed_output,
        deduplicated_output_path=args.deduplicated_output,
        limit=args.limit,
        max_completion_tokens=args.max_completion_tokens,
        max_rouge_l_similarity=args.max_rouge_l_similarity,
    )
