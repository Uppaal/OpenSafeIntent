import sys
import argparse
from tqdm import tqdm
from pathlib import Path
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
        normalize_datapoint,
        save_json,
    )
except ImportError:
    from utils import (
        MISSING_VALUE,
        extract_json_object_with_keys,
        get_generation_value,
        load_json,
        load_text,
        normalize_datapoint,
        save_json,
    )


PROMPT_DIR = REPO_ROOT / "prompts"
DATA_GENERATION_PROMPT_DIR = PROMPT_DIR / "data_generation"
STAGE_3_PROMPT_PATH = DATA_GENERATION_PROMPT_DIR / "stage_3.txt"
HARM_DOMAIN_DEFINITIONS_PATH = PROMPT_DIR / "definitions" / "harm_domain.json"
TASK_TYPE_DEFINITIONS_PATH = PROMPT_DIR / "definitions" / "task_type.json"
STAGE_2_OUTPUT_DIR = DATASET_OUTPUT_DIR / "stage_2"
STAGE_2_OUTPUT_PATH = STAGE_2_OUTPUT_DIR / "stage_2_outputs.json"
STAGE_2_TYPO_OUTPUT_PATH = STAGE_2_OUTPUT_DIR / "stage_2_outputs.json"
STAGE_3_OUTPUT_DIR = DATASET_OUTPUT_DIR / "stage_3"
STAGE_3_OUTPUT_PATH = STAGE_3_OUTPUT_DIR / "stage_3_outputs.json"
DEFAULT_MAX_COMPLETION_TOKENS = 1500

REQUIRED_RESPONSE_KEYS = (
    "underlying_topic",
    "benign_task",
    "dual_use_task",
    "malicious_task",
    "benign_prompt",
    "dual_use_prompt",
    "malicious_prompt",
    "self_check",
    "failure_notes",
)


def resolve_default_stage_2_input():
    if STAGE_2_OUTPUT_PATH.exists():
        return STAGE_2_OUTPUT_PATH
    if STAGE_2_TYPO_OUTPUT_PATH.exists():
        return STAGE_2_TYPO_OUTPUT_PATH
    return STAGE_2_OUTPUT_PATH


def load_stage_2_outputs(input_path=None, limit=None):
    input_path = Path(input_path or resolve_default_stage_2_input())
    rows = [normalize_datapoint(row) for row in load_json(input_path)]
    return rows[:limit] if limit is not None else rows


def load_prompt_template(prompt_path=STAGE_3_PROMPT_PATH):
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


def get_required_generation_value(row, key):
    value = get_generation_value(row, key)
    if value == MISSING_VALUE:
        raise ValueError(f"Datapoint is missing generation_metadata.{key}")
    return value


def build_stage_3_prompt(
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

    topic_summary = get_required_generation_value(row, "topic_summary")
    harm_domain = get_required_generation_value(row, "harm_domain")
    task_type = get_required_generation_value(row, "task_type")

    if harm_domain not in harm_domain_definitions:
        raise ValueError(f"Unknown harm_domain {harm_domain!r}")
    if task_type not in task_type_definitions:
        raise ValueError(f"Unknown task_type {task_type!r}")

    return (
        prompt_template.replace("{{TOPIC_SUMMARY}}", topic_summary)
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
    )


def normalize_failure_notes(failure_notes):
    if failure_notes is None:
        return []
    if isinstance(failure_notes, list):
        return failure_notes
    return [str(failure_notes)]


def parse_stage_3_response(raw_response):
    parsed = extract_json_object_with_keys(
        raw_response,
        required_keys=REQUIRED_RESPONSE_KEYS,
    )
    self_check = parsed.get("self_check")
    if not isinstance(self_check, dict):
        raise ValueError(f"Model response self_check was not an object: {raw_response!r}")

    parsed["failure_notes"] = normalize_failure_notes(parsed.get("failure_notes"))
    return parsed


def call_llm_for_prompt_triplet(
    row,
    prompt_template=None,
    harm_domain_definitions=None,
    task_type_definitions=None,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
):
    prompt = build_stage_3_prompt(
        row=row,
        prompt_template=prompt_template,
        harm_domain_definitions=harm_domain_definitions,
        task_type_definitions=task_type_definitions,
    )
    raw_response = get_required_api_response(
        prompt,
        max_completion_tokens=max_completion_tokens,
    )
    return parse_stage_3_response(raw_response)


def format_success_datapoint(row, generated_prompts):
    output_row = normalize_datapoint(row)
    output_row["generated_prompts"] = generated_prompts
    return output_row


def empty_generated_prompts_with_error(error):
    return {
        "underlying_topic": "",
        "benign_task": "",
        "dual_use_task": "",
        "malicious_task": "",
        "benign_prompt": "",
        "dual_use_prompt": "",
        "malicious_prompt": "",
        "self_check": {},
        "failure_notes": [f"stage_3_error: {error}"],
    }


def format_failure_datapoint(row, error):
    output_row = normalize_datapoint(row)
    output_row["generated_prompts"] = empty_generated_prompts_with_error(error)
    return output_row


def get_label_pair(row):
    return (
        get_generation_value(row, "task_type"),
        get_generation_value(row, "harm_domain"),
    )


def has_failure_notes(row):
    failure_notes = row.get("generated_prompts", {}).get("failure_notes", [])
    return bool(normalize_failure_notes(failure_notes))


def calculate_stage_3_stats(rows, harm_domains=None, task_types=None):
    status_counts = defaultdict(lambda: Counter({"succeeded": 0, "failed": 0}))
    self_check_counts = defaultdict(Counter)

    for harm_domain in harm_domains or []:
        for task_type in task_types or []:
            status_counts[(harm_domain, task_type)]

    for row in rows:
        task_type, harm_domain = get_label_pair(row)
        status = "failed" if has_failure_notes(row) else "succeeded"
        status_counts[(harm_domain, task_type)][status] += 1

        self_check = row.get("generated_prompts", {}).get("self_check", {})
        if isinstance(self_check, dict):
            for field, value in self_check.items():
                self_check_counts[field][str(value)] += 1

    return status_counts, self_check_counts


def print_stage_3_stats(status_counts, self_check_counts):
    print("\nStage 3 Success/Failure by Harm Domain x Task Type")
    print("-" * 100)
    print(
        f"{'Harm Domain':<30} "
        f"{'Task Type':<30} "
        f"{'Succeeded':>10} "
        f"{'Failed':>10} "
        f"{'Total':>10}"
    )
    print("-" * 100)

    for (harm_domain, task_type), counts in sorted(status_counts.items()):
        succeeded = counts["succeeded"]
        failed = counts["failed"]
        total = succeeded + failed
        print(
            f"{harm_domain:<30} "
            f"{task_type:<30} "
            f"{succeeded:>10} "
            f"{failed:>10} "
            f"{total:>10}"
        )

    print("\nSelf-Check Field Distribution")
    print("-" * 80)
    if not self_check_counts:
        print("No self_check fields found.")
        return

    for field, counts in sorted(self_check_counts.items()):
        total = sum(counts.values())
        print(f"\n{field}")
        for value, count in counts.most_common():
            percentage = (count / total) * 100 if total else 0
            print(f"  {value}: {count} ({percentage:.1f}%)")


def generate_stage_3_dataset(
    input_path=None,
    output_path=STAGE_3_OUTPUT_PATH,
    limit=None,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
):
    input_path = Path(input_path or resolve_default_stage_2_input())
    output_path = Path(output_path)
    rows = load_stage_2_outputs(input_path=input_path, limit=limit)
    prompt_template = load_prompt_template()
    harm_domain_definitions = load_json(HARM_DOMAIN_DEFINITIONS_PATH)
    task_type_definitions = load_json(TASK_TYPE_DEFINITIONS_PATH)

    output_rows = []
    for row in tqdm(rows, desc="Generating prompt triplets"):
        try:
            generated_prompts = call_llm_for_prompt_triplet(
                row=row,
                prompt_template=prompt_template,
                harm_domain_definitions=harm_domain_definitions,
                task_type_definitions=task_type_definitions,
                max_completion_tokens=max_completion_tokens,
            )
            output_rows.append(format_success_datapoint(row, generated_prompts))
        except Exception as error:
            output_rows.append(format_failure_datapoint(row, error))

    save_json(output_rows, output_path)
    print(f"Saved stage 3 outputs to: {output_path.resolve()}")

    status_counts, self_check_counts = calculate_stage_3_stats(
        output_rows,
        harm_domains=harm_domain_definitions.keys(),
        task_types=task_type_definitions.keys(),
    )
    print_stage_3_stats(status_counts, self_check_counts)

    return {
        "rows": output_rows,
        "status_counts": dict(status_counts),
        "self_check_counts": {
            field: dict(counts) for field, counts in self_check_counts.items()
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate benign/dual-use/malicious prompt triplets from stage 2 outputs."
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=STAGE_3_OUTPUT_PATH)
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
        help="Maximum tokens to request from the API model for each triplet.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_stage_3_dataset(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        max_completion_tokens=args.max_completion_tokens,
    )
