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
from project_config import (
    DATASET_OUTPUT_DIR,
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_TEMPERATURE,
)

try:
    from .utils import MISSING_VALUE, get_generation_value, load_json, load_text, save_json
    from .utils import extract_json_object_with_keys
except ImportError:
    from utils import MISSING_VALUE, get_generation_value, load_json, load_text, save_json
    from utils import extract_json_object_with_keys


PROMPT_DIR = REPO_ROOT / "prompts"
STAGE_4_PROMPT_DIR = PROMPT_DIR / "data_generation" / "stage_4"
BASE_REPAIR_PROMPT_PATH = STAGE_4_PROMPT_DIR / "base_prompt.txt"
HARM_DOMAIN_DEFINITIONS_PATH = PROMPT_DIR / "definitions" / "harm_domain.json"
TASK_TYPE_DEFINITIONS_PATH = PROMPT_DIR / "definitions" / "task_type.json"
STAGE_3_OUTPUT_DIR = DATASET_OUTPUT_DIR / "stage_3"
STAGE_3_VERIFICATION_INPUT_PATH = (
    STAGE_3_OUTPUT_DIR / "stage_3_intent_verification.json"
)
STAGE_4_OUTPUT_DIR = DATASET_OUTPUT_DIR / "stage_4"
STAGE_4_OUTPUT_PATH = STAGE_4_OUTPUT_DIR / "stage_4_outputs.json"
DEFAULT_MAX_COMPLETION_TOKENS = 4096

PROMPT_FIELDS = ("benign_prompt", "dual_use_prompt", "malicious_prompt")
EXPECTED_DECISIONS = {
    "benign_prompt_decision": "benign",
    "dual_use_prompt_decision": "dual_use",
    "malicious_prompt_decision": "malicious",
}
REPAIR_MODES = (
    {
        "name": "benign_prompt_decision=dual_use",
        "decision_field": "benign_prompt_decision",
        "observed_decision": "dual_use",
        "prompt_field": "benign_prompt",
        "issue_path": STAGE_4_PROMPT_DIR / "benign_is_dual_use.txt",
    },
    {
        "name": "dual_use_prompt_decision=benign",
        "decision_field": "dual_use_prompt_decision",
        "observed_decision": "benign",
        "prompt_field": "dual_use_prompt",
        "issue_path": STAGE_4_PROMPT_DIR / "dual_use_is_benign.txt",
    },
    {
        "name": "dual_use_prompt_decision=malicious",
        "decision_field": "dual_use_prompt_decision",
        "observed_decision": "malicious",
        "prompt_field": "dual_use_prompt",
        "issue_path": STAGE_4_PROMPT_DIR / "dual_use_is_malicious.txt",
    },
    {
        "name": "malicious_prompt_decision=dual_use",
        "decision_field": "malicious_prompt_decision",
        "observed_decision": "dual_use",
        "prompt_field": "malicious_prompt",
        "issue_path": STAGE_4_PROMPT_DIR / "malicious_is_dual_use.txt",
    },
)
REQUIRED_REPAIR_RESPONSE_KEYS = (
    "underlying_topic",
    "benign_prompt",
    "dual_use_prompt",
    "malicious_prompt",
    "repair_note",
    "self_check",
)


def load_stage_3_verification_outputs(
    input_path=STAGE_3_VERIFICATION_INPUT_PATH,
    limit=None,
):
    rows = load_json(input_path)
    return rows[:limit] if limit is not None else rows


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


def normalize_decision(decision):
    if not isinstance(decision, str):
        return decision

    return decision.strip().lower().replace("-", "_").replace(" ", "_")


def get_verification_decision(row, decision_field):
    verification = row.get("verification_intent", {})
    if not isinstance(verification, dict):
        return None

    return normalize_decision(verification.get(decision_field))


def has_stage_3_generation_error(row):
    generated_prompts = row.get("generated_prompts", {})
    if not isinstance(generated_prompts, dict):
        return True

    failure_notes = generated_prompts.get("failure_notes", [])
    if not failure_notes:
        return False

    return not any(
        isinstance(generated_prompts.get(prompt_field), str)
        and generated_prompts[prompt_field].strip()
        for prompt_field in PROMPT_FIELDS
    )


def verification_is_correct(row):
    return all(
        get_verification_decision(row, decision_field) == expected_decision
        for decision_field, expected_decision in EXPECTED_DECISIONS.items()
    )


def get_repair_modes(row):
    return [
        repair_mode
        for repair_mode in REPAIR_MODES
        if get_verification_decision(row, repair_mode["decision_field"])
        == repair_mode["observed_decision"]
    ]


def has_unsupported_verification_error(row):
    repaired_decision_fields = {
        repair_mode["decision_field"] for repair_mode in get_repair_modes(row)
    }
    return any(
        get_verification_decision(row, decision_field) != expected_decision
        and decision_field not in repaired_decision_fields
        for decision_field, expected_decision in EXPECTED_DECISIONS.items()
    )


def get_required_generated_prompt(row, prompt_field):
    generated_prompts = row.get("generated_prompts", {})
    if not isinstance(generated_prompts, dict):
        raise ValueError("Datapoint generated_prompts was not an object.")

    value = generated_prompts.get(prompt_field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Datapoint is missing generated_prompts.{prompt_field}")

    return value


def get_required_generation_value(row, field):
    value = get_generation_value(row, field)
    if value == MISSING_VALUE:
        raise ValueError(f"Datapoint is missing generation_metadata.{field}")
    return value


def build_repair_prompt(
    row,
    issue_text,
    base_prompt_template=None,
    harm_domain_definitions=None,
    task_type_definitions=None,
):
    base_prompt_template = base_prompt_template or load_text(BASE_REPAIR_PROMPT_PATH)
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
        base_prompt_template.replace("{{HARM_DOMAIN}}", harm_domain)
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
            "{{UNDERLYING_TOPIC}}",
            get_required_generated_prompt(row, "underlying_topic"),
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
        .replace("{{ISSUE}}", issue_text)
    )


def parse_repair_response(raw_response):
    repaired = extract_json_object_with_keys(
        raw_response,
        required_keys=REQUIRED_REPAIR_RESPONSE_KEYS,
    )
    for prompt_field in ("underlying_topic", *PROMPT_FIELDS):
        value = repaired[prompt_field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Repair response contained an empty {prompt_field}.")

    if not isinstance(repaired["self_check"], dict):
        raise ValueError("Repair response self_check was not an object.")

    return repaired


def call_llm_for_repair(
    row,
    repair_mode,
    base_prompt_template=None,
    harm_domain_definitions=None,
    task_type_definitions=None,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
):
    prompt = build_repair_prompt(
        row=row,
        issue_text=load_text(repair_mode["issue_path"]),
        base_prompt_template=base_prompt_template,
        harm_domain_definitions=harm_domain_definitions,
        task_type_definitions=task_type_definitions,
    )
    raw_response = get_required_api_response(
        prompt,
        max_completion_tokens=max_completion_tokens,
    )
    return parse_repair_response(raw_response)


def apply_targeted_repair(row, repair_mode, repaired_prompts):
    output_row = deepcopy(row)
    generated_prompts = dict(output_row["generated_prompts"])
    prompt_field = repair_mode["prompt_field"]
    generated_prompts[prompt_field] = repaired_prompts[prompt_field]
    output_row["generated_prompts"] = generated_prompts
    return output_row


def repair_datapoint(
    row,
    repair_modes,
    base_prompt_template=None,
    harm_domain_definitions=None,
    task_type_definitions=None,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
):
    repaired_row = deepcopy(row)
    for repair_mode in repair_modes:
        repaired_prompts = call_llm_for_repair(
            row=repaired_row,
            repair_mode=repair_mode,
            base_prompt_template=base_prompt_template,
            harm_domain_definitions=harm_domain_definitions,
            task_type_definitions=task_type_definitions,
            max_completion_tokens=max_completion_tokens,
        )
        repaired_row = apply_targeted_repair(
            repaired_row,
            repair_mode,
            repaired_prompts,
        )

    return repaired_row


def process_stage_4_rows(
    rows,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
):
    base_prompt_template = load_text(BASE_REPAIR_PROMPT_PATH)
    harm_domain_definitions = load_json(HARM_DOMAIN_DEFINITIONS_PATH)
    task_type_definitions = load_json(TASK_TYPE_DEFINITIONS_PATH)
    processed_rows = []
    stats = Counter()

    for index, row in enumerate(tqdm(rows, desc="Repairing prompt triplets")):
        if has_stage_3_generation_error(row):
            stats["skipped_stage_3_error"] += 1
            continue

        if verification_is_correct(row):
            processed_rows.append(row)
            stats["retained"] += 1
            continue

        repair_modes = get_repair_modes(row)
        for repair_mode in repair_modes:
            stats[f"failure_mode:{repair_mode['name']}"] += 1

        if not repair_modes or has_unsupported_verification_error(row):
            stats["skipped_unsupported_verification"] += 1
            continue

        try:
            processed_rows.append(
                repair_datapoint(
                    row=row,
                    repair_modes=repair_modes,
                    base_prompt_template=base_prompt_template,
                    harm_domain_definitions=harm_domain_definitions,
                    task_type_definitions=task_type_definitions,
                    max_completion_tokens=max_completion_tokens,
                )
            )
            stats["repaired"] += 1
            stats["repair_calls"] += len(repair_modes)
        except Exception as error:
            stats["dropped_repair_error"] += 1
            tqdm.write(f"Dropped datapoint {index} after repair error: {error}")

    stats["input_rows"] = len(rows)
    stats["output_rows"] = len(processed_rows)
    return processed_rows, stats


def print_stage_4_stats(stats):
    print("\nStage 4 Processing Summary")
    print("-" * 40)
    for field in (
        "input_rows",
        "retained",
        "repaired",
        "repair_calls",
        "skipped_stage_3_error",
        "skipped_unsupported_verification",
        "dropped_repair_error",
        "output_rows",
    ):
        print(f"{field}: {stats.get(field, 0)}")

    total_failure_modes = sum(
        stats.get(f"failure_mode:{repair_mode['name']}", 0)
        for repair_mode in REPAIR_MODES
    )
    print("\nConfigured Failure Mode Distribution")
    print("-" * 40)
    for repair_mode in REPAIR_MODES:
        failure_mode = repair_mode["name"]
        count = stats.get(f"failure_mode:{failure_mode}", 0)
        percentage = (count / total_failure_modes) * 100 if total_failure_modes else 0
        print(f"{failure_mode}: {count} ({percentage:.1f}%)")


def generate_stage_4_dataset(
    input_path=STAGE_3_VERIFICATION_INPUT_PATH,
    output_path=STAGE_4_OUTPUT_PATH,
    limit=None,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
):
    rows = load_stage_3_verification_outputs(input_path, limit=limit)
    processed_rows, stats = process_stage_4_rows(
        rows,
        max_completion_tokens=max_completion_tokens,
    )

    save_json(processed_rows, output_path)
    print(f"Saved stage 4 pilot_dataset to: {Path(output_path).resolve()}")
    print_stage_4_stats(stats)
    return {"rows": processed_rows, "stats": dict(stats)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repair incorrect prompt intents from stage 3 verification pilot_dataset."
    )
    parser.add_argument("--input", type=Path, default=STAGE_3_VERIFICATION_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=STAGE_4_OUTPUT_PATH)
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
        help="Maximum tokens to request from the API model for each repair.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_stage_4_dataset(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        max_completion_tokens=args.max_completion_tokens,
    )
