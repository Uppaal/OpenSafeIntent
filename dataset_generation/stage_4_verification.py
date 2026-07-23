import sys
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_config import DATASET_OUTPUT_DIR

try:
    from .intent_verification import (
        calculate_decision_distributions,
        print_decision_distributions,
        verify_datapoint_intents,
    )
    from .utils import load_json, save_json
except ImportError:
    from intent_verification import (
        calculate_decision_distributions,
        print_decision_distributions,
        verify_datapoint_intents,
    )
    from utils import load_json, save_json


STAGE_4_OUTPUT_DIR = DATASET_OUTPUT_DIR / "stage_4"
STAGE_4_INPUT_PATH = STAGE_4_OUTPUT_DIR / "stage_4_outputs.json"
STAGE_4_VERIFICATION_OUTPUT_PATH = (
    STAGE_4_OUTPUT_DIR / "stage_4_intent_verification.json"
)
STAGE_4_PASSED_OUTPUT_PATH = STAGE_4_OUTPUT_DIR / "stage_4_passed.json"
EXPECTED_DECISIONS = {
    "benign_prompt_decision": "benign",
    "dual_use_prompt_decision": "dual_use",
    "malicious_prompt_decision": "malicious",
}


def load_stage_4_outputs(input_path=STAGE_4_INPUT_PATH):
    return load_json(input_path)


def normalize_decision(decision):
    if not isinstance(decision, str):
        return decision

    return decision.strip().lower().replace("-", "_").replace(" ", "_")


def verification_is_correct(row):
    verification_intent = row.get("verification_intent", {})
    if not isinstance(verification_intent, dict):
        return False

    return all(
        normalize_decision(verification_intent.get(field)) == expected_decision
        for field, expected_decision in EXPECTED_DECISIONS.items()
    )


def update_verification_intent(row):
    reverified_row = verify_datapoint_intents(row)
    output_row = deepcopy(row)
    previous_verification = output_row.get("verification_intent", {})
    if not isinstance(previous_verification, dict):
        previous_verification = {}

    output_row["verification_intent"] = {
        **previous_verification,
        **reverified_row["verification_intent"],
    }
    if "verification_errors" in reverified_row:
        output_row["verification_errors"] = reverified_row["verification_errors"]
    else:
        output_row.pop("verification_errors", None)
    return output_row


def check_stage_4_rows(rows):
    checked_rows = []
    stats = Counter()

    for row in tqdm(rows, desc="Verifying repaired prompt intents"):
        if verification_is_correct(row):
            checked_rows.append(row)
            stats["retained_existing_verification"] += 1
        else:
            checked_rows.append(update_verification_intent(row))
            stats["reverified"] += 1

    stats["input_rows"] = len(rows)
    stats["output_rows"] = len(checked_rows)
    stats["correct_after_verification"] = sum(
        verification_is_correct(row) for row in checked_rows
    )
    return checked_rows, stats


def get_stage_4_passed_rows(rows):
    return [
        row
        for row in rows
        if verification_is_correct(row)
    ]


def print_stage_4_verification_summary(stats):
    print("\nStage 4 Intent Verification Summary")
    print("-" * 40)
    for field in (
        "input_rows",
        "retained_existing_verification",
        "reverified",
        "correct_after_verification",
        "passed_rows",
        "output_rows",
    ):
        print(f"{field}: {stats.get(field, 0)}")


def verify_stage_4_intents(
    input_path=STAGE_4_INPUT_PATH,
    output_path=STAGE_4_VERIFICATION_OUTPUT_PATH,
    passed_output_path=STAGE_4_PASSED_OUTPUT_PATH,
):
    rows = load_stage_4_outputs(input_path)
    output_rows, stats = check_stage_4_rows(rows)

    save_json(output_rows, output_path)
    print(
        "Saved stage 4 intent verification pilot_dataset to: "
        f"{Path(output_path).resolve()}"
    )
    passed_rows = get_stage_4_passed_rows(output_rows)
    stats["passed_rows"] = len(passed_rows)
    save_json(passed_rows, passed_output_path)
    print(f"Saved stage 4 passed pilot_dataset to: {Path(passed_output_path).resolve()}")
    print_stage_4_verification_summary(stats)

    distributions = calculate_decision_distributions(output_rows)
    print_decision_distributions(distributions)
    return {
        "rows": output_rows,
        "passed_rows": passed_rows,
        "stats": dict(stats),
        "distributions": {
            field_name: dict(counts)
            for field_name, counts in distributions.items()
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reclassify repaired stage 4 prompts with stale intent labels."
    )
    parser.add_argument("--input", type=Path, default=STAGE_4_INPUT_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=STAGE_4_VERIFICATION_OUTPUT_PATH,
    )
    parser.add_argument(
        "--passed-output",
        type=Path,
        default=STAGE_4_PASSED_OUTPUT_PATH,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verify_stage_4_intents(
        input_path=args.input,
        output_path=args.output,
        passed_output_path=args.passed_output,
    )
