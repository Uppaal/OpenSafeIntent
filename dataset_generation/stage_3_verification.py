import sys
import argparse
from tqdm import tqdm
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OpenSafeIntent.project_config import DATASET_OUTPUT_DIR

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


STAGE_3_OUTPUT_DIR = DATASET_OUTPUT_DIR / "stage_3"
STAGE_3_INPUT_PATH = STAGE_3_OUTPUT_DIR / "stage_3_outputs.json"
STAGE_3_VERIFICATION_OUTPUT_PATH = (
    STAGE_3_OUTPUT_DIR / "stage_3_intent_verification.json"
)


def load_stage_3_outputs(input_path=STAGE_3_INPUT_PATH):
    return load_json(input_path)


def verify_stage_3_intents(
    input_path=STAGE_3_INPUT_PATH,
    output_path=STAGE_3_VERIFICATION_OUTPUT_PATH,
):
    rows = load_stage_3_outputs(input_path)
    output_rows = [
        verify_datapoint_intents(row)
        for row in tqdm(rows, desc="Verifying prompt intents")
    ]

    save_json(output_rows, output_path)
    print(f"Saved stage 3 intent verification outputs to: {Path(output_path).resolve()}")

    distributions = calculate_decision_distributions(output_rows)
    print_decision_distributions(distributions)
    return {
        "rows": output_rows,
        "distributions": {
            field_name: dict(counts)
            for field_name, counts in distributions.items()
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify the intended use of each generated stage 3 prompt."
    )
    parser.add_argument("--input", type=Path, default=STAGE_3_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=STAGE_3_VERIFICATION_OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verify_stage_3_intents(input_path=args.input, output_path=args.output)
