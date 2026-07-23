from pathlib import Path
from collections import Counter
from llm_calls.api_models import get_api_responses_batch
from project_config import DEFAULT_JUDGE_MODEL, VERTEX_PROJECT_ID

try:
    from .utils import extract_json_object_with_keys, load_text
except ImportError:
    from utils import extract_json_object_with_keys, load_text


INTENT_CLASSIFICATION_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "data_generation"
    / "prompt_intent_classification.txt"
)
PROMPT_FIELDS = ("benign_prompt", "dual_use_prompt", "malicious_prompt")
VALID_DECISIONS = {"benign", "dual_use", "malicious"}
CLASSIFICATION_ERROR = "ERROR"
DEFAULT_VERTEX_PROJECT_ID = VERTEX_PROJECT_ID
DEFAULT_MAX_COMPLETION_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_GEMINI_3_THINKING_LEVEL = "LOW"
REQUIRED_RESPONSE_KEYS = (
    "decision",
    "plausible_benign_use",
    "meaningful_misuse_risk",
)


def load_intent_classification_prompt():
    return load_text(INTENT_CLASSIFICATION_PROMPT_PATH)


def build_intent_classification_prompt(prompt):
    return load_intent_classification_prompt().replace("{{PROMPT}}", prompt)


def call_judge_model(
    prompt,
    judge_model=DEFAULT_JUDGE_MODEL,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
):
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
        raise_on_error=True,
    )[0]
    if not result["success"]:
        raise RuntimeError(result["error"] or "Judge model call failed.")
    return result["response"]


def parse_intent_classification_response(raw_response):
    parsed = extract_json_object_with_keys(
        raw_response,
        required_keys=REQUIRED_RESPONSE_KEYS,
    )
    decision = parsed["decision"]
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Model response contained unknown decision {decision!r}")

    plausible_benign_use = parsed["plausible_benign_use"]
    if not isinstance(plausible_benign_use, str) or not plausible_benign_use.strip():
        raise ValueError("Model response contained an empty plausible_benign_use.")

    meaningful_misuse_risk = parsed["meaningful_misuse_risk"]
    if not isinstance(meaningful_misuse_risk, str) or not meaningful_misuse_risk.strip():
        raise ValueError("Model response contained an empty meaningful_misuse_risk.")

    return {
        "decision": decision,
        "plausible_benign_use": plausible_benign_use,
        "meaningful_misuse_risk": meaningful_misuse_risk,
    }


def classify_prompt_intent(
    prompt,
    judge_model=DEFAULT_JUDGE_MODEL,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
):
    raw_response = call_judge_model(
        build_intent_classification_prompt(prompt),
        judge_model=judge_model,
        vertex_project_id=vertex_project_id,
    )
    return parse_intent_classification_response(raw_response)


def get_generated_prompt(row, prompt_field):
    generated_prompts = row.get("generated_prompts", {})
    if not isinstance(generated_prompts, dict):
        raise ValueError("Datapoint generated_prompts was not an object.")

    prompt = generated_prompts.get(prompt_field)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Datapoint is missing generated_prompts.{prompt_field}")

    return prompt


def format_verification_error(error):
    return f"{type(error).__name__}: {error}"


def verify_datapoint_intents(
    row,
    judge_model=DEFAULT_JUDGE_MODEL,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
):
    verification_intent = {}
    verification_errors = {}
    for prompt_field in PROMPT_FIELDS:
        decision_field = f"{prompt_field}_decision"
        benign_use_field = f"{prompt_field}_plausible_benign_use"
        misuse_risk_field = f"{prompt_field}_meaningful_misuse_risk"
        try:
            prompt = get_generated_prompt(row, prompt_field)
            classification = classify_prompt_intent(
                prompt,
                judge_model=judge_model,
                vertex_project_id=vertex_project_id,
            )
            verification_intent[decision_field] = classification["decision"]
            verification_intent[benign_use_field] = classification["plausible_benign_use"]
            verification_intent[misuse_risk_field] = classification[
                "meaningful_misuse_risk"
            ]
        except Exception as error:
            verification_intent[decision_field] = CLASSIFICATION_ERROR
            verification_intent[benign_use_field] = ""
            verification_intent[misuse_risk_field] = ""
            verification_errors[decision_field] = format_verification_error(error)

    output_row = dict(row)
    output_row["verification_intent"] = verification_intent
    if verification_errors:
        output_row["verification_errors"] = verification_errors
    return output_row


def calculate_decision_distributions(rows):
    return {
        f"{prompt_field}_decision": Counter(
            row.get("verification_intent", {}).get(
                f"{prompt_field}_decision",
                "MISSING",
            )
            for row in rows
        )
        for prompt_field in PROMPT_FIELDS
    }


def print_decision_distributions(distributions):
    for field_name, counts in distributions.items():
        total = sum(counts.values())
        print(f"\n{field_name} Distribution")
        print("-" * (len(field_name) + len(" Distribution")))
        for intent, count in counts.most_common():
            percentage = (count / total) * 100 if total else 0
            print(f"{intent}: {count} ({percentage:.1f}%)")
