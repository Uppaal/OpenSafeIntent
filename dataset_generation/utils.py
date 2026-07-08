import json
from pathlib import Path


MISSING_VALUE = "MISSING"


def load_text(input_path):
    return Path(input_path).read_text(encoding="utf-8")


def load_json(input_path):
    with Path(input_path).open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def save_json(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(rows, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def missing_if_empty(value):
    return MISSING_VALUE if value is None or value == "" else value


def normalize_datapoint(row, backfilled=None):
    if "origin_metadata" in row and "generation_metadata" in row:
        origin_metadata = row.get("origin_metadata", {})
        generation_metadata = row.get("generation_metadata", {})
    else:
        annotation = row.get("annotation", {})
        origin_metadata = {
            "source_dataset": row.get("source_dataset", ""),
            "source_split": row.get("source_split", ""),
            "unsafe_prompt": row.get("unsafe_prompt", ""),
        }
        generation_metadata = {
            "topic_summary": annotation.get("topic_summary", row.get("topic_summary", "")),
            "is_prompt_safe": annotation.get("is_prompt_safe", row.get("is_prompt_safe", "")),
            "harm_domain": annotation.get("harm_domain", row.get("harm_domain", "")),
            "task_type": annotation.get("task_type", row.get("task_type", "")),
        }

    backfilled = generation_metadata.get("backfilled", False) if backfilled is None else backfilled

    return {
        "origin_metadata": {
            "source_dataset": origin_metadata.get("source_dataset", ""),
            "source_split": origin_metadata.get("source_split", ""),
            "unsafe_prompt": origin_metadata.get("unsafe_prompt", ""),
        },
        "generation_metadata": {
            "topic_summary": generation_metadata.get("topic_summary", ""),
            "is_prompt_safe": generation_metadata.get("is_prompt_safe", ""),
            "harm_domain": generation_metadata.get("harm_domain", ""),
            "task_type": generation_metadata.get("task_type", ""),
            "backfilled": bool(backfilled),
        },
    }


def get_generation_value(row, key):
    return missing_if_empty(row.get("generation_metadata", {}).get(key))


def get_annotation_value(row, key):
    annotation = row.get("generation_metadata") or row.get("annotation", {})
    return missing_if_empty(annotation.get(key))


def extract_json_object(raw_response):
    if raw_response is None:
        raise ValueError("Model response was empty.")

    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Model response did not contain a JSON object: {raw_response!r}")

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError(f"Model response did not contain valid JSON: {raw_response!r}") from error

    if not isinstance(parsed, dict):
        raise ValueError(f"Model response JSON was not an object: {raw_response!r}")

    return parsed


def extract_json_object_with_keys(raw_response, required_keys):
    parsed = extract_json_object(raw_response)
    missing_keys = [key for key in required_keys if key not in parsed]
    if missing_keys:
        raise ValueError(
            f"Model response JSON was missing required keys {missing_keys}: {raw_response!r}"
        )

    return parsed
