import sys
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_calls.api_models import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_SECONDS,
    DEFAULT_RETRY_MAX_SECONDS,
    DEFAULT_VERTEX_PROJECT_ID,
    SUPPORTED_MODEL_NAMES,
    get_api_responses_batch,
)
from OpenSafeIntent.project_config import (
    DATASET_OUTPUT_DIR,
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_COMPLETION_TOKENS,
)

try:
    from .utils import (
        extract_json_object_with_keys,
        get_annotation_value,
        load_json,
        load_text,
        save_json,
    )
except ImportError:
    from utils import (
        extract_json_object_with_keys,
        get_annotation_value,
        load_json,
        load_text,
        save_json,
    )


DATASET_NAME = "PKU-Alignment/PKU-SafeRLHF"
DEFAULT_SPLIT = "train"
DEFAULT_NUM_DATAPOINTS = 1
DEFAULT_MODEL_NAME = DEFAULT_GENERATOR_MODEL
PROMPT_PATH = (
    REPO_ROOT / "prompts" / "data_generation" / "stage_1.txt"
)
STAGE_1_OUTPUT_DIR = DATASET_OUTPUT_DIR / "stage_1"
OUTPUT_PATH = STAGE_1_OUTPUT_DIR / "stage_1_outputs.json"


def load_stage_1_outputs(output_path=OUTPUT_PATH):
    return load_json(output_path)


def load_prompt_template(prompt_path=PROMPT_PATH):
    return load_text(prompt_path)


def print_distribution(title, counts, total):
    print(f"\n{title}")
    print("-" * len(title))

    for label, count in counts.most_common():
        percentage = (count / total) * 100 if total else 0
        print(f"{label}: {count} ({percentage:.1f}%)")


def report_stage_1_distributions(output_path=OUTPUT_PATH):
    def _format_distribution(counts, total):
        return {
            label: {
                "count": count,
                "%": round((count / total) * 100, 2) if total else 0.0,
            }
            for label, count in counts.items()
        }

    rows = load_stage_1_outputs(output_path)
    total = len(rows)

    task_type_counts = Counter(get_annotation_value(row, "task_type") for row in rows)
    harm_domain_counts = Counter(get_annotation_value(row, "harm_domain") for row in rows)
    is_prompt_safe_counts = Counter(
        get_annotation_value(row, "is_prompt_safe") for row in rows
    )

    print(f"Loaded {total} stage 1 rows from: {Path(output_path).resolve()}")
    print_distribution("Task Type Distribution", task_type_counts, total)
    print_distribution("Harm Domain Distribution", harm_domain_counts, total)
    print_distribution("Is Prompt Safe Distribution", is_prompt_safe_counts, total)

    return {
        "total": total,
        "task_type": _format_distribution(task_type_counts, total),
        "harm_domain": _format_distribution(harm_domain_counts, total),
        "is_prompt_safe": _format_distribution(is_prompt_safe_counts, total),
    }


def iter_unsafe_prompts(dataset_name=DATASET_NAME, split=DEFAULT_SPLIT):
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    seen = set()

    for row in dataset:
        unsafe_prompt = row.get("prompt")
        if not unsafe_prompt or unsafe_prompt in seen:
            continue

        seen.add(unsafe_prompt)
        yield {
            "source_dataset": dataset_name,
            "source_split": split,
            "unsafe_prompt": unsafe_prompt,
        }


def annotate_datapoint(
    datapoint,
    prompt_template,
    model_name=DEFAULT_MODEL_NAME,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens=MAX_COMPLETION_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_base_seconds=DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds=DEFAULT_RETRY_MAX_SECONDS,
):
    model_prompt = prompt_template.replace("{{unsafe_prompt}}", datapoint["unsafe_prompt"])
    result = get_api_responses_batch(
        [model_prompt],
        model_name=model_name,
        vertex_project_id=vertex_project_id,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        system_prompt="You are a helpful assistant.",
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
        raise_on_error=True,
    )[0]
    raw_response = result["response"]
    success = result["success"]
    if not success:
        raise RuntimeError("Provider filtered the stage 1 annotation request.")

    annotation = extract_json_object_with_keys(
        raw_response,
        required_keys=("topic_summary", "is_prompt_safe", "harm_domain", "task_type"),
    )

    return {
        "origin_metadata": {
            "source_dataset": datapoint["source_dataset"],
            "source_split": datapoint["source_split"],
            "unsafe_prompt": datapoint["unsafe_prompt"],
        },
        "generation_metadata": {
            "topic_summary": annotation.get("topic_summary", ""),
            "is_prompt_safe": annotation.get("is_prompt_safe", ""),
            "harm_domain": annotation.get("harm_domain", ""),
            "task_type": annotation.get("task_type", ""),
        },
    }


def generate_stage_1_dataset(
    num_datapoints=DEFAULT_NUM_DATAPOINTS,
    output_path=OUTPUT_PATH,
    dataset_name=DATASET_NAME,
    split=DEFAULT_SPLIT,
    model_name=DEFAULT_MODEL_NAME,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens=MAX_COMPLETION_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_base_seconds=DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds=DEFAULT_RETRY_MAX_SECONDS,
):
    prompt_template = load_prompt_template()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    prompt_iter = iter_unsafe_prompts(dataset_name=dataset_name, split=split)

    for datapoint in tqdm(prompt_iter, total=num_datapoints, desc="Annotating prompts"):
        annotated = annotate_datapoint(
            datapoint,
            prompt_template,
            model_name=model_name,
            vertex_project_id=vertex_project_id,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
        )
        rows.append(annotated)

        if len(rows) >= num_datapoints:
            break

    save_json(rows, output_path)

    print(f"Saved stage 1 dataset to: {output_path.resolve()}")

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Generate stage 1 annotations for unsafe prompts.")
    parser.add_argument("--num-datapoints", type=int, default=DEFAULT_NUM_DATAPOINTS)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODEL_NAMES,
        default=DEFAULT_MODEL_NAME,
        help="API-backed model used for stage 1 annotation.",
    )
    parser.add_argument(
        "--vertex-project-id",
        default=DEFAULT_VERTEX_PROJECT_ID,
        help="Google Cloud project ID used for Vertex models.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=MAX_COMPLETION_TOKENS,
        help="Maximum tokens to request for each annotation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for annotation calls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries for transient provider errors.",
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
    generate_stage_1_dataset(
        num_datapoints=args.num_datapoints,
        output_path=args.output,
        dataset_name=args.dataset,
        split=args.split,
        model_name=args.model,
        vertex_project_id=args.vertex_project_id,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        retry_max_seconds=args.retry_max_seconds,
    )
