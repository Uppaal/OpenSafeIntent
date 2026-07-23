import re
import sys
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_generation.utils import (  # noqa: E402
    extract_json_object,
    load_json,
    load_text,
    save_json,
)
from project_config import (  # noqa: E402
    ANALYSIS_OUTPUT_DIR,
    DEFAULT_GENERATOR_MODEL,
    RESPONSE_OUTPUT_DIR,
)
from llm_calls.api_models import (  # noqa: E402
    DEFAULT_MAX_RETRIES as DEFAULT_API_MAX_RETRIES,
    DEFAULT_RETRY_BASE_SECONDS as DEFAULT_API_RETRY_BASE_SECONDS,
    DEFAULT_RETRY_MAX_SECONDS as DEFAULT_API_RETRY_MAX_SECONDS,
    DEFAULT_VERTEX_PROJECT_ID,
    get_api_responses_batch,
)
from llm_calls.local import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    build_local_generation_backend,
    safe_model_output_name,
)
from llm_calls.models import (  # noqa: E402
    LOCAL_ALIAS_TO_MODEL,
    LOCAL_MODEL_ALIASES,
    LOCAL_MODEL_NAMES,
)


MODEL_RESPONSES_INPUT_DIR = RESPONSE_OUTPUT_DIR
PROMPT_PATH = REPO_ROOT / "prompts" / "metrics" / "intent_classification.txt"
DEFAULT_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "intent_classification"
DUAL_USE_PROMPT_FIELD = "dual_use_prompt"
GENERATED_PROMPTS_FIELD = "generated_prompts"
HARMFULNESS_FIELD = "response_scores_harmfulness"
INTENT_CLASSIFICATION_FIELD = "dual_use_intent_classification"
FAILURES_FIELD = "intent_classification_failures"
VALID_STRATEGIES = ("FULL", "CONSTRAINED", "REFUSE")
VALID_STRATEGY_SET = set(VALID_STRATEGIES)
ERROR_STRATEGY = "ERROR"
DEFAULT_API_MAX_COMPLETION_TOKENS = 64
DEFAULT_LOCAL_MAX_COMPLETION_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_GENERATOR_RESPONSE_ALIAS = DEFAULT_GENERATOR_MODEL.replace(".", "_")
API_MODEL_ALIASES = {
    DEFAULT_GENERATOR_RESPONSE_ALIAS: DEFAULT_GENERATOR_MODEL,
}
RESPONSE_FILE_ALIASES = {
    DEFAULT_GENERATOR_MODEL: DEFAULT_GENERATOR_RESPONSE_ALIAS,
    **LOCAL_MODEL_ALIASES,
}


def default_input_path(model_alias):
    response_alias = RESPONSE_FILE_ALIASES.get(model_alias, model_alias)
    safe_name = safe_model_output_name(response_alias)
    candidates = (
        MODEL_RESPONSES_INPUT_DIR / f"{safe_name}_scored.json",
        MODEL_RESPONSES_INPUT_DIR / f"{safe_name}.json",
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_output_path(input_path):
    input_path = Path(input_path)
    return DEFAULT_OUTPUT_DIR / f"{input_path.stem}_intent_classification.json"


def load_rows(input_path, limit=None):
    rows = load_json(input_path)
    if not isinstance(rows, list):
        raise ValueError("Model response input must be a list of datapoints.")
    return rows[:limit] if limit is not None else rows


def require_numeric_score(value, field_name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")
    return float(value)


def dual_use_harmfulness(row):
    harmfulness_scores = row.get(HARMFULNESS_FIELD)
    if not isinstance(harmfulness_scores, dict):
        raise ValueError(f"Datapoint is missing object field {HARMFULNESS_FIELD}.")

    return require_numeric_score(
        harmfulness_scores.get(DUAL_USE_PROMPT_FIELD),
        f"{HARMFULNESS_FIELD}.{DUAL_USE_PROMPT_FIELD}",
    )


def get_dual_use_prompt(row):
    generated_prompts = row.get(GENERATED_PROMPTS_FIELD)
    if not isinstance(generated_prompts, dict):
        raise ValueError(f"Datapoint is missing object field {GENERATED_PROMPTS_FIELD}.")

    prompt = generated_prompts.get(DUAL_USE_PROMPT_FIELD)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            f"Datapoint is missing {GENERATED_PROMPTS_FIELD}."
            f"{DUAL_USE_PROMPT_FIELD}."
        )

    return prompt


def build_filtered_tasks(rows):
    tasks = []
    stats = Counter()

    for row_index, row in enumerate(
        tqdm(rows, desc="Filtering harmful dual-use rows", unit="row")
    ):
        if not isinstance(row, dict):
            stats["non_object_datapoints_skipped"] += 1
            continue

        try:
            harmfulness = dual_use_harmfulness(row)
            if harmfulness <= 0:
                stats["safe_dual_use_rows_skipped"] += 1
                continue
            tasks.append(
                {
                    "row_index": row_index,
                    "row": row,
                    "prompt": get_dual_use_prompt(row),
                    "dual_use_harmfulness": harmfulness,
                }
            )
        except ValueError as error:
            stats["invalid_datapoints_skipped"] += 1
            tqdm.write(f"Skipping datapoint {row_index}: {error}")

    stats["filtered_rows"] = len(tasks)
    return tasks, stats


def format_intent_prompt(template, prompt):
    if "{{prompt}}" in template:
        return template.replace("{{prompt}}", prompt)
    if "{{PROMPT}}" in template:
        return template.replace("{{PROMPT}}", prompt)
    return template.format(prompt=prompt)


def parse_strategy(raw_response):
    try:
        parsed = extract_json_object(raw_response)
    except ValueError:
        return ERROR_STRATEGY

    strategy = parsed.get("strategy")
    if not isinstance(strategy, str):
        return ERROR_STRATEGY

    strategy = strategy.strip().upper()
    if strategy not in VALID_STRATEGY_SET:
        return ERROR_STRATEGY
    return strategy


def classification_record(raw_response, error=None):
    strategy = ERROR_STRATEGY if error is not None else parse_strategy(raw_response)
    record = {
        "strategy": strategy,
        "raw_response": raw_response if isinstance(raw_response, str) else "",
    }
    if error is not None:
        record["error"] = summarize_error(error)
    elif strategy == ERROR_STRATEGY:
        record["error"] = "Model response did not contain a valid strategy JSON object."
    return record


def summarize_error(error):
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


def add_classification(task, classification):
    output_row = deepcopy(task["row"])
    output_row[INTENT_CLASSIFICATION_FIELD] = classification
    if classification["strategy"] == ERROR_STRATEGY:
        output_row[FAILURES_FIELD] = [
            {
                "prompt": DUAL_USE_PROMPT_FIELD,
                "metric": "intent_classification",
                "error": classification.get("error", ""),
            }
        ]
    return output_row


def save_progress(output_rows, output_path):
    if output_path is not None:
        save_json(output_rows, output_path)


def resolve_local_model_name(model_name):
    return LOCAL_ALIAS_TO_MODEL.get(model_name, model_name)


def resolve_api_model_name(model_name):
    return API_MODEL_ALIASES.get(model_name, model_name)


def classify_api_tasks(
    tasks,
    model_name,
    prompt_template,
    output_path=None,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens=DEFAULT_API_MAX_COMPLETION_TOKENS,
    max_retries=DEFAULT_API_MAX_RETRIES,
    retry_base_seconds=DEFAULT_API_RETRY_BASE_SECONDS,
    retry_max_seconds=DEFAULT_API_RETRY_MAX_SECONDS,
):
    api_model_name = resolve_api_model_name(model_name)
    prompts = [
        format_intent_prompt(prompt_template, task["prompt"])
        for task in tasks
    ]
    batch_results = get_api_responses_batch(
        prompts,
        model_name=api_model_name,
        vertex_project_id=vertex_project_id,
        max_completion_tokens=max_completion_tokens,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
        desc=f"Classifying intent with {model_name}",
        show_progress=True,
    )
    output_rows = []
    stats = Counter()

    for task, result in zip(tasks, batch_results):
        raw_response = result["response"] or ""
        stats["model_calls"] += result["api_calls"]
        stats["retry_attempts"] += result["retry_attempts"]

        if result["success"]:
            classification = classification_record(raw_response)
            stats["successful_calls"] += 1
        elif result["filtered"]:
            classification = classification_record(raw_response)
            stats["filtered_calls"] += 1
        else:
            classification = classification_record("", error=result["error"])
            stats["failed_calls"] += 1

        stats[classification["strategy"]] += 1
        output_rows.append(add_classification(task, classification))
        save_progress(output_rows, output_path)

    return output_rows, stats


def classify_local_tasks(
    tasks,
    model_name,
    prompt_template,
    output_path=None,
    max_completion_tokens=DEFAULT_LOCAL_MAX_COMPLETION_TOKENS,
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    dtype="auto",
    gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len=2048,
    batch_size=DEFAULT_BATCH_SIZE,
    temperature=DEFAULT_TEMPERATURE,
    top_p=1.0,
    use_chat_template=True,
    chat_template_path=None,
    trust_remote_code=False,
):
    resolved_model_name = resolve_local_model_name(model_name)
    backend = build_local_generation_backend(
        model_name=resolved_model_name,
        max_completion_tokens=max_completion_tokens,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        temperature=temperature,
        top_p=top_p,
        use_chat_template=use_chat_template,
        chat_template_path=chat_template_path,
        trust_remote_code=trust_remote_code,
    )
    output_rows = []
    stats = Counter()

    with tqdm(
        total=len(tasks),
        desc=f"Classifying intent with {model_name}",
        unit="prompt",
    ) as progress:
        for start in range(0, len(tasks), batch_size):
            task_batch = tasks[start : start + batch_size]
            prompts = [
                format_intent_prompt(prompt_template, task["prompt"])
                for task in task_batch
            ]
            try:
                raw_responses = backend.generate(prompts)
            except Exception as error:
                raw_responses = [None] * len(task_batch)
                classifications = [
                    classification_record("", error=error) for _ in task_batch
                ]
                stats["failed_batches"] += 1
            else:
                classifications = [
                    classification_record(raw_response)
                    for raw_response in raw_responses
                ]
                stats["local_generations"] += len(raw_responses)
                stats["successful_calls"] += len(raw_responses)

            for task, classification in zip(task_batch, classifications):
                stats[classification["strategy"]] += 1
                output_rows.append(add_classification(task, classification))

            save_progress(output_rows, output_path)
            progress.update(len(task_batch))

    return output_rows, stats


def model_is_local(model_name):
    return model_name in LOCAL_MODEL_NAMES


def format_count_percentage(count, denominator):
    if denominator == 0:
        return f"{count} (N/A)"
    return f"{count} ({100.0 * count / denominator:.1f}%)"


def print_summary(stats, model_name, input_path, output_path):
    input_rows = stats.get("input_rows", 0)
    total_prompts_considered = stats.get("filtered_rows", 0)
    print(f"\n{model_name} Intent Classification Summary")
    print("-" * 56)
    print(f"input_path: {Path(input_path).resolve()}")
    print(f"output_path: {Path(output_path).resolve()}")
    print(f"input_rows: {input_rows}")
    print(f"total_prompts_considered: {total_prompts_considered}")
    for field in (
        "safe_dual_use_rows_skipped",
        "invalid_datapoints_skipped",
        "non_object_datapoints_skipped",
    ):
        print(
            f"{field}: "
            f"{format_count_percentage(stats.get(field, 0), input_rows)}"
        )
    for field in (
        "successful_calls",
        "filtered_calls",
        "failed_calls",
        "failed_batches",
        "retry_attempts",
        "output_rows",
    ):
        print(
            f"{field}: "
            f"{format_count_percentage(stats.get(field, 0), total_prompts_considered)}"
        )

    print("\nStrategies")
    print("-" * 56)
    for strategy in (*VALID_STRATEGIES, ERROR_STRATEGY):
        print(
            f"{strategy}: "
            f"{format_count_percentage(stats.get(strategy, 0), total_prompts_considered)}"
        )


def classify_intents(
    model_name,
    input_path=None,
    output_path=None,
    limit=None,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens=None,
    max_retries=DEFAULT_API_MAX_RETRIES,
    retry_base_seconds=DEFAULT_API_RETRY_BASE_SECONDS,
    retry_max_seconds=DEFAULT_API_RETRY_MAX_SECONDS,
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    dtype="auto",
    gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len=None,
    batch_size=DEFAULT_BATCH_SIZE,
    temperature=DEFAULT_TEMPERATURE,
    top_p=1.0,
    use_chat_template=True,
    chat_template_path=None,
    trust_remote_code=False,
):
    if input_path is None:
        input_path = default_input_path(model_name)
    if output_path is None:
        output_path = default_output_path(input_path)

    rows = load_rows(input_path=input_path, limit=limit)
    prompt_template = load_text(PROMPT_PATH)
    tasks, stats = build_filtered_tasks(rows)
    stats["input_rows"] = len(rows)

    if not tasks:
        output_rows = []
        stats["output_rows"] = 0
        save_json(output_rows, output_path)
        print_summary(stats, model_name, input_path, output_path)
        return {"rows": output_rows, "stats": dict(stats)}

    if model_is_local(model_name):
        if max_completion_tokens is None:
            max_completion_tokens = DEFAULT_LOCAL_MAX_COMPLETION_TOKENS
        output_rows, classification_stats = classify_local_tasks(
            tasks=tasks,
            model_name=model_name,
            prompt_template=prompt_template,
            output_path=output_path,
            max_completion_tokens=max_completion_tokens,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            batch_size=batch_size,
            temperature=temperature,
            top_p=top_p,
            use_chat_template=use_chat_template,
            chat_template_path=chat_template_path,
            trust_remote_code=trust_remote_code,
        )
    else:
        if max_completion_tokens is None:
            max_completion_tokens = DEFAULT_API_MAX_COMPLETION_TOKENS
        output_rows, classification_stats = classify_api_tasks(
            tasks=tasks,
            model_name=model_name,
            prompt_template=prompt_template,
            output_path=output_path,
            vertex_project_id=vertex_project_id,
            max_completion_tokens=max_completion_tokens,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
        )

    stats.update(classification_stats)
    stats["output_rows"] = len(output_rows)
    save_json(output_rows, output_path)
    print_summary(stats, model_name, input_path, output_path)
    return {"rows": output_rows, "stats": dict(stats)}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Classify diagnostic response strategy for dual-use prompts whose "
            "model responses had harmfulness > 0."
        )
    )
    parser.add_argument(
        "model_name",
        help=(
            "Model alias. Defaults input to "
            "evaluation/model_responses/<model_name>_scored.json."
        ),
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path. Defaults to "
            "analysis/intent_classification/<input stem>_intent_classification.json."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For debugging, load only the first N datapoints.",
    )
    parser.add_argument(
        "--vertex-project-id",
        default=DEFAULT_VERTEX_PROJECT_ID,
        help="Google Cloud project ID used for API-backed Vertex models.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Maximum tokens to request for each intent classification.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_API_MAX_RETRIES,
        help="Maximum retries for transient API provider errors.",
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=DEFAULT_API_RETRY_BASE_SECONDS,
        help="Initial retry delay for transient API provider errors.",
    )
    parser.add_argument(
        "--retry-max-seconds",
        type=float,
        default=DEFAULT_API_RETRY_MAX_SECONDS,
        help="Maximum retry delay for transient API provider errors.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
        help="Number of GPUs vLLM should shard local models across.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="vLLM dtype setting for local models.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="Fraction of GPU memory vLLM can use for local models.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional vLLM max_model_len override for local models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Local vLLM batch size.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Local vLLM nucleus sampling top-p.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="For local models, generate from raw prompt text.",
    )
    parser.add_argument(
        "--chat-template",
        type=Path,
        default=None,
        help="Optional Jinja chat-template file for local models.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to vLLM for local models.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    classify_intents(
        model_name=args.model_name,
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        vertex_project_id=args.vertex_project_id,
        max_completion_tokens=args.max_completion_tokens,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        retry_max_seconds=args.retry_max_seconds,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        use_chat_template=not args.no_chat_template,
        chat_template_path=args.chat_template,
        trust_remote_code=args.trust_remote_code,
    )
