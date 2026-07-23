import sys
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_generation.utils import load_json, save_json
from llm_calls.local import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    build_local_generation_backend,
    safe_model_output_name,
)
from project_config import RESPONSE_OUTPUT_DIR


FINAL_DATASET_PATH = REPO_ROOT / "dataset.json"
MODEL_RESPONSES_OUTPUT_DIR = RESPONSE_OUTPUT_DIR
PROMPT_FIELDS = ("benign_prompt", "dual_use_prompt", "malicious_prompt")
PARAPHRASE_FIELD = "dual_use_paraphrases"


def load_final_dataset(input_path=FINAL_DATASET_PATH, limit=None):
    rows = load_json(input_path)
    if not isinstance(rows, list):
        raise ValueError("Evaluation dataset must be a list of datapoints.")

    return rows[:limit] if limit is not None else rows


def get_required_generated_prompt(row, prompt_field):
    generated_prompts = row.get("generated_prompts", {})
    if not isinstance(generated_prompts, dict):
        raise ValueError("Datapoint generated_prompts was not an object.")

    prompt = generated_prompts.get(prompt_field)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Datapoint is missing generated_prompts.{prompt_field}.")

    return prompt


def get_required_paraphrases(row):
    paraphrases = row.get(PARAPHRASE_FIELD)
    if not isinstance(paraphrases, list):
        raise ValueError(f"Datapoint {PARAPHRASE_FIELD} was not a list.")

    for index, prompt in enumerate(paraphrases):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Datapoint {PARAPHRASE_FIELD}[{index}] contained no prompt text."
            )

    return paraphrases


def get_datapoint_prompts(row):
    prompts = [
        (prompt_field, get_required_generated_prompt(row, prompt_field))
        for prompt_field in PROMPT_FIELDS
    ]
    prompts.extend(
        (PARAPHRASE_FIELD, prompt)
        for prompt in get_required_paraphrases(row)
    )
    return prompts


def add_model_responses(row, model_responses):
    output_row = deepcopy(row)
    output_row["model_responses"] = model_responses
    return output_row


def default_output_path(model_name):
    return MODEL_RESPONSES_OUTPUT_DIR / f"{safe_model_output_name(model_name)}.json"


def assign_response(model_responses, prompt_field, response):
    if prompt_field == PARAPHRASE_FIELD:
        model_responses[PARAPHRASE_FIELD].append(response)
    else:
        model_responses[prompt_field] = response


def flush_completed_rows(output_rows, pending_rows, output_path=None):
    completed_indexes = []
    for row_index, pending in pending_rows.items():
        if pending["remaining"] != 0:
            continue

        output_rows[row_index] = add_model_responses(
            pending["row"],
            pending["model_responses"],
        )
        completed_indexes.append(row_index)

    for row_index in completed_indexes:
        del pending_rows[row_index]

    if output_path is not None and completed_indexes:
        save_json([row for row in output_rows if row is not None], output_path)


def iter_chunks(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def process_rows(
    rows,
    backend,
    output_path=None,
    batch_size=DEFAULT_BATCH_SIZE,
):
    output_rows = [None] * len(rows)
    pending_rows = {}
    generation_tasks = []
    stats = Counter()

    for row_index, row in enumerate(rows):
        try:
            prompts = get_datapoint_prompts(row)
        except Exception as error:
            stats["skipped_datapoints"] += 1
            stats["datapoint_errors"] += 1
            tqdm.write(f"Skipping datapoint {row_index}: {error}")
            continue

        pending_rows[row_index] = {
            "row": row,
            "remaining": len(prompts),
            "model_responses": {
                PROMPT_FIELDS[0]: "",
                PROMPT_FIELDS[1]: "",
                PROMPT_FIELDS[2]: "",
                PARAPHRASE_FIELD: [],
            },
        }
        for prompt_field, prompt in prompts:
            generation_tasks.append(
                {
                    "row_index": row_index,
                    "prompt_field": prompt_field,
                    "prompt": prompt,
                }
            )

    progress = tqdm(
        total=len(generation_tasks),
        desc="Generating local model responses",
    )
    for task_batch in iter_chunks(generation_tasks, batch_size):
        prompts = [task["prompt"] for task in task_batch]
        responses = backend.generate(prompts)
        for task, response in zip(task_batch, responses):
            pending = pending_rows[task["row_index"]]
            assign_response(
                pending["model_responses"],
                task["prompt_field"],
                response,
            )
            pending["remaining"] -= 1
            stats["successful_calls"] += 1

        stats["local_generations"] += len(task_batch)
        progress.update(len(task_batch))
        flush_completed_rows(output_rows, pending_rows, output_path=output_path)

    progress.close()

    output_rows = [row for row in output_rows if row is not None]
    stats["input_rows"] = len(rows)
    stats["evaluated_rows"] = len(output_rows)
    stats["output_rows"] = len(output_rows)
    return output_rows, stats


def print_summary(stats, model_name):
    print(f"\n{model_name} Local Evaluation Summary")
    print("-" * 40)
    for field in (
        "input_rows",
        "evaluated_rows",
        "local_generations",
        "successful_calls",
        "skipped_datapoints",
        "datapoint_errors",
        "output_rows",
    ):
        print(f"{field}: {stats.get(field, 0)}")


def generate_model_responses(
    model_name,
    input_path=FINAL_DATASET_PATH,
    output_path=None,
    limit=None,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    dtype="auto",
    gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len=None,
    batch_size=DEFAULT_BATCH_SIZE,
    temperature=0.0,
    top_p=1.0,
    use_chat_template=True,
    chat_template_path=None,
    trust_remote_code=False,
):
    if output_path is None:
        output_path = default_output_path(model_name)

    rows = load_final_dataset(input_path=input_path, limit=limit)
    backend = build_local_generation_backend(
        model_name=model_name,
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
    output_rows, stats = process_rows(
        rows=rows,
        backend=backend,
        output_path=output_path,
        batch_size=batch_size,
    )
    save_json(output_rows, output_path)
    print(f"Saved model responses to: {Path(output_path).resolve()}")
    print_summary(stats, model_name)
    return {"rows": output_rows, "stats": dict(stats)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate local vLLM responses for each final datapoint."
    )
    parser.add_argument(
        "model_name",
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument("--input", type=Path, default=FINAL_DATASET_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path; defaults to "
            "evaluation/model_responses/<safe_model_name>.json."
        ),
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
        help="Maximum new tokens to generate for each prompt.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
        help="Number of GPUs vLLM should shard the model across.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="vLLM dtype setting, for example auto, float16, or bfloat16.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="Fraction of GPU memory vLLM can use.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional vLLM max_model_len override.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of prompts to submit to vLLM per generate call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Defaults to greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Generate from raw prompt text instead of a user-only chat template.",
    )
    parser.add_argument(
        "--chat-template",
        type=Path,
        default=None,
        help=(
            "Optional Jinja chat-template file to use instead of the "
            "tokenizer's built-in chat template."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to vLLM.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_model_responses(
        model_name=args.model_name,
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        max_completion_tokens=args.max_completion_tokens,
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
