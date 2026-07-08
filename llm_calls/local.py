import os
import re
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from dataset_generation.utils import load_text
from llm_calls.models import LOCAL_MODEL_CONFIGS
from OpenSafeIntent.project_config import (
    DEFAULT_BATCH_SIZE as CONFIG_DEFAULT_BATCH_SIZE,
    DEFAULT_GPU_MEMORY_UTILIZATION as CONFIG_DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_TENSOR_PARALLEL_SIZE as CONFIG_DEFAULT_TENSOR_PARALLEL_SIZE,
    MAX_COMPLETION_TOKENS as CONFIG_MAX_COMPLETION_TOKENS,
    get_config_section,
)


DEFAULT_MAX_COMPLETION_TOKENS = CONFIG_MAX_COMPLETION_TOKENS
DEFAULT_TENSOR_PARALLEL_SIZE = CONFIG_DEFAULT_TENSOR_PARALLEL_SIZE
DEFAULT_GPU_MEMORY_UTILIZATION = CONFIG_DEFAULT_GPU_MEMORY_UTILIZATION
DEFAULT_BATCH_SIZE = CONFIG_DEFAULT_BATCH_SIZE
DEFAULT_VLLM_USE_V1 = "0"
HF_TOKEN_CONFIG_KEYS = (
    "huggingface_token",
    "huggingface_hub_token",
    "hf_token",
)


@dataclass
class LocalGenerationBackend:
    llm: Any
    sampling_params: Any
    use_chat_template: bool = True
    chat_template: Optional[str] = None

    def generate(self, prompts: Sequence[str]) -> list[str]:
        return generate_responses_for_prompts(
            llm=self.llm,
            prompts=prompts,
            sampling_params=self.sampling_params,
            use_chat_template=self.use_chat_template,
            chat_template=self.chat_template,
        )


def safe_model_output_name(model_name):
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "__", model_name).strip("._-")
    return safe_name or "local_model"


def resolve_local_model_config(model_name):
    config = LOCAL_MODEL_CONFIGS.get(model_name)
    if config is None:
        return model_name, {}

    resolved_name = config.model_name
    vllm_kwargs = dict(config.vllm_kwargs or {})
    return resolved_name, vllm_kwargs


def get_config_string(config, keys):
    for key in keys:
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def get_hf_token():
    return get_config_string(get_config_section("keys"), HF_TOKEN_CONFIG_KEYS)


def configure_local_vllm_runtime():
    os.environ.setdefault("VLLM_USE_V1", DEFAULT_VLLM_USE_V1)


def import_vllm():
    try:
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise ImportError(
            "vLLM is required for local inference. Install it in the runtime "
            "environment before running this script."
        ) from error

    return LLM, SamplingParams


def validate_tensor_parallel_size(tensor_parallel_size):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        num_visible = len([x for x in visible.split(",") if x.strip()])
        if tensor_parallel_size > num_visible:
            raise ValueError(
                f"--tensor-parallel-size={tensor_parallel_size} but only "
                f"{num_visible} GPUs are visible via CUDA_VISIBLE_DEVICES={visible!r}."
            )


def build_llm(
    model_name,
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    dtype="auto",
    gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len=None,
    trust_remote_code=False,
    hf_token=None,
    extra_vllm_kwargs=None,
):
    LLM, _ = import_vllm()
    kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
        "hf_token": hf_token,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if extra_vllm_kwargs is not None:
        kwargs.update(extra_vllm_kwargs)

    return LLM(**kwargs)


def build_sampling_params(
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    temperature=0.0,
    top_p=1.0,
):
    _, SamplingParams = import_vllm()
    return SamplingParams(
        max_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def format_generation_prompts(
    llm,
    prompts,
    use_chat_template=True,
    chat_template=None,
):
    if not use_chat_template:
        return prompts

    tokenizer = llm.get_tokenizer()
    if chat_template is None and not getattr(tokenizer, "chat_template", None):
        tqdm.write(
            "Tokenizer has no chat template; generating from raw prompt text."
        )
        return prompts

    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            chat_template=chat_template,
        )
        for prompt in prompts
    ]


def generate_responses_for_prompts(
    llm,
    prompts,
    sampling_params,
    use_chat_template=True,
    chat_template=None,
):
    formatted_prompts = format_generation_prompts(
        llm,
        prompts,
        use_chat_template=use_chat_template,
        chat_template=chat_template,
    )
    outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=False)
    return [output.outputs[0].text for output in outputs]


def build_local_generation_backend(
    model_name,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    dtype="auto",
    gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len=None,
    temperature=0.0,
    top_p=1.0,
    use_chat_template=True,
    chat_template_path=None,
    trust_remote_code=False,
    configure_runtime=True,
):
    if configure_runtime:
        configure_local_vllm_runtime()

    resolved_model_name, extra_vllm_kwargs = resolve_local_model_config(model_name)
    chat_template = (
        load_text(Path(chat_template_path))
        if chat_template_path is not None
        else None
    )
    validate_tensor_parallel_size(tensor_parallel_size)
    llm = build_llm(
        model_name=resolved_model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        hf_token=get_hf_token(),
        extra_vllm_kwargs=extra_vllm_kwargs,
    )
    sampling_params = build_sampling_params(
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return LocalGenerationBackend(
        llm=llm,
        sampling_params=sampling_params,
        use_chat_template=use_chat_template,
        chat_template=chat_template,
    )
