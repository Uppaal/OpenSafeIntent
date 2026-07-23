from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


Provider = Literal[
    "vertex_maas",
    "vertex_claude",
    "vertex_gemini",
    "vertex_gemini_embedding",
    "vertex_mistral",
]
EmbeddingTaskType = Literal[
    "UNSPECIFIED",
    "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
    "CODE_RETRIEVAL_QUERY",
]

GeminiThinkingLevel = Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"]

@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for one managed/non-deployed Vertex model.
    """

    provider: Provider
    model_id: str
    location: str
    request_model: Optional[str] = None
    supports_temperature: bool = True
    gemini_thinking_level: Optional[GeminiThinkingLevel] = None
    gemini_thinking_budget: Optional[int] = None


@dataclass(frozen=True)
class LocalModelConfig:
    """
    Configuration for one local/Hugging Face model.
    """

    model_name: str
    response_alias: Optional[str] = None
    vllm_kwargs: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class BatchPromptResult:
    index: int
    prompt: str
    success: bool
    response_text: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class BatchEmbeddingResult:
    index: int
    text: str
    success: bool
    embedding: Optional[List[float]] = None
    error: Optional[str] = None


MINISTRAL_VLLM_KWARGS: Dict[str, str] = {
    "tokenizer_mode": "mistral",
    "config_format": "mistral",
    "load_format": "mistral",
}

LOCAL_MODEL_CONFIGS: Dict[str, LocalModelConfig] = {
    "Qwen/Qwen3-4B": LocalModelConfig(
        model_name="Qwen/Qwen3-4B",
        response_alias="Qwen__Qwen3-4B",
    ),
    "qwen3_5_4b": LocalModelConfig(
        model_name="Qwen/Qwen3.5-4B",
    ),
    "Qwen/Qwen3.5-4B": LocalModelConfig(
        model_name="Qwen/Qwen3.5-4B",
    ),
    "qwen3_32b": LocalModelConfig(
        model_name="Qwen/Qwen3-32B",
        response_alias="Qwen__Qwen3-32B",
    ),
    "Qwen/Qwen3-32B": LocalModelConfig(
        model_name="Qwen/Qwen3-32B",
        response_alias="Qwen__Qwen3-32B",
    ),
    "deepseek_r1_distill_llama_8b": LocalModelConfig(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        response_alias="deepseek-ai__DeepSeek-R1-Distill-Llama-8B",
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": LocalModelConfig(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        response_alias="deepseek-ai__DeepSeek-R1-Distill-Llama-8B",
    ),
    "mistralai/Mistral-Small-24B-Instruct-2501": LocalModelConfig(
        model_name="mistralai/Mistral-Small-24B-Instruct-2501",
        response_alias="mistralai__Mistral-Small-24B-Instruct-2501",
    ),
    "ministral_3_8b_instruct_2512": LocalModelConfig(
        model_name="mistralai/Ministral-3-8B-Instruct-2512",
        vllm_kwargs=MINISTRAL_VLLM_KWARGS,
    ),
    "mistralai/Ministral-3-8B-Instruct-2512": LocalModelConfig(
        model_name="mistralai/Ministral-3-8B-Instruct-2512",
        vllm_kwargs=MINISTRAL_VLLM_KWARGS,
    ),
    "meta-llama/Llama-3.1-8B-Instruct": LocalModelConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        response_alias="meta-llama__Llama-3.1-8B-Instruct",
    ),
}

LOCAL_MODEL_ALIASES: Dict[str, str] = {
    config.model_name: config.response_alias
    for config in LOCAL_MODEL_CONFIGS.values()
    if config.response_alias is not None
}
LOCAL_ALIAS_TO_MODEL: Dict[str, str] = {
    alias: model_name for model_name, alias in LOCAL_MODEL_ALIASES.items()
}
LOCAL_MODEL_NAMES = frozenset(LOCAL_MODEL_ALIASES) | frozenset(LOCAL_ALIAS_TO_MODEL)


DEFAULT_GENERATION_MODELS: Dict[str, ModelConfig] = {
    "llama4_scout": ModelConfig(
        provider="vertex_maas",
        model_id="llama-4-scout-17b-16e-instruct-maas",
        location="us-east5",
    ),
    "llama4_maverick": ModelConfig(
        provider="vertex_maas",
        model_id="llama-4-maverick-17b-128e-instruct-maas",
        location="us-east5",
    ),
    "llama33_70b": ModelConfig(
        provider="vertex_maas",
        model_id="llama-3.3-70b-instruct-maas",
        location="us-central1",
    ),
    "gpt_oss_120b": ModelConfig(
        provider="vertex_maas",
        model_id="gpt-oss-120b-maas",
        location="global",
        request_model="openai/gpt-oss-120b-maas",
    ),
    "gpt_oss_20b": ModelConfig(
        provider="vertex_maas",
        model_id="gpt-oss-20b-maas",
        location="global",
        request_model="openai/gpt-oss-20b-maas",
    ),
    "mistral_small_2503": ModelConfig(
        provider="vertex_mistral",
        model_id="mistral-small-2503",
        location="us-central1",
    ),
    "mistral_medium_3": ModelConfig(
        provider="vertex_mistral",
        model_id="mistral-medium-3",
        location="us-central1",
    ),
    "gemma_4_26b_a4b_it": ModelConfig(
        provider="vertex_maas",
        model_id="gemma-4-26b-a4b-it-maas",
        location="global",
        request_model="google/gemma-4-26b-a4b-it-maas",
    ),
    "qwen3_next_80b_a3b_instruct": ModelConfig(
        provider="vertex_maas",
        model_id="qwen3-next-80b-a3b-instruct-maas",
        location="global",
        request_model="qwen/qwen3-next-80b-a3b-instruct-maas",
    ),
    "deepseek_v3_1": ModelConfig(
        provider="vertex_maas",
        model_id="deepseek-v3.1-maas",
        location="us-west2",
        request_model="deepseek-ai/deepseek-v3.1-maas",
    ),
    "deepseek_r1_0528": ModelConfig(
        provider="vertex_maas",
        model_id="deepseek-r1-0528-maas",
        location="us-central1",
        request_model="deepseek-ai/deepseek-r1-0528-maas",
    ),
    "claude_haiku_4_5": ModelConfig(
        provider="vertex_claude",
        model_id="claude-haiku-4-5",
        location="us-east5",
    ),
    "claude_sonnet_4_6": ModelConfig(
        provider="vertex_claude",
        model_id="claude-sonnet-4-6",
        location="us-east5",
    ),
    "claude_opus_4_8": ModelConfig(
        provider="vertex_claude",
        model_id="claude-opus-4-8",
        location="global",
        supports_temperature=False,
    ),
    "gemini_2_5_flash": ModelConfig(
        provider="vertex_gemini",
        model_id="gemini-2.5-flash",
        location="us-central1",
    ),
    "gemini_2_5_pro": ModelConfig(
        provider="vertex_gemini",
        model_id="gemini-2.5-pro",
        location="us-central1",
    ),
    "gemini_2_0_flash": ModelConfig(
        provider="vertex_gemini",
        model_id="gemini-2.0-flash",
        location="us-central1",
    ),
    "gemini_3_flash_preview": ModelConfig(
        provider="vertex_gemini",
        model_id="gemini-3-flash-preview",
        location="global",
        gemini_thinking_level="MINIMAL",
    ),
    "gemini_3_pro_preview": ModelConfig(
        provider="vertex_gemini",
        model_id="gemini-3-pro-preview",
        location="global",
        gemini_thinking_level="LOW",
    ),
    "gemini_3_1_flash_lite": ModelConfig(
        provider="vertex_gemini",
        model_id="gemini-3.1-flash-lite",
        location="global",
        gemini_thinking_level="MINIMAL",
    ),
    "gemini_3_1_pro_preview": ModelConfig(
        provider="vertex_gemini",
        model_id="gemini-3.1-pro-preview",
        location="global",
    ),
}


DEFAULT_EMBEDDING_MODELS: Dict[str, ModelConfig] = {
    "gemini_embedding_001": ModelConfig(
        provider="vertex_gemini_embedding",
        model_id="gemini-embedding-001",
        location="global",
    ),
}


DEFAULT_MODELS: Dict[str, ModelConfig] = {
    **DEFAULT_GENERATION_MODELS,
    **DEFAULT_EMBEDDING_MODELS,
}
