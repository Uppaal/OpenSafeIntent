import re
import sys
import time
from tqdm import tqdm
from typing import Any, Callable, Dict, Iterable, List, Optional

from llm_calls.models import DEFAULT_GENERATION_MODELS
from project_config import (
    DEFAULT_GENERATOR_MODEL,
    MAX_COMPLETION_TOKENS,
    VERTEX_PROJECT_ID,
)


AZURE_MODEL_NAME = DEFAULT_GENERATOR_MODEL
DEFAULT_VERTEX_PROJECT_ID = VERTEX_PROJECT_ID
DEFAULT_MAX_COMPLETION_TOKENS = MAX_COMPLETION_TOKENS
DEFAULT_MAX_RETRIES = 10
DEFAULT_RETRY_BASE_SECONDS = 60
DEFAULT_RETRY_MAX_SECONDS = 300
VERTEX_MODEL_NAMES = tuple(sorted(DEFAULT_GENERATION_MODELS))
SUPPORTED_MODEL_NAMES = tuple(
    dict.fromkeys((DEFAULT_GENERATOR_MODEL, *VERTEX_MODEL_NAMES))
)
FILTERED_RESPONSE_TEXT = "I cannot answer that."
FILTER_ERROR_MARKERS = (
    "content_filter",
    "cyber_policy",
    "responsibleaipolicyviolation",
    "content was flagged",
    "cybersecurity risk",
    "response was filtered",
    "prompt was blocked",
    "safety",
)
RETRYABLE_ERROR_MARKERS = (
    "429",
    "resource_exhausted",
    "resource exhausted",
    "rate_limit",
    "rate limit",
    "too many requests",
    "quota exceeded",
    "retryable gemini",
    "overloaded",
    "service unavailable",
    "currently unavailable",
    "temporarily unavailable",
    "try again later",
)


def log_progress(message):
    tqdm.write(message, file=sys.stderr)


def is_vertex_model(model_name):
    return model_name in DEFAULT_GENERATION_MODELS


def is_configured_azure_model(model_name):
    return model_name == AZURE_MODEL_NAME and not is_vertex_model(model_name)


def get_supported_models_text():
    return ", ".join(SUPPORTED_MODEL_NAMES)


def resolve_deployment_name(model_name):
    if is_vertex_model(model_name):
        raise ValueError(f"{model_name!r} is a Vertex model alias, not an Azure model.")

    if model_name != AZURE_MODEL_NAME:
        raise ValueError(
            f"Unsupported model {model_name!r}. Supported models: "
            f"{get_supported_models_text()}."
        )

    from llm_calls.azure import MODEL_NAME

    return MODEL_NAME


def call_azure_model(
    prompt,
    deployment_name,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    temperature=None,
    system_prompt=None,
):
    from llm_calls.azure import get_client

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "model": deployment_name,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = get_client().chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError("Azure OpenAI response contained no text content.")

    return content


def call_vertex_model(
    prompt,
    client,
    model_name,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    temperature=None,
    thinking_level=None,
):
    kwargs = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_completion_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if thinking_level is not None:
        kwargs["thinking_level"] = thinking_level

    content = client.generate(**kwargs)
    if not isinstance(content, str):
        raise ValueError("Vertex model response contained no text content.")

    return content


def build_model_caller(
    model_name,
    vertex_project_id=DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    temperature=None,
    system_prompt=None,
    thinking_level=None,
):
    if model_name in VERTEX_MODEL_NAMES:
        from llm_calls.vertex import VertexLLMClient

        client = VertexLLMClient(project_id=vertex_project_id)
        client.validate_credentials()

        def call_model(prompt):
            return call_vertex_model(
                prompt,
                client=client,
                model_name=model_name,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                thinking_level=thinking_level,
            )

        return call_model

    if is_configured_azure_model(model_name):
        deployment_name = resolve_deployment_name(model_name)

        def call_model(prompt):
            return call_azure_model(
                prompt,
                deployment_name=deployment_name,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            )

        return call_model

    raise ValueError(
        f"Unsupported model {model_name!r}. Supported models: "
        f"{get_supported_models_text()}."
    )


def get_error_text(error):
    error_parts = [str(error)]
    body = getattr(error, "body", None)
    if body is not None:
        error_parts.append(str(body))

    response = getattr(error, "response", None)
    response_text = getattr(response, "text", None)
    if response_text:
        error_parts.append(str(response_text))

    return "\n".join(error_parts).lower()


def is_provider_filter_error(error):
    error_text = get_error_text(error)

    return any(marker in error_text for marker in FILTER_ERROR_MARKERS)


def is_retryable_provider_error(error):
    error_type = error.__class__.__name__
    if error_type == "VertexTransientResponseError":
        return True
    if error_type in {
        "VertexContentFilteredError",
        "VertexGeminiMaxTokensError",
        "VertexResponseFormatError",
    }:
        return False

    error_text = get_error_text(error)
    status_code = getattr(getattr(error, "response", None), "status_code", None)
    return status_code == 429 or any(
        marker in error_text for marker in RETRYABLE_ERROR_MARKERS
    )


def get_retry_delay(attempt, retry_base_seconds, retry_max_seconds):
    return min(retry_max_seconds, retry_base_seconds * (2**attempt))


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


def get_api_model_response(
    prompt: str,
    model_name: str,
    vertex_project_id: str = DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None,
    thinking_level: Optional[str] = None,
    model_caller: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Execute one prompt against an API-backed model alias.

    `model_caller` lets batch callers reuse an already-initialized provider client
    while still routing through this single prompt-level entrypoint.
    """

    call_model = model_caller or build_model_caller(
        model_name=model_name,
        vertex_project_id=vertex_project_id,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        thinking_level=thinking_level,
    )
    return call_model(prompt)


def _success_result(index, prompt, response, retry_attempts):
    return {
        "index": index,
        "prompt": prompt,
        "response": response,
        "success": True,
        "filtered": False,
        "retry_attempts": retry_attempts,
        "api_calls": 1 + retry_attempts,
        "error": None,
        "error_type": None,
    }


def _failure_result(index, prompt, error, retry_attempts, filtered=False):
    response = FILTERED_RESPONSE_TEXT if filtered else None
    return {
        "index": index,
        "prompt": prompt,
        "response": response,
        "success": False,
        "filtered": filtered,
        "retry_attempts": retry_attempts,
        "api_calls": 1 + retry_attempts,
        "error": summarize_error(error),
        "error_type": error.__class__.__name__,
    }


def get_api_responses_batch(
    prompts: Iterable[str],
    model_name: str,
    vertex_project_id: str = DEFAULT_VERTEX_PROJECT_ID,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None,
    thinking_level: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
    desc: Optional[str] = None,
    show_progress: bool = False,
    log_failures: bool = True,
    raise_on_error: bool = False,
) -> List[Dict[str, Any]]:
    """
    Execute a batch of prompts through the provider-agnostic API model layer.

    The batch wrapper owns retry/backoff, provider filtering, and per-prompt
    exception capture. Set `raise_on_error=True` when callers want unrecoverable
    provider failures to abort the surrounding workflow.
    """

    prompt_list = list(prompts)
    if not prompt_list:
        return []

    call_model = build_model_caller(
        model_name=model_name,
        vertex_project_id=vertex_project_id,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        thinking_level=thinking_level,
    )
    indexed_prompts = enumerate(prompt_list)
    if show_progress:
        indexed_prompts = tqdm(
            indexed_prompts,
            total=len(prompt_list),
            desc=desc or f"Calling {model_name}",
        )

    results = []
    for index, prompt in indexed_prompts:
        for attempt in range(max_retries + 1):
            try:
                response = get_api_model_response(
                    prompt=prompt,
                    model_name=model_name,
                    vertex_project_id=vertex_project_id,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt,
                    thinking_level=thinking_level,
                    model_caller=call_model,
                )
                results.append(_success_result(index, prompt, response, attempt))
                break
            except Exception as error:
                if is_provider_filter_error(error):
                    result = _failure_result(
                        index=index,
                        prompt=prompt,
                        error=error,
                        retry_attempts=attempt,
                        filtered=True,
                    )
                    if log_failures:
                        log_progress(
                            f"Provider filtered prompt {index}: {result['error']}"
                        )
                    results.append(result)
                    break

                if is_retryable_provider_error(error) and attempt < max_retries:
                    retry_delay = get_retry_delay(
                        attempt,
                        retry_base_seconds=retry_base_seconds,
                        retry_max_seconds=retry_max_seconds,
                    )
                    log_progress(
                        "Transient provider error; retrying in "
                        f"{retry_delay} seconds: {summarize_error(error)}"
                    )
                    time.sleep(retry_delay)
                    continue

                if raise_on_error:
                    raise

                result = _failure_result(
                    index=index,
                    prompt=prompt,
                    error=error,
                    retry_attempts=attempt,
                    filtered=False,
                )
                if log_failures:
                    log_progress(f"API prompt {index} failed: {result['error']}")
                results.append(result)
                break

    return results


def call_model_for_prompt(
    call_model,
    prompt,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_base_seconds=DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds=DEFAULT_RETRY_MAX_SECONDS,
):
    for attempt in range(max_retries + 1):
        try:
            return call_model(prompt), True, attempt
        except Exception as error:
            if is_provider_filter_error(error):
                return FILTERED_RESPONSE_TEXT, False, attempt

            if not is_retryable_provider_error(error) or attempt == max_retries:
                raise

            retry_delay = get_retry_delay(
                attempt,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
            )
            log_progress(
                "Transient provider error; retrying in "
                f"{retry_delay} seconds: {summarize_error(error)}"
            )
            time.sleep(retry_delay)
