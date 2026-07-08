import json
import requests
import google.auth
import google.auth.transport.requests
from typing import Any, Dict, List, Optional
from google.auth import exceptions as google_auth_exceptions
from llm_calls.models import (
    BatchEmbeddingResult,
    BatchPromptResult,
    DEFAULT_MODELS,
    EmbeddingTaskType,
    ModelConfig,
)


ADC_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
GEMINI_THINKING_LEVELS = {"MINIMAL", "LOW", "MEDIUM", "HIGH"}

class VertexContentFilteredError(RuntimeError):
    """Raised when Vertex returns a successful response with no usable text."""

class VertexGeminiMaxTokensError(RuntimeError):
    """Raised when Gemini exhausts the output budget before producing text."""

class VertexTransientResponseError(RuntimeError):
    """Raised when Vertex returns a retryable provider condition in a 200 body."""


class VertexResponseFormatError(RuntimeError):
    """Raised when Vertex returns a response shape this client cannot parse."""


class VertexLLMClient:
    """
    Client for managed/non-deployed Vertex LLMs.

    This deliberately does NOT support deployed endpoints.
    """

    def __init__(
        self,
        project_id: str,
        models: Optional[Dict[str, ModelConfig]] = None,
        timeout_seconds: int = 120,
    ):
        self.project_id = project_id
        self.models = models or DEFAULT_MODELS
        self.timeout_seconds = timeout_seconds
        self._credentials = None
        self._auth_request = google.auth.transport.requests.Request()

    def _get_access_token(self) -> str:
        try:
            if self._credentials is None:
                self._credentials, _ = google.auth.default(scopes=ADC_SCOPES)

            if not self._credentials.valid:
                self._credentials.refresh(self._auth_request)

        except (
            google_auth_exceptions.DefaultCredentialsError,
            google_auth_exceptions.RefreshError,
        ) as exc:
            raise RuntimeError(
                "Vertex authentication failed while refreshing ADC.\n"
                f"Underlying error: {exc!r}"
            ) from exc

        return self._credentials.token

    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        thinking_level: Optional[str] = None,
    ) -> str:
        if model not in self.models:
            known = ", ".join(sorted(self.models))
            raise ValueError(f"Unknown model alias '{model}'. Known aliases: {known}")

        config = self.models[model]
        self._validate_no_deployed_endpoint(config)

        if config.provider == "vertex_maas":
            return self._call_vertex_maas(config, prompt, max_tokens, temperature)
        if config.provider == "vertex_claude":
            return self._call_vertex_claude(config, prompt, max_tokens, temperature)
        if config.provider == "vertex_gemini":
            return self._call_vertex_gemini(
                config,
                prompt,
                max_tokens,
                temperature,
                thinking_level=thinking_level,
            )
        if config.provider == "vertex_mistral":
            return self._call_vertex_mistral(config, prompt, max_tokens, temperature)
        if config.provider == "vertex_gemini_embedding":
            raise ValueError(
                f"Model alias '{model}' is an embedding model. Use embed() instead."
            )

        raise ValueError(f"Unsupported provider: {config.provider}")

    def embed(
        self,
        model: str,
        text: str,
        task_type: Optional[EmbeddingTaskType] = None,
        output_dimensionality: Optional[int] = None,
        title: Optional[str] = None,
        auto_truncate: Optional[bool] = None,
    ) -> List[float]:
        if model not in self.models:
            known = ", ".join(sorted(self.models))
            raise ValueError(f"Unknown model alias '{model}'. Known aliases: {known}")

        config = self.models[model]
        self._validate_no_deployed_endpoint(config)

        if config.provider != "vertex_gemini_embedding":
            raise ValueError(
                f"Model alias '{model}' is not an embedding model. Use generate() instead."
            )

        return self._call_vertex_gemini_embedding(
            config=config,
            text=text,
            task_type=task_type,
            output_dimensionality=output_dimensionality,
            title=title,
            auto_truncate=auto_truncate,
        )

    def generate_batch(
        self,
        model: str,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.2,
        thinking_level: Optional[str] = None,
    ) -> List[BatchPromptResult]:
        results: List[BatchPromptResult] = []

        for index, prompt in enumerate(prompts):
            try:
                response_text = self.generate(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking_level=thinking_level,
                )
                results.append(
                    BatchPromptResult(
                        index=index,
                        prompt=prompt,
                        success=True,
                        response_text=response_text,
                    )
                )
            except Exception as exc:
                results.append(
                    BatchPromptResult(
                        index=index,
                        prompt=prompt,
                        success=False,
                        error=str(exc),
                    )
                )

        return results

    def embed_batch(
        self,
        model: str,
        texts: List[str],
        task_type: Optional[EmbeddingTaskType] = None,
        output_dimensionality: Optional[int] = None,
        title: Optional[str] = None,
        auto_truncate: Optional[bool] = None,
    ) -> List[BatchEmbeddingResult]:
        results: List[BatchEmbeddingResult] = []

        for index, text in enumerate(texts):
            try:
                embedding = self.embed(
                    model=model,
                    text=text,
                    task_type=task_type,
                    output_dimensionality=output_dimensionality,
                    title=title,
                    auto_truncate=auto_truncate,
                )
                results.append(
                    BatchEmbeddingResult(
                        index=index,
                        text=text,
                        success=True,
                        embedding=embedding,
                    )
                )
            except Exception as exc:
                results.append(
                    BatchEmbeddingResult(
                        index=index,
                        text=text,
                        success=False,
                        error=str(exc),
                    )
                )

        return results

    def validate_credentials(self) -> None:
        self._get_access_token()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json",
        }

    def _api_host(self, location: str) -> str:
        return (
            "aiplatform.googleapis.com"
            if location == "global"
            else f"{location}-aiplatform.googleapis.com"
        )

    def _validate_no_deployed_endpoint(self, config: ModelConfig) -> None:
        suspicious_values = [config.model_id, config.location]

        for value in suspicious_values:
            lowered = value.lower()

            if "/endpoints/" in lowered:
                raise ValueError(
                    "This client is for managed/non-deployed models only. "
                    "Detected a deployed endpoint resource."
                )
            if lowered.startswith("projects/"):
                raise ValueError(
                    "This client is for managed/non-deployed models only. "
                    "Detected a full Google Cloud resource path."
                )
            if value.strip().isdigit():
                raise ValueError(
                    "This client is for managed/non-deployed models only. "
                    "Detected a numeric endpoint ID."
                )

    def _build_gemini_thinking_config(
        self,
        thinking_level: Optional[str],
        thinking_budget: Optional[int],
    ) -> Dict[str, Any]:
        if thinking_level is not None and thinking_budget is not None:
            raise ValueError(
                "Configure either Gemini thinking_level or thinking_budget, not both."
            )

        if thinking_level is not None:
            normalized_level = thinking_level.strip().upper()
            if normalized_level not in GEMINI_THINKING_LEVELS:
                known = ", ".join(sorted(GEMINI_THINKING_LEVELS))
                raise ValueError(
                    f"Unsupported Gemini thinking level {thinking_level!r}. "
                    f"Expected one of: {known}."
                )
            return {"thinkingLevel": normalized_level}

        if thinking_budget is not None:
            if not isinstance(thinking_budget, int) or thinking_budget < 0:
                raise ValueError(
                    "Gemini thinking_budget must be a non-negative integer."
                )
            return {"thinkingBudget": thinking_budget}

        return {}

    def _call_vertex_maas(
        self,
        config: ModelConfig,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        api_host = self._api_host(config.location)
        url = (
            f"https://{api_host}/v1beta1/"
            f"projects/{self.project_id}/locations/{config.location}/"
            f"endpoints/openapi/chat/completions"
        )
        request_model = config.request_model or f"meta/{config.model_id}"

        payload: Dict[str, Any] = {
            "model": request_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        if request_model.startswith("openai/gpt-oss-"):
            payload["reasoning_effort"] = "low"

        response = requests.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        self._raise_for_response(response, url, payload)

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise RuntimeError(
                "Unexpected Vertex MaaS response format:\n"
                f"{json.dumps(data, indent=2)}"
            ) from exc

    def _call_vertex_claude(
        self,
        config: ModelConfig,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        url = (
            f"https://{config.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{config.location}/"
            f"publishers/anthropic/models/{config.model_id}:rawPredict"
        )

        payload: Dict[str, Any] = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        response = requests.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        self._raise_for_response(response, url, payload)

        data = response.json()
        try:
            content = data.get("content", [])
            text_parts = [
                item.get("text", "")
                for item in content
                if item.get("type") == "text" or "text" in item
            ]
            return "\n".join(part for part in text_parts if part).strip()
        except Exception as exc:
            raise RuntimeError(
                "Unexpected Claude-on-Vertex response format:\n"
                f"{json.dumps(data, indent=2)}"
            ) from exc

    def _call_vertex_gemini(
        self,
        config: ModelConfig,
        prompt: str,
        max_tokens: int,
        temperature: float,
        thinking_level: Optional[str] = None,
    ) -> str:
        api_host = self._api_host(config.location)
        url = (
            f"https://{api_host}/v1/"
            f"projects/{self.project_id}/locations/{config.location}/"
            f"publishers/google/models/{config.model_id}:generateContent"
        )

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        effective_thinking_level = (
            thinking_level
            if thinking_level is not None
            else config.gemini_thinking_level
        )

        thinking_config = self._build_gemini_thinking_config(
            thinking_level=effective_thinking_level,
            thinking_budget=config.gemini_thinking_budget,
        )

        if thinking_config:
            payload["generationConfig"]["thinkingConfig"] = thinking_config

        response = requests.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        self._raise_for_response(response, url, payload)

        data = response.json()
        return self._extract_gemini_text(data)

    def _call_vertex_gemini_embedding(
        self,
        config: ModelConfig,
        text: str,
        task_type: Optional[EmbeddingTaskType],
        output_dimensionality: Optional[int],
        title: Optional[str],
        auto_truncate: Optional[bool],
    ) -> List[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding text must be a non-empty string.")
        if output_dimensionality is not None and output_dimensionality <= 0:
            raise ValueError("output_dimensionality must be a positive integer.")
        if title is not None and not title.strip():
            raise ValueError("title must be non-empty when provided.")

        api_host = self._api_host(config.location)
        url = (
            f"https://{api_host}/v1/"
            f"projects/{self.project_id}/locations/{config.location}/"
            f"publishers/google/models/{config.model_id}:predict"
        )

        instance: Dict[str, Any] = {"content": text}
        if task_type is not None:
            instance["task_type"] = task_type
        if title is not None:
            instance["title"] = title

        payload: Dict[str, Any] = {"instances": [instance]}
        parameters: Dict[str, Any] = {}
        if output_dimensionality is not None:
            parameters["outputDimensionality"] = output_dimensionality
        if auto_truncate is not None:
            parameters["autoTruncate"] = auto_truncate
        if parameters:
            payload["parameters"] = parameters

        response = requests.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        self._raise_for_response(response, url, payload)

        data = response.json()
        return self._extract_embedding_values(data)

    def _extract_gemini_text(self, data: Dict[str, Any]) -> str:
        if "error" in data:
            error_text = self._format_provider_json(data["error"])
            if self._looks_retryable(error_text):
                raise VertexTransientResponseError(
                    "Retryable Gemini-on-Vertex response body:\n"
                    f"{self._format_provider_json(data)}"
                )
            raise VertexResponseFormatError(
                "Gemini-on-Vertex response contained an error body:\n"
                f"{self._format_provider_json(data)}"
            )

        prompt_feedback = data.get("promptFeedback")
        if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
            raise VertexContentFilteredError(
                "Gemini-on-Vertex prompt was blocked: "
                f"{self._format_provider_json(prompt_feedback)}"
            )

        try:
            candidates = data["candidates"]
        except Exception as exc:
            raise VertexResponseFormatError(
                "Unexpected Gemini-on-Vertex response format:\n"
                f"{self._format_provider_json(data)}"
            ) from exc

        if not isinstance(candidates, list):
            raise VertexResponseFormatError(
                "Unexpected Gemini-on-Vertex candidates format:\n"
                f"{self._format_provider_json(data)}"
            )
        if not candidates:
            raise VertexContentFilteredError(
                "Gemini-on-Vertex returned no candidates:\n"
                f"{self._format_provider_json(data)}"
            )

        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise VertexResponseFormatError(
                "Unexpected Gemini-on-Vertex candidate format:\n"
                f"{self._format_provider_json(candidate)}"
            )

        content = candidate.get("content")
        if not isinstance(content, dict):
            finish_reason = candidate.get("finishReason")
            message = (
                candidate.get("finishMessage")
                or candidate.get("citationMetadata")
                or candidate.get("safetyRatings")
            )
            detail = self._format_provider_json(candidate)
            if self._looks_retryable(detail):
                raise VertexTransientResponseError(
                    "Retryable Gemini-on-Vertex candidate with no content:\n"
                    f"{detail}"
                )
            if finish_reason == "MAX_TOKENS":
                raise VertexGeminiMaxTokensError(
                    "Gemini-on-Vertex exhausted maxOutputTokens before returning "
                    f"text content (finishReason={finish_reason}, detail={message}):\n"
                    f"{detail}"
                )
            if finish_reason:
                raise VertexContentFilteredError(
                    "Gemini-on-Vertex returned no text content "
                    f"(finishReason={finish_reason}, detail={message}):\n"
                    f"{detail}"
                )
            raise VertexResponseFormatError(
                "Unexpected Gemini-on-Vertex candidate format:\n"
                f"{detail}"
            )

        parts = content.get("parts")
        if not isinstance(parts, list):
            finish_reason = candidate.get("finishReason")
            detail = self._format_provider_json(candidate)
            if finish_reason == "MAX_TOKENS":
                raise VertexGeminiMaxTokensError(
                    "Gemini-on-Vertex exhausted maxOutputTokens before returning "
                    f"text parts (finishReason={finish_reason}):\n"
                    f"{detail}"
                )
            raise VertexContentFilteredError(
                "Gemini-on-Vertex returned no text parts "
                f"(finishReason={finish_reason or 'unknown'}):\n"
                f"{detail}"
            )

        text = "\n".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and part.get("text")
        ).strip()
        if text:
            return text

        finish_reason = candidate.get("finishReason")
        detail = self._format_provider_json(candidate)
        if self._looks_retryable(detail):
            raise VertexTransientResponseError(
                "Retryable Gemini-on-Vertex candidate with empty text:\n"
                f"{detail}"
            )
        if finish_reason == "MAX_TOKENS":
            raise VertexGeminiMaxTokensError(
                "Gemini-on-Vertex exhausted maxOutputTokens before returning "
                f"non-empty text (finishReason={finish_reason}):\n"
                f"{detail}"
            )
        raise VertexContentFilteredError(
            "Gemini-on-Vertex returned empty text "
            f"(finishReason={finish_reason or 'unknown'}):\n"
            f"{detail}"
        )

    def _extract_embedding_values(self, data: Dict[str, Any]) -> List[float]:
        if "error" in data:
            raise VertexResponseFormatError(
                "Gemini embedding response contained an error body:\n"
                f"{self._format_provider_json(data)}"
            )

        values = None
        embedding = data.get("embedding")
        if isinstance(embedding, dict):
            values = embedding.get("values")

        if values is None:
            predictions = data.get("predictions")
            if isinstance(predictions, list) and predictions:
                first_prediction = predictions[0]
                if isinstance(first_prediction, dict):
                    prediction_embedding = first_prediction.get("embeddings")
                    if isinstance(prediction_embedding, dict):
                        values = prediction_embedding.get("values")
                    elif "values" in first_prediction:
                        values = first_prediction.get("values")

        if not isinstance(values, list) or not values:
            raise VertexResponseFormatError(
                "Unexpected Gemini embedding response format:\n"
                f"{self._format_provider_json(data)}"
            )

        try:
            return [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise VertexResponseFormatError(
                "Gemini embedding response contained non-numeric values:\n"
                f"{self._format_provider_json(data)}"
            ) from exc

    def _call_vertex_mistral(
            self,
            config: ModelConfig,
            prompt: str,
            max_tokens: int,
            temperature: float,
    ) -> str:
        api_host = self._api_host(config.location)
        url = (
            f"https://{api_host}/v1/"
            f"projects/{self.project_id}/locations/{config.location}/"
            f"publishers/mistralai/models/{config.model_id}:rawPredict"
        )

        payload = {
            "model": config.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        response = requests.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        self._raise_for_response(response, url, payload)

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise ValueError(
                f"Could not parse Mistral response: {self._format_provider_json(data)}"
            ) from exc

    @staticmethod
    def _format_provider_json(data: Any) -> str:
        return json.dumps(data, indent=2, ensure_ascii=False)

    @staticmethod
    def _looks_retryable(text: str) -> bool:
        lowered = text.lower()
        retryable_markers = (
            "429",
            "resource_exhausted",
            "resource exhausted",
            "rate_limit",
            "rate limit",
            "too many requests",
            "quota exceeded",
            "temporarily unavailable",
            "try again later",
            "overloaded",
            "service unavailable",
            "currently unavailable",
        )
        return any(marker in lowered for marker in retryable_markers)

    def _raise_for_response(
        self,
        response: requests.Response,
        url: str,
        payload: Dict[str, Any],
    ) -> None:
        if response.status_code == 200:
            return

        safe_payload = dict(payload)
        if "messages" in safe_payload:
            safe_payload["messages"] = "[omitted]"
        if "contents" in safe_payload:
            safe_payload["contents"] = "[omitted]"
        if "content" in safe_payload:
            safe_payload["content"] = "[omitted]"
        if "instances" in safe_payload:
            safe_payload["instances"] = "[omitted]"

        raise RuntimeError(
            "Vertex request failed.\n\n"
            f"Status: {response.status_code}\n"
            f"URL: {url}\n"
            f"Payload summary: {json.dumps(safe_payload, indent=2)}\n"
            f"Response:\n{response.text}"
        )
