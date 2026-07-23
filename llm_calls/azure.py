from openai import AzureOpenAI
from project_config import (
    CONFIG_PATH,
    DEFAULT_TEMPERATURE,
    DEFAULT_GENERATOR_MODEL,
    MAX_COMPLETION_TOKENS,
    get_config_value as get_project_config_value,
)


AZURE_OPENAI_ENDPOINT = get_project_config_value("keys", "azure_openai_endpoint")
AZURE_OPENAI_API_KEY = get_project_config_value("keys", "azure_openai_api_key")
AZURE_OPENAI_API_VERSION = get_project_config_value(
    "keys",
    "azure_openai_api_version",
    "2024-12-01-preview",
)

MODEL_NAME = DEFAULT_GENERATOR_MODEL
_client = None


def get_client():
    global _client
    if _client is None:
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
            raise RuntimeError(
                "Add azure_openai_endpoint and azure_openai_api_key to "
                f"{CONFIG_PATH} before calling Azure OpenAI."
            )

        _client = AzureOpenAI(
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
        )

    return _client


def get_response(
    prompt,
    model_name=MODEL_NAME,
    max_completion_tokens=MAX_COMPLETION_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
):
    response = get_client().chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        model=model_name,
    )

    return response.choices[0].message.content
