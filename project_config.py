import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.json"


def load_config():
    config_path = CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Project config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8-sig") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Project config must be a JSON object: {config_path}")

    return config


CONFIG = load_config()


def get_config_section(section_name):
    section = CONFIG.get(section_name)
    if not isinstance(section, dict):
        raise KeyError(f"Missing required config.json section: {section_name}")

    return section


def get_config_value(section_name, config_name, default=None):
    return get_config_section(section_name).get(config_name, default)


def get_required_config_value(section_name, config_name):
    value = get_config_section(section_name).get(config_name)
    if value in (None, ""):
        raise KeyError(
            f"Missing required config.json field: {section_name}.{config_name}"
        )
    return value


def get_config_path(config_name):
    value = get_required_config_value("filepaths", config_name)
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


VERTEX_PROJECT_ID = get_required_config_value("keys", "vertex_project_id")
DEFAULT_JUDGE_MODEL = get_required_config_value("models", "default_judge_model")
DEFAULT_GENERATOR_MODEL = get_required_config_value(
    "models",
    "default_generator_model",
)
MAX_COMPLETION_TOKENS = get_required_config_value(
    "api_hyperparams",
    "MAX_COMPLETION_TOKENS",
)
DEFAULT_TEMPERATURE = get_required_config_value(
    "api_hyperparams",
    "DEFAULT_TEMPERATURE",
)
DEFAULT_TENSOR_PARALLEL_SIZE = get_required_config_value(
    "api_hyperparams",
    "DEFAULT_TENSOR_PARALLEL_SIZE",
)
DEFAULT_GPU_MEMORY_UTILIZATION = get_required_config_value(
    "api_hyperparams",
    "DEFAULT_GPU_MEMORY_UTILIZATION",
)
DEFAULT_BATCH_SIZE = get_required_config_value(
    "api_hyperparams",
    "DEFAULT_BATCH_SIZE",
)
DATASET_OUTPUT_DIR = get_config_path("dataset_output_dir")
RESPONSE_OUTPUT_DIR = get_config_path("response_output_dir")
ANALYSIS_OUTPUT_DIR = get_config_path("analysis_output_dir")
