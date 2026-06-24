"""
Shared benchmark configuration loader.
"""
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_CONFIG_PATH = "benchmark_config.json"


def load_benchmark_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load benchmark configuration from JSON."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Benchmark config must be a JSON object: {path}")
    return config


def get_section(config: Dict[str, Any], section_name: str) -> Dict[str, Any]:
    """Return a required top-level config section."""
    section = config.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid '{section_name}' section in benchmark config")
    return section


def get_enabled_models(section: Dict[str, Any], section_name: str) -> List[Dict[str, Any]]:
    """Return enabled model configs from a benchmark section."""
    models = section.get("models", [])
    if not isinstance(models, list):
        raise ValueError(f"'{section_name}.models' must be a list")

    enabled_models = []
    for index, model in enumerate(models, 1):
        if not isinstance(model, dict):
            raise ValueError(f"'{section_name}.models[{index}]' must be an object")
        if model.get("enabled", True):
            if not model.get("name") or not model.get("model_id"):
                raise ValueError(
                    f"'{section_name}.models[{index}]' must include non-empty name and model_id"
                )
            enabled_models.append(model)
    return enabled_models


def get_required_value(section: Dict[str, Any], key: str, section_name: str):
    """Return a required value from a section."""
    value = section.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required '{section_name}.{key}' in benchmark config")
    return value
