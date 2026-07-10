from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import yaml

from llm_refinery.core.runs import stable_hash


class ConfigError(ValueError):
    """Raised when a configuration document is invalid."""


def load_yaml_mapping(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")
    return config_path, raw


def reject_unknown_keys(raw: dict[str, Any], allowed: set[str], *, context: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"{context} has unknown field(s): {', '.join(unknown)}")


def coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def coerce_command(value: Any) -> list[str]:
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ConfigError(f"command must be a string or list, got {type(value).__name__}")


def coerce_arg_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ConfigError(f"extra args must be a string or list, got {type(value).__name__}")


__all__ = [
    "ConfigError",
    "coerce_arg_list",
    "coerce_command",
    "coerce_list",
    "load_yaml_mapping",
    "reject_unknown_keys",
    "stable_hash",
]
