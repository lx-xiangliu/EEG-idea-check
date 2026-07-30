"""Configuration and path helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError("Configuration root must be a mapping")
    return result


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def with_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    result = deepcopy(config)
    mapping = {
        "mode": ("data", "mode"),
        "output_dir": ("project", "output_dir"),
        "max_subjects": ("data", "max_subjects"),
        "max_epochs_per_subject": ("data", "max_epochs_per_subject"),
        "bootstrap_iterations": ("probe", "bootstrap_iterations"),
        "checkpoint": ("model", "checkpoint"),
        "seed": ("project", "seed"),
    }
    for name, value in overrides.items():
        if value is not None and name in mapping:
            section, key = mapping[name]
            result[section][key] = value
    return result

