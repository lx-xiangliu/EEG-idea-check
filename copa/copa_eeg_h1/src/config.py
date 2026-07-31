"""YAML configuration and CLI override helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    return config


def apply_overrides(
    config: dict[str, Any],
    *,
    max_subjects: int | None = None,
    max_epochs_per_subject: int | None = None,
    seed: int | None = None,
    mode: str | None = None,
    output_dir: str | None = None,
    bootstrap_iterations: int | None = None,
) -> dict[str, Any]:
    result = deepcopy(config)
    if max_subjects is not None:
        result["data"]["max_subjects"] = int(max_subjects)
    if max_epochs_per_subject is not None:
        result["data"]["max_epochs_per_subject"] = int(max_epochs_per_subject)
    if seed is not None:
        result["project"]["seed"] = int(seed)
    if mode is not None:
        result["data"]["mode"] = mode
    if output_dir is not None:
        result["project"]["output_dir"] = output_dir
    if bootstrap_iterations is not None:
        result["evaluation"]["bootstrap_iterations"] = int(bootstrap_iterations)
    return result


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
