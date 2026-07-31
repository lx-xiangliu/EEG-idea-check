from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    mode: str = "hierarchical"
    n_train: int = 192
    n_val: int = 64
    n_test: int = 96
    seq_len: int = 8
    eeg_dim: int = 12
    audio_dim: int = 12
    latent_dim: int = 3
    n_subjects_train: int = 8
    n_subjects_val: int = 2
    n_subjects_test: int = 4
    noise_std: float = 0.25
    nuisance_std: float = 0.8
    delay: int = 0


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 24
    eeg_layers: int = 4
    audio_layers: int = 6
    n_heads: int = 4
    ff_mult: int = 2
    projection_dim: int = 16
    dropout: float = 0.0
    mapper_temperature: float = 0.55


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    method: str = "dda_monotonic"
    normalize_residuals: bool = True
    device: str = "cpu"
    num_workers: int = 0
    checkpoint_dir: str = "outputs/checkpoints"


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    unknown = set(payload) - {"data", "model", "train"}
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    return ExperimentConfig(
        data=DataConfig(**payload.get("data", {})),
        model=ModelConfig(**payload.get("model", {})),
        train=TrainConfig(**payload.get("train", {})),
    )
