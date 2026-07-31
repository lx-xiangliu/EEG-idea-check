from dataclasses import replace

import pytest

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from src.data import make_synthetic_bundle
from src.training import train_synthetic


def test_two_cpu_runs_are_deterministic() -> None:
    cfg = ExperimentConfig(
        data=DataConfig(
            n_train=32,
            n_val=16,
            n_test=24,
            n_subjects_train=3,
            n_subjects_val=2,
            n_subjects_test=2,
        ),
        model=ModelConfig(),
        train=TrainConfig(seed=7, epochs=1, batch_size=16, method="dda_fixed"),
    )
    first = train_synthetic(cfg, make_synthetic_bundle(cfg.data, cfg.train.seed), save_checkpoint=False)
    second = train_synthetic(cfg, make_synthetic_bundle(cfg.data, cfg.train.seed), save_checkpoint=False)
    assert first.probe_accuracy == pytest.approx(second.probe_accuracy, abs=1e-12)
    assert first.best_val_loss == pytest.approx(second.best_val_loss, abs=1e-12)
