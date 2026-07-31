from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.config import DataConfig


@dataclass(frozen=True)
class SyntheticBundle:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    split_metadata: dict[str, dict[str, np.ndarray]]
    mode: str


def _mix_matrix(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    matrix = rng.normal(size=(rows, cols))
    q, _ = np.linalg.qr(matrix)
    return q[:, :cols].astype(np.float32)


def _build_split(
    rng: np.random.Generator,
    n: int,
    subject_ids: np.ndarray,
    stimulus_offset: int,
    cfg: DataConfig,
    eeg_mix: np.ndarray,
    audio_mix: np.ndarray,
    subject_table: np.ndarray,
) -> tuple[TensorDataset, dict[str, np.ndarray]]:
    z = rng.normal(size=(n, cfg.latent_dim)).astype(np.float32)
    stimulus_ids = np.arange(stimulus_offset, stimulus_offset + n, dtype=np.int64)
    chosen_subjects = rng.choice(subject_ids, size=n, replace=True)
    t = np.linspace(0.0, 1.0, cfg.seq_len, dtype=np.float32)
    temporal = np.stack(
        [
            np.sin(2 * np.pi * t),
            np.cos(2 * np.pi * t),
            np.sin(4 * np.pi * t + 0.35),
        ],
        axis=-1,
    )
    latent_time = z[:, None, :] * (1.0 + 0.25 * temporal[None, :, :])
    if cfg.mode == "flat":
        latent_time = latent_time.mean(axis=-1, keepdims=True).repeat(cfg.latent_dim, axis=-1)
    elif cfg.mode == "nonmonotonic":
        latent_time = latent_time[..., [2, 0, 1]]
    elif cfg.mode == "parallel":
        latent_time[..., 1] += 0.65 * latent_time[..., 0]
        latent_time[..., 2] += 0.65 * latent_time[..., 0]
    elif cfg.mode not in {"hierarchical", "teacher_shuffled"}:
        raise ValueError(f"Unknown synthetic mode: {cfg.mode}")

    audio = latent_time @ audio_mix.T
    audio += 0.08 * np.tanh(audio)
    eeg_signal = np.tanh(latent_time @ eeg_mix.T)
    nuisance = subject_table[chosen_subjects][:, None, :]
    eeg = eeg_signal + cfg.nuisance_std * nuisance
    eeg += cfg.noise_std * rng.normal(size=eeg.shape).astype(np.float32)
    audio += (cfg.noise_std * 0.5) * rng.normal(size=audio.shape).astype(np.float32)
    if cfg.delay:
        eeg = np.roll(eeg, shift=cfg.delay, axis=1)

    dataset = TensorDataset(
        torch.from_numpy(eeg.astype(np.float32)),
        torch.from_numpy(audio.astype(np.float32)),
        torch.from_numpy(z.astype(np.float32)),
        torch.from_numpy(chosen_subjects.astype(np.int64)),
    )
    metadata = {
        "subject_id": chosen_subjects,
        "stimulus_id": stimulus_ids,
    }
    return dataset, metadata


def make_synthetic_bundle(cfg: DataConfig, seed: int) -> SyntheticBundle:
    rng = np.random.default_rng(seed)
    eeg_mix = _mix_matrix(rng, cfg.eeg_dim, cfg.latent_dim)
    audio_mix = _mix_matrix(rng, cfg.audio_dim, cfg.latent_dim)
    total_subjects = cfg.n_subjects_train + cfg.n_subjects_val + cfg.n_subjects_test
    subject_table = rng.normal(size=(total_subjects, cfg.eeg_dim)).astype(np.float32)
    subject_table /= np.linalg.norm(subject_table, axis=1, keepdims=True).clip(1e-6)
    train_ids = np.arange(0, cfg.n_subjects_train)
    val_ids = np.arange(cfg.n_subjects_train, cfg.n_subjects_train + cfg.n_subjects_val)
    test_ids = np.arange(cfg.n_subjects_train + cfg.n_subjects_val, total_subjects)
    train, train_meta = _build_split(
        rng, cfg.n_train, train_ids, 0, cfg, eeg_mix, audio_mix, subject_table
    )
    val, val_meta = _build_split(
        rng, cfg.n_val, val_ids, 1_000_000, cfg, eeg_mix, audio_mix, subject_table
    )
    test, test_meta = _build_split(
        rng, cfg.n_test, test_ids, 2_000_000, cfg, eeg_mix, audio_mix, subject_table
    )
    return SyntheticBundle(
        train=train,
        val=val,
        test=test,
        split_metadata={"train": train_meta, "val": val_meta, "test": test_meta},
        mode=cfg.mode,
    )
