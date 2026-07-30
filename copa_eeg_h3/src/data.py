"""Subject-grouped BNCI2014_001 and deterministic synthetic EEG loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .config import resolve_path


BNCI_CHANNELS = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
    "P2", "POz",
)


@dataclass(frozen=True)
class EEGDataset:
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    sample_ids: np.ndarray
    sfreq: float
    channel_names: tuple[str, ...]
    mode: str
    dataset_name: str
    loader: str

    def validate(self) -> None:
        n = len(self.X)
        if self.X.ndim != 3 or any(len(v) != n for v in (self.y, self.subjects, self.sample_ids)):
            raise ValueError("Expected aligned X[epoch,channel,time], y, subject, sample_id")
        if len(self.channel_names) != self.X.shape[1] or np.unique(self.subjects).size < 2:
            raise ValueError("Invalid channels or fewer than two subjects")
        if np.unique(self.sample_ids).size != n or not np.isfinite(self.X).all():
            raise ValueError("Sample IDs must be unique and EEG finite")


def make_synthetic_eeg(
    n_subjects: int = 4,
    epochs_per_subject: int = 8,
    n_channels: int = 8,
    sfreq: float = 250.0,
    epoch_seconds: float = 2.0,
    seed: int = 42,
) -> EEGDataset:
    if n_subjects < 2 or epochs_per_subject < 4 or n_channels < 4:
        raise ValueError("Synthetic data needs >=2 subjects, >=4 epochs/subject, >=4 channels")
    rng = np.random.default_rng(seed)
    n_times = int(round(sfreq * epoch_seconds))
    t = np.arange(n_times) / sfreq
    base_names = ("Fp1", "Fp2", "C3", "Cz", "C4", "P3", "Pz", "P4")
    names = tuple(base_names[i] if i < len(base_names) else f"EEG{i+1:02d}" for i in range(n_channels))
    X, y, groups = [], [], []
    for subject in range(1, n_subjects + 1):
        srng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        mixing = np.eye(n_channels) + srng.normal(0, 0.025, (n_channels, n_channels))
        for epoch in range(epochs_per_subject):
            label = epoch % 2
            phase = srng.uniform(0, 2 * np.pi, n_channels)
            alpha = np.sin(2 * np.pi * 10 * t[None] + phase[:, None])
            beta = np.sin(2 * np.pi * 20 * t[None] + phase[:, None] / 2)
            signal = srng.normal(0, 0.65, (n_channels, n_times))
            signal += 0.3 * srng.normal(size=n_times)[None] + 0.15 * alpha + 0.07 * beta
            c3, c4 = names.index("C3"), names.index("C4")
            signal[c3] += (0.65 if label == 0 else 0.15) * alpha[c3]
            signal[c4] += (0.15 if label == 0 else 0.65) * alpha[c4]
            X.append((mixing @ signal * srng.uniform(0.9, 1.1)).astype(np.float32))
            y.append(label)
            groups.append(subject)
    result = EEGDataset(
        np.stack(X), np.asarray(y), np.asarray(groups), np.arange(len(X)),
        sfreq, names, "synthetic", "synthetic_flow_check", "generator",
    )
    result.validate()
    return result


def _balanced_limit(X, y, subjects, max_epochs: int | None, seed: int):
    if max_epochs is None:
        return X, y, subjects
    rng, keep = np.random.default_rng(seed), []
    for subject in np.unique(subjects):
        subject_idx = np.flatnonzero(subjects == subject)
        for position, label in enumerate(np.unique(y[subject_idx])):
            candidates = subject_idx[y[subject_idx] == label].copy()
            rng.shuffle(candidates)
            count = max_epochs // 2 + int(position < max_epochs % 2)
            keep.extend(candidates[:count].tolist())
    idx = np.asarray(sorted(keep), dtype=int)
    return X[idx], y[idx], subjects[idx]


def _load_local_bnci(subjects: list[int], mat_dir: Path, max_epochs: int | None, seed: int) -> EEGDataset:
    epochs, labels, groups = [], [], []
    for subject in subjects:
        paths = [mat_dir / f"A{subject:02d}T.mat", mat_dir / f"A{subject:02d}E.mat"]
        if not all(path.exists() for path in paths):
            raise FileNotFoundError(f"Missing official BNCI files for subject {subject}: {mat_dir}")
        for path in paths:
            mat = loadmat(path, squeeze_me=True, struct_as_record=False)
            for run in np.atleast_1d(mat["data"]):
                for onset, label in zip(np.atleast_1d(run.trial).astype(int), np.atleast_1d(run.y).astype(int)):
                    if label not in (1, 2):
                        continue
                    start = onset - 1 + int(round(2.0 * float(run.fs)))
                    segment = np.asarray(run.X[start:start + 1001, :22], dtype=np.float32).T
                    if segment.shape == (22, 1001):
                        epochs.append(segment * 1e-6)
                        labels.append(label - 1)
                        groups.append(subject)
    X, y, group = np.stack(epochs), np.asarray(labels), np.asarray(groups)
    X, y, group = _balanced_limit(X, y, group, max_epochs, seed)
    result = EEGDataset(
        X, y, group, np.arange(len(X)), 250.0, BNCI_CHANNELS,
        "real", "BNCI2014_001", "local_official_mat",
    )
    result.validate()
    return result


def load_dataset(config: dict[str, Any], seed: int) -> EEGDataset:
    data = config["data"]
    max_subjects = int(data.get("max_subjects", 5))
    max_epochs = data.get("max_epochs_per_subject")
    if data.get("mode", "real") == "synthetic":
        synth = data["synthetic"]
        return make_synthetic_eeg(
            min(max_subjects, int(synth["n_subjects"])),
            min(int(synth["epochs_per_subject"]), int(max_epochs or 10**9)),
            int(synth["n_channels"]), float(synth["sfreq"]),
            float(synth["epoch_seconds"]), seed,
        )
    subjects = [int(v) for v in data.get("subjects", [1, 2, 3, 4, 5])][:max_subjects]
    return _load_local_bnci(
        subjects, resolve_path(data["h1_mat_cache"]),
        None if max_epochs is None else int(max_epochs), seed,
    )

