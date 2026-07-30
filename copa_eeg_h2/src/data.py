"""Subject-grouped EEG loading with MOABB, local-MAT, and synthetic paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import warnings

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
        if self.X.ndim != 3:
            raise ValueError(f"X must have [epochs, channels, samples], got {self.X.shape}")
        n = self.X.shape[0]
        if any(len(v) != n for v in (self.y, self.subjects, self.sample_ids)):
            raise ValueError("X/y/subjects/sample_ids length mismatch")
        if len(self.channel_names) != self.X.shape[1]:
            raise ValueError("Channel-name count does not match X")
        if np.unique(self.subjects).size < 2:
            raise ValueError("At least two subjects are required")
        if np.unique(self.sample_ids).size != n:
            raise ValueError("sample_ids must be unique")
        if not np.isfinite(self.X).all():
            raise ValueError("EEG contains NaN or Inf")


def make_synthetic_eeg(
    n_subjects: int = 5,
    epochs_per_subject: int = 16,
    n_channels: int = 8,
    sfreq: float = 250.0,
    epoch_seconds: float = 2.0,
    seed: int = 42,
) -> EEGDataset:
    if n_subjects < 2 or epochs_per_subject < 4 or n_channels < 4:
        raise ValueError("Synthetic mode requires >=2 subjects, >=4 epochs, >=4 channels")
    rng = np.random.default_rng(seed)
    n_times = int(round(sfreq * epoch_seconds))
    time = np.arange(n_times) / sfreq
    names = ("Fp1", "Fp2", "C3", "Cz", "C4", "P3", "Pz", "P4")
    channel_names = tuple(names[i] if i < len(names) else f"EEG{i+1:02d}" for i in range(n_channels))
    epochs, labels, groups = [], [], []
    for subject in range(1, n_subjects + 1):
        srng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        mix = np.eye(n_channels) + srng.normal(0, 0.025, (n_channels, n_channels))
        for epoch_index in range(epochs_per_subject):
            label = epoch_index % 2
            white = srng.normal(0, 0.65, (n_channels, n_times))
            common = srng.normal(0, 0.3, n_times)
            phases = srng.uniform(0, 2 * np.pi, n_channels)
            alpha = np.sin(2 * np.pi * 10 * time[None, :] + phases[:, None])
            beta = np.sin(2 * np.pi * 20 * time[None, :] + phases[:, None] / 2)
            signal = white + 0.35 * common + 0.15 * alpha + 0.07 * beta
            c3, c4 = channel_names.index("C3"), channel_names.index("C4")
            signal[c3] += (0.7 if label == 0 else 0.15) * alpha[c3]
            signal[c4] += (0.15 if label == 0 else 0.7) * alpha[c4]
            epochs.append((mix @ signal * srng.uniform(0.9, 1.1)).astype(np.float32))
            labels.append(label)
            groups.append(subject)
    result = EEGDataset(
        np.stack(epochs),
        np.asarray(labels, dtype=int),
        np.asarray(groups, dtype=int),
        np.arange(len(epochs), dtype=int),
        float(sfreq),
        channel_names,
        "synthetic",
        "synthetic_flow_check",
        "generator",
    )
    result.validate()
    return result


def _balanced_limit(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    max_epochs: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_epochs is None:
        return X, y, subjects
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for subject in np.unique(subjects):
        subject_idx = np.flatnonzero(subjects == subject)
        selected: list[int] = []
        classes = np.unique(y[subject_idx])
        for class_position, label in enumerate(classes):
            candidates = subject_idx[y[subject_idx] == label].copy()
            rng.shuffle(candidates)
            count = max_epochs // len(classes) + int(class_position < max_epochs % len(classes))
            selected.extend(candidates[:count].tolist())
        keep.extend(sorted(selected))
    idx = np.asarray(keep, dtype=int)
    return X[idx], y[idx], subjects[idx]


def _load_local_bnci_mat(
    subjects: list[int],
    mat_dir: Path,
    max_epochs: int | None,
    seed: int,
) -> EEGDataset:
    """Read official BNCI2014_001 MAT files already downloaded by H1.

    Trials are cropped to four seconds from the cue marker. Only labels 1 and 2
    (left/right hand) and the first 22 EEG channels are retained.
    """
    epochs, labels, groups = [], [], []
    n_times = 1001
    for subject in subjects:
        paths = [mat_dir / f"A{subject:02d}T.mat", mat_dir / f"A{subject:02d}E.mat"]
        if not all(path.exists() for path in paths):
            raise FileNotFoundError(f"Missing BNCI MAT files for subject {subject} under {mat_dir}")
        for path in paths:
            mat = loadmat(path, squeeze_me=True, struct_as_record=False)
            for run in np.atleast_1d(mat["data"]):
                trials = np.atleast_1d(run.trial)
                run_labels = np.atleast_1d(run.y)
                if trials.size == 0:
                    continue
                for onset, label in zip(trials.astype(int), run_labels.astype(int)):
                    if label not in (1, 2):
                        continue
                    # BNCI2014_001 declares the motor-imagery interval as
                    # 2–6 s relative to the MATLAB trial marker. MATLAB
                    # positions are one-based; 1001 samples matches MOABB's
                    # inclusive four-second epoch.
                    start = onset - 1 + int(round(2.0 * float(run.fs)))
                    segment = np.asarray(
                        run.X[start : start + n_times, :22], dtype=np.float32
                    ).T
                    if segment.shape == (22, n_times):
                        # Official MAT signals are in microvolts; match MNE units.
                        epochs.append(segment * 1e-6)
                        labels.append(label - 1)
                        groups.append(subject)
    X = np.stack(epochs)
    y = np.asarray(labels, dtype=int)
    subject_array = np.asarray(groups, dtype=int)
    X, y, subject_array = _balanced_limit(X, y, subject_array, max_epochs, seed)
    result = EEGDataset(
        X, y, subject_array, np.arange(len(X)), 250.0, BNCI_CHANNELS,
        "real", "BNCI2014_001", "local_official_mat",
    )
    result.validate()
    return result


def _load_moabb(
    subjects: list[int],
    cache_dir: Path,
    max_epochs: int | None,
    seed: int,
) -> EEGDataset:
    os.environ.setdefault("MNE_DATA", str(cache_dir))
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    epochs, labels, metadata = MotorImagery(
        events=["left_hand", "right_hand"], n_classes=2
    ).get_data(dataset=BNCI2014_001(), subjects=subjects, return_epochs=True)
    mapping = {"left_hand": 0, "right_hand": 1}
    X = np.asarray(epochs.get_data(), dtype=np.float32)
    y = np.asarray([mapping[str(label)] for label in labels], dtype=int)
    group = metadata["subject"].to_numpy().astype(int)
    X, y, group = _balanced_limit(X, y, group, max_epochs, seed)
    result = EEGDataset(
        X, y, group, np.arange(len(X)), float(epochs.info["sfreq"]),
        tuple(epochs.ch_names), "real", "BNCI2014_001", "moabb",
    )
    result.validate()
    return result


def load_dataset(config: dict[str, Any], seed: int, cache_dir: Path) -> EEGDataset:
    data = config["data"]
    max_subjects = int(data.get("max_subjects", 5))
    subjects = [int(s) for s in data.get("subjects", [1, 2, 3, 4, 5])][:max_subjects]
    max_epochs = data.get("max_epochs_per_subject")
    if str(data.get("mode", "real")).lower() == "synthetic":
        synthetic = data["synthetic"]
        return make_synthetic_eeg(
            n_subjects=min(max_subjects, int(synthetic["n_subjects"])),
            epochs_per_subject=min(int(synthetic["epochs_per_subject"]), int(max_epochs or 10**9)),
            n_channels=int(synthetic["n_channels"]),
            sfreq=float(synthetic["sfreq"]),
            epoch_seconds=float(synthetic["epoch_seconds"]),
            seed=seed,
        )
    errors: list[str] = []
    try:
        return _load_moabb(subjects, cache_dir, max_epochs, seed)
    except Exception as exc:
        errors.append(f"MOABB: {type(exc).__name__}: {exc}")
    try:
        return _load_local_bnci_mat(
            subjects, resolve_path(data["h1_mat_cache"]), max_epochs, seed
        )
    except Exception as exc:
        errors.append(f"local MAT: {type(exc).__name__}: {exc}")
    if bool(data.get("fallback_to_synthetic", False)):
        warnings.warn("Real loading failed; using synthetic smoke data. " + " | ".join(errors))
        synthetic = data["synthetic"]
        return make_synthetic_eeg(
            n_subjects=min(max_subjects, int(synthetic["n_subjects"])),
            epochs_per_subject=min(int(synthetic["epochs_per_subject"]), int(max_epochs or 10**9)),
            n_channels=int(synthetic["n_channels"]),
            sfreq=float(synthetic["sfreq"]),
            epoch_seconds=float(synthetic["epoch_seconds"]),
            seed=seed,
        )
    raise RuntimeError("Real-data loading failed; no silent fallback. " + " | ".join(errors))
