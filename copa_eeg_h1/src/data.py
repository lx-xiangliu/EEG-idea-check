"""Dataset loading and deterministic synthetic fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import warnings

import numpy as np


@dataclass(frozen=True)
class EEGDataset:
    """In-memory epoched EEG with subject-level grouping."""

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    sfreq: float
    channel_names: tuple[str, ...]
    mode: str
    dataset_name: str

    def validate(self) -> None:
        if self.X.ndim != 3:
            raise ValueError(f"X must be [epochs, channels, samples], got {self.X.shape}")
        n_epochs, n_channels, _ = self.X.shape
        if len(self.y) != n_epochs or len(self.subjects) != n_epochs:
            raise ValueError("X, y, and subjects must contain the same number of epochs")
        if len(self.channel_names) != n_channels:
            raise ValueError("channel_names does not match X channel dimension")
        if not np.isfinite(self.X).all():
            raise ValueError("EEG contains NaN or Inf")
        if np.unique(self.subjects).size < 2:
            raise ValueError("At least two subjects are required for subject-disjoint evaluation")


def _balanced_labels(n_epochs: int) -> np.ndarray:
    labels = np.arange(n_epochs, dtype=int) % 2
    return labels


def make_synthetic_eeg(
    n_subjects: int = 3,
    epochs_per_subject: int = 24,
    n_channels: int = 8,
    sfreq: float = 250.0,
    epoch_seconds: float = 2.0,
    seed: int = 42,
) -> EEGDataset:
    """Create flow-check data with subject shifts and a motor-imagery-like signal.

    This generator is deliberately useful for software validation, not for
    scientific inference.
    """

    if n_subjects < 2 or epochs_per_subject < 4 or n_channels < 4:
        raise ValueError("Synthetic data requires >=2 subjects, >=4 epochs, >=4 channels")
    rng = np.random.default_rng(seed)
    n_times = int(round(sfreq * epoch_seconds))
    time = np.arange(n_times) / sfreq
    canonical = ["Fp1", "Fp2", "C3", "Cz", "C4", "P3", "Pz", "P4"]
    channel_names = tuple(
        canonical[i] if i < len(canonical) else f"EEG{i + 1:02d}"
        for i in range(n_channels)
    )
    epochs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for subject in range(1, n_subjects + 1):
        subject_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        subject_gain = subject_rng.uniform(0.85, 1.2)
        subject_offset = subject_rng.normal(0.0, 0.08, size=(n_channels, 1))
        y_subject = _balanced_labels(epochs_per_subject)
        for label in y_subject:
            white = subject_rng.normal(0.0, 0.8, size=(n_channels, n_times))
            common = subject_rng.normal(0.0, 0.35, size=n_times)
            signal = white + 0.35 * common[None, :]
            phases = subject_rng.uniform(0, 2 * np.pi, size=n_channels)
            alpha = np.sin(2 * np.pi * 10.0 * time[None, :] + phases[:, None])
            beta = np.sin(2 * np.pi * 20.0 * time[None, :] + phases[:, None] / 2)
            # Opposing C3/C4 modulation mimics left/right motor imagery.
            if "C3" in channel_names and "C4" in channel_names:
                c3, c4 = channel_names.index("C3"), channel_names.index("C4")
            else:
                c3, c4 = 0, min(1, n_channels - 1)
            signal += 0.18 * alpha + 0.07 * beta
            signal[c3] += (0.8 if label == 0 else 0.2) * alpha[c3]
            signal[c4] += (0.2 if label == 0 else 0.8) * alpha[c4]
            signal = subject_gain * signal + subject_offset
            epochs.append(signal.astype(np.float64))
        labels.append(y_subject)
        groups.append(np.full(epochs_per_subject, subject, dtype=int))
    dataset = EEGDataset(
        X=np.stack(epochs),
        y=np.concatenate(labels),
        subjects=np.concatenate(groups),
        sfreq=float(sfreq),
        channel_names=channel_names,
        mode="synthetic",
        dataset_name="synthetic_flow_check",
    )
    dataset.validate()
    return dataset


def _limit_epochs_per_subject(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    max_epochs: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_epochs is None:
        return X, y, subjects
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for subject in np.unique(subjects):
        idx = np.flatnonzero(subjects == subject)
        # Preserve approximate class balance while keeping selection deterministic.
        per_class: list[np.ndarray] = []
        classes = np.unique(y[idx])
        base = max_epochs // len(classes)
        remainder = max_epochs % len(classes)
        for position, label in enumerate(classes):
            candidates = idx[y[idx] == label].copy()
            rng.shuffle(candidates)
            per_class.append(candidates[: base + (position < remainder)])
        chosen = np.concatenate(per_class)
        rng.shuffle(chosen)
        selected.extend(chosen.tolist())
    selected_array = np.asarray(sorted(selected), dtype=int)
    return X[selected_array], y[selected_array], subjects[selected_array]


def load_bnci2014_001(
    subjects: list[int],
    cache_dir: str | Path,
    max_epochs_per_subject: int | None,
    seed: int,
) -> EEGDataset:
    """Load left-vs-right epochs through MOABB with project-local caching."""

    cache_path = Path(cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MNE_DATA", str(cache_path))
    os.environ.setdefault("MNE_DATASETS_BNCI_PATH", str(cache_path))
    # Keep MNE/MOABB configuration inside the execution sandbox instead of
    # attempting to create ~/.mne on managed systems.
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "Real-data mode requires MOABB and MNE. Install requirements-real.txt "
            "or run scripts/run_smoke_test.py for the synthetic flow check."
        ) from exc

    dataset = BNCI2014_001()
    paradigm = MotorImagery(events=["left_hand", "right_hand"], n_classes=2)
    epochs, labels, metadata = paradigm.get_data(
        dataset=dataset, subjects=subjects, return_epochs=True
    )
    X = epochs.get_data()
    mapping = {"left_hand": 0, "right_hand": 1}
    try:
        y = np.asarray([mapping[str(label)] for label in labels], dtype=int)
    except KeyError as exc:
        raise RuntimeError(f"Unexpected BNCI2014_001 label: {exc}") from exc
    subject_values = metadata["subject"].to_numpy()
    subjects_array = np.asarray([int(value) for value in subject_values], dtype=int)
    X, y, subjects_array = _limit_epochs_per_subject(
        np.asarray(X, dtype=np.float64),
        y,
        subjects_array,
        max_epochs_per_subject,
        seed,
    )
    channel_names = tuple(str(name) for name in epochs.ch_names)
    result = EEGDataset(
        X=X,
        y=y,
        subjects=subjects_array,
        sfreq=float(epochs.info["sfreq"]),
        channel_names=channel_names,
        mode="real",
        dataset_name="BNCI2014_001",
    )
    result.validate()
    return result


def load_dataset(config: dict[str, Any], seed: int, cache_dir: Path) -> EEGDataset:
    data_config = config["data"]
    mode = str(data_config.get("mode", "real")).lower()
    max_subjects = int(data_config.get("max_subjects", 3))
    subjects = [int(s) for s in data_config.get("subjects", [1, 2, 3])][:max_subjects]
    max_epochs = data_config.get("max_epochs_per_subject")
    if mode == "synthetic":
        synthetic = data_config["synthetic"]
        n_subjects = min(int(synthetic["n_subjects"]), max_subjects)
        epochs_per_subject = int(synthetic["epochs_per_subject"])
        if max_epochs is not None:
            epochs_per_subject = min(epochs_per_subject, int(max_epochs))
        return make_synthetic_eeg(
            n_subjects=n_subjects,
            epochs_per_subject=epochs_per_subject,
            n_channels=int(synthetic["n_channels"]),
            sfreq=float(synthetic["sfreq"]),
            epoch_seconds=float(synthetic["epoch_seconds"]),
            seed=seed,
        )
    if mode != "real":
        raise ValueError(f"Unknown data mode: {mode}")
    try:
        return load_bnci2014_001(subjects, cache_dir, max_epochs, seed)
    except Exception:
        if not bool(data_config.get("fallback_to_synthetic", False)):
            raise
        warnings.warn(
            "MOABB real-data loading failed; using synthetic flow-check data. "
            "These outputs are not scientific results.",
            RuntimeWarning,
        )
        synthetic = data_config["synthetic"]
        return make_synthetic_eeg(
            n_subjects=min(int(synthetic["n_subjects"]), max_subjects),
            epochs_per_subject=min(
                int(synthetic["epochs_per_subject"]), int(max_epochs or 10**9)
            ),
            n_channels=int(synthetic["n_channels"]),
            sfreq=float(synthetic["sfreq"]),
            epoch_seconds=float(synthetic["epoch_seconds"]),
            seed=seed,
        )
