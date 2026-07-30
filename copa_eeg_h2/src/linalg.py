"""Numerically stable spectral and subspace metrics."""

from __future__ import annotations

import numpy as np
from scipy.linalg import subspace_angles
from sklearn.utils.extmath import randomized_svd


EPS = np.finfo(np.float64).eps


def singular_values(matrix: np.ndarray) -> np.ndarray:
    """Compute all singular values through the smaller Gram matrix."""
    X = np.asarray(matrix, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("SVD input must be a finite 2-D matrix")
    if X.size == 0:
        raise ValueError("SVD input is empty")
    gram = X.T @ X if X.shape[1] <= X.shape[0] else X @ X.T
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.maximum(eigenvalues, 0)
    return np.sqrt(eigenvalues[::-1])


def spectrum_metrics(matrix: np.ndarray, ranks: list[int], thresholds: list[float]) -> dict:
    values = singular_values(matrix)
    energy = values**2
    total = float(energy.sum())
    if total <= EPS:
        return {
            "singular_values": values,
            "rho": {int(rank): 0.0 for rank in ranks},
            "threshold_ranks": {float(threshold): 0 for threshold in thresholds},
            "effective_rank": 0.0,
            "stable_rank": 0.0,
        }
    cumulative = np.cumsum(energy) / total
    probabilities = energy[energy > EPS] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return {
        "singular_values": values,
        "rho": {
            int(rank): float(cumulative[min(int(rank), len(cumulative)) - 1])
            for rank in ranks
        },
        "threshold_ranks": {
            float(threshold): int(np.searchsorted(cumulative, threshold, side="left") + 1)
            for threshold in thresholds
        },
        "effective_rank": float(np.exp(entropy)),
        "stable_rank": float(total / max(energy[0], EPS)),
    }


def top_right_subspace(matrix: np.ndarray, rank: int, seed: int = 0) -> np.ndarray:
    X = np.asarray(matrix, dtype=np.float64)
    usable_rank = min(int(rank), X.shape[0], X.shape[1])
    if usable_rank < 1:
        raise ValueError("Subspace rank must be positive")
    if usable_rank >= min(X.shape) - 1 or X.shape[1] <= 96:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        return vt[:usable_rank].T
    _, _, vt = randomized_svd(X, n_components=usable_rank, random_state=seed)
    return vt.T


def explained_by_subspace(matrix: np.ndarray, basis: np.ndarray) -> float:
    X = np.asarray(matrix, dtype=np.float64)
    denominator = float(np.sum(X * X))
    if denominator <= EPS:
        return 0.0
    projected = X @ basis
    return float(np.sum(projected * projected) / denominator)


def random_subspace(dimension: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dimension, min(rank, dimension))))
    return q


def principal_angle_summary(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    angles = np.degrees(subspace_angles(a, b))
    return float(np.mean(angles)), float(np.max(angles)), float(np.min(angles))
