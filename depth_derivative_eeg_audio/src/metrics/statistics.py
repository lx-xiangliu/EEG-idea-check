from __future__ import annotations

import numpy as np


def subject_bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("subject_bootstrap_ci needs at least two subject-level values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_permutation_test(
    left: np.ndarray,
    right: np.ndarray,
    n_permutations: int = 20_000,
    seed: int = 0,
) -> tuple[float, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("paired_permutation_test requires paired one-dimensional arrays")
    difference = left - right
    observed = float(difference.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, len(difference)))
    null = (signs * difference).mean(axis=1)
    p_value = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_permutations + 1))
    scale = difference.std(ddof=1)
    effect = observed / scale if scale > 0 else float("inf") if observed != 0 else 0.0
    return p_value, float(effect)
