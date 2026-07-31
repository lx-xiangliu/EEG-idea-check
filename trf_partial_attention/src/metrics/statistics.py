from __future__ import annotations

import numpy as np


def bootstrap_ci(values: np.ndarray, seed: int = 0, n_boot: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_permutation_pvalue(a: np.ndarray, b: np.ndarray, seed: int = 0, n_perm: int = 10000) -> float:
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(difference.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(difference)))
    null = abs((signs * difference).mean(axis=1))
    return float((1 + (null >= observed).sum()) / (n_perm + 1))


def retrieval_metrics(scores: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, axis=1)
    ranks = np.array([int(np.flatnonzero(order[i] == correct[i])[0]) + 1 for i in range(len(correct))])
    return {
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "median_rank": float(np.median(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
    }

