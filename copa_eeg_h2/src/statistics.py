"""Subject-level bootstrap, Wilcoxon tests, Holm correction, and effect sizes."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, wilcoxon


def subject_bootstrap_difference(
    paired_by_subject: dict[int, float],
    control_by_subject: dict[int, float],
    iterations: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    subjects = sorted(set(paired_by_subject).intersection(control_by_subject))
    if not subjects:
        return float("nan"), float("nan"), float("nan")
    differences = np.asarray(
        [paired_by_subject[s] - control_by_subject[s] for s in subjects], dtype=float
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    for index in range(iterations):
        estimates[index] = rng.choice(differences, size=len(differences), replace=True).mean()
    alpha = 1 - confidence
    return (
        float(differences.mean()),
        float(np.quantile(estimates, alpha / 2)),
        float(np.quantile(estimates, 1 - alpha / 2)),
    )


def rank_biserial(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[values != 0]
    if values.size == 0:
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    positive = ranks[values > 0].sum()
    negative = ranks[values < 0].sum()
    return float((positive - negative) / (positive + negative))


def paired_wilcoxon(
    paired_by_subject: dict[int, float],
    control_by_subject: dict[int, float],
    minimum_subjects: int,
) -> tuple[float, float, int]:
    subjects = sorted(set(paired_by_subject).intersection(control_by_subject))
    if len(subjects) < minimum_subjects:
        return float("nan"), float("nan"), len(subjects)
    differences = np.asarray(
        [paired_by_subject[s] - control_by_subject[s] for s in subjects], dtype=float
    )
    if np.allclose(differences, 0):
        return 1.0, 0.0, len(subjects)
    return (
        float(wilcoxon(differences, alternative="greater").pvalue),
        rank_biserial(differences),
        len(subjects),
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    result = np.full(len(p_values), np.nan)
    finite = [index for index, value in enumerate(p_values) if np.isfinite(value)]
    ordered = sorted(finite, key=lambda index: p_values[index])
    running = 0.0
    count = len(ordered)
    for position, index in enumerate(ordered):
        adjusted = min(1.0, (count - position) * p_values[index])
        running = max(running, adjusted)
        result[index] = running
    return result.tolist()
