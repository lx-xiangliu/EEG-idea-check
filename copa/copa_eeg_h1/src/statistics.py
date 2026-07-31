"""Paired cross-operator statistics with small-sample safeguards."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from scipy.stats import rankdata, wilcoxon


def rank_biserial_effect(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    denominator = positive + negative
    return float((positive - negative) / denominator) if denominator else 0.0


def holm_correction(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted.tolist()


def paired_cross_operator_test(
    cross_results: dict[str, Any], minimum_subjects: int = 5
) -> dict[str, Any]:
    names = cross_results["operator_names"]
    cell_scores = cross_results["subject_balanced_accuracy"]
    subject_ids = sorted(
        {
            subject
            for scores in cell_scores.values()
            for subject in scores
        },
        key=int,
    )
    diagonal_by_subject: list[float] = []
    off_diagonal_by_subject: list[float] = []
    for subject in subject_ids:
        diagonal = [
            cell_scores[f"{name}__to__{name}"][subject] for name in names
        ]
        off_diagonal = [
            cell_scores[f"{train}__to__{test}"][subject]
            for train in names
            for test in names
            if train != test
        ]
        diagonal_by_subject.append(float(np.mean(diagonal)))
        off_diagonal_by_subject.append(float(np.mean(off_diagonal)))
    diagonal_array = np.asarray(diagonal_by_subject)
    off_diagonal_array = np.asarray(off_diagonal_by_subject)
    differences = diagonal_array - off_diagonal_array
    base: dict[str, Any] = {
        "test": "wilcoxon_signed_rank",
        "unit": "subject",
        "n_subjects": len(subject_ids),
        "minimum_subjects": minimum_subjects,
        "diagonal_by_subject": dict(zip(subject_ids, diagonal_by_subject)),
        "off_diagonal_by_subject": dict(zip(subject_ids, off_diagonal_by_subject)),
        "mean_paired_difference": float(differences.mean()),
        "effect_size": {
            "name": "matched_pairs_rank_biserial",
            "value": rank_biserial_effect(differences),
        },
    }
    if len(subject_ids) < minimum_subjects:
        base.update(
            {
                "status": "insufficient_sample_size",
                "statistic": None,
                "p_value": None,
                "message": (
                    f"Only {len(subject_ids)} subjects; at least {minimum_subjects} "
                    "are required before reporting significance."
                ),
            }
        )
        return base
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test = wilcoxon(diagonal_array, off_diagonal_array, alternative="greater")
        base.update(
            {
                "status": "ok",
                "statistic": float(test.statistic),
                "p_value": float(test.pvalue),
            }
        )
    except ValueError as exc:
        base.update(
            {
                "status": "not_testable",
                "statistic": None,
                "p_value": None,
                "message": str(exc),
            }
        )
    return base


def per_operator_tests(
    cross_results: dict[str, Any], minimum_subjects: int = 5
) -> dict[str, Any]:
    """Outgoing diagonal-vs-cross tests with Holm correction."""

    names = cross_results["operator_names"]
    scores = cross_results["subject_balanced_accuracy"]
    subject_ids = sorted(next(iter(scores.values())).keys(), key=int)
    results: dict[str, Any] = {}
    raw_p_values: list[float] = []
    tested_names: list[str] = []
    for train in names:
        diagonal = np.asarray(
            [scores[f"{train}__to__{train}"][subject] for subject in subject_ids]
        )
        cross = np.asarray(
            [
                np.mean(
                    [
                        scores[f"{train}__to__{test}"][subject]
                        for test in names
                        if test != train
                    ]
                )
                for subject in subject_ids
            ]
        )
        differences = diagonal - cross
        item: dict[str, Any] = {
            "n_subjects": len(subject_ids),
            "mean_difference": float(differences.mean()),
            "effect_size": rank_biserial_effect(differences),
        }
        if len(subject_ids) < minimum_subjects:
            item.update({"status": "insufficient_sample_size", "p_value": None})
        else:
            try:
                test_result = wilcoxon(diagonal, cross, alternative="greater")
                item.update(
                    {
                        "status": "ok",
                        "statistic": float(test_result.statistic),
                        "p_value": float(test_result.pvalue),
                    }
                )
                raw_p_values.append(float(test_result.pvalue))
                tested_names.append(train)
            except ValueError as exc:
                item.update(
                    {"status": "not_testable", "p_value": None, "message": str(exc)}
                )
        results[train] = item
    for name, adjusted in zip(tested_names, holm_correction(raw_p_values)):
        results[name]["p_value_holm"] = adjusted
    return {"correction": "Holm", "comparisons": results}
