"""Subject-disjoint probes, subject bootstrap intervals, and controls."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def _models(seed: int, names: list[str]):
    result = {}
    if "LogisticRegression" in names:
        result["LogisticRegression"] = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    if "LinearSVC" in names:
        result["LinearSVC"] = LinearSVC(class_weight="balanced", random_state=seed, dual="auto", max_iter=5000)
    return result


def _bootstrap(values: dict[int, float], iterations: int, confidence: float, seed: int):
    ordered = np.asarray([values[key] for key in sorted(values)], dtype=float)
    if not len(ordered):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.asarray([rng.choice(ordered, len(ordered), replace=True).mean() for _ in range(iterations)])
    alpha = 1 - confidence
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def grouped_probe(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    model_names: list[str],
    iterations: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    classes = np.unique(y)
    if len(classes) < 2:
        return rows
    for model_name, estimator in _models(seed, model_names).items():
        by_subject_bacc, by_subject_f1 = {}, {}
        for subject in np.unique(subjects):
            train, test = subjects != subject, subjects == subject
            if len(np.unique(y[train])) < 2 or not np.all(np.isin(np.unique(y[test]), np.unique(y[train]))):
                continue
            pipeline = make_pipeline(StandardScaler(), clone(estimator))
            pipeline.fit(X[train], y[train])
            prediction = pipeline.predict(X[test])
            by_subject_bacc[int(subject)] = balanced_accuracy_score(y[test], prediction)
            by_subject_f1[int(subject)] = f1_score(y[test], prediction, average="macro", zero_division=0)
        if not by_subject_bacc:
            continue
        b_low, b_high = _bootstrap(by_subject_bacc, iterations, confidence, seed + 11)
        f_low, f_high = _bootstrap(by_subject_f1, iterations, confidence, seed + 17)
        rows.append({
            "model": model_name,
            "balanced_accuracy": float(np.mean(list(by_subject_bacc.values()))),
            "balanced_accuracy_ci_low": b_low,
            "balanced_accuracy_ci_high": b_high,
            "macro_f1": float(np.mean(list(by_subject_f1.values()))),
            "macro_f1_ci_low": f_low,
            "macro_f1_ci_high": f_high,
            "n_subject_folds": len(by_subject_bacc),
            "split": "leave_one_subject_out",
            "scaler_fit": "training_subjects_only",
            "chance_level": 1.0 / len(classes),
        })
    return rows


def run_main_probes(store, config, seed):
    analysis = config["probe"]
    model_names = list(analysis["models"])
    iterations, confidence = int(analysis["bootstrap_iterations"]), float(analysis["confidence_level"])
    operator_rows, task_rows, family_rows, operator_ovr_rows = [], [], [], []
    metadata = store.metadata
    for block, X, indices in store.finalized():
        rows = [metadata[int(index)] for index in indices]
        subjects = np.asarray([row["subject_id"] for row in rows])
        operator_y = np.asarray([row["operator_label"] for row in rows])
        task_y = np.asarray([row["task_label"] for row in rows])
        base = {"representation": block.representation, "layer": block.layer, "head": block.head if block.head is not None else "all"}
        # Head-level output is intentionally limited to attention probability
        # probes; other per-head summaries remain in the compressed cache for
        # targeted follow-up without multiplying thousands of CPU fits.
        if block.head is not None and block.representation != "attention_probs":
            continue
        selected_models = ["LogisticRegression"] if block.head is not None else model_names
        for result in grouped_probe(X, operator_y, subjects, selected_models, iterations, confidence, seed + block.layer + 101):
            operator_rows.append({**base, **result})
        if block.head is not None:
            continue
        for result in grouped_probe(X, task_y, subjects, model_names, iterations, confidence, seed + block.layer + 211):
            task_rows.append({**base, **result})
        # Family probes are one-vs-rest and exclude identity as a positive family.
        families = sorted({row["operator_family"] for row in rows} - {"identity"})
        for position, family in enumerate(families):
            binary = np.asarray([int(row["operator_family"] == family) for row in rows])
            for result in grouped_probe(X, binary, subjects, ["LogisticRegression"], iterations, confidence, seed + position + 307):
                family_rows.append({**base, "operator_family": family, "contrast": "one_vs_rest", **result})
        if block.representation in {
            "q", "k", "v", "attention_probs", "attention_output",
            "residual", "mlp_output", "final", "pooled_representation",
        }:
            for position, operator in enumerate(sorted(set(operator_y) - {"identity"})):
                binary = np.asarray([int(value == operator) for value in operator_y])
                for result in grouped_probe(
                    X, binary, subjects, ["LogisticRegression"], iterations,
                    confidence, seed + position + 401,
                ):
                    operator_ovr_rows.append({
                        **base, "operator_label": operator,
                        "contrast": "one_vs_rest", **result,
                    })
    return operator_rows, task_rows, family_rows, operator_ovr_rows


def shuffled_label_control(X, operator_y, subjects, config, seed):
    rng = np.random.default_rng(seed)
    shuffled = operator_y.copy()
    rng.shuffle(shuffled)
    results = grouped_probe(
        X, shuffled, subjects, ["LogisticRegression"],
        int(config["bootstrap_iterations"]), float(config["confidence_level"]), seed,
    )
    return results[0] if results else {}


def duplicate_identity_control(X, subjects, n_labels, config, seed):
    duplicated = np.repeat(X, n_labels, axis=0)
    pseudo = np.tile(np.arange(n_labels), len(X))
    groups = np.repeat(subjects, n_labels)
    results = grouped_probe(
        duplicated, pseudo, groups, ["LogisticRegression"],
        int(config["bootstrap_iterations"]), float(config["confidence_level"]), seed,
    )
    return results[0] if results else {}


def paired_feature_distances(X, rows):
    """Distances that distinguish source identity from operator identity."""
    by_source, by_operator = defaultdict(list), defaultdict(list)
    for index, row in enumerate(rows):
        by_source[row["source_epoch_id"]].append(index)
        by_operator[row["operator_label"]].append(index)
    same_source, same_operator = [], []
    for indices in by_source.values():
        for left, right in zip(indices[:-1], indices[1:]):
            same_source.append(cosine(X[left], X[right]))
    for indices in by_operator.values():
        ordered = sorted(indices, key=lambda i: rows[i]["source_epoch_id"])
        for left, right in zip(ordered[:-1], ordered[1:]):
            if rows[left]["source_epoch_id"] != rows[right]["source_epoch_id"]:
                same_operator.append(cosine(X[left], X[right]))
    return float(np.nanmean(same_source)), float(np.nanmean(same_operator))
