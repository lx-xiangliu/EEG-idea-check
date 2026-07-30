"""Subject-disjoint evaluation and subject-level bootstrap confidence intervals."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable
import warnings

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from .models import make_operator_model, make_task_model


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


def metric_bundle(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[int], target_names: list[str]
) -> dict[str, Any]:
    metrics = _metric_values(y_true, y_pred)
    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=labels
    ).tolist()
    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return metrics


def subject_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, list[float]]:
    """Bootstrap complete subjects, never individual epochs."""

    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"balanced_accuracy": [], "macro_f1": []}
    for _ in range(iterations):
        drawn = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in drawn])
        values = _metric_values(y_true[indices], y_pred[indices])
        for name, value in values.items():
            samples[name].append(value)
    alpha = 1.0 - confidence_level
    return {
        name: [
            float(np.quantile(values, alpha / 2)),
            float(np.quantile(values, 1 - alpha / 2)),
        ]
        for name, values in samples.items()
    }


def evaluate_operator_probe(
    features_by_operator: dict[str, np.ndarray],
    subjects: np.ndarray,
    model_names: list[str],
    n_splits: int,
    bootstrap_iterations: int,
    confidence_level: float,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    operator_names = list(features_by_operator)
    X = np.concatenate([features_by_operator[name] for name in operator_names], axis=0)
    y = np.concatenate(
        [np.full(len(subjects), index, dtype=int) for index in range(len(operator_names))]
    )
    groups = np.tile(subjects, len(operator_names))
    split_count = min(n_splits, np.unique(groups).size)
    if split_count < 2:
        raise ValueError("Operator probe requires at least two subjects")
    splitter = GroupKFold(n_splits=split_count)
    splits = list(splitter.split(X, y, groups))
    # Verify the central anti-leakage invariant explicitly.
    for train, test in splits:
        if np.intersect1d(groups[train], groups[test]).size:
            raise RuntimeError("Subject leakage detected in operator probe split")

    model_results: dict[str, Any] = {}
    for model_offset, model_name in enumerate(model_names):
        predictions = np.empty_like(y)
        fold_subjects: list[dict[str, list[int]]] = []
        for fold, (train, test) in enumerate(splits):
            model = make_operator_model(model_name, seed + fold, n_jobs)
            model.fit(X[train], y[train])
            predictions[test] = model.predict(X[test])
            fold_subjects.append(
                {
                    "train": np.unique(groups[train]).astype(int).tolist(),
                    "test": np.unique(groups[test]).astype(int).tolist(),
                }
            )
        result = metric_bundle(
            y, predictions, list(range(len(operator_names))), operator_names
        )
        result["confidence_intervals"] = subject_bootstrap_ci(
            y,
            predictions,
            groups,
            bootstrap_iterations,
            confidence_level,
            seed + 10_000 + model_offset,
        )
        result["fold_subjects"] = fold_subjects
        result["predictions"] = predictions.astype(int).tolist()
        model_results[model_name] = result
    return {
        "operator_names": operator_names,
        "chance_balanced_accuracy": 1.0 / len(operator_names),
        "models": model_results,
    }


def evaluate_cross_operator(
    features_by_operator: dict[str, np.ndarray],
    task_labels: np.ndarray,
    subjects: np.ndarray,
    model_name: str,
    bootstrap_iterations: int,
    confidence_level: float,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    operator_names = list(features_by_operator)
    logo = LeaveOneGroupOut()
    base = next(iter(features_by_operator.values()))
    splits = list(logo.split(base, task_labels, subjects))
    if len(splits) < 2:
        raise ValueError("Cross-operator evaluation requires at least two subjects")
    collected: dict[tuple[str, str], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"true": [], "pred": [], "groups": []}
    )
    fold_subjects: list[dict[str, list[int]]] = []
    for fold, (train, test) in enumerate(splits):
        if np.intersect1d(subjects[train], subjects[test]).size:
            raise RuntimeError("Subject leakage detected in cross-operator split")
        fold_subjects.append(
            {
                "train": np.unique(subjects[train]).astype(int).tolist(),
                "test": np.unique(subjects[test]).astype(int).tolist(),
            }
        )
        for train_index, train_operator in enumerate(operator_names):
            model = make_task_model(
                model_name, seed + fold * len(operator_names) + train_index, n_jobs
            )
            model.fit(features_by_operator[train_operator][train], task_labels[train])
            for test_operator in operator_names:
                predicted = model.predict(features_by_operator[test_operator][test])
                cell = collected[(train_operator, test_operator)]
                cell["true"].append(task_labels[test])
                cell["pred"].append(np.asarray(predicted, dtype=int))
                cell["groups"].append(subjects[test])

    n_operators = len(operator_names)
    balanced = np.zeros((n_operators, n_operators), dtype=float)
    macro_f1 = np.zeros_like(balanced)
    cell_results: dict[str, Any] = {}
    subject_scores: dict[str, dict[str, float]] = {}
    for row, train_operator in enumerate(operator_names):
        for column, test_operator in enumerate(operator_names):
            cell = collected[(train_operator, test_operator)]
            y_true = np.concatenate(cell["true"])
            y_pred = np.concatenate(cell["pred"])
            groups = np.concatenate(cell["groups"])
            metrics = metric_bundle(y_true, y_pred, [0, 1], ["left_hand", "right_hand"])
            metrics["confidence_intervals"] = subject_bootstrap_ci(
                y_true,
                y_pred,
                groups,
                bootstrap_iterations,
                confidence_level,
                seed + 100_000 + row * n_operators + column,
            )
            key = f"{train_operator}__to__{test_operator}"
            per_subject: dict[str, float] = {}
            for subject in np.unique(groups):
                idx = groups == subject
                per_subject[str(int(subject))] = _metric_values(
                    y_true[idx], y_pred[idx]
                )["balanced_accuracy"]
            subject_scores[key] = per_subject
            cell_results[key] = metrics
            balanced[row, column] = metrics["balanced_accuracy"]
            macro_f1[row, column] = metrics["macro_f1"]

    diagonal = np.diag(balanced)
    off_diagonal_mask = ~np.eye(n_operators, dtype=bool)
    diagonal_mean = float(diagonal.mean())
    off_diagonal_mean = float(balanced[off_diagonal_mask].mean())
    outgoing_drop = {
        operator: float(
            balanced[index, index]
            - np.delete(balanced[index], index).mean()
        )
        for index, operator in enumerate(operator_names)
    }
    incoming_drop = {
        operator: float(
            balanced[index, index]
            - np.delete(balanced[:, index], index).mean()
        )
        for index, operator in enumerate(operator_names)
    }
    return {
        "operator_names": operator_names,
        "task_model": model_name,
        "balanced_accuracy_matrix": balanced.tolist(),
        "macro_f1_matrix": macro_f1.tolist(),
        "diagonal_mean": diagonal_mean,
        "off_diagonal_mean": off_diagonal_mean,
        "average_cross_operator_drop": diagonal_mean - off_diagonal_mean,
        "outgoing_drop": outgoing_drop,
        "incoming_drop": incoming_drop,
        "cells": cell_results,
        "subject_balanced_accuracy": subject_scores,
        "fold_subjects": fold_subjects,
    }
