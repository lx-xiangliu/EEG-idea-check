"""CPU-only sklearn model factories."""

from __future__ import annotations

from typing import Any

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def make_operator_model(name: str, seed: int, n_jobs: int = 1) -> Any:
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "linear_svc":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=seed,
                        max_iter=5000,
                        tol=1e-2,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        )
    raise ValueError(f"Unknown operator model: {name}")


def make_task_model(name: str, seed: int, n_jobs: int = 1) -> Any:
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "lda":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        )
    raise ValueError(
        f"Unknown task model: {name}. Optional CSP/pyRiemann models require their extras."
    )
