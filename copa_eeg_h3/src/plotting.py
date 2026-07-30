"""Compact evidence plots for H3."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _matrix(rows, representations):
    layers = sorted({int(row["layer"]) for row in rows})
    matrix = np.full((len(layers), len(representations)), np.nan)
    for li, layer in enumerate(layers):
        for ri, representation in enumerate(representations):
            values = [
                float(row["balanced_accuracy"]) for row in rows
                if int(row["layer"]) == layer
                and row["representation"] == representation
                and row["head"] == "all"
                and row["model"] == "LogisticRegression"
            ]
            if values:
                matrix[li, ri] = np.mean(values)
    return layers, matrix


def heatmap(rows, path: Path, title: str) -> None:
    representations = [
        "input_embedding", "q", "k", "v", "attention_logits",
        "attention_probs", "attention_output", "residual", "mlp_output",
        "pooled_representation",
    ]
    layers, matrix = _matrix(rows, representations)
    figure, axis = plt.subplots(figsize=(12, 4.5))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(representations)), labels=representations, rotation=40, ha="right")
    axis.set_yticks(range(len(layers)), labels=layers)
    axis.set_xlabel("representation")
    axis.set_ylabel("layer (-1=input)")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="balanced accuracy")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def qkv_plot(operator_rows, path: Path) -> None:
    labels, means, lows, highs = [], [], [], []
    for representation in ("q", "k", "v"):
        candidates = [
            row for row in operator_rows
            if row["representation"] == representation and row["head"] == "all"
            and row["model"] == "LogisticRegression"
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda row: float(row["balanced_accuracy"]))
        labels.append(f"{representation.upper()} L{best['layer']}")
        means.append(float(best["balanced_accuracy"]))
        lows.append(float(best["balanced_accuracy"]) - float(best["balanced_accuracy_ci_low"]))
        highs.append(float(best["balanced_accuracy_ci_high"]) - float(best["balanced_accuracy"]))
    figure, axis = plt.subplots(figsize=(6.5, 4))
    axis.bar(labels, means, yerr=np.asarray([lows, highs]), capsize=4, color=["#4477AA", "#66CCEE", "#228833"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("operator balanced accuracy")
    axis.set_title("Best subject-disjoint Q/K/V operator decoding")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def family_paths(evidence_rows, path: Path) -> None:
    families = [row["operator_family"] for row in evidence_rows]
    paths = ["Q", "K", "V", "Attention", "Residual", "Final"]
    matrix = np.asarray([[float(row[path]) for path in paths] for row in evidence_rows])
    figure, axis = plt.subplots(figsize=(8.5, max(3.5, 0.65 * len(families))))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="magma")
    axis.set_xticks(range(len(paths)), labels=paths)
    axis.set_yticks(range(len(families)), labels=families)
    axis.set_title("Operator-family leakage paths")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", color="white")
    figure.colorbar(image, ax=axis, label="balanced accuracy")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def head_heatmap(operator_rows, path: Path) -> None:
    rows = [
        row for row in operator_rows
        if row["representation"] == "attention_probs"
        and row["head"] != "all" and row["model"] == "LogisticRegression"
    ]
    layers = sorted({int(row["layer"]) for row in rows})
    heads = sorted({int(row["head"]) for row in rows})
    matrix = np.full((len(layers), len(heads)), np.nan)
    for li, layer in enumerate(layers):
        for hi, head in enumerate(heads):
            values = [float(row["balanced_accuracy"]) for row in rows if int(row["layer"]) == layer and int(row["head"]) == head]
            if values:
                matrix[li, hi] = np.mean(values)
    figure, axis = plt.subplots(figsize=(6, 4))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="cividis")
    axis.set_xticks(range(len(heads)), labels=heads)
    axis.set_yticks(range(len(layers)), labels=layers)
    axis.set_xlabel("head")
    axis.set_ylabel("layer")
    axis.set_title("Attention-probability operator decoding")
    figure.colorbar(image, ax=axis, label="balanced accuracy")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)

