"""Headless publication-style figures for H2."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_cumulative_spectra(rows: list[dict], output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    embeddings = sorted({row["embedding"] for row in rows})
    for embedding in embeddings:
        subset = [
            row for row in rows
            if row["embedding"] == embedding and row["control"] == "paired"
            and row["scope"] == "overall"
        ]
        fig, ax = plt.subplots(figsize=(8.4, 5.3))
        for operator in sorted({row["operator"] for row in subset}):
            operator_rows = sorted(
                [row for row in subset if row["operator"] == operator],
                key=lambda row: int(row["rank"]),
            )
            ax.plot(
                [int(row["rank"]) for row in operator_rows],
                [float(row["explained_variance"]) for row in operator_rows],
                marker="o",
                linewidth=1.5,
                label=operator,
            )
        ax.set(xlabel="rank r", ylabel="cumulative explained energy", ylim=(0, 1.02))
        ax.set_title(f"Paired operator effects — {embedding}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(figures / f"cumulative_spectrum_{embedding}.png", dpi=180)
        plt.close(fig)


def _heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> None:
    fig_width = max(6.5, 0.65 * len(col_labels) + 2.3)
    fig_height = max(5.2, 0.52 * len(row_labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_transfer_heatmap(rows: list[dict], output_dir: Path) -> None:
    figures = output_dir / "figures"
    operators = sorted({row["operator"] for row in rows})
    embeddings = sorted({row["embedding"] for row in rows})
    matrix = np.full((len(operators), len(embeddings)), np.nan)
    for i, operator in enumerate(operators):
        for j, embedding in enumerate(embeddings):
            values = [
                float(row["explained_test"])
                for row in rows
                if row["operator"] == operator and row["embedding"] == embedding
                and row["baseline"] == "train_subspace"
            ]
            if values:
                matrix[i, j] = np.mean(values)
    _heatmap(
        matrix, operators, embeddings, "LOSO train-subspace explained energy",
        figures / "transfer_heatmap.png", vmin=0, vmax=1,
    )


def plot_operator_matrices(
    specificity_rows: list[dict],
    angle_rows: list[dict],
    output_dir: Path,
) -> None:
    figures = output_dir / "figures"
    operators = sorted(
        set(row["source_operator"] for row in specificity_rows)
        | set(row["target_operator"] for row in specificity_rows)
    )
    specificity = np.full((len(operators), len(operators)), np.nan)
    angles = np.full_like(specificity, np.nan)
    for i, source in enumerate(operators):
        for j, target in enumerate(operators):
            values = [
                float(row["explained_test"]) for row in specificity_rows
                if row["source_operator"] == source and row["target_operator"] == target
            ]
            if values:
                specificity[i, j] = np.mean(values)
            angle_values = [
                float(row["mean_angle_deg"]) for row in angle_rows
                if row.get("comparison_type") == "operator"
                and row["source_operator"] == source and row["target_operator"] == target
            ]
            if angle_values:
                angles[i, j] = np.mean(angle_values)
    _heatmap(
        specificity, operators, operators, "Operator A subspace explains operator B",
        figures / "operator_specificity_heatmap.png", vmin=0, vmax=1,
    )
    _heatmap(
        angles, operators, operators, "Principal angles between operator subspaces (degrees)",
        figures / "principal_angle_heatmap.png", vmin=0, vmax=90, cmap="magma",
    )
    similarity = 1 - angles / 90
    _heatmap(
        similarity, operators, operators, "Operator subspace similarity (1 - angle/90)",
        figures / "subspace_similarity_heatmap.png", vmin=0, vmax=1,
    )
