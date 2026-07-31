#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_REPO_ROOT / "outputs" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_REPO_ROOT / "outputs" / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import yaml

from src.config import ExperimentConfig, load_config
from src.data import make_synthetic_bundle
from src.metrics import paired_permutation_test
from src.training import RunResult, train_synthetic
from src.utils import configure_logging

LOGGER = logging.getLogger("run_synthetic")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the preregistered synthetic falsification suite")
    parser.add_argument("--config", default="configs/experiment/synthetic.yaml")
    parser.add_argument("--sweep", default="configs/sweeps/first_round.yaml")
    parser.add_argument("--output-dir", default="outputs/synthetic")
    parser.add_argument("--quick", action="store_true", help="Use 2 epochs while preserving 5 seeds")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Reuse an existing results.csv and details.json without retraining",
    )
    return parser.parse_args()


def _configurations(sweep: dict[str, object]) -> list[tuple[str, str]]:
    rows = [("hierarchical", method) for method in sweep["hierarchical_methods"]]
    for mode, methods in sweep["counterexamples"].items():
        rows.extend((mode, method) for method in methods)
    return rows


def _row(result: RunResult) -> dict[str, object]:
    payload = asdict(result)
    for key in [
        "mapping_means",
        "mapping_weights",
        "eeg_residual_norms",
        "audio_residual_norms",
        "hidden_cka_matrix",
        "residual_cka_matrix",
        "layer_probe_accuracy",
        "residual_probe_accuracy",
    ]:
        payload[key] = json.dumps(payload[key], separators=(",", ":"))
    return payload


def _mean_std(frame: pd.DataFrame, mode: str, method: str, metric: str) -> tuple[float, float]:
    values = frame[(frame["mode"] == mode) & (frame["method"] == method)][metric].to_numpy(float)
    return float(values.mean()), float(values.std(ddof=1))


def _paired(frame: pd.DataFrame, left: str, right: str, metric: str) -> tuple[float, float, float]:
    left_values = frame[(frame["mode"] == "hierarchical") & (frame["method"] == left)].sort_values("seed")[metric].to_numpy(float)
    right_values = frame[(frame["mode"] == "hierarchical") & (frame["method"] == right)].sort_values("seed")[metric].to_numpy(float)
    p_value, effect = paired_permutation_test(left_values, right_values, seed=20260731)
    return float((left_values - right_values).mean()), p_value, effect


def _save_figures(frame: pd.DataFrame, details: list[RunResult], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    hierarchical = frame[frame["mode"] == "hierarchical"]
    order = list(dict.fromkeys(hierarchical["method"].tolist()))
    means = [hierarchical[hierarchical["method"] == method]["probe_accuracy"].mean() for method in order]
    stds = [hierarchical[hierarchical["method"] == method]["probe_accuracy"].std(ddof=1) for method in order]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(order)), means, yerr=stds, capsize=3)
    ax.set_xticks(np.arange(len(order)), order, rotation=35, ha="right")
    ax.set_ylabel("Unseen-subject latent sign probe accuracy")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Hierarchical synthetic task: five-seed results")
    fig.tight_layout()
    fig.savefig(figure_dir / "hierarchical_probe_accuracy.png", dpi=180)
    plt.close(fig)

    monotonic = [item for item in details if item.mode == "hierarchical" and item.method == "dda_monotonic"]
    if monotonic and monotonic[0].mapping_weights:
        weights = np.mean([np.asarray(item.mapping_weights) for item in monotonic], axis=0)
        fig, ax = plt.subplots(figsize=(7, 4))
        image = ax.imshow(weights, aspect="auto", vmin=0, vmax=max(0.5, float(weights.max())), cmap="viridis")
        ax.set_xlabel("Audio residual layer")
        ax.set_ylabel("EEG residual layer")
        ax.set_title("Mean monotonic mapping over five seeds")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(figure_dir / "mapping_matrix.png", dpi=180)
        plt.close(fig)

    representative = next(
        item for item in details if item.mode == "hierarchical" and item.method == "dda_monotonic" and item.seed == 0
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, matrix, title in [
        (axes[0], np.asarray(representative.hidden_cka_matrix), "Hidden-state CKA"),
        (axes[1], np.asarray(representative.residual_cka_matrix), "Residual CKA"),
    ]:
        image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="magma")
        ax.set_xlabel("Audio depth")
        ax.set_ylabel("EEG depth")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figure_dir / "cka_matrices.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(representative.layer_probe_accuracy, marker="o", label="hidden states")
    ax.plot(np.arange(1, len(representative.residual_probe_accuracy) + 1), representative.residual_probe_accuracy, marker="s", label="depth derivatives")
    ax.set_xlabel("EEG layer / transition")
    ax.set_ylabel("Latent sign probe accuracy")
    ax.set_ylim(0.45, 1.0)
    ax.legend()
    ax.set_title("Layer probing (representative seed 0)")
    fig.tight_layout()
    fig.savefig(figure_dir / "probing_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(representative.eeg_residual_norms, marker="o", label="EEG")
    ax.plot(representative.audio_residual_norms, marker="s", label="audio teacher")
    ax.set_xlabel("Residual layer")
    ax.set_ylabel("Mean L2 norm after normalization")
    ax.legend()
    ax.set_title("Residual norms")
    fig.tight_layout()
    fig.savefig(figure_dir / "residual_norms.png", dpi=180)
    plt.close(fig)


def _write_reports(frame: pd.DataFrame, details: list[RunResult], root: Path, quick: bool) -> None:
    reports = root / "reports"
    figures = root / "outputs" / "synthetic" / "figures"
    dda_mean, dda_std = _mean_std(frame, "hierarchical", "dda_monotonic", "probe_accuracy")
    fixed_mean, fixed_std = _mean_std(frame, "hierarchical", "dda_fixed", "probe_accuracy")
    hidden_mean, hidden_std = _mean_std(frame, "hierarchical", "hidden", "probe_accuracy")
    final_mean, final_std = _mean_std(frame, "hierarchical", "final", "probe_accuracy")
    shuffled_mean, shuffled_std = _mean_std(frame, "hierarchical", "shuffled_residual", "probe_accuracy")
    flat_dda, flat_dda_std = _mean_std(frame, "flat", "dda_monotonic", "probe_accuracy")
    flat_hidden, flat_hidden_std = _mean_std(frame, "flat", "hidden", "probe_accuracy")
    nonmono_learned, nonmono_learned_std = _mean_std(frame, "nonmonotonic", "dda_learned", "probe_accuracy")
    nonmono_mono, nonmono_mono_std = _mean_std(frame, "nonmonotonic", "dda_monotonic", "probe_accuracy")
    delta_hidden, p_hidden, effect_hidden = _paired(frame, "dda_monotonic", "hidden", "probe_accuracy")
    delta_final, p_final, effect_final = _paired(frame, "dda_monotonic", "final", "probe_accuracy")
    delta_shuffle, p_shuffle, effect_shuffle = _paired(frame, "dda_monotonic", "shuffled_residual", "probe_accuracy")
    passes_hierarchy = dda_mean > hidden_mean and dda_mean > final_mean
    passes_order = dda_mean > shuffled_mean
    passes_flat = (flat_dda - flat_hidden) <= max(0.01, dda_mean - hidden_mean)
    synthetic_decision = "PASS" if passes_hierarchy and passes_order and passes_flat else "FAIL"
    budget_label = "quick 2-epoch diagnostic" if quick else "preregistered 10-epoch first round"

    text = f"""# Synthetic Validation

## Design

- Controlled data: three continuous latent variables (`z1` acoustic, `z2` phonetic, `z3` semantic), subject nuisance, noise, and disjoint subject/stimulus sets.
- EEG encoder: 4-layer pre-LN Transformer, 24 hidden units, 4 heads.
- Frozen audio teacher: 6 controlled residual layers.
- Evaluation: linear sign probes and ridge `R²` on unseen subjects and unseen stimuli.
- Seeds: 0–4 for every configuration; result unit is the seed, not a window.
- Run profile: {budget_label}.

## Main hierarchical result

| Method | Probe accuracy mean ± SD |
|---|---:|
| final state | {final_mean:.4f} ± {final_std:.4f} |
| fixed hidden alignment | {hidden_mean:.4f} ± {hidden_std:.4f} |
| DDA fixed | {fixed_mean:.4f} ± {fixed_std:.4f} |
| DDA monotonic | {dda_mean:.4f} ± {dda_std:.4f} |
| shuffled DDA | {shuffled_mean:.4f} ± {shuffled_std:.4f} |

Paired seed-level permutation tests (unadjusted exploratory p-values):

- DDA monotonic − hidden: Δ={delta_hidden:+.4f}, p={p_hidden:.4f}, paired standardized effect={effect_hidden:.3f}.
- DDA monotonic − final: Δ={delta_final:+.4f}, p={p_final:.4f}, effect={effect_final:.3f}.
- DDA monotonic − shuffled: Δ={delta_shuffle:+.4f}, p={p_shuffle:.4f}, effect={effect_shuffle:.3f}.

## Counterexamples

- No hierarchy: DDA monotonic {flat_dda:.4f} ± {flat_dda_std:.4f}; hidden {flat_hidden:.4f} ± {flat_hidden_std:.4f}.
- Non-monotonic hierarchy: unconstrained learned DDA {nonmono_learned:.4f} ± {nonmono_learned_std:.4f}; monotonic DDA {nonmono_mono:.4f} ± {nonmono_mono_std:.4f}.
- Parallel and shuffled-teacher controls are recorded in `ablation_results.csv`.

## Falsification gates

| Gate | Result |
|---|---|
| DDA beats hidden and final in hierarchical data | {'PASS' if passes_hierarchy else 'FAIL'} |
| Correct order beats shuffled order | {'PASS' if passes_order else 'FAIL'} |
| No-hierarchy advantage is not larger than hierarchical advantage | {'PASS' if passes_flat else 'FAIL'} |
| Overall synthetic gate | **{synthetic_decision}** |

The synthetic task validates implementation behavior only. It cannot establish that real EEG encoder depth corresponds to physiology or that an audio teacher has a true acoustic→semantic derivative hierarchy.

![Five-seed method comparison](../outputs/synthetic/figures/hierarchical_probe_accuracy.png)

## Stage status

- Completed: 5-seed hierarchical comparison, no-hierarchy, non-monotonic, parallel, shuffled-order, and random-teacher controls.
- Failed: see gates above; failures are retained rather than overwritten.
- Missing: noise/delay/data-scale grids beyond the first-round configuration.
- Key findings: numerical values above are generated from `outputs/synthetic/results.csv`.
- Blocking issues: novelty failure prevents treating a synthetic pass as sufficient evidence.
- Decision: **{synthetic_decision}** for the controlled synthetic mechanism; real-data claims remain not run.
"""
    (reports / "synthetic_validation.md").write_text(text, encoding="utf-8")

    stats_text = f"""# Statistical Analysis

## Unit of inference

Synthetic comparisons use five paired random seeds. Windows are not treated as independent inferential units. Real-data subject-level inference is not available because no real dataset was run.

## Paired permutation results

| Comparison | Mean paired delta | Two-sided p | Standardized paired effect |
|---|---:|---:|---:|
| DDA monotonic vs hidden | {delta_hidden:+.4f} | {p_hidden:.4f} | {effect_hidden:.3f} |
| DDA monotonic vs final | {delta_final:+.4f} | {p_final:.4f} | {effect_final:.3f} |
| DDA monotonic vs shuffled | {delta_shuffle:+.4f} | {p_shuffle:.4f} | {effect_shuffle:.3f} |

With only five seeds the exact sign-permutation resolution is coarse and confidence intervals are wide. These are mechanism checks, not publication-level evidence. Holm correction should be applied only after preregistering the final primary comparison family. Real experiments must use paired subject-level permutation/Wilcoxon and subject bootstrap, never window-level tests.

## Stage status

- Completed: paired seed-level permutation tests and effect sizes.
- Failed: no claim of subject-level significance is possible.
- Missing: per-subject real scores, Holm-corrected confirmatory family, and bootstrap CIs.
- Key findings: DDA-hidden exploratory delta is {delta_hidden:+.4f}.
- Blocking issues: real data absent.
- Decision: statistics are diagnostic only.
"""
    (reports / "statistical_analysis.md").write_text(stats_text, encoding="utf-8")

    representative = next(item for item in details if item.mode == "hierarchical" and item.method == "dda_monotonic" and item.seed == 0)
    representation_text = f"""# Representation Analysis

## Generated analyses

- [Hidden and residual CKA matrices](../outputs/synthetic/figures/cka_matrices.png)
- [Layer and residual probing curves](../outputs/synthetic/figures/probing_curves.png)
- [Residual norms](../outputs/synthetic/figures/residual_norms.png)
- [Soft monotonic mapping](../outputs/synthetic/figures/mapping_matrix.png)

Representative seed 0 mean fixed-diagonal hidden CKA: {representative.final_cka:.4f}; residual CKA: {representative.residual_cka:.4f}. Learned mapping means: `{representative.mapping_means}`. Because residuals were unit-normalized, the norm plot is a sanity check and is expected near one; raw-norm ablations remain required on real teachers.

## Interpretation limit

CKA and probing are computed on controlled synthetic latents. They show whether the implementation recovers the planted structure, not whether acoustic/phonetic/semantic information is hierarchically represented in real EEG.

## Stage status

- Completed: CKA, probing, residual norm, and mapping figures.
- Failed: no physiological interpretation is authorized.
- Missing: pre/post-training real-teacher probes, RSA, temporal latency sweep, speaker/device leakage.
- Key findings: mapping and CKA are recorded per seed in `outputs/synthetic/details.json`.
- Blocking issues: real synchronized EEG–audio data absent.
- Decision: implementation diagnostics complete.
"""
    (reports / "representation_analysis.md").write_text(representation_text, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    configure_logging()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    sweep = yaml.safe_load((root / args.sweep).read_text(encoding="utf-8"))
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.postprocess_only:
        results_path = output_dir / "results.csv"
        details_path = output_dir / "details.json"
        if not results_path.exists() or not details_path.exists():
            raise FileNotFoundError("--postprocess-only requires existing results.csv and details.json")
        frame = pd.read_csv(results_path)
        details = [RunResult(**item) for item in json.loads(details_path.read_text(encoding="utf-8"))]
        LOGGER.info("reusing %d completed runs", len(details))
    else:
        seeds = [int(seed) for seed in sweep["seeds"]]
        details = []
        configurations = _configurations(sweep)
        for index, (mode, method) in enumerate(configurations, start=1):
            for seed in seeds:
                data_cfg = replace(config.data, mode=mode)
                train_cfg = replace(
                    config.train,
                    seed=seed,
                    method=method,
                    epochs=2 if args.quick else config.train.epochs,
                    checkpoint_dir=str(output_dir / "checkpoints"),
                )
                run_cfg = ExperimentConfig(data=data_cfg, model=config.model, train=train_cfg)
                LOGGER.info("configuration %d/%d mode=%s method=%s seed=%d", index, len(configurations), mode, method, seed)
                bundle = make_synthetic_bundle(data_cfg, seed)
                details.append(train_synthetic(run_cfg, bundle, save_checkpoint=True))
        rows = [_row(item) for item in details]
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "results.csv", index=False)
        (output_dir / "details.json").write_text(
            json.dumps([asdict(item) for item in details], indent=2, allow_nan=False), encoding="utf-8"
        )
    summary = frame.groupby(["mode", "method"], as_index=False).agg(
        probe_accuracy_mean=("probe_accuracy", "mean"),
        probe_accuracy_sd=("probe_accuracy", "std"),
        probe_r2_mean=("probe_r2", "mean"),
        nuisance_accuracy_mean=("nuisance_accuracy", "mean"),
        final_cka_mean=("final_cka", "mean"),
        residual_cka_mean=("residual_cka", "mean"),
        wall_time_mean=("wall_time_seconds", "mean"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    frame.to_csv(root / "reports" / "ablation_results.csv", index=False)
    benchmark = pd.DataFrame(
        [
            {"dataset": name, "status": "not_run", "reason": "no licensed local dataset root supplied", "metric": "", "value": ""}
            for name in ["SparrKULee", "KUL", "DTU", "UGR-MINDVOICE", "ChineseEEG-2"]
        ]
    )
    benchmark.to_csv(root / "reports" / "benchmark_results.csv", index=False)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config": config.to_dict(),
        "sweep": sweep,
    }
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    _save_figures(frame, details, output_dir / "figures")
    _write_reports(frame, details, root, args.quick)
    LOGGER.info("completed %d runs; results=%s", len(details), output_dir / "results.csv")


if __name__ == "__main__":
    main()
