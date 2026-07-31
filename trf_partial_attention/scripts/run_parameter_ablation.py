#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import torch

from _common import ROOT
from src.analysis.synthetic_benchmark import _design, _pair_scores, _probe_r2, _subset_r1
from src.data import generate_synthetic_batch
from src.trf import ResidualMaker


@dataclass
class Row:
    seed: int
    family: str
    setting: str
    held_out_subject_r1: float
    semantic_probe_r2: float
    acoustic_probe_r2: float
    residual_energy: float


def evaluate(seed: int, family: str, setting: str, lags: list[int], ridge: float, features: int, intercept: bool) -> Row:
    batch = generate_synthetic_batch(seed, "acoustic_semantic")
    acoustic = batch.acoustic[..., :features]
    eeg_c, eeg_mask = _design(acoustic, lags)
    audio_c, audio_mask = _design(acoustic, [0])
    mask = eeg_mask & audio_mask
    method = "qr" if ridge == 0 else "ridge"
    residualizer = ResidualMaker(ridge=ridge, method=method, add_intercept=intercept)
    q = residualizer(batch.eeg, eeg_c, mask) * mask.unsqueeze(-1)
    k = residualizer(batch.audio, audio_c, mask) * mask.unsqueeze(-1)
    scores = _pair_scores(q, k)
    subjects, stories = batch.subject.numpy(), batch.story.numpy()
    held_subject = subjects >= subjects.max() - 1
    held_story = stories == stories.max()
    train = (~held_subject & ~held_story).nonzero()[0]
    test = (held_subject & ~held_story).nonzero()[0]
    energy = float(q.square().sum() / batch.eeg.square().sum().clamp_min(1e-12))
    return Row(
        seed, family, setting, _subset_r1(scores, batch, test, True),
        _probe_r2(q, batch.semantic, train, test),
        _probe_r2(q, batch.acoustic.mean(1), train, test), energy,
    )


def main() -> None:
    rows: list[Row] = []
    for seed in range(5):
        for ridge in [0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]:
            rows.append(evaluate(seed, "ridge", str(ridge), [0, 1, 2, 3, 4], ridge, 3, False))
        lag_sets = {"no_lag": [0], "0_1": [0, 1], "0_2": [0, 1, 2], "0_4": [0, 1, 2, 3, 4], "neg1_4": [-1, 0, 1, 2, 3, 4], "wrong_3_5": [3, 4, 5]}
        for name, lags in lag_sets.items():
            rows.append(evaluate(seed, "lags", name, lags, 1e-3, 3, False))
        for features in [1, 2, 3]:
            rows.append(evaluate(seed, "feature_count", str(features), [0, 1, 2, 3, 4], 1e-3, features, False))
        for intercept in [False, True]:
            rows.append(evaluate(seed, "intercept", str(intercept).lower(), [0, 1, 2, 3, 4], 1e-3, 3, intercept))
    frame = pd.DataFrame(asdict(row) for row in rows)
    output = ROOT / "outputs" / "synthetic" / "parameter_ablation.csv"
    frame.to_csv(output, index=False)
    print(frame.groupby(["family", "setting"]).mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()

