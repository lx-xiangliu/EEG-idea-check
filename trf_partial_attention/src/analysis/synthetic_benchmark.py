"""Zero-training falsification benchmark for residualized EEG-audio similarity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from src.data import SyntheticBatch, generate_synthetic_batch
from src.trf import LaggedDesignBuilder, ResidualMaker


METHODS = [
    "standard_attention", "standard_contrastive", "input_residualization",
    "pooled_residualization", "loss_partial_correlation", "q_only", "k_only",
    "qk_tpa", "qkv_tpa", "random_subspace", "shuffled_covariates", "no_lag",
    "oracle_residualization", "nonlinear_oracle",
]
CONDITIONS = ["acoustic_only", "acoustic_semantic", "weak_semantic", "nonlinear", "wrong_lag", "random_nuisance"]


@dataclass
class Result:
    seed: int
    condition: str
    method: str
    held_out_subject_r1: float
    held_out_story_r1: float
    random_negative_r1: float
    semantic_probe_r2: float
    acoustic_probe_r2: float
    alignment_margin: float
    residual_energy: float
    wall_seconds: float


def _design(acoustic: torch.Tensor, lags: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    builder = LaggedDesignBuilder()
    return builder(acoustic, torch.tensor(lags, dtype=torch.long))


def _residualize(x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ResidualMaker(ridge=1e-3, method="ridge")(x, c, mask)


def _representations(batch: SyntheticBatch, method: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    q, k = batch.eeg, batch.audio
    eeg_c, eeg_mask = _design(batch.acoustic, list(range(0, batch.true_lag + 3)))
    audio_c, audio_mask = _design(batch.acoustic, [0])
    mask = eeg_mask & audio_mask
    if method in {"standard_attention", "standard_contrastive"}:
        qr, kr = q, k
    elif method in {"input_residualization", "loss_partial_correlation", "qk_tpa", "qkv_tpa"}:
        qr, kr = _residualize(q, eeg_c, mask), _residualize(k, audio_c, mask)
    elif method == "q_only":
        qr, kr = _residualize(q, eeg_c, mask), k
    elif method == "k_only":
        qr, kr = q, _residualize(k, audio_c, mask)
    elif method == "no_lag":
        no_lag_c, no_lag_mask = _design(batch.acoustic, [0])
        common = no_lag_mask & audio_mask
        qr, kr, mask = _residualize(q, no_lag_c, common), _residualize(k, audio_c, common), common
    elif method == "shuffled_covariates":
        generator = torch.Generator().manual_seed(seed + 9103)
        permutation = torch.randperm(len(batch.acoustic), generator=generator)
        shuffled, shuffled_mask = _design(batch.acoustic[permutation], list(range(0, batch.true_lag + 3)))
        mask = shuffled_mask & audio_mask
        qr, kr = _residualize(q, shuffled, mask), _residualize(k, audio_c, mask)
    elif method == "random_subspace":
        generator = torch.Generator().manual_seed(seed + 2207)
        random_c = torch.randn(eeg_c.shape, generator=generator)
        qr, kr = _residualize(q, random_c, mask), _residualize(k, random_c[..., : audio_c.shape[-1]], mask)
    elif method == "oracle_residualization":
        # The generator is linear in lagged acoustic latents except in the nonlinear condition.
        qr, kr = _residualize(q, eeg_c, mask), _residualize(k, audio_c, mask)
    elif method == "nonlinear_oracle":
        nonlinear = torch.cat([eeg_c, eeg_c.square()], dim=-1)
        audio_nonlinear = torch.cat([audio_c, audio_c.square()], dim=-1)
        qr, kr = _residualize(q, nonlinear, mask), _residualize(k, audio_nonlinear, mask)
    elif method == "pooled_residualization":
        # Pooled residualization cannot remove within-window acoustic dynamics.
        qr, kr = q - q.mean(1, keepdim=True), k - k.mean(1, keepdim=True)
    else:
        raise ValueError(method)
    weights = mask.unsqueeze(-1).to(q.dtype)
    energy = float((qr.square() * weights).sum() / (q.square() * weights).sum().clamp_min(1e-12))
    return qr * weights, kr * weights, energy


def _pair_scores(q: torch.Tensor, k: torch.Tensor) -> np.ndarray:
    qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    kn = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.einsum("ntd,mtd->nm", qn, kn).div(q.shape[1]).cpu().numpy()


def _subset_r1(scores: np.ndarray, batch: SyntheticBatch, query_indices: np.ndarray, matched: bool) -> float:
    hits: list[float] = []
    groups = batch.acoustic_group.numpy()
    subjects = batch.subject.numpy()
    stories = batch.story.numpy()
    for i in query_indices:
        if matched:
            candidates = np.flatnonzero((groups == groups[i]) & (subjects == subjects[i]))
        else:
            candidates = np.flatnonzero(subjects == subjects[i])
        prediction = candidates[np.argmax(scores[i, candidates])]
        hits.append(float(stories[prediction] == stories[i]))
    return float(np.mean(hits))


def _probe_r2(x: torch.Tensor, target: torch.Tensor, train: np.ndarray, test: np.ndarray) -> float:
    features = x.mean(dim=1).cpu().numpy()
    y = target.cpu().numpy()
    model = Ridge(alpha=1.0).fit(features[train], y[train])
    return float(r2_score(y[test], model.predict(features[test]), multioutput="variance_weighted"))


def evaluate(seed: int, condition: str, method: str) -> Result:
    started = perf_counter()
    batch = generate_synthetic_batch(seed, condition)
    q, k, energy = _representations(batch, method, seed)
    scores = _pair_scores(q, k)
    subjects, stories = batch.subject.numpy(), batch.story.numpy()
    held_subject = subjects >= subjects.max() - 1
    held_story = stories == stories.max()
    train = np.flatnonzero(~held_subject & ~held_story)
    test = np.flatnonzero(held_subject & ~held_story)
    story_test = np.flatnonzero(~held_subject & held_story)
    matched = _subset_r1(scores, batch, test, matched=True)
    story_r1 = _subset_r1(scores, batch, story_test, matched=True)
    random_r1 = _subset_r1(scores, batch, test, matched=False)
    correct = np.arange(len(scores))
    diagonal = scores[correct, correct]
    shifted = scores[correct, np.roll(correct, 1)]
    acoustic_target = batch.acoustic.mean(dim=1)
    return Result(
        seed, condition, method, matched, story_r1, random_r1,
        _probe_r2(q, batch.semantic, train, test),
        _probe_r2(q, acoustic_target, train, test),
        float(np.mean(diagonal - shifted)), energy, perf_counter() - started,
    )


def run_benchmark(seeds: list[int] | None = None) -> pd.DataFrame:
    seeds = list(range(5)) if seeds is None else seeds
    rows = [asdict(evaluate(seed, condition, method)) for seed in seeds for condition in CONDITIONS for method in METHODS]
    return pd.DataFrame(rows)


def summarize_gates(frame: pd.DataFrame) -> dict[str, object]:
    means = frame.groupby(["condition", "method"]).mean(numeric_only=True)
    b = means.loc["acoustic_semantic"]
    a = means.loc["acoustic_only"]
    e = means.loc["wrong_lag"]
    checks = {
        "B_semantic_improves": bool(b.loc["qk_tpa", "semantic_probe_r2"] > b.loc["standard_attention", "semantic_probe_r2"]),
        "A_no_false_semantics": bool(a.loc["qk_tpa", "semantic_probe_r2"] <= 0.05),
        "true_beats_shuffled": bool(b.loc["qk_tpa", "held_out_subject_r1"] > b.loc["shuffled_covariates", "held_out_subject_r1"]),
        "true_beats_random": bool(b.loc["qk_tpa", "held_out_subject_r1"] > b.loc["random_subspace", "held_out_subject_r1"]),
        "qk_beats_input": bool(b.loc["qk_tpa", "held_out_subject_r1"] > b.loc["input_residualization", "held_out_subject_r1"] + 1e-6),
        "lag_beats_no_lag": bool(e.loc["qk_tpa", "held_out_subject_r1"] > e.loc["no_lag", "held_out_subject_r1"]),
    }
    return {"checks": checks, "passed": int(sum(checks.values())), "total": len(checks), "decision": "PASS" if all(checks.values()) else "PIVOT_OR_STOP"}

