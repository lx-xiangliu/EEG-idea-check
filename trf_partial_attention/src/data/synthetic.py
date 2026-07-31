"""Synthetic EEG-audio process with explicit acoustic, phonetic and semantic latents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SyntheticBatch:
    eeg: Tensor
    audio: Tensor
    acoustic: Tensor
    phonetic: Tensor
    semantic: Tensor
    subject: Tensor
    story: Tensor
    acoustic_group: Tensor
    true_lag: int


def generate_synthetic_batch(
    seed: int,
    condition: str,
    n_subjects: int = 8,
    n_stories: int = 8,
    time_steps: int = 48,
    dimension: int = 16,
) -> SyntheticBatch:
    """Generate one sample per subject/story and shared acoustic templates.

    Every two stories reuse the same low-level acoustic template.  Retrieval
    within those pairs is therefore an acoustically matched test of semantic
    signal rather than an envelope-identification shortcut.
    """
    valid = {"acoustic_only", "acoustic_semantic", "weak_semantic", "nonlinear", "wrong_lag", "random_nuisance"}
    if condition not in valid:
        raise ValueError(f"unknown synthetic condition: {condition}")
    generator = torch.Generator().manual_seed(seed)
    n_groups = max(2, n_stories // 2)
    acoustic_dim, phonetic_dim, semantic_dim = 3, 3, 4

    def smooth_noise(shape: tuple[int, ...]) -> Tensor:
        raw = torch.randn(shape, generator=generator)
        for _ in range(3):
            raw = (torch.roll(raw, 1, dims=-2) + 2 * raw + torch.roll(raw, -1, dims=-2)) / 4
        return raw

    acoustic_templates = smooth_noise((n_groups, time_steps, acoustic_dim))
    phonetic_templates = smooth_noise((n_stories, time_steps, phonetic_dim))
    semantic_vectors = torch.randn((n_stories, semantic_dim), generator=generator)
    semantic_vectors = semantic_vectors / semantic_vectors.norm(dim=-1, keepdim=True)
    basis, _ = torch.linalg.qr(torch.randn((dimension, dimension), generator=generator))
    wa = basis[:acoustic_dim]
    wp = basis[acoustic_dim : acoustic_dim + phonetic_dim]
    ws = basis[acoustic_dim + phonetic_dim : acoustic_dim + phonetic_dim + semantic_dim]
    subject_offsets = 0.35 * torch.randn((n_subjects, dimension), generator=generator)

    true_lag = 4 if condition == "wrong_lag" else 2
    semantic_gain = {
        "acoustic_only": 0.0,
        "acoustic_semantic": 0.65,
        "weak_semantic": 0.12,
        "nonlinear": 0.65,
        "wrong_lag": 0.65,
        "random_nuisance": 0.65,
    }[condition]
    phonetic_gain = 0.0 if condition == "acoustic_only" else 0.40
    rows: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, int, int]] = []
    for subject in range(n_subjects):
        for story in range(n_stories):
            group = story % n_groups
            za = acoustic_templates[group] + 0.04 * torch.randn((time_steps, acoustic_dim), generator=generator)
            zp = phonetic_templates[story]
            zs = semantic_vectors[story].expand(time_steps, -1)
            delayed = torch.roll(za, true_lag, dims=0)
            delayed[:true_lag] = 0
            audio = 2.4 * za @ wa + 0.45 * zp @ wp + 0.65 * zs @ ws
            eeg = 2.7 * delayed @ wa + phonetic_gain * zp @ wp + semantic_gain * zs @ ws
            if condition == "nonlinear":
                nonlinear = delayed.square() - delayed.square().mean(dim=0, keepdim=True)
                eeg = eeg + 1.1 * nonlinear @ wa
            eeg = eeg + subject_offsets[subject] + 0.30 * torch.randn((time_steps, dimension), generator=generator)
            audio = audio + 0.15 * torch.randn((time_steps, dimension), generator=generator)
            rows.append((eeg, audio, za, zp, semantic_vectors[story], subject, story, group))
    return SyntheticBatch(
        eeg=torch.stack([r[0] for r in rows]),
        audio=torch.stack([r[1] for r in rows]),
        acoustic=torch.stack([r[2] for r in rows]),
        phonetic=torch.stack([r[3] for r in rows]),
        semantic=torch.stack([r[4] for r in rows]),
        subject=torch.tensor([r[5] for r in rows]),
        story=torch.tensor([r[6] for r in rows]),
        acoustic_group=torch.tensor([r[7] for r in rows]),
        true_lag=true_lag,
    )
