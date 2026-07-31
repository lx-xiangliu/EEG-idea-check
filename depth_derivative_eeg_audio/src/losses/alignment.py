from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.models.depth import DepthDerivativeExtractor, MonotonicDepthMapper, SoftDepthMapper
from src.utils.repro import assert_finite


def _pool(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError(f"Expected [batch,time,features], got {tuple(value.shape)}")
    return value.mean(dim=1)


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(f"cosine inputs must have equal shape: {left.shape} vs {right.shape}")
    loss = 1.0 - F.cosine_similarity(left, right, dim=-1, eps=1e-6).mean()
    assert_finite("cosine distance", loss)
    return loss


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.reshape(left.shape[0], -1)
    right = right.reshape(right.shape[0], -1)
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    cross = left.T @ right
    numerator = (cross * cross).sum()
    left_norm = torch.linalg.norm(left.T @ left)
    right_norm = torch.linalg.norm(right.T @ right)
    return numerator / (left_norm * right_norm).clamp_min(1e-8)


@dataclass
class AlignmentDiagnostics:
    means: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    entropy: torch.Tensor | None = None


class AlignmentObjective(nn.Module):
    METHODS = {
        "final",
        "hidden",
        "all_hidden",
        "matryoshka",
        "dda_fixed",
        "dda_learned",
        "dda_monotonic",
        "shuffled_residual",
        "random_teacher",
        "no_audio",
    }

    def __init__(
        self,
        method: str,
        d_model: int,
        projection_dim: int,
        eeg_layers: int,
        audio_layers: int,
        normalize_residuals: bool = True,
        mapper_temperature: float = 0.55,
    ) -> None:
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(f"Unknown method {method!r}; choose from {sorted(self.METHODS)}")
        self.method = method
        self.normalize_residuals = normalize_residuals
        self.eeg_projector = nn.Linear(d_model, projection_dim, bias=False)
        self.audio_projector = nn.Linear(d_model, projection_dim, bias=False)
        nn.init.orthogonal_(self.eeg_projector.weight)
        nn.init.orthogonal_(self.audio_projector.weight)
        self.extractor = DepthDerivativeExtractor()
        self.mapper: MonotonicDepthMapper | SoftDepthMapper | None = None
        if method == "dda_monotonic":
            self.mapper = MonotonicDepthMapper(eeg_layers, audio_layers, mapper_temperature)
        elif method == "dda_learned":
            self.mapper = SoftDepthMapper(eeg_layers, audio_layers)
        self.eeg_layers = eeg_layers
        self.audio_layers = audio_layers

    def _pair_loss(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return cosine_distance(self.eeg_projector(_pool(left)), self.audio_projector(_pool(right)))

    def _fixed_indices(self, count: int, target_count: int) -> list[int]:
        return [round(i * (target_count - 1) / max(count - 1, 1)) for i in range(count)]

    def forward(
        self,
        eeg_states: list[torch.Tensor],
        audio_states: list[torch.Tensor],
        eeg_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AlignmentDiagnostics]:
        if len(eeg_states) != self.eeg_layers + 1:
            raise ValueError("Incorrect EEG hidden-state count")
        if len(audio_states) != self.audio_layers + 1:
            raise ValueError("Incorrect audio hidden-state count")
        diagnostics = AlignmentDiagnostics()
        if self.method == "final":
            return self._pair_loss(eeg_states[-1], audio_states[-1]), diagnostics
        if self.method == "matryoshka":
            eeg = self.eeg_projector(_pool(eeg_states[-1]))
            audio = self.audio_projector(_pool(audio_states[-1]))
            dims = sorted({max(2, eeg.shape[-1] // 4), max(2, eeg.shape[-1] // 2), eeg.shape[-1]})
            return torch.stack([cosine_distance(eeg[:, :dim], audio[:, :dim]) for dim in dims]).mean(), diagnostics
        if self.method == "hidden":
            indices = self._fixed_indices(self.eeg_layers + 1, self.audio_layers + 1)
            return torch.stack([self._pair_loss(eeg, audio_states[index]) for eeg, index in zip(eeg_states, indices)]).mean(), diagnostics
        if self.method == "all_hidden":
            losses = []
            for eeg in eeg_states:
                candidates = torch.stack([self._pair_loss(eeg, audio) for audio in audio_states])
                losses.append(torch.logsumexp(-candidates / 0.2, dim=0) * -0.2)
            return torch.stack(losses).mean(), diagnostics
        if self.method == "no_audio":
            if eeg_input is None:
                raise ValueError("no_audio objective requires eeg_input")
            first = self.eeg_projector(_pool(eeg_states[0]))
            last = self.eeg_projector(_pool(eeg_states[-1]))
            invariance = cosine_distance(last, first.detach())
            centered = last - last.mean(dim=0, keepdim=True)
            std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
            variance = F.relu(1.0 - std).mean()
            return invariance + 0.1 * variance, diagnostics

        eeg_residuals = self.extractor(eeg_states, normalize=self.normalize_residuals)
        audio_residuals = self.extractor(audio_states, normalize=self.normalize_residuals)
        if self.method == "shuffled_residual":
            order = list(reversed(range(len(audio_residuals))))
            audio_residuals = [audio_residuals[index] for index in order]
        if self.method in {"dda_fixed", "shuffled_residual", "random_teacher"}:
            indices = self._fixed_indices(self.eeg_layers, self.audio_layers)
            mapped = [audio_residuals[index] for index in indices]
            diagnostics.means = torch.tensor(indices, device=eeg_states[0].device, dtype=eeg_states[0].dtype)
        else:
            if self.mapper is None:
                raise RuntimeError("Learned DDA method is missing a mapper")
            mapped, raw = self.mapper(eeg_residuals, audio_residuals)
            diagnostics = AlignmentDiagnostics(
                means=raw["means"], weights=raw["weights"], entropy=raw["entropy"]
            )
        losses = [self._pair_loss(eeg, audio) for eeg, audio in zip(eeg_residuals, mapped)]
        return torch.stack(losses).mean(), diagnostics
