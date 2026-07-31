from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.utils.repro import assert_finite


class DepthDerivativeExtractor(nn.Module):
    def forward(
        self,
        hidden_states: list[torch.Tensor],
        normalize: bool = True,
    ) -> list[torch.Tensor]:
        if len(hidden_states) < 2:
            raise ValueError("At least two hidden states are required to compute a depth derivative")
        reference_shape = hidden_states[0].shape
        if any(state.shape != reference_shape for state in hidden_states):
            shapes = [tuple(state.shape) for state in hidden_states]
            raise ValueError(f"Hidden-state shapes must match before differencing: {shapes}")
        residuals = [right - left for left, right in zip(hidden_states, hidden_states[1:])]
        if normalize:
            residuals = [F.normalize(residual, dim=-1, eps=1e-6) for residual in residuals]
        for index, residual in enumerate(residuals):
            assert_finite(f"depth derivative {index}", residual)
        return residuals


def _stack_residuals(residuals: list[torch.Tensor]) -> torch.Tensor:
    if not residuals:
        raise ValueError("Residual list is empty")
    shape = residuals[0].shape
    if any(item.shape != shape for item in residuals):
        raise ValueError("All residual tensors must have the same shape")
    return torch.stack(residuals, dim=0)


class MonotonicDepthMapper(nn.Module):
    def __init__(self, eeg_layers: int, audio_layers: int, temperature: float = 0.55) -> None:
        super().__init__()
        if eeg_layers < 1 or audio_layers < 1:
            raise ValueError("Layer counts must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eeg_layers = eeg_layers
        self.audio_layers = audio_layers
        self.temperature = temperature
        self.increments = nn.Parameter(torch.zeros(eeg_layers))

    def mapping_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        positive = F.softplus(self.increments) + 1e-5
        cumulative = torch.cumsum(positive, dim=0)
        means = (self.audio_layers - 1) * cumulative / cumulative[-1]
        grid = torch.arange(self.audio_layers, device=means.device, dtype=means.dtype)
        logits = -0.5 * ((grid[None, :] - means[:, None]) / self.temperature) ** 2
        weights = torch.softmax(logits, dim=-1)
        return weights, means

    def forward(
        self,
        eeg_residuals: list[torch.Tensor],
        audio_residuals: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        if len(eeg_residuals) != self.eeg_layers:
            raise ValueError(f"Expected {self.eeg_layers} EEG residuals, got {len(eeg_residuals)}")
        if len(audio_residuals) != self.audio_layers:
            raise ValueError(f"Expected {self.audio_layers} audio residuals, got {len(audio_residuals)}")
        audio = _stack_residuals(audio_residuals)
        weights, means = self.mapping_weights()
        mapped_stack = torch.einsum("lm,mbtd->lbtd", weights, audio)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
        diagnostics = {"weights": weights, "means": means, "entropy": entropy}
        return list(mapped_stack.unbind(dim=0)), diagnostics


class SoftDepthMapper(nn.Module):
    def __init__(self, eeg_layers: int, audio_layers: int) -> None:
        super().__init__()
        self.eeg_layers = eeg_layers
        self.audio_layers = audio_layers
        initial = torch.full((eeg_layers, audio_layers), -2.0)
        for layer in range(eeg_layers):
            index = round(layer * (audio_layers - 1) / max(eeg_layers - 1, 1))
            initial[layer, index] = 2.0
        self.logits = nn.Parameter(initial)

    def forward(
        self,
        eeg_residuals: list[torch.Tensor],
        audio_residuals: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        if len(eeg_residuals) != self.eeg_layers or len(audio_residuals) != self.audio_layers:
            raise ValueError("Residual layer count does not match mapper configuration")
        weights = torch.softmax(self.logits, dim=-1)
        audio = _stack_residuals(audio_residuals)
        mapped = torch.einsum("lm,mbtd->lbtd", weights, audio)
        grid = torch.arange(self.audio_layers, device=weights.device, dtype=weights.dtype)
        means = (weights * grid).sum(dim=-1)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
        return list(mapped.unbind(dim=0)), {"weights": weights, "means": means, "entropy": entropy}
