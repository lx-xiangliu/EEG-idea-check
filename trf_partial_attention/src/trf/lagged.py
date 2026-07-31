"""Lag-expanded acoustic design matrices."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LaggedDesignBuilder(nn.Module):
    """Build C(t-lag) along the time axis without circular wraparound."""

    def forward(
        self,
        acoustic_features: Tensor,
        lags_in_samples: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if acoustic_features.ndim != 3:
            raise ValueError("acoustic_features must have shape [batch, time, features]")
        if lags_in_samples.ndim != 1:
            raise ValueError("lags_in_samples must be one-dimensional")
        b, t, f = acoustic_features.shape
        if valid_mask is None:
            valid_mask = torch.ones((b, t), dtype=torch.bool, device=acoustic_features.device)
        if valid_mask.shape != (b, t):
            raise ValueError("valid_mask must have shape [batch, time]")

        blocks: list[Tensor] = []
        masks: list[Tensor] = []
        index = torch.arange(t, device=acoustic_features.device)
        for lag_tensor in lags_in_samples:
            lag = int(lag_tensor.item())
            source = index - lag
            inside = (source >= 0) & (source < t)
            safe = source.clamp(0, t - 1)
            block = acoustic_features[:, safe, :]
            block = block * inside.view(1, t, 1)
            source_valid = valid_mask[:, safe] & inside.view(1, t)
            blocks.append(block)
            masks.append(source_valid)
        design = torch.cat(blocks, dim=-1) if blocks else acoustic_features.new_empty((b, t, 0))
        design_valid = valid_mask & torch.stack(masks, dim=0).all(dim=0) if masks else valid_mask
        return design, design_valid

