"""TRF-partial scaled dot-product attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.trf import ResidualMaker


class TRFPartialAttention(nn.Module):
    def __init__(
        self,
        ridge: float = 1e-4,
        method: str = "ridge",
        residualize_query: bool = True,
        residualize_key: bool = True,
        residualize_value: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.residualizer = ResidualMaker(ridge=ridge, method=method)
        self.residualize_query = residualize_query
        self.residualize_key = residualize_key
        self.residualize_value = residualize_value
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        eeg_covariates: Tensor,
        audio_covariates: Tensor,
        query_mask: Tensor | None = None,
        key_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query, key and value must have shape [batch, time, features]")
        if query.shape[0] != key.shape[0] or key.shape[:2] != value.shape[:2]:
            raise ValueError("incompatible query/key/value shapes")
        if query.shape[-1] != key.shape[-1]:
            raise ValueError("query and key feature dimensions must match")
        b, tq, _ = query.shape
        tk = key.shape[1]
        if query_mask is None:
            query_mask = torch.ones((b, tq), dtype=torch.bool, device=query.device)
        if key_mask is None:
            key_mask = torch.ones((b, tk), dtype=torch.bool, device=key.device)

        q = self.residualizer(query, eeg_covariates, query_mask) if self.residualize_query else query
        k = self.residualizer(key, audio_covariates, key_mask) if self.residualize_key else key
        v = self.residualizer(value, audio_covariates, key_mask) if self.residualize_value else value
        scores = q @ k.transpose(-2, -1) / math.sqrt(query.shape[-1])
        scores = scores.masked_fill(~key_mask[:, None, :], -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0)
        attention = self.dropout(attention) * query_mask[:, :, None]
        output = (attention @ v) * query_mask[:, :, None]
        diagnostics = {
            "query_residual_energy": self._energy_ratio(q, query, query_mask),
            "key_residual_energy": self._energy_ratio(k, key, key_mask),
            "value_residual_energy": self._energy_ratio(v, value, key_mask),
        }
        if return_attention:
            diagnostics["attention"] = attention
        return output, diagnostics

    @staticmethod
    def _energy_ratio(residual: Tensor, original: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1).to(original.dtype)
        numerator = (residual.square() * weights).sum(dim=(-2, -1))
        denominator = (original.square() * weights).sum(dim=(-2, -1)).clamp_min(1e-12)
        return numerator / denominator

