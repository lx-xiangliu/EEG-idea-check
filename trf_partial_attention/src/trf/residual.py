"""Masked, batched residualization without constructing a T x T matrix."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualMaker(nn.Module):
    def __init__(self, ridge: float = 1e-4, method: str = "qr", add_intercept: bool = False):
        super().__init__()
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        if method not in {"qr", "ridge"}:
            raise ValueError("method must be 'qr' or 'ridge'")
        self.ridge = float(ridge)
        self.method = method
        self.add_intercept = add_intercept

    def forward(self, x: Tensor, covariates: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if x.ndim != 3 or covariates.ndim != 3:
            raise ValueError("x and covariates must have shape [batch, time, features]")
        if x.shape[:2] != covariates.shape[:2]:
            raise ValueError("x and covariates must share batch and time dimensions")
        b, t, _ = x.shape
        if valid_mask is None:
            valid_mask = torch.ones((b, t), dtype=torch.bool, device=x.device)
        if valid_mask.shape != (b, t):
            raise ValueError("valid_mask must have shape [batch, time]")
        c = covariates.detach()
        outputs: list[Tensor] = []
        for i in range(b):
            mask = valid_mask[i]
            xi = x[i, mask]
            ci = c[i, mask].to(dtype=x.dtype)
            if self.add_intercept:
                ci = torch.cat([torch.ones_like(ci[:, :1]), ci], dim=-1)
            ri = self._residualize_valid(xi, ci)
            full = torch.zeros_like(x[i])
            full[mask] = ri
            outputs.append(full)
        return torch.stack(outputs, dim=0)

    def _residualize_valid(self, x: Tensor, c: Tensor) -> Tensor:
        if c.numel() == 0 or c.shape[0] == 0:
            return x
        if self.method == "qr" and self.ridge == 0.0:
            q, r = torch.linalg.qr(c, mode="reduced")
            diagonal = torch.diagonal(r, 0, -2, -1).abs()
            tolerance = torch.finfo(c.dtype).eps * max(c.shape) * diagonal.max().clamp_min(1.0)
            rank = int((diagonal > tolerance).sum().item())
            q = q[:, :rank]
            return x - q @ (q.transpose(-2, -1) @ x)
        gram = c.transpose(-2, -1) @ c
        identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        penalty = self.ridge if self.ridge > 0 else torch.finfo(gram.dtype).eps
        beta = torch.linalg.solve(gram + penalty * identity, c.transpose(-2, -1) @ x)
        return x - c @ beta
