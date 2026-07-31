from __future__ import annotations

import torch
from torch import Tensor, nn


class SmallEEGEncoder(nn.Module):
    """A small temporal encoder used only for the CPU smoke test."""

    def __init__(self, channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, eeg: Tensor) -> Tensor:
        if eeg.ndim != 3:
            raise ValueError("eeg must have shape [batch, time, channels]")
        return self.network(eeg.transpose(1, 2)).transpose(1, 2)

