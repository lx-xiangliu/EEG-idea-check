from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


@dataclass
class SmokeTrainer:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    gradient_clip: float = 1.0

    def step(self, loss: Tensor) -> float:
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite gradient norm")
        self.optimizer.step()
        return float(norm)

    def save(self, path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: Path) -> int:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return int(checkpoint["epoch"])

