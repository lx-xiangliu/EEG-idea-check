from __future__ import annotations

import torch
from torch import Tensor


def symmetric_contrastive_loss(eeg: Tensor, audio: Tensor, temperature: float = 0.1) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    eeg = torch.nn.functional.normalize(eeg, dim=-1)
    audio = torch.nn.functional.normalize(audio, dim=-1)
    logits = eeg @ audio.transpose(-2, -1) / temperature
    labels = torch.arange(len(eeg), device=eeg.device)
    return (torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.T, labels)) / 2

