#!/usr/bin/env python3
"""CPU smoke trainer; real-data training requires an explicit manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from _common import ROOT
from src.data import generate_synthetic_batch
from src.losses import symmetric_contrastive_loss
from src.models import SmallEEGEncoder
from src.training import SmokeTrainer
from src.utils import set_deterministic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "smoke" / "checkpoint.pt")
    args = parser.parse_args()
    set_deterministic(0)
    batch = generate_synthetic_batch(0, "acoustic_semantic", n_subjects=3, n_stories=4)
    model = SmallEEGEncoder(batch.eeg.shape[-1], hidden=batch.audio.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = SmokeTrainer(model, optimizer)
    start = trainer.load(args.resume) + 1 if args.resume else 0
    for epoch in range(start, args.epochs):
        eeg_embedding = model(batch.eeg).mean(1)
        audio_embedding = batch.audio.mean(1)
        loss = symmetric_contrastive_loss(eeg_embedding, audio_embedding)
        grad_norm = trainer.step(loss)
        print(f"epoch={epoch} loss={float(loss.detach()):.6f} grad_norm={grad_norm:.6f}")
        trainer.save(args.checkpoint, epoch)


if __name__ == "__main__":
    main()
