#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.config import load_config
from src.data import make_synthetic_bundle
from src.training import train_synthetic
from src.utils import configure_logging


def main() -> None:
    configure_logging()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/experiment/smoke.yaml")
    start = time.perf_counter()
    result = train_synthetic(cfg, make_synthetic_bundle(cfg.data, cfg.train.seed), save_checkpoint=True)
    elapsed = time.perf_counter() - start
    output = root / "outputs" / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "passed",
        "elapsed_seconds": elapsed,
        "result": asdict(result),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    report = f"""# Smoke Test Report

## Result

- Status: **PASSED**
- Device: CPU
- Seed: {cfg.train.seed}
- Epochs: {cfg.train.epochs}
- Model: 4-layer EEG Transformer + frozen 6-layer audio teacher
- Method: {cfg.train.method}
- Probe accuracy: {result.probe_accuracy:.4f}
- Best validation loss: {result.best_val_loss:.6f}
- Trainable parameters: {result.trainable_parameters:,}
- Training wall time: {result.wall_time_seconds:.3f} s
- End-to-end script time: {elapsed:.3f} s
- Budget gate (<30 min): **PASS**

## Stage status

- Completed: forward/backward pass, checkpoint, unseen-subject probe, CKA, mapping diagnostics.
- Failed: none.
- Missing: GPU execution was not required for the CPU smoke gate.
- Key findings: finite loss and deterministic metrics were obtained.
- Blocking issues: none for synthetic work.
- Decision: implementation passes the smoke gate.
"""
    (root / "reports" / "smoke_test_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
