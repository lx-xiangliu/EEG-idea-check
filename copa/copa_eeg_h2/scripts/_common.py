"""Shared CLI helpers."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "cache" / "matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from src.config import load_config, with_overrides


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    result.add_argument("--mode", choices=["real", "synthetic"])
    result.add_argument("--output-dir")
    result.add_argument("--max-subjects", type=int)
    result.add_argument("--max-epochs-per-subject", type=int)
    result.add_argument("--bootstrap-iterations", type=int)
    result.add_argument("--seed", type=int)
    return result


def resolved_config(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    return with_overrides(
        config,
        mode=args.mode,
        output_dir=args.output_dir,
        max_subjects=args.max_subjects,
        max_epochs_per_subject=args.max_epochs_per_subject,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
