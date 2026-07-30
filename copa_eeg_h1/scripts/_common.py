"""CLI bootstrap shared by project scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import apply_overrides, load_config, resolve_project_path


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="YAML configuration path",
    )
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--max-epochs-per-subject", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    return parser


def configured(args: argparse.Namespace, mode: str | None = None):
    config = load_config(args.config)
    config = apply_overrides(
        config,
        max_subjects=args.max_subjects,
        max_epochs_per_subject=args.max_epochs_per_subject,
        seed=args.seed,
        mode=mode,
        output_dir=args.output_dir,
    )
    output_value = args.output_dir or config["project"]["output_dir"]
    return config, resolve_project_path(output_value)
