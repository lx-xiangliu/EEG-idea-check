#!/usr/bin/env python3
"""Run both H1 experiments in real-data or explicit synthetic mode."""

from __future__ import annotations

from _common import common_parser, configured
from src.pipeline import run_experiment


def main() -> None:
    parser = common_parser("Run the complete COPA EEG H1 experiment")
    parser.add_argument("--mode", choices=["real", "synthetic"])
    args = parser.parse_args()
    config, output_dir = configured(args, mode=args.mode)
    result = run_experiment(config, output_dir)
    print(f"Complete H1 run written to: {output_dir}")
    print(f"Data mode: {result['mode']}")


if __name__ == "__main__":
    main()
