#!/usr/bin/env python3
"""Run only the train-operator by test-operator task matrix."""

from __future__ import annotations

from _common import common_parser, configured
from src.pipeline import run_experiment


def main() -> None:
    parser = common_parser("Run cross-operator task generalization")
    parser.add_argument("--mode", choices=["real", "synthetic"])
    args = parser.parse_args()
    config, output_dir = configured(args, mode=args.mode)
    result = run_experiment(config, output_dir, run_operator=False, run_cross=True)
    print(f"Cross-operator evaluation complete: {output_dir}")
    print(f"Data mode: {result['mode']}")


if __name__ == "__main__":
    main()
