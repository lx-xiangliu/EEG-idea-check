#!/usr/bin/env python3
"""Run only the subject-disjoint operator classification probe."""

from __future__ import annotations

from _common import common_parser, configured
from src.pipeline import run_experiment


def main() -> None:
    parser = common_parser("Run the operator separability experiment")
    parser.add_argument("--mode", choices=["real", "synthetic"])
    args = parser.parse_args()
    config, output_dir = configured(args, mode=args.mode)
    result = run_experiment(config, output_dir, run_operator=True, run_cross=False)
    print(f"Operator probe complete: {output_dir}")
    print(f"Data mode: {result['mode']}")


if __name__ == "__main__":
    main()
