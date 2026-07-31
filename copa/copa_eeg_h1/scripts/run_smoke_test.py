#!/usr/bin/env python3
"""Run the complete deterministic synthetic flow check."""

from __future__ import annotations

from _common import common_parser, configured
from src.config import apply_overrides
from src.pipeline import run_experiment


def main() -> None:
    parser = common_parser("Run a 5-10 minute synthetic smoke test")
    args = parser.parse_args()
    config, output_dir = configured(args, mode="synthetic")
    smoke = config["smoke"]
    if args.max_subjects is None:
        config["data"]["max_subjects"] = int(smoke["max_subjects"])
    if args.max_epochs_per_subject is None:
        config["data"]["max_epochs_per_subject"] = int(
            smoke["max_epochs_per_subject"]
        )
    config = apply_overrides(
        config, bootstrap_iterations=int(smoke["bootstrap_iterations"])
    )
    result = run_experiment(config, output_dir)
    probe = result["operator_probe"]["models"]["logistic_regression"][
        "balanced_accuracy"
    ]
    drop = result["cross_operator"]["average_cross_operator_drop"]
    print(f"Synthetic smoke test complete: {output_dir}")
    print("SCIENTIFIC STATUS: flow check only; not evidence for or against H1")
    print(f"Operator probe balanced accuracy: {probe:.4f}")
    print(f"Average cross-operator drop: {drop:.4f}")


if __name__ == "__main__":
    main()
