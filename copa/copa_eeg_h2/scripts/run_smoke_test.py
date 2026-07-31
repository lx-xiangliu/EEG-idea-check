#!/usr/bin/env python
"""Run a small synthetic flow check; never scientific evidence."""

from _common import parser, resolved_config
from src.pipeline import H2Pipeline


def main() -> None:
    cli = parser("COPA EEG H2 synthetic smoke test")
    cli.set_defaults(mode="synthetic", output_dir="outputs/smoke")
    args = cli.parse_args()
    config = resolved_config(args)
    smoke = config["smoke"]
    if args.max_subjects is None:
        config["data"]["max_subjects"] = int(smoke["max_subjects"])
    if args.max_epochs_per_subject is None:
        config["data"]["max_epochs_per_subject"] = int(smoke["max_epochs_per_subject"])
    if args.bootstrap_iterations is None:
        config["analysis"]["bootstrap_iterations"] = int(smoke["bootstrap_iterations"])
    config["analysis"]["random_subspace_repeats"] = int(smoke["random_subspace_repeats"])
    H2Pipeline(config).run()


if __name__ == "__main__":
    main()
