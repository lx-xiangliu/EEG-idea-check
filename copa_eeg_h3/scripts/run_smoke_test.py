#!/usr/bin/env python
"""Run the synthetic architecture flow check."""

from _common import parser, resolved_config
from src.pipeline import H3Pipeline


def main():
    cli = parser("COPA EEG H3 synthetic smoke test")
    cli.set_defaults(mode="synthetic", output_dir="outputs/smoke")
    args = cli.parse_args()
    config = resolved_config(args)
    smoke = config["smoke"]
    if args.max_subjects is None:
        config["data"]["max_subjects"] = int(smoke["max_subjects"])
    if args.max_epochs_per_subject is None:
        config["data"]["max_epochs_per_subject"] = int(smoke["max_epochs_per_subject"])
    if args.bootstrap_iterations is None:
        config["probe"]["bootstrap_iterations"] = int(smoke["bootstrap_iterations"])
    H3Pipeline(config).run()


if __name__ == "__main__":
    main()

