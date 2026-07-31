#!/usr/bin/env python
"""Run real or explicitly selected H2 experiment."""

from _common import parser, resolved_config
from src.pipeline import H2Pipeline


def main() -> None:
    args = parser("COPA EEG H2 full experiment").parse_args()
    H2Pipeline(resolved_config(args)).run()


if __name__ == "__main__":
    main()
