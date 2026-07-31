#!/usr/bin/env python
"""Run H3 on real data or an explicitly selected mode."""

from _common import parser, resolved_config
from src.pipeline import H3Pipeline


def main():
    args = parser("COPA EEG H3 full experiment").parse_args()
    H3Pipeline(resolved_config(args)).run()


if __name__ == "__main__":
    main()

