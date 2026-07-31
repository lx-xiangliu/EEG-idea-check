#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.results)
    print(frame.groupby(["condition", "method"]).mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()

