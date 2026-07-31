#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a completed result CSV without retraining")
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    path = Path(args.results)
    if not path.exists():
        raise FileNotFoundError(f"Result file does not exist: {path}")
    frame = pd.read_csv(path)
    required = {"mode", "method", "seed", "probe_accuracy", "probe_r2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Result file is missing columns: {sorted(missing)}")
    summary = frame.groupby(["mode", "method"])[["probe_accuracy", "probe_r2"]].agg(["mean", "std", "count"])
    print(json.dumps(json.loads(summary.to_json()), indent=2))


if __name__ == "__main__":
    main()
