#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from _common import ROOT
from src.analysis import run_benchmark, summarize_gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "synthetic")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = run_benchmark(list(range(args.seeds)))
    frame.to_csv(args.output / "benchmark_results.csv", index=False)
    summary = summarize_gates(frame)
    (args.output / "gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    subset = frame[frame.condition == "acoustic_semantic"].groupby("method").mean(numeric_only=True)
    subset[["held_out_subject_r1", "semantic_probe_r2", "acoustic_probe_r2"]].plot.bar(figsize=(12, 6))
    plt.ylabel("score")
    plt.title("Acoustic + semantic condition (5-seed mean)")
    plt.tight_layout()
    plt.savefig(args.output / "method_comparison.png", dpi=180)
    plt.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

