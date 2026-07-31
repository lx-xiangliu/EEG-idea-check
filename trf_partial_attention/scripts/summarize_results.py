#!/usr/bin/env python3
from __future__ import annotations

from _common import ROOT
import pandas as pd


def main() -> None:
    raw = pd.read_csv(ROOT / "outputs" / "synthetic" / "benchmark_results.csv")
    numeric = [column for column in raw.columns if column not in {"seed", "condition", "method"}]
    aggregate = raw.groupby(["condition", "method"])[numeric].agg(["mean", "std"]).reset_index()
    aggregate.columns = ["_".join(part for part in column if part) if isinstance(column, tuple) else column for column in aggregate.columns]
    aggregate.to_csv(ROOT / "reports" / "benchmark_results.csv", index=False)
    controls = ["standard_attention", "input_residualization", "q_only", "k_only", "qk_tpa", "qkv_tpa", "random_subspace", "shuffled_covariates", "no_lag", "oracle_residualization", "nonlinear_oracle"]
    aggregate[aggregate.method.isin(controls)].to_csv(ROOT / "reports" / "ablation_results.csv", index=False)
    print(f"raw rows={len(raw)}, aggregate rows={len(aggregate)}")


if __name__ == "__main__":
    main()

