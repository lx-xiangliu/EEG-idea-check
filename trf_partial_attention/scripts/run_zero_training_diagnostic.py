#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from _common import ROOT
from src.analysis.synthetic_benchmark import evaluate


def main() -> None:
    rows = [evaluate(seed, "acoustic_semantic", method).__dict__ for seed in range(5) for method in ["standard_attention", "qk_tpa", "no_lag"]]
    output = ROOT / "outputs" / "zero_training_proxy"
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostic.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} synthetic-proxy rows to {output}")


if __name__ == "__main__":
    main()

