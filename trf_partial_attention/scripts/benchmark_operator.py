#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import torch

from _common import ROOT
from src.attention import TRFPartialAttention


def main() -> None:
    torch.manual_seed(0)
    rows = []
    for t, p, d in [(64, 8, 32), (128, 16, 64), (256, 32, 64)]:
        q = torch.randn(2, t, d)
        k = torch.randn(2, t, d)
        v = torch.randn(2, t, d)
        c = torch.randn(2, t, p)
        standard = TRFPartialAttention(residualize_query=False, residualize_key=False)
        partial = TRFPartialAttention(ridge=1e-3)
        for name, module in [("standard", standard), ("tpa", partial)]:
            for _ in range(2):
                module(q, k, v, c, c)
            started = perf_counter()
            repeats = 10
            for _ in range(repeats):
                module(q, k, v, c, c)
            elapsed = (perf_counter() - started) / repeats
            rows.append({"device": "cpu", "T": t, "p": p, "d": d, "method": name, "wall_seconds": elapsed})
    output = ROOT / "outputs" / "operator_benchmark.json"
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

