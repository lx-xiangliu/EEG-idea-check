# TRF-Partial Attention — Falsification-First Validation

This repository implements and tests time-axis acoustic residualization before EEG–audio attention. The current research decision is **PIVOT**, not GO: the synthetic gate fails architecture necessity, random-subspace specificity, and lag necessity. No real EEG–audio result is claimed.

## What is implemented

- PyTorch acoustic features: envelope, onset, F0, energy, spectral flux and centroid.
- Masked lag-expanded design matrices with positive and negative lags.
- Batched QR/ridge residualization without constructing a `T × T` matrix.
- Q/K/V-selective `TRFPartialAttention` with residual-energy diagnostics.
- Synthetic acoustic/phonetic/semantic/subject generator.
- 6 conditions, 14 methods, 5 seeds; parameter/lag/ridge/feature/intercept ablations.
- Split-leakage audit, deterministic mode, NaN checks, gradient clipping, checkpoint/resume.
- 62-paper novelty matrix and all requested reports.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

The verified run reused the already-installed environment at `../depth_derivative_eeg_audio/.venv` to avoid duplicating large PyTorch packages.

## Verify

```bash
python -m pytest -q
python scripts/train.py --epochs 3
python scripts/train.py --epochs 4 --resume outputs/smoke/checkpoint.pt
python scripts/run_zero_training_diagnostic.py
python scripts/run_synthetic_validation.py --seeds 5
python scripts/run_parameter_ablation.py
python scripts/benchmark_operator.py
python scripts/summarize_results.py
```

Verified: 11 tests passed; 420 synthetic configurations completed; checkpoint resume completed.

## Real data safety gate

`scripts/prepare_data.py` and teacher extraction intentionally stop unless a real manifest/model is explicitly configured. There is no silent dataset download, teacher substitution, or synthetic-to-real fallback. A real manifest must include subject, story, segment, start and duration fields and pass `audit_split_leakage`.

## Key outputs

- `reports/final_recommendation.md`: fixed-structure decision and exact next steps.
- `reports/novelty_matrix.csv`: 62 audited papers.
- `reports/benchmark_results.csv`: 5-seed aggregate synthetic results.
- `outputs/synthetic/benchmark_results.csv`: all 420 raw rows.
- `outputs/synthetic/parameter_ablation.csv`: ridge/lag/feature/intercept grid.
- `outputs/synthetic/method_comparison.png`: synthetic method comparison.

## Interpretation boundary

The implementation is linear nuisance adjustment. It must not be described as a general conditional-mutual-information estimator or causal adjustment. Real-data claims require passive-listening data, held-out subjects and stories, acoustically matched negatives, and subject-level inference.

