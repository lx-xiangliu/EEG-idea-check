# Smoke Test Report

## Environment

- macOS CPU execution; Python 3.12.13 in an existing isolated environment.
- PyTorch 2.13.0, NumPy 2.3.5, SciPy 1.18.0, pandas 2.2.3, scikit-learn 1.9.0.
- No CUDA device or real EEG–audio dataset was used.

## Verification

- Unit tests: **11 passed in 1.06 s** after the final residualization correction.
- Covered: acoustic features, lag sign/mask, QR and ridge residualization, rank deficiency, padding, attention output, key/query masks, split leakage, determinism, and gradients.
- Gradient test confirms gradients reach Q/K/V while covariates are detached.
- Smoke trainer: loss 4.0882 → 3.7012 → 3.3705 across epochs 0–2; resumed checkpoint at epoch 3 with loss 3.0789.
- NaN checks and gradient clipping executed; checkpoint save/resume verified.

## Operator timing

CPU forward microbenchmark overhead was about 2.2–2.5× for tested small shapes. This is acceptable for a diagnostic operator, but GPU memory and full-training throughput remain unmeasured.

## Decision

Engineering smoke gate: **PASS**. Research Gate 2: **FAIL**.

