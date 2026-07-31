# Synthetic Validation

## Design

The generator has explicit acoustic, phonetic, semantic and subject latents. Two stories reuse one acoustic template, making the primary two-candidate retrieval comparison acoustically matched (chance R@1 = 0.5). Five deterministic seeds were run for 6 conditions × 14 methods = 420 configurations. The raw table is `outputs/synthetic/benchmark_results.csv`.

## Gate results

| Required check | Result | Evidence |
|---|---|---|
| B: TPA improves semantic recovery | Fail | Semantic R² 0.326 vs 0.417 standard. |
| A: no false higher-level information | Pass | Acoustic-only Q+K semantic R² -0.087 and matched R@1 0.543, near chance. |
| True C beats shuffled C | Pass | Matched R@1 0.929 vs 0.900 in B. |
| True C beats random subspace | Fail | Random projection R@1 0.943 and semantic R² 0.506. |
| Q+K beats input residualization | Fail | Identical by construction in this score-level zero-training setting. |
| Lag-expanded beats no-lag | Fail | 0.929 vs 0.957 matched R@1 in B; wrong-lag condition also fails. |

Gate 2 score: **2/6, PIVOT_OR_STOP**.

## Condition findings

- Acoustic only: standard and Q+K are near matched chance; no fabricated semantic recovery. This is the strongest correctness result.
- Acoustic + semantic: Q+K greatly reduces acoustic leakage but also lowers semantic probe and does not improve retrieval.
- Weak semantic: Q+K semantic R² is negative and random projection is stronger, showing a poor detection boundary.
- Nonlinear leakage: linear Q+K leaves a similar retrieval pattern; the aggressive nonlinear oracle removes almost all energy and damages performance.
- Wrong lag: the wide lag design removes more energy but no-lag performs at least as well, rejecting TRF necessity in this generator.
- Random nuisance: random projection is competitive, supporting a regularization/subspace-dimension explanation.

## Parameter ablations

- Feature count 1 → 3: acoustic probe falls 0.520 → 0.006, but semantic R² falls 0.487 → 0.326 and residual energy falls 0.656 → 0.227.
- Lag set: no-lag is best or tied on matched R@1 (0.957); adding 0–4 lags gives 0.929; adding a negative lag gives 0.900.
- Ridge from 0 to 0.1 is relatively insensitive in this well-conditioned synthetic setting.
- Adding an intercept changes semantic R² from 0.326 to approximately 0 and matched R@1 from 0.929 to 0.886. Intercept handling is a critical preregistration choice.

## Interpretation

The operator controls linear acoustic information, but the experiment falsifies the stronger claim that this automatically reveals or improves higher-level correspondence. The score-level identity with input residualization directly triggers Architecture Failure E.

