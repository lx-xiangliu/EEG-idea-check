# Statistical Analysis

## Scope

Only synthetic experiments were available. Five seeds were paired across methods. These seed-level tests do not substitute for real subject-level inference.

## Primary synthetic comparison (acoustic + semantic)

Q+K TPA minus standard:

- held-out-subject matched R@1: 0.929 vs 0.943, difference -0.014; exact Wilcoxon p = 1.000.
- semantic probe R²: 0.326 vs 0.417, difference -0.092; Wilcoxon p = 0.625.
- acoustic probe R²: 0.006 vs 0.557, difference -0.550; Wilcoxon p = 0.0625.

With only five pairs, the smallest nonzero two-sided exact Wilcoxon p-value is limited. The leakage decrease is large but does not meet a conventional 0.05 threshold, and it is accompanied by lower semantic recovery.

## Controls

- Input residualization and Q+K score residualization are numerically identical in this zero-training score benchmark, so no architecture effect can be estimated.
- True C vs shuffled C matched R@1: +0.029.
- True C vs random subspace: -0.014.
- Lag-expanded vs no-lag: -0.029.

No multiple-comparison-adjusted positive TPA claim is warranted.

## Preregistered real-data plan

Primary endpoint remains held-out-subject Recall@1 with acoustically matched negatives. Use subject averages, paired subject permutation, subject bootstrap 95% CI, Wilcoxon sensitivity analysis, effect size, and Holm correction. Windows must never be treated as independent inferential units.

