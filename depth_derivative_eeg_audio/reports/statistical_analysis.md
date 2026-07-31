# Statistical Analysis

## Unit of inference

Synthetic comparisons use five paired random seeds. Windows are not treated as independent inferential units. Real-data subject-level inference is not available because no real dataset was run.

## Paired permutation results

| Comparison | Mean paired delta | Two-sided p | Standardized paired effect |
|---|---:|---:|---:|
| DDA monotonic vs hidden | -0.0007 | 1.0000 | -0.088 |
| DDA monotonic vs final | +0.0014 | 0.6215 | 0.174 |
| DDA monotonic vs shuffled | -0.0021 | 0.8120 | -0.208 |

With only five seeds the exact sign-permutation resolution is coarse and confidence intervals are wide. These are mechanism checks, not publication-level evidence. Holm correction should be applied only after preregistering the final primary comparison family. Real experiments must use paired subject-level permutation/Wilcoxon and subject bootstrap, never window-level tests.

## Stage status

- Completed: paired seed-level permutation tests and effect sizes.
- Failed: no claim of subject-level significance is possible.
- Missing: per-subject real scores, Holm-corrected confirmatory family, and bootstrap CIs.
- Key findings: DDA-hidden exploratory delta is -0.0007.
- Blocking issues: real data absent.
- Decision: statistics are diagnostic only.
