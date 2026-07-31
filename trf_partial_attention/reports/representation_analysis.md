# Representation Analysis

## Leakage–information trade-off

In the acoustic + semantic synthetic condition, Q+K residualization retains 24.1% of EEG representation energy. Acoustic probe R² drops from 0.557 to 0.006, while semantic R² drops from 0.417 to 0.326. The method removes the intended linear shortcut, but it is not selective enough to improve higher-level information.

The feature-count ablation makes the trade-off explicit:

| Acoustic dimensions controlled | Residual energy | Acoustic R² | Semantic R² | Matched R@1 |
|---:|---:|---:|---:|---:|
| 1 | 0.656 | 0.520 | 0.487 | 0.971 |
| 2 | 0.411 | 0.351 | 0.434 | 0.929 |
| 3 | 0.227 | 0.006 | 0.326 | 0.929 |

## Mechanism controls

- Random subspace retains 66.0% energy and matches/exceeds standard retrieval, so some apparent benefits can arise from generic geometry/regularization.
- Shuffled C does worse than true C on matched R@1, but has higher semantic probe than true C; no single metric establishes specificity.
- No-lag retains 71.7% energy and performs better than the wide lag design, rejecting the claim that lag expansion is necessary in this test.
- Intercept residualization erases time-constant story semantics, a concrete information-destruction mechanism.

## Not produced

Real layer-wise leakage curves, CKA/RSA, scalp/TRF topographies and learned attention maps were not produced because no real data or teacher model was configured. The generated method-comparison figure is synthetic and located at `outputs/synthetic/method_comparison.png`.

