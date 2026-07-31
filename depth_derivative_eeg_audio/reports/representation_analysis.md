# Representation Analysis

## Generated analyses

- [Hidden and residual CKA matrices](../outputs/synthetic/figures/cka_matrices.png)
- [Layer and residual probing curves](../outputs/synthetic/figures/probing_curves.png)
- [Residual norms](../outputs/synthetic/figures/residual_norms.png)
- [Soft monotonic mapping](../outputs/synthetic/figures/mapping_matrix.png)

Representative seed 0 mean fixed-diagonal hidden CKA: 0.7776; residual CKA: 0.7421. Learned mapping means: `[1.1838057041168213, 2.483213186264038, 3.791762351989746, 5.0]`. Because residuals were unit-normalized, the norm plot is a sanity check and is expected near one; raw-norm ablations remain required on real teachers.

## Interpretation limit

CKA and probing are computed on controlled synthetic latents. They show whether the implementation recovers the planted structure, not whether acoustic/phonetic/semantic information is hierarchically represented in real EEG.

## Stage status

- Completed: CKA, probing, residual norm, and mapping figures.
- Failed: no physiological interpretation is authorized.
- Missing: pre/post-training real-teacher probes, RSA, temporal latency sweep, speaker/device leakage.
- Key findings: mapping and CKA are recorded per seed in `outputs/synthetic/details.json`.
- Blocking issues: real synchronized EEG–audio data absent.
- Decision: implementation diagnostics complete.
