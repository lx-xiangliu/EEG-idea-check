# Zero-Training Diagnostic

## Scope

No synchronized real EEG–audio dataset or licensed model checkpoint was configured. Therefore the required WavLM/HuBERT, Whisper, and BEATs/CLAP layer scan was **not run**. The values below are a synthetic proxy only and cannot satisfy real-data Gate 1.

## Synthetic proxy (acoustic + semantic condition, 5 seeds)

| Method | Matched R@1 | Semantic probe R² | Acoustic probe R² | Alignment margin | Residual energy |
|---|---:|---:|---:|---:|---:|
| Standard score | 0.943 ± 0.060 | 0.417 ± 0.360 | 0.557 ± 0.221 | 0.418 ± 0.067 | 1.000 |
| Q+K residualized score | 0.929 ± 0.051 | 0.326 ± 0.060 | 0.006 ± 0.264 | 0.154 ± 0.024 | 0.241 ± 0.064 |
| No-lag residualization | 0.957 ± 0.064 | 0.400 ± 0.331 | -0.015 ± 0.469 | 0.177 ± 0.018 | 0.717 ± 0.067 |

The residual score remains above the acoustic-only condition and retains measurable synthetic semantic information, but the properly lagged version is not better than no-lag and does not improve retrieval.

## Gate 1

**INCONCLUSIVE / NOT PASSED.** Synthetic evidence cannot establish that real EEG retains audio-teacher dependence after acoustic adjustment. Expensive real-data training is blocked until a subject/story-disjoint real diagnostic passes.

## Required real diagnostic

Use at least one passive-listening dataset, frozen WavLM/HuBERT, Whisper, and BEATs/CLAP-family teachers, train-only normalization, held-out story scoring, layer × lag scans, and subject-level permutation/bootstrap inference. Do not tune layer/lag on test subjects.

