# Theory Review

## What the operator is

For each sample, TPA performs ridge-stabilized linear nuisance regression along the **time axis**:

\[
X^\perp = X-C(C^\top C+\lambda I)^{-1}C^\top X.
\]

It then substitutes residualized Q and/or K into scaled dot-product attention. The implementation never forms a `T × T` residual-maker matrix.

## What it is not

- An attention score is not a mutual-information estimator.
- Linear residualization does not generally estimate `I(E;A|C)`.
- Observational residualization is not a causal intervention.
- Zero linear covariance after projection does not imply nonlinear conditional independence.

In a jointly Gaussian model, the covariance of linear residuals is the partial covariance (a Schur complement), and zero partial correlation corresponds to conditional independence. Outside that setting, TPA can only be called **conditioning-inspired partialled alignment**.

## FWL conditions and ridge

Classical Frisch–Waugh–Lovell equivalence assumes linear least squares, a common sample space, the same nuisance design, and ordinary projection onto the column space of `C`. With `lambda > 0`, the ridge residual-maker is symmetric but not idempotent, so exact FWL equivalence is lost. With rank-deficient `C`, QR/SVD rank handling or ridge is needed. Estimating `C`-to-representation regression within short windows can have high variance when `p >= T`.

## Shapes and masks

`Q_E: [B,T_E,d]`, `C_E: [B,T_E,p_E]`, `K_A,V_A: [B,T_A,d]`, and `C_A: [B,T_A,p_A]`. Projection is per sample on the time dimension. Padding rows must be excluded from the solve and zeroed afterward. EEG and audio projections are independent, so `T_E` need not equal `T_A`; attention still yields `[B,T_E,T_A]`.

The primary complexity is `O(T p^2 + p^3 + T p d)`, plus ordinary attention. Explicit `M_C` would add `O(T^2 d)` and is not implemented.

## Q/K/V choices

| Variant | Interpretation | Main risk |
|---|---|---|
| Q only | EEG matching direction ignores linearly predictable acoustic tracking. | Audio K can still dominate score geometry. |
| K only | Audio matching direction removes low-level covariates. | EEG Q still carries the shortcut. |
| Q+K | Symmetric partialled score; recommended hypothesis test. | May remove legitimate phonetic/semantic components correlated with acoustics. |
| Q+K+V | Scores and delivered content are adjusted. | Highest information-destruction risk; poor default for tasks needing acoustic detail. |

Leaving V intact does **not** guarantee acoustic invariance of the output: attention weights can route unadjusted acoustic values. It only separates score control from content delivery.

## Intercept and centering

The mathematical prompt does not include an intercept, so the implementation defaults to `add_intercept=False`. Adding an intercept projects out all time-constant components within a window; in the synthetic generator, this destroys story-level semantic offsets. Real experiments must preregister centering/intercept behavior rather than silently adding one.

## Nonlinear and semantic overlap limitations

Teacher embeddings may encode envelope, pitch and speaker information nonlinearly. Linear residualization can leave such leakage. Conversely, phoneme onsets, syllables and word boundaries are physically coupled to acoustic onsets; aggressive removal can delete the very higher-level signal the method is intended to preserve. Pitch also encodes speaker identity and prosody. This makes residual energy, nonlinear leakage probes, and task-specific covariate ablations mandatory.

## Compute measurement

On the current CPU, forward-only microbenchmarks gave approximately 2.2–2.5× wall-time overhead for the tested small tensors: 0.103 vs 0.255 ms (`T=64,p=8,d=32`), 0.166 vs 0.404 ms (`128,16,64`), and 0.286 vs 0.617 ms (`256,32,64`). These are CPU microbenchmarks, not GPU training estimates.

## Theoretical decision

The operator is mathematically valid as ridge partialling, but the conditional-MI and causal interpretations are rejected. The strongest defensible claim is a minimal, differentiable, time-axis acoustic nuisance adjustment for attention scores.

