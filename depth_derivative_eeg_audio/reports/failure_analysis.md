# Failure Analysis

## Executive finding

The current Depth-Derivative Alignment (DDA) hypothesis failed its preregistered minimum gates. In the planted hierarchical synthetic task, monotonic DDA did not outperform fixed hidden-state or final-state alignment, and correct teacher order did not outperform reversed residual order. Combined with the direct ACL 2025 Feature Dynamics Distillation prior, this is a stop condition for the current top-level research claim.

## Observed failures

| Failure criterion | Evidence | Status |
|---|---|---|
| Novelty failure | FDD 2025 explicitly matches adjacent-layer first differences and feature trajectories | **Triggered** |
| Synthetic failure | DDA monotonic 0.8757±0.0206 vs hidden 0.8764±0.0267; paired Δ −0.0007, p=1.0000 | **Triggered** |
| Explanation failure | Correct-order DDA 0.8757±0.0206 vs reversed 0.8778±0.0142; Δ −0.0021, p=0.8120 | **Triggered** |
| Control failure | all-hidden 0.8799±0.0252 and shuffled DDA 0.8778±0.0142 are at least as good as proposed | **Triggered** |
| Real-data failure | No real-data run was started after earlier gates failed | Not evaluated |
| Generalization failure | Synthetic test is subject/stimulus-disjoint, but all methods were similar | No unique DDA benefit |

## Strongest negative evidence

1. Correct depth order was unnecessary under the chosen controlled hierarchy. That directly undermines the mechanistic claim that performance depends on ordered layer evolution.
2. DDA's mean difference from hidden alignment was effectively zero relative to seed variation.
3. A no-audio encoder already reached 0.8743±0.0320, showing the sign-probe task is largely solvable from the EEG input itself and has limited headroom for alignment objectives.
4. Random/cross-sample teacher supervision reached 0.8660±0.0360 and even higher `R²` than several semantic teacher objectives, indicating that regularization and representation reshaping can mimic “teacher benefit.”
5. Subject nuisance leakage remained high for DDA (0.7375 mean) and was lower for fixed/all-hidden alignment (0.5667/0.5583). DDA did not establish the claimed nuisance-history reduction.

## Alternative explanations

### Ordinary multi-layer regularization

The similarity of hidden, all-hidden, DDA, and shuffled DDA supports the simpler explanation that any multi-layer objective regularizes the encoder. Correct derivative semantics are not required.

### Probe ceiling and task sufficiency

The no-audio probe is already strong. The synthetic EEG transformation preserves the latent signs, so final-state representations can solve the probe without ordered audio supervision. This limits the ability to show a DDA-specific gain, but it is not a reason to retroactively change the benchmark: the claimed method should not need a hand-crafted task where final states are made intentionally defective.

### Projection-head absorption

The matched linear projectors can ignore some teacher-specific components. That makes hidden and derivative alignment more similar, exactly as predicted by the theory review. Freezing or constraining projectors is a future mechanistic experiment, not grounds to discard this negative run.

### Network depth is not the planted latent hierarchy

The teacher has a planted order, but the EEG Transformer is free to encode all latents early. DDA did not induce a reliably ordered EEG trajectory. This is a substantive failure of the intended inductive bias, not merely an implementation error; shape, gradient, mapping, and determinism tests passed.

## What was ruled out

- Silent CPU/GPU fallback: device was explicitly CPU.
- Split leakage: subject and stimulus identities are disjoint by construction; automated tests cover leakage.
- Seed selection: all preregistered seeds 0–4 are retained.
- NaN or optimizer failure: finite guards remained silent, checkpoints were produced, and validation losses improved.
- Parameter-count advantage: methods share the encoder and matched projectors; monotonic mapping adds only four scalars.
- Single-run accident: conclusions use five paired seeds for each configuration.

## Why real-data training was stopped

The prompt requires stopping expansion after novelty or synthetic failure. Real-data downloads would add license acceptance, tens of gigabytes, teacher extraction, and GPU compute without the minimum controlled mechanism having passed. Starting them now would violate the specified gate and invite post-hoc hypothesis rescue.

## Legitimate future pivot

A new project could pivot from “DDA improves EEG–audio learning” to a diagnostic question: **Which basis-invariant measures of speech-model transformations predict auditory EEG beyond accumulated states?** That would emphasize CKA/RSA/cross-covariance analysis, latency, and negative controls rather than claim a novel derivative-distillation objective.

## Stage status

- Completed: novelty, synthetic, explanation, control, leakage, and alternative-explanation audit.
- Failed: the core positive DDA claim.
- Missing: real-data outcomes by design after stop gate.
- Key findings: no ordered-derivative advantage and high nuisance leakage.
- Blocking issues: both novelty and synthetic gates failed.
- Decision: **STOP current method; consider a distinct diagnostic pivot.**
