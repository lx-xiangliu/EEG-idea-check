# Theory and Mathematical Feasibility Review

## 1. What the four residual definitions actually mean

Let a block be pre-LN Transformer:

`U = H + Attn(LN(H))`, `H_next = U + FFN(LN(U))`.

- **A: output difference** `H_next - H` equals the sum of the attention update and the FFN update evaluated after the attention update. It is not a single branch output.
- **B: block residual branch** `F(H)` equals A only for a conceptual one-branch residual block `H_next = H + F(H)`. It is not literally equivalent to a standard two-sublayer Transformer unless the two branches are grouped into one composite `F`.
- **C: separate attention/FFN differences** is the most faithful computation-level definition for Transformers, but doubles targets and supervision and may create a capacity advantage over hidden-state baselines.
- **D: normalized difference** `LN(H_next)-LN(H)` is not the normalized version of A: LayerNorm is nonlinear and removes mean/scale components. It measures a change in direction/relative coordinates, discarding some magnitude information.

For post-LN blocks, `H_next = LN(H + F(H))`; hence `H_next-H` mixes the branch computation with normalization and is never equal to `F(H)` in general. For Conformer the difference aggregates attention, convolution, and two FFN branches. For CNN/SSM/Mamba it is well-defined only when successive states share time grid and width; downsampling or changing channel dimensions requires an explicit resampling/projection operator, making the derivative basis- and implementation-dependent.

## 2. Identifiability

Hidden states are not identifiable coordinates. If two encoders use invertible transforms `Q_E` and `Q_A`, then raw residual cosine/MSE can change even when task-relevant geometry is preserved. A projection head solves dimensional mismatch but introduces another non-identifiability: a sufficiently flexible projector can learn ordinary cross-modal distillation and erase the meaning of “residual.”

Minimum controls:

1. shared low-capacity linear projectors with matched parameter counts;
2. orthogonal/whitened projections or explicit covariance regularization;
3. coordinate-free alternatives: linear CKA, pairwise distance/RSA, covariance, cross-covariance, CCA, or subspace principal angles;
4. projector-only baseline trained on hidden states with exactly the same number of teacher layers;
5. frozen random teacher and cross-sample shuffle to test whether the projector bypasses semantics.

Raw cosine is acceptable only after temporal alignment and variance control. It measures direction, not magnitude. MSE confounds basis, scale, and direction. CKA/RSA are more invariant but discard sample-wise directional information. No single metric is privileged; the first study should use normalized cosine plus a geometry-based control.

## 3. Layer mapping schemes

| Mapping | Params | Stability | Interpretability | Collapse risk | Main failure |
|---|---:|---|---|---|---|
| Fixed linear | 0 | high | high | none | assumes equal abstraction speed |
| Learnable scalar + interpolation | L | medium | high | boundary collapse | non-smooth floor/ceil bookkeeping |
| Dense softmax | L×M | medium-low | medium | all rows pick same layer | ignores order unless regularized |
| Sinkhorn / OT | L×M plus iterations | medium | medium | diffuse or permutation-like coupling | extra compute and temperature sensitivity |
| Monotonic cumulative softplus | L | high | high | near-duplicate adjacent means | wrongly excludes feedback/non-monotone hierarchy |

The proposed cumulative-softplus formula should end at `M-1`, not `M`, to index `M` residual layers safely. A stable form is:

`mu_l = (M-1) * cumsum(softplus(u))[l] / sum(softplus(u))`.

Linear interpolation between neighboring audio residuals keeps gradients with respect to `mu`, but the integer boundary is piecewise differentiable. A soft Gaussian kernel around `mu` is smoother and exposes entropy as a collapse diagnostic.

## 4. Supportive reasons

1. Accumulated states contain previous computations; a transition target can emphasize what a block changes.
2. Speech-model layers show coarse acoustic-to-contextual trends and correlate differently with the auditory pathway.
3. A training-only auxiliary loss can add no inference cost.
4. Finite differences partially remove layer-persistent nuisance components when those components are approximately constant with depth.
5. Soft many-to-many mapping is more plausible than strict one-to-one matching.

## 5. Reasons against

1. A residual update is an optimization artifact, not automatically “new semantic information.”
2. Reparameterizing adjacent blocks can move the same function across layers and change deltas without changing the model.
3. Residual norms depend on LayerScale, normalization, optimizer, depth, and width.
4. Speech SSL hierarchies are imperfect: early features persist, HG may not prefer a unique depth, and feedback/parallel processing violate monotonicity.
5. EEG encoder depth is architectural, not physiological time or cortical depth.
6. Temporal latency and network depth are different axes; a model may trade one for the other.
7. Multi-layer targets provide more supervision than final-state baselines.
8. A projector can absorb the modality gap and reduce the method to ordinary multi-layer regression.
9. Differencing can amplify independent layer noise: `Var(e_{l+1}-e_l)=Var(e_{l+1})+Var(e_l)-2Cov`.
10. If hidden states are highly correlated across depth, deltas may have low signal-to-noise and be dominated by numerical/normalization effects.

## 6. Minimum conditions for the hypothesis to hold

1. Audio teacher residuals must contain reproducible layer-dependent information beyond residual norm.
2. EEG windows must contain corresponding acoustic/phonetic/semantic information at usable SNR.
3. EEG and audio temporal grids must be aligned within the tolerance of the loss.
4. A low-capacity mapping must bridge bases without learning the whole task.
5. Correct layer order must outperform shuffled/reversed order.
6. Improvement must remain after matched multi-layer supervision, normalization, parameter, and FLOP controls.
7. Gains must appear on unseen subjects and unseen stimuli, not only overlapping windows.

## 7. Explicit failure conditions

- Residual probes do not show a stronger or clearer hierarchy than hidden-state probes.
- DDA fails to beat hidden alignment in hierarchical synthetic data.
- DDA also improves equally in no-hierarchy synthetic data.
- Shuffling audio residual order causes no loss.
- Learned maps collapse to one teacher layer or vary radically across seeds.
- Hidden-state alignment with identical normalization/projectors matches DDA.
- Any gain disappears on subject- or stimulus-disjoint splits.
- Residual norm alone predicts the outcome.

## 8. Testable predictions

1. In hierarchical synthetic data, monotonic DDA should improve mean `z1/z2/z3` probe score over final/all-hidden alignment and recover ordered map means.
2. In no-hierarchy data, DDA should not outperform matched multi-layer hidden alignment.
3. In non-monotonic data, unconstrained soft mapping should beat monotonic mapping.
4. In parallel hierarchy, dense/OT mapping should beat scalar one-to-one mapping.
5. Shuffling teacher layer order should reduce DDA more than final-state alignment.
6. After DDA, subject-ID probe accuracy should fall without hurting stimulus retrieval.
7. Correct-order advantage should persist after per-layer unit normalization; otherwise the effect is likely norm matching.

## 9. The simpler explanation to beat

The simplest competing explanation is: **DDA is normalized multi-layer supervision plus a trainable projector.** This explanation predicts similar gains for normalized hidden alignment, random layer pairing, uniform multi-layer averaging, and no-hierarchy synthetic data. The proposed interpretation survives only if correct ordered deltas uniquely matter after those controls.

## Stage status

- Completed: residual equivalence, pre/post-LN, identifiability, mapping, risks, conditions, failures, and predictions.
- Failed: equivalence claim A=B for ordinary two-sublayer or post-LN Transformers.
- Missing: formal generalization bound; such a bound would require assumptions not justified by EEG data.
- Key findings: delta comparison is basis-dependent and requires low-capacity projections plus geometry controls.
- Blocking issues: no theoretical guarantee identifies model depth with physiology.
- Decision: mathematically implementable, mechanistically unproven, and highly falsifiable.
