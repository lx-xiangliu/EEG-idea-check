# Reviewer Critique

## Reviewer A — Methods

### Summary

The submission proposes aligning adjacent-layer EEG and audio feature differences with a soft monotonic layer map. The implementation and negative controls are unusually transparent, but the core objective substantially overlaps ACL 2025 Feature Dynamics Distillation and fails the authors' own synthetic gate.

### Strengths

- Clear mathematical definitions and pre/post-LN caveats.
- Matched projectors, five seeds, shuffled/random teacher controls, and explicit stop rules.
- Minimal training-only modification and no proposed inference overhead.
- Negative results are preserved.

### Weaknesses

- The headline objective is not new after FDD.
- Correct residual order does not matter empirically.
- Monotonic mapping is assumed rather than identified and its means are partly determined by its parameterization.
- Trainable projection heads can reduce DDA to ordinary multi-layer distillation.
- No benefit over all-hidden supervision; no real data.

### Questions

1. What claim remains beyond applying FDD to a new modality pair?
2. Why should model depth be comparable across architectures after arbitrary reparameterization?
3. Does a shared orthogonal projector change the shuffled-order result?
4. Can a basis-invariant transformation measure outperform hidden-state CKA without training?

- Score: **3/10**
- Confidence: **5/5**
- Tendency: **Reject**

## Reviewer B — EEG / Neuroscience

### Summary

The work imports an appealing acoustic→phonetic→semantic hierarchy into EEG encoder depth. Existing speech/brain studies support coarse correlations but also show region-specific exceptions, persistent representations, and parallel processing. The current experiment contains no biological EEG.

### Strengths

- Avoids claiming that network depth literally equals cortical depth in the theory report.
- Plans subject/stimulus-disjoint evaluation and nuisance leakage probes.
- Correctly identifies volume conduction, latency, and subject variation as threats.

### Weaknesses

- No evidence that scalp EEG contains separable phonetic and semantic residual updates.
- Model depth, neural latency, cortical hierarchy, and laminar depth are conflated in the motivating narrative.
- A monotonic feed-forward map is biologically questionable given feedback and parallel pathways.
- Subject leakage remains high in the proposed synthetic model.
- Overt-speech datasets would introduce EMG/articulation confounds; listened-speech semantics may be weak at short windows.

### Questions

1. What preregistered EEG probe would falsify H2 before downstream training?
2. How will latency be separated from layer depth?
3. How will ocular/EMG and story identity shortcuts be excluded?
4. Why expect four EEG Transformer transitions to correspond to six speech-model transitions?

- Score: **2/10**
- Confidence: **4/5**
- Tendency: **Reject**

## Reviewer C — Experiments and Reproducibility

### Summary

The repository is reproducible for its synthetic scope and has strong leakage tests. However, the main empirical claim is negative and the requested benchmark/ablation breadth is absent because the authors correctly stopped after early failure.

### Strengths

- CPU smoke test, deterministic seeds, checkpoints, environment record, and machine-readable outputs.
- Five paired seeds and no use of windows as inferential units.
- Subject and stimulus splits are disjoint in the synthetic data.
- All 90 preregistered runs are retained.

### Weaknesses

- Only one small synthetic scale and one primary noise level.
- No three-teacher, depth, channel, delay, data-scale, or real-dataset ablations.
- Five seeds still give coarse permutation-test resolution.
- The synthetic sign probe has high no-audio performance and limited headroom.
- Benchmark CSV correctly says `not_run`, so there is no evidence for venue-level claims.

### Questions

1. Are raw and normalized residual norms both reported in a future real run?
2. Will the official ICASSP held-out-subject/story scores be reported separately?
3. How will FLOPs and projector capacity be exactly matched?
4. What is the single confirmatory primary endpoint and correction family?

- Score: **3/10**
- Confidence: **5/5**
- Tendency: **Reject**

## Meta-review

The reviewers agree that the engineering and falsification discipline are strengths. They also agree that direct prior work and a failed synthetic/order mechanism leave no publishable positive method claim. The appropriate outcome is rejection of the current method and preservation of the repository as a negative-result audit.
