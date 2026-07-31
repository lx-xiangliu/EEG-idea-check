# Reviewer Critique

## Reviewer A — Multimodal representation

- Summary: a clean application of residualization to Q/K, but conditional-MI language is overstated.
- Strengths: minimal operator; explicit random/shuffled controls; reproducible code.
- Weaknesses: concept erasure and conditional contrastive prior art are dense; input residualization is equivalent in the current test.
- Questions: what non-commuting architecture makes attention-level placement necessary? Why not condition negatives/loss?
- Score: **4/10**; confidence **4/5**; tendency: **reject**.

## Reviewer B — EEG / neuroscience

- Summary: the shortcut question is important, but envelope/onset are not pure nuisances.
- Strengths: lag-aware design and explicit negative result policy.
- Weaknesses: no real EEG; no evidence of phonetic/semantic residual dependence; fixed linear TRF may miss subject latency and nonlinear tracking.
- Questions: how are latency, production artifacts, comprehension and acoustic–phonetic coupling handled?
- Score: **3/10**; confidence **5/5**; tendency: **reject**.

## Reviewer C — experiments

- Summary: engineering tests are solid, but the empirical paper is not yet present.
- Strengths: five seeds, leakage tests, checkpointing, leakage manifest audit, hard stop rules.
- Weaknesses: no real dataset, no three teachers/encoders, no full attention training, no subject-level real statistics, and synthetic random projection is competitive.
- Questions: can the result survive held-out subject + story, same-speaker acoustic matching, and matched compute?
- Score: **3/10**; confidence **5/5**; tendency: **reject**.

