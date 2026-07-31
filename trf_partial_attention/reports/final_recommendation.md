# Final Recommendation

## 1. Idea summary

TRF-Partial Attention residualizes EEG queries and audio keys against low-level acoustic covariates (with an EEG lag expansion) before cross-modal attention, aiming to measure correspondence not explained by easy acoustic tracking.

## 2. Closest prior work

The closest families are acoustic-controlled EEG TRFs, brain/model feature-removal studies, INLP/LEACE concept erasure, conditional contrastive learning, and causal/deconfounded vision-language attention. No audited paper directly combines lagged acoustic `C_E` with both `M_C Q` and `M_C K` for EEG–audio attention.

## 3. What is genuinely new

The exact EEG–audio operator placement and the joint evaluation protocol—acoustically matched negatives, true/shuffled/random C, lag necessity, and input-vs-attention residualization—were not found as one prior method.

## 4. What is not new

TRFs, ridge residualization, FWL logic, nuisance subspace projection, concept erasure, conditional contrastive objectives, hard negatives, and deconfounded attention are established.

## 5. Mathematical validity

The operator is valid linear ridge partialling along time. It is not a general estimator of conditional mutual information and is not causal adjustment. Exact FWL equivalence is lost under ridge. Nonlinear leakage and collateral removal remain.

## 6. Zero-training diagnostic

No real diagnostic was run because no licensed synchronized EEG–audio data or teacher checkpoints were configured. The synthetic proxy retains some semantic signal after adjustment but cannot pass real Gate 1.

## 7. Synthetic validation

Five seeds × six conditions × fourteen methods were run. Gate 2 passed only 2/6 requirements. The method avoids fabricating semantic signal in acoustic-only data and true C beats shuffled C, but it fails semantic improvement, random-subspace specificity, attention-level necessity, and lag necessity.

## 8. Real-data results

**未运行真实数据实验。** Held-out-subject/story retrieval, three audio teachers, three EEG encoders, LoRA/full fine-tuning, few-shot transfer, and real subject-level statistics remain missing.

## 9. Strongest positive evidence

Linear acoustic leakage falls sharply in the acoustic+semantic synthetic condition (probe R² 0.557 → 0.006), while 24.1% residual energy and semantic R² 0.326 remain. The acoustic-only condition stays near chance and does not manufacture semantic information.

## 10. Strongest negative evidence

Primary matched R@1 falls 0.943 → 0.929; semantic R² falls 0.417 → 0.326; random projection is competitive; no-lag is better; and input residualization is identical in the score-level test. Adding an intercept can erase story semantics entirely.

## 11. Alternative explanations

Generic subspace regularization, residual energy differences, acoustic–phonetic coupling, window-constant semantic removal, and unadjusted V routing can explain apparent behavior without conditional high-level EEG–audio dependence.

## 12. Key failure modes

Architecture failure, random-subspace mechanism failure, lag-necessity failure, information destruction, and no held-out-subject primary gain are triggered synthetically. Real Gate 1 remains untested.

## 13. Required additional experiments

1. Obtain passive-listening data with subject and story IDs and run the zero-training teacher/layer/lag diagnostic.
2. If and only if real residual dependence is significant, compare a non-commuting attention implementation to input residualization at matched capacity.
3. Preregister intercept/centering, lag selection, primary endpoint and train-only nuisance estimation.
4. Use same-speaker acoustically matched negatives and subject-level inference.
5. Add nonlinear leakage probes and a value-path ablation.

## 14. Scores

| Dimension | Score / 6 | Evidence |
|---|---:|---|
| Novelty | 3 | Exact combination not found; all constituents established. |
| Motivation | 4 | Strong shortcut rationale and adjacent brain evidence. |
| Minimality | 5 | Local operator, small parameter cost. |
| Empirical strength | 1 | Synthetic only; primary result negative. |
| Mechanistic evidence | 2 | Good controls, but several fail. |
| Venue potential | 2 | Protocol may suit ICASSP/Interspeech after real evidence; not ready for a top ML venue. |

Venue outlook: ICLR/NeurIPS/ICML 2/6; ACL 2/6; Interspeech 3/6; ICASSP 3/6; CVPR 1/6.

## 15. Decision

# PIVOT

Stop the current claim that an attention-level operator is necessary. Preserve the stronger research contribution as a shortcut-audit protocol plus simple, explicit residualization baselines. Resume TPA architecture work only if real Gate 1 passes and a non-commuting attention implementation beats input/loss-level controls.

## 16. Exact next steps

1. Request SparrKULee access and inspect ChineseEEG-2 passive-listening annotations.
2. Build a leakage-free subject/story manifest; run the supplied audit.
3. Extract low-level C and frozen WavLM, Whisper and BEATs/CLAP-family layers.
4. Run held-out-story partial alignment before any training.
5. If significant after correction, implement trained standard vs input-residualized vs TPA models with five seeds.
6. Stop permanently if true C does not beat shuffled/random C or if TPA does not beat input residualization on held-out-subject matched retrieval.

## Answers to the ten required questions

1. Real residual EEG–audio association: **unknown; not measured**. Synthetic: yes, but weaker.
2. Real phonetic/lexical/semantic content: **unknown**. Synthetic semantic information remains but decreases.
3. TRF lag expansion necessary: **no in the current synthetic test**.
4. Q/K better than input residualization: **no; identical in the score-level test**.
5. True C better than shuffled/random: **better than shuffled, not random**.
6. Hard-negative retrieval improves: **no**.
7. Held-out subject/story improves: **subject no; story at ceiling and not persuasive**.
8. Is it nuisance projection application: **principally yes**, with a specific operator/protocol.
9. Top-conference novelty sufficient: **not with current evidence**.
10. Direct falsifier: true-C TPA failing to beat input residualization and shuffled/random controls on held-out subject/story matched retrieval.

