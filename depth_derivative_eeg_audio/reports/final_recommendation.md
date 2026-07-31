# Final Recommendation

## 1. Idea summary

Align adjacent-layer representation changes between a trainable EEG encoder and a frozen audio teacher, optionally through a soft monotonic depth map, instead of aligning only cumulative hidden states.

## 2. Closest prior work

ACL 2025 [Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation](https://aclanthology.org/2025.acl-long.1125/) directly matches feature trajectories and first-order adjacent-layer deltas. EEG/MEG–speech representation alignment, multi-level cross-modal alignment, adaptive teacher–student layer mapping, and transformation-centric brain analyses also exist separately.

## 3. What is genuinely new

- The checked corpus did not contain the exact combination of EEG–audio derivative alignment and a soft monotonic depth map.
- The proposed subject/stimulus-disjoint falsification framework and explicit shuffled/no-hierarchy controls are a useful application-specific evaluation package.

## 4. What is not new

- Adjacent-layer finite-difference distillation.
- Feature-trajectory matching.
- Multi-layer teacher supervision and learned layer mapping.
- Frozen speech representations supervising non-invasive brain encoders.
- The idea that model transformations can be more informative than accumulated embeddings.

## 5. Mathematical validity

The loss is implementable when adjacent states share shape and temporal grid. It is basis-dependent, not invariant to network reparameterization, and only equals a block residual output in restricted residual architectures. Projection heads can absorb the modality gap and make the method ordinary distillation. Monotonicity is a modeling constraint, not a neuroscientific consequence.

## 6. Synthetic validation

Ninety runs covered 18 mode/method configurations with five seeds each. In the hierarchical condition:

- final: 0.8743±0.0206;
- hidden: 0.8764±0.0267;
- DDA fixed: 0.8743±0.0200;
- DDA monotonic: 0.8757±0.0206;
- shuffled DDA: 0.8778±0.0142.

DDA monotonic minus hidden was −0.0007 (paired permutation p=1.0000). Correct-order DDA did not beat reversed order. The synthetic and explanation gates failed.

## 7. Real-data results

**Not run.** No licensed local EEG–audio dataset was supplied, and the prompt requires stopping expansion after novelty or synthetic failure. `benchmark_results.csv` records this explicitly.

## 8. Strongest positive evidence

- The implementation can optimize the proposed loss, recover finite mappings, and run deterministically.
- The modification is small (four mapping scalars for monotonic DDA beyond matched projectors) and adds no intended inference cost.
- Speech-model depth has documented coarse correspondence with auditory/cortical hierarchies.

## 9. Strongest negative evidence

- Direct prior art exists for the core feature-delta objective.
- No measurable advantage over hidden or final alignment in the controlled hierarchy.
- Reversing teacher residual order did not hurt performance.
- Subject nuisance leakage was higher for DDA than fixed/all-hidden alignment in this suite.
- No-audio representations nearly matched proposed performance.

## 10. Alternative explanations

Ordinary multi-layer regularization, normalization, projector capacity, and easy latent recoverability explain the observed results without ordered depth derivatives. The evidence does not require a hierarchy-matching mechanism.

## 11. Key failure modes

Basis non-identifiability; residual-norm confounding; projector bypass; non-monotonic/parallel brain processing; teacher hierarchy exceptions; depth/latency conflation; subject/story leakage; low semantic EEG SNR; extra multi-layer supervision; and selective teacher choice.

## 12. Required additional experiments

No additional large experiment is justified for the current method claim. A distinct diagnostic pivot would first test basis-invariant teacher transformation hierarchy and real auditory EEG probes without training a DDA encoder.

## 13. Scores

| Dimension | Score / 6 | Evidence |
|---|---:|---|
| Novelty | 2 | Exact EEG–audio combination is new, but FDD directly covers the core objective. |
| Motivation | 3 | Coarse speech/brain hierarchy evidence exists; residual semantics and EEG depth correspondence do not. |
| Minimality | 5 | Small training-only objective and four monotonic scalars; matched projectors still add capacity. |
| Empirical strength | 1 | Reproducible synthetic study is negative; no real data. |
| Mechanistic evidence | 2 | Strong controls were run, but they falsified rather than supported the mechanism. |
| Venue potential | 1 | Current positive method claim is not competitive. |

Venue fit if submitted now: ICLR 1/6, NeurIPS 1/6, ICML 1/6, CVPR 0/6, ACL 1/6, Interspeech 1/6, ICASSP 1/6. A negative-result or benchmark venue may value the audit, but that is a different paper.

## 14. Decision

**STOP：不建议继续当前方法。**

This decision concerns DDA as the headline method, not the broader question of how speech-model computations relate to auditory EEG.

## 15. Exact next steps

1. Archive the current 90-run suite as the immutable negative baseline.
2. Drop claims of a new depth-derivative paradigm and cite FDD explicitly.
3. If continuing the broader topic, redefine it as a diagnostic representation-analysis study.
4. Probe raw teacher hidden states and transformations for envelope, pitch, phoneme, word, semantics, and speaker before any EEG training.
5. Prefer basis-invariant CKA/RSA/cross-covariance and latency-aware encoding over coordinate MSE.
6. Use SparrKULee official held-out-subject and held-out-story splits if licensed local data are later supplied.
7. Pre-register one primary endpoint and a subject-level inference plan before accessing test labels.
8. Require correct-order advantage over shuffled/reversed order before reviving any hierarchy claim.
9. Require nuisance leakage to fall relative to matched all-hidden supervision.
10. Do not launch multi-teacher or multi-dataset scaling unless those gates pass in a new preregistered project.
