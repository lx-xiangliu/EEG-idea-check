# Literature Review

## Scope and method

Search window: 2018-01-01 through 2026-07-31. The audit used targeted web searches, primary paper pages, and six cached OpenAlex queries. The machine-readable evidence is `novelty_matrix.csv`; cached query responses are under `source_metadata/openalex/`. We screened 62 unique papers and inspected abstracts/method descriptions for the closest works. This is a reproducible novelty audit, not a claim that no unindexed paper exists.

| Family | Count | Main lesson |
|---|---:|---|
| TRF / speech EEG | 10 | Lagged acoustic regression and acoustic/linguistic variance partitioning are established. |
| EEG–audio | 10 | Envelope reconstruction, match–mismatch and auditory attention are mature; many benchmarks reward low-level tracking. |
| Nuisance / projection | 14 | Linear concept erasure and null-space projection are established and have known collateral-removal risks. |
| Conditional multimodal | 10 | Conditional sampling/losses and hard negatives are established alternatives to modifying attention. |
| Attention deconfounding | 7 | “Deconfounded/causal attention” exists in vision-language work, generally via intervention or reweighting rather than `M_C Q, M_C K`. |
| Brain/model variance partitioning | 11 | Higher-level tracking beyond acoustics is sometimes measurable, but highly task-, region- and representation-dependent. |

## Closest evidence

1. [Neural Markers of Speech Comprehension](https://pubmed.ncbi.nlm.nih.gov/34858238/) explicitly evaluates linguistic EEG tracking while controlling speech acoustics. It validates the scientific question, but does not residualize attention queries and keys.
2. [Speech language models lack important brain-relevant semantics](https://aclanthology.org/2024.acl-long.462/) removes low-level stimulus features before brain alignment. It is the closest mechanism-level warning: residual speech-model alignment need not contain late-language semantics.
3. [Null It Out (INLP)](https://aclanthology.org/2020.acl-main.647/) and [LEACE](https://openreview.net/forum?id=awIpKpwTwF) establish projection-based concept removal. They make “orthogonal nuisance removal” prior art, even though their sample/feature geometry differs from TPA.
4. [Conditional Contrastive Learning with Kernel](https://openreview.net/forum?id=AAJLBoGt0XM) establishes conditioning through negative/positive sampling and kernel weights. It is a strong loss-level alternative.
5. [Deconfounded Visual Grounding](https://arxiv.org/abs/2112.15324), [Causal Attention for Vision-Language Tasks](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_Causal_Attention_for_Vision-Language_Tasks_CVPR_2021_paper.html), and related work establish attention deconfounding as a broad label, but the audited methods do not apply a lag-expanded observed acoustic design to both Q and K.
6. [Eelbrain/TRF tutorial](https://pmc.ncbi.nlm.nih.gov/articles/PMC10783870/) shows envelope alone can explain much of predictable EEG variance and that onset/spectro-temporal features add unique variance. This supports a richer `C`, while warning that “envelope only” is not a complete shortcut control.

## Answers to the six novelty questions

1. No audited paper explicitly computes both `M_C Q` and `M_C K` immediately before EEG–audio attention. This is “not found”, not a proof of absence.
2. Lag-expanded acoustic designs are standard for EEG TRFs, but no audited EEG–audio attention paper uses that design specifically as the Q/K nuisance subspace.
3. FWL, residualization and concept erasure are established. Calling their use “an approximation to conditional mutual information” is too strong without a probabilistic estimator or restrictive assumptions.
4. Several EEG/fMRI papers report linguistic effects after acoustic controls; others show that residual speech-model alignment lacks higher-level semantics. Therefore Gate 1 is empirical, not guaranteed.
5. Conditional/hard-negative sampling is established; acoustically matched EEG–audio negatives appear to be an underused but natural protocol contribution.
6. The contribution is best described as an EEG–audio-specific combination and evaluation protocol, with a potentially new local operator. It is not a new statistical principle.

## Stage status

- Completed: 62-paper matrix; six-family coverage; closest-overlap analysis.
- Failed: no evidence supporting a claim that residualization itself is new or causal.
- Missing: exhaustive citation-graph/full-text search of paywalled papers.
- Key findings: direct operator overlap not found; adjacent prior art is dense.
- Blocking issues: none for synthetic validation.
- Decision: novelty grade **B (borderline C)**.

