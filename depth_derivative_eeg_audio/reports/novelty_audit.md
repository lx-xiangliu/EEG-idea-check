# Novelty Audit

## Executive finding

**Novelty grade: C.** The EEG–audio instantiation and its proposed monotonic soft layer map were not found verbatim, but the core mathematical objective—aligning feature trajectories and adjacent-layer first differences—was published as Feature Dynamics Distillation (FDD) at ACL 2025. Multi-level cross-modal alignment, adaptive layer mapping, and network-transformation/brain correspondence also predate this proposal. The defensible contribution is therefore a controlled transfer and mechanistic test in EEG–audio, not a new general paradigm.

## Direct answers

1. **Existing explicit alignment of `H[l+1]-H[l]`?** Yes. FDD defines layer feature delta KD as a finite difference across adjacent features and matches teacher/student deltas alongside the trajectory.
2. **Existing layer-transition / trajectory alignment?** Yes. FDD is direct; TinyBERT, PKD, Universal-KD, C2KD, and TALAS cover multi-layer or learned layer mapping; transformation-centric brain papers analyze layer computations.
3. **Existing EEG–audio learned monotonic layer mapping?** None found in the checked set. This is the narrowest apparently new element.
4. **Depth as an abstraction trajectory in other modalities?** Yes. Speech/brain hierarchy work, hierarchical cross-modal prompt learning, and multiple distillation papers do so, with important counterexamples.
5. **Classification:** combination innovation / existing-method transfer with a new application-specific constraint and falsification package.

## Closest 10 works

## Paper 1

- 标题：[Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation](https://aclanthology.org/2025.acl-long.1125/)
- 年份与 venue：2025, ACL Long Paper
- 任务：LLM knowledge distillation
- 核心方法：同时匹配层级 feature trajectory 与 adjacent-layer first-order delta。
- 与本 idea 的重叠：核心公式、ODE/深度轨迹动机、层间一阶差分训练目标直接重叠。
- 关键区别：单模态师生蒸馏，而非 EEG–audio；未主张脑/声学层级，也未使用本文的单调软映射。
- novelty 风险：**critical**。
- 论文中是否必须主动讨论：是；必须将其列为最接近方法并避免“首次提出 depth derivative alignment”。

## Paper 2

- 标题：[Shared functional specialization in transformer-based language models and the human brain](https://www.nature.com/articles/s41467-024-49173-5)
- 年份与 venue：2024, Nature Communications
- 任务：Transformer computations 与自然语言 fMRI 对齐
- 核心方法：分解并分析 layer/head transformations，而不只分析累计 embedding。
- 与本 idea 的重叠：支持“对齐变化/计算而非只对齐状态”的神经科学动机。
- 关键区别：分析性 encoding study，不训练 EEG–audio encoder，也不匹配 adjacent-layer delta。
- novelty 风险：高（动机与机制表述重叠）。
- 论文中是否必须主动讨论：是。

## Paper 3

- 标题：[Dissecting neural computations in the human auditory pathway using deep neural networks for speech](https://www.nature.com/articles/s41593-023-01468-4)
- 年份与 venue：2023, Nature Neuroscience
- 任务：HuBERT/wav2vec 层与听觉通路神经响应对齐
- 核心方法：逐层 neural encoding；浅层更拟合 AN/IC，深层更拟合 STG。
- 与本 idea 的重叠：为 audio teacher 层级与脑层级关系提供最直接支持。
- 关键区别：没有 EEG encoder 训练和 derivative matching；HG 结果与多层持久表征反驳严格串行映射。
- novelty 风险：高（动机已存在）。
- 论文中是否必须主动讨论：是。

## Paper 4

- 标题：[Toward a realistic model of speech processing in the brain with self-supervised learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d81ecfc8fb18e833a3fa0a35d92532b8-Abstract-Conference.html)
- 年份与 venue：2022, NeurIPS
- 任务：wav2vec 2.0 与跨语言 fMRI 对齐
- 核心方法：比较模型层级与皮层层级。
- 与本 idea 的重叠：将 speech model depth 解释为 coarse functional hierarchy。
- 关键区别：状态 encoding，而非残差训练。
- novelty 风险：中高。
- 论文中是否必须主动讨论：是。

## Paper 5

- 标题：[Decoding speech perception from non-invasive brain recordings](https://www.nature.com/articles/s42256-023-00714-5)
- 年份与 venue：2023, Nature Machine Intelligence
- 任务：从 EEG/MEG 检索/识别语音感知内容
- 核心方法：用冻结 speech representation 监督非侵入脑信号 encoder。
- 与本 idea 的重叠：EEG/MEG–audio foundation representation alignment 与 retrieval 评测。
- 关键区别：主要对齐选定状态/目标，不学习 adjacent-layer derivative mapping。
- novelty 风险：高（应用框架相近）。
- 论文中是否必须主动讨论：是。

## Paper 6

- 标题：[Contrastive representation learning with transformers for robust auditory EEG decoding](https://www.nature.com/articles/s41598-025-13646-4)
- 年份与 venue：2025, Scientific Reports
- 任务：ICASSP auditory EEG match–mismatch 与 envelope regression
- 核心方法：CLIP-style EEG–speech contrastive pretraining，speech target 使用 wav2vec 层。
- 与本 idea 的重叠：同一模态、同一典型数据与下游任务、冻结 speech representations。
- 关键区别：单层/累计表示，不对齐层间变化。
- novelty 风险：高；是最重要的真实任务 baseline。
- 论文中是否必须主动讨论：是。

## Paper 7

- 标题：[Relate Auditory Speech to EEG by Shallow-Deep Attention-Based Network](https://arxiv.org/abs/2303.10897)
- 年份与 venue：2023, ICASSP
- 任务：Auditory EEG match–mismatch
- 核心方法：浅层与深层 EEG/speech features 联合相似度。
- 与本 idea 的重叠：显式利用不同深度的跨模态信息。
- 关键区别：不是层间差分，也没有 learned monotonic mapping。
- novelty 风险：中高。
- 论文中是否必须主动讨论：是。

## Paper 8

- 标题：[Universal-KD](https://aclanthology.org/2021.emnlp-main.603/)
- 年份与 venue：2021, EMNLP
- 任务：跨架构中间层蒸馏
- 核心方法：可解释的 output-grounded intermediate layer alignment。
- 与本 idea 的重叠：不同深度/架构之间的层映射与 projector。
- 关键区别：不匹配 adjacent-layer derivative；非跨模态。
- novelty 风险：高（mapping 和 projector 退化风险已有先例）。
- 论文中是否必须主动讨论：是。

## Paper 9

- 标题：[Hierarchical Cross-modal Prompt Learning](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Hierarchical_Cross-modal_Prompt_Learning_for_Vision-Language_Models_ICCV_2025_paper.html)
- 年份与 venue：2025, ICCV
- 任务：vision-language adaptation
- 核心方法：层级知识 mapper 与浅/深层双向跨模态信息流。
- 与本 idea 的重叠：learned hierarchical cross-modal depth interactions。
- 关键区别：prompt learning，不是 derivative loss；无单调 EEG–audio 对应。
- novelty 风险：中高。
- 论文中是否必须主动讨论：是。

## Paper 10

- 标题：[Matryoshka Representation Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html)
- 年份与 venue：2022, NeurIPS
- 任务：coarse-to-fine nested representation
- 核心方法：多个嵌套容量共享监督，推理无额外开销。
- 与本 idea 的重叠：层级/渐进表征与极小训练修改的叙事。
- 关键区别：按 embedding dimension 嵌套，不对齐网络深度或跨模态残差。
- novelty 风险：中；必须作为公平 baseline，而非 direct prior。
- 论文中是否必须主动讨论：是。

## Evidence chain for grade C

1. **Objective-level overlap:** FDD already matches adjacent-layer first differences.
2. **Mapping overlap:** intermediate/cross-layer KD already learns or selects heterogeneous teacher–student layer correspondences.
3. **Application overlap:** frozen speech representations have already supervised non-invasive EEG/MEG and auditory EEG contrastive models.
4. **Motivation overlap:** transformation-centric brain analyses already argue that computations can be more informative than accumulated embeddings.
5. **Remaining novelty:** jointly applying delta matching, subject/stimulus-disjoint EEG–audio evaluation, and a soft monotonic mapping with explicit shuffled-order/no-hierarchy falsifiers.
6. **Why not D:** no checked work combines all of those in EEG–audio, so there remains a testable application-specific contribution.
7. **Why not B/A:** the core loss and central “trajectory derivative” interpretation are already explicit in a major prior paper.

## Required claim rewrite

Avoid: “We introduce a new paradigm that aligns representation evolution.”

Defensible: “We test whether feature-dynamics distillation, adapted to paired EEG–audio with a soft monotonic layer map, provides a better inductive bias than matched-capacity state alignment under subject- and stimulus-disjoint controls.”

## Stage status

- Completed: closest-10 audit and grade with evidence chain.
- Failed: A/B-level generic novelty claim.
- Missing: patent search and citation-graph snowballing beyond the targeted corpus.
- Key findings: FDD is direct objective overlap; EEG–audio monotonic mapping remains unverified as prior art.
- Blocking issues: novelty blocks top-venue positioning without unusually strong mechanistic and real-data evidence.
- Decision: **CONDITIONAL CONTINUE** for minimal falsification only; no large-scale expansion yet.
