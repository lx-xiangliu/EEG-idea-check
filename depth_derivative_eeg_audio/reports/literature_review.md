# Literature Review

## Scope and audit protocol

- Search date: 2026-07-31 (Asia/Shanghai).
- Coverage target: 2020-present, with two older historical anchors (Patient KD 2019 and Nested Dropout 2014).
- Sources: official proceedings, journal full text, ACL Anthology, OpenReview, arXiv abstracts, and official repositories. Search-result pages and secondary blogs were not used as evidence.
- Reading labels: **full** means the accessible article/PDF method and relevant result sections were inspected; **abstract** means official title/authors/abstract/method summary were checked; **code** means an official repository or code link was also inspected.
- The machine-readable 50-paper audit is in `novelty_matrix.csv`. It contains 10 EEG foundation-model papers, 10 EEG–audio/brain–speech papers, 10 cross-modal alignment papers, 10 distillation/trajectory papers, 5 speech–brain hierarchy papers, and 5 Matryoshka/ordered-representation papers.
- Limitation: this is a reproducible targeted audit, not a claim of an exhaustive systematic review. Papers labeled **abstract** were not represented as fully read.

## 1. EEG foundation models (10)

| ID | Paper; authors | Year / venue | Checked | Relevance |
|---|---|---|---|---|
| E01 | [BENDR](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.653659/full); Demetres Kostas, Stephane Aroca-Ouellette, Frank Rudzicz | 2021, Frontiers in Human Neuroscience | full + code | Contrastive EEG pretraining; no audio-layer alignment. |
| E02 | [MAEEG](https://arxiv.org/abs/2211.02625); Hsiang-Yun Sherry Chien, Hanlin Goh, Christopher M. Sandino, Joseph Y. Cheng | 2022, NeurIPS TS4H | abstract + code | Masked reconstruction baseline. |
| E03 | [BIOT](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract.html); Chaoqi Yang, M. Brandon Westover, Jimeng Sun | 2023, NeurIPS | full + code | Cross-dataset biosignal encoder. |
| E04 | [LaBraM](https://arxiv.org/abs/2405.18765); Wei-Bang Jiang, Li-Ming Zhao, Bao-Liang Lu | 2024, arXiv | abstract + code | Neural tokenizer and masked-code prediction. |
| E05 | [EEG2Rep](https://arxiv.org/abs/2402.17772); Navid Mohammadi Foumani, Geoffrey Mackellar, Soheila Ghane, Saad Irtza, Nam Nguyen, Mahsa Salehi | 2024, arXiv | abstract + code | Latent-space masked prediction. |
| E06 | [EEGPT](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html); Guangyu Wang, Wenchao Liu, Yuhong He, Cong Xu, Lin Ma, Haifeng Li | 2024, NeurIPS | full + code | Uses representation alignment internally, but not adjacent-layer deltas or audio. |
| E07 | [CBraMod](https://openreview.net/forum?id=NPNUHgHF2w); Jiquan Wang, Sha Zhao, Zhiling Luo, Yangxuan Zhou, Haiteng Jiang, Shijian Li, Tao Li, Gang Pan | 2025, ICLR | full + code | Criss-cross spatial/temporal modeling; useful backbone. |
| E08 | [REVE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/20a917f77773ac0fa8bea2bdd6606b66-Abstract-Conference.html); Yassine El Ouahidi, Jonathan Lys, Philipp Thölke, Nicolas Farrugia, Bastien Pasdeloup, Vincent Gripon, Karim Jerbi, Giulia Lioi | 2025, NeurIPS | full + code | Strong heterogeneous-pretraining comparator. |
| E09 | [BrainOmni](https://papers.neurips.cc/paper_files/paper/2025/file/3aef4a18c2646b9d11d9ab4d9bec72c8-Paper-Conference.pdf); Qinfan Xiao, Ziyun Cui, Chi Zhang, Siqi Chen, Wen Wu, Andrew Thwaites, Alexandra Woolgar, Bowen Zhou, Chao Zhang | 2025, NeurIPS | full + code | Cross-device EEG/MEG tokenizer; no EEG–audio derivative objective. |
| E10 | [CSBrain](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7e199ad8ae40eb19b2980f61f659cb07-Abstract-Conference.html); Yuchen Zhou et al. | 2025, NeurIPS | abstract | Cross-scale hierarchy is adjacent motivation but not layer-transition alignment. |

## 2. EEG–audio, brain–speech, and auditory decoding (10)

| ID | Paper; authors | Year / venue | Checked | Relevance |
|---|---|---|---|---|
| A01 | [Auditory stimulus-response modeling with a match-mismatch task](https://orbit.dtu.dk/en/publications/auditory-stimulus-response-modeling-with-a-match-mismatch-task); Alain de Cheveigné, Malcolm Slaney, Søren A. Fuglsang, Jens Hjortkjær | 2021, JNE | abstract | Defines a clean retrieval-like target task. |
| A02 | [VLAAI](https://www.nature.com/articles/s41598-022-27332-2); Bernd Accou, Jonas Vanthornhout, Hugo Van hamme, Tom Francart et al. | 2023, Scientific Reports | full + code | Strong subject-independent envelope decoder. |
| A03 | [Eeg2vec](https://arxiv.org/abs/2305.13957); Qiushi Zhu, Xiaoying Zhao, Jie Zhang, Yu Gu, Chao Weng, Yuchen Hu | 2023, arXiv | abstract | Auditory EEG SSL for match–mismatch and regression. |
| A04 | [SDANet](https://arxiv.org/abs/2303.10897); Fan Cui, Liyong Guo, Lang He, Jiyao Liu, Ercheng Pei, Yujun Wang, Dongmei Jiang | 2023, ICASSP | abstract | Shallow/deep features overlap with hierarchical supervision, not residual deltas. |
| A05 | [Word-boundary match–mismatch](https://arxiv.org/abs/2307.00366); Akshara Soman, Vidhi Sinha, Sriram Ganapathy | 2023, ICASSP | abstract | Shows temporal segmentation can masquerade as representation improvement. |
| A06 | [Decoding speech perception from non-invasive brain recordings](https://www.nature.com/articles/s42256-023-00714-5); Alexandre Défossez, Charlotte Caucheteux, Jérémy Rapin, Ori Kabeli, Jean-Rémi King | 2023, Nature Machine Intelligence | full + code | Frozen speech-model targets aligned to EEG/MEG; close application-level prior. |
| A07 | [ICASSP 2023 Auditory EEG Decoding Challenge](https://signalprocessingsociety.org/publications-resources/data-challenges/auditory-eeg-decoding-challenge-icassp-2023); challenge organizers at KU Leuven/IEEE SPS | 2023, IEEE SPS | full + data docs | Official held-out-story and held-out-subject benchmark. |
| A08 | [Contrastive representation learning with transformers for robust auditory EEG decoding](https://www.nature.com/articles/s41598-025-13646-4); Lies Bollens, Bernd Accou, Hugo Van hamme, Tom Francart et al. | 2025, Scientific Reports | full + code | Strong CLIP-style EEG–speech latent alignment; uses one wav2vec layer. |
| A09 | [BrainECHO](https://openreview.net/pdf/be13e0fd8110cf10c1c61b143358b163e55d8ae1.pdf); authors listed in official manuscript | 2025, OpenReview manuscript | full | Brain–audio latent reconstruction; publication status remains uncertain. |
| A10 | [Joint Text-Audio Alignment for EEG-to-Text Decoding](https://arxiv.org/abs/2607.25626); authors listed on arXiv | 2026, arXiv | abstract | Very recent EEG–audio alignment; no adjacent-layer derivative reported in abstract. |

## 3. Cross-modal and hierarchical alignment (10)

| ID | Paper; authors | Year / venue | Checked | Relevance |
|---|---|---|---|---|
| C01 | [CLIP](https://proceedings.mlr.press/v139/radford21a.html); Alec Radford et al. | 2021, ICML | full + code | Canonical final-state contrastive baseline. |
| C02 | [ALIGN](https://proceedings.mlr.press/v139/jia21b.html); Chao Jia et al. | 2021, ICML | abstract | Scaling reference for final-state alignment. |
| C03 | [VATT](https://proceedings.neurips.cc/paper/2021/hash/cb3213ada48302953cb0f166464ab356-Abstract.html); Hassan Akbari et al. | 2021, NeurIPS | full | Audio–video–text multi-branch pretraining. |
| C04 | [Geometric Multimodal Contrastive Learning](https://proceedings.mlr.press/v162/poklukar22a.html); Petra Poklukar, Miguel Vasco, Hang Yin, Francisco S. Melo, Ana Paiva, Danica Kragic | 2022, ICML | full | Supports geometry/subspace controls instead of basis-dependent MSE. |
| C05 | [Representation Codebook Alignment](https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Multi-Modal_Alignment_Using_Representation_Codebook_CVPR_2022_paper.html); Jiali Duan et al. | 2022, CVPR | full | Aligns cluster geometry when raw coordinates are unstable. |
| C06 | [Speech–text multi-granularity alignment](https://aclanthology.org/2023.ccl-1.7/); Ling Zhou, Guojiang Dong, Zhengtao Yu, Shengxiang Gao, Wenjun Wang, Houli Ma | 2023, CCL | abstract | Multi-level teacher supervision control. |
| C07 | [Multi-Level Cross-Modal Alignment for Speech Relation Extraction](https://aclanthology.org/2024.emnlp-main.668/); Liang Zhang et al. | 2024, EMNLP | full | Direct multi-level speech–text alignment prior. |
| C08 | [HiCroPL](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Hierarchical_Cross-modal_Prompt_Learning_for_Vision-Language_Models_ICCV_2025_paper.html); Hao Zheng, Shunzhi Yang, Zhuoxin He, Jinfeng Yang, Zhenhua Huang | 2025, ICCV | full + code | Learns hierarchical cross-modal knowledge flow, but not deltas. |
| C09 | [M3-JEPA](https://proceedings.mlr.press/v267/lei25b.html); Hongyang Lei et al. | 2025, ICML | full | Predictive latent geometry and modality-collapse alternative. |
| C10 | [Understanding the Modality Gap](https://aclanthology.org/2025.emnlp-main.262/); Xiang et al. | 2025, EMNLP | full | Shows layerwise direction and magnitude behave differently. |

## 4. Residual, trajectory, and intermediate-layer distillation (10)

| ID | Paper; authors | Year / venue | Checked | Relevance |
|---|---|---|---|---|
| R01 | [TinyBERT](https://aclanthology.org/2020.findings-emnlp.372/); Xiaoqi Jiao et al. | 2020, Findings EMNLP | full | Strong all-layer hidden/attention distillation baseline. |
| R02 | [MiniLMv2](https://aclanthology.org/2021.findings-acl.188/); Wenhui Wang et al. | 2021, Findings ACL | full | Relation geometry instead of raw coordinate matching. |
| R03 | [Universal-KD](https://aclanthology.org/2021.emnlp-main.603/); Li et al. | 2021, EMNLP | full | Cross-architecture intermediate-layer mapping. |
| R04 | [Marginal Utility Diminishes](https://aclanthology.org/2021.acl-long.228/); Yuanxin Liu, Fandong Meng, Zheng Lin, Weiping Wang, Jie Zhou | 2021, ACL | full | Warns that benefit may come merely from extra hidden-state supervision. |
| R05 | [Patient Knowledge Distillation](https://aclanthology.org/D19-1441/); Siqi Sun, Yu Cheng, Zhe Gan, Jingjing Liu | 2019, EMNLP-IJCNLP | full | Historical multi-layer mapping anchor. |
| R06 | [Intermediate Layer Distillation with the Reused Teacher Classifier](https://aclanthology.org/2024.findings-emnlp.422/); authors on ACL record | 2024, Findings EMNLP | full | Normalized hidden matching and deterministic layer selection. |
| R07 | [What Mechanisms Does Knowledge Distillation Distill?](https://proceedings.mlr.press/v243/wu24a.html); Cindy Wu, Ekdeep Singh Lubana, Bruno Kacper Mlodozeniec, Robert Kirk, David Krueger | 2024, UniReps | full | Jacobian and contrastive geometry are mechanistic alternatives. |
| R08 | [Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation](https://aclanthology.org/2025.acl-long.1125/); Guoqiang Gong, Jiaxing Wang, Jin Xu, Deping Xiang, Zicheng Zhang, Leqi Shen, Yifeng Zhang, Junhua Shu, Zhaolong Xing, Zhen Chen, Pengzhang Liu, Ke Zhang | 2025, ACL | full | **Directly matches adjacent-layer first differences and feature trajectories.** |
| R09 | [C2KD](https://aclanthology.org/2025.findings-acl.917/); authors on ACL record | 2025, Findings ACL | full | Learned cross-layer/cross-head mapping. |
| R10 | [TALAS](https://aclanthology.org/2026.acl-long.1509/); authors on ACL record | 2026, ACL | full | Adaptive teacher-anchored layer alignment; mapping control. |

## 5. Speech-model layers and brain hierarchy (5)

| ID | Paper; authors | Year / venue | Checked | Relevance |
|---|---|---|---|---|
| B01 | [Toward a realistic model of speech processing in the brain](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d81ecfc8fb18e833a3fa0a35d92532b8-Abstract-Conference.html); Juliette Millet, Charlotte Caucheteux, Pierre Orhan, Yves Boubenec, Alexandre Gramfort, Ewan Dunbar, Christophe Pallier, Jean-Rémi King | 2022, NeurIPS | full + code | wav2vec depth correlates with cortical hierarchy. |
| B02 | [Dissecting neural computations in the human auditory pathway](https://www.nature.com/articles/s41593-023-01468-4); Yuanning Li, Gopala K. Anumanchipalli, Abdelrahman Mohamed, Peili Chen, Laurel H. Carney, Junfeng Lu, Jinsong Wu, Edward F. Chang et al. | 2023, Nature Neuroscience | full | Early HuBERT layers fit AN/IC; layer 10 fits STG, but HG is an exception and salient signals persist across layers. |
| B03 | [Predictive coding hierarchy](https://www.nature.com/articles/s41562-022-01516-2); Charlotte Caucheteux, Alexandre Gramfort, Jean-Rémi King | 2023, Nature Human Behaviour | full + code | Supports multi-timescale brain hierarchy, not a strict feed-forward one-to-one depth map. |
| B04 | [Shared functional specialization](https://www.nature.com/articles/s41467-024-49173-5); Sreejan Kumar, Theodore R. Sumers, Takateru Yamakoshi, Ariel Goldstein, Uri Hasson, Kenneth A. Norman, Thomas L. Griffiths, Robert D. Hawkins, Samuel A. Nastase et al. | 2024, Nature Communications | full | Directly studies layer transformations; major mechanistic prior. |
| B05 | [Temporal structure corresponds to layered hierarchy](https://www.nature.com/articles/s41467-025-65518-0); Ariel Goldstein, Eric Ham, Mariano Schain, Samuel A. Nastase et al. | 2025, Nature Communications | full | Deeper LLM layers peak later in ECoG, but depth is a coarse correlate rather than an identified causal hierarchy. |

## 6. Matryoshka and ordered representations (5)

| ID | Paper; authors | Year / venue | Checked | Relevance |
|---|---|---|---|---|
| M01 | [Matryoshka Representation Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html); Aditya Kusupati et al. | 2022, NeurIPS | full + code | Required matched-capacity coarse-to-fine baseline. |
| M02 | [Nested Dropout](https://proceedings.mlr.press/v32/rippel14.html); Oren Rippel, Michael Gelbart, Ryan Adams | 2014, ICML | full | Historical ordered-representation anchor. |
| M03 | [Matryoshka Multimodal Models](https://openreview.net/forum?id=ii17KMFtmI); Mu Cai, Jianwei Yang, Jianfeng Gao, Yong Jae Lee | 2024, NeurIPS VLM Workshop | abstract | Nested visual-token granularity. |
| M04 | [Matryoshka Query Transformer](https://openreview.net/forum?id=B1vGiSgELw); Wenbo Hu, Zi-Yi Dou, Liunian Harold Li, Amita Kamath, Nanyun Peng, Kai-Wei Chang | 2024, NeurIPS | full | Flexible nested visual token count. |
| M05 | [fMRLRec](https://aclanthology.org/2024.findings-emnlp.786/); Yueqi Wang, Zhenrui Yue, Huimin Zeng, Dong Wang, Julian McAuley | 2024, Findings EMNLP | full | Multimodal Matryoshka control. |

## Literature-level conclusions

1. No checked work applies adjacent-layer first-difference alignment with a learned monotonic depth map specifically to EEG–audio.
2. The generic training objective is not new: ACL 2025 FDD explicitly aligns a feature trajectory and its first-order finite difference across adjacent layers.
3. Cross-modal multi-level alignment, adaptive layer mapping, and transformation-based brain analyses already exist separately. The proposal is therefore a **combination/application contribution**, not a new general representation-learning paradigm.
4. Brain/speech evidence supports a loose depth–hierarchy correlation, but also documents exceptions, parallel persistence, and region-specific optima; strict monotonicity is a falsifiable modeling choice, not a neuroscientific fact.
5. The decisive empirical question is whether first differences outperform matched-capacity multi-layer hidden/geometry supervision under shuffled-order, random-teacher, and no-hierarchy controls.

## Stage status

- Completed: targeted 50-paper metadata and overlap audit; official links; closest-prior identification.
- Failed: claim that adjacent-layer derivative matching is a wholly new objective.
- Missing: exhaustive citation-chaining and independent second-reviewer screening.
- Key findings: FDD 2025 is direct objective-level prior; no verified EEG–audio monotonic derivative alignment was found.
- Blocking issues: none for minimal synthetic validation; novelty blocks a “new paradigm” claim.
- Decision: continue only as a minimal, application-specific falsification study.
