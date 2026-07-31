# H5——分离“等价算子”与“有损算子”是否优于统一强一致性

你是一名负责自监督 EEG 表征学习、小模型训练和严谨消融实验的研究工程师。请创建一个 CPU-only 的 Tiny COPA-EEG 验证项目，用于检验：

H5：将 EEG 观测算子区分为 equivalence-preserving operators 和 information-degrading operators，并采用不同训练目标，是否优于对所有 operator views 使用统一的强表示一致性。

该项目只做前期机制验证，不追求训练大型 Foundation Model。不要只给伪代码，直接实现模型、数据、训练、评估、消融、测试和 README。

# 一、核心比较

必须比较以下方法：

A. No pretraining：
- 随机初始化 encoder，直接 linear probe 或轻量监督训练。

B. Identity reconstruction：
- 仅对 identity view 做 masked reconstruction 或 autoencoding。

C. All-view hard consistency：
- 对所有 operator view 使用：
  ||z(X) - z(T_o(X))||^2

D. All-view contrastive：
- 将所有同源 operator views 当作正样本，不区分信息损失。

E. COPA split objective：
- 对 equivalence operators 使用强一致性；
- 对 lossy operators 使用可见信息预测或 confidence-weighted latent prediction；
- 不要求完整表示相等。

F. COPA split objective + lightweight operator leakage regularization，可选。

# 二、数据

首选 BNCI2014_001：
- left hand vs right hand；
- 默认 3 名受试者 smoke test；
- 正式实验配置支持全部受试者；
- subject-disjoint；
- 每个原始 epoch 构造 operator-paired views。

# 三、算子分组

默认定义：

Equivalence-preserving：
1. identity
2. CAR vs 合法 reference 变换
3. channel permutation，同时同步更新 channel metadata
4. unit/global gain change，可配置是否归入 equivalence

Lossy：
1. random channel dropout
2. regional channel dropout
3. 4–30 Hz filter
4. 8–30 Hz filter
5. downsample-and-upsample
6. additive device-like noise，可选

注意：
- gain 是否属于严格等价依赖 normalization；必须将它作为可配置消融；
- reference 是否真正等价依赖输入是否保留参考电极信息；需要在 README 说明；
- 不允许在代码中把这些分类写成不可修改常量。

# 四、Tiny EEG Transformer

模型必须足够小，可以 CPU 训练：

默认：
- input channels：由数据决定
- temporal patch length：可配置
- d_model = 64
- depth = 2
- num_heads = 4
- mlp_dim = 128
- projection rank r ∈ {2, 4, 8}
- dropout = 0.1
- pooled representation = mean pooling 或 CLS

提供：
1. vanilla self-attention；
2. 可选 Operator-Partial Attention；
3. operator metadata encoder；
4. 不增加第二条 EEG encoder 分支。

teacher-student 可采用：
- EMA teacher；
- 或 stop-gradient target encoder；
- 为节省 CPU，允许共享主干并仅用 EMA 参数副本。

# 五、预训练目标

## 1. Hard consistency baseline

L_hard = ||z_identity - z_operator||^2

对所有 operator 相同处理。

## 2. COPA equivalence objective

若 o ∈ O_eq：

L_eq = ||z_identity - z_operator||^2

可加入 token-level consistency。

## 3. COPA lossy objective

若 o ∈ O_lossy，不要求完整 latent 相同。

至少实现一种主方法和一种备选方法：

主方法：confidence-weighted latent prediction

L_lossy = w(X,o) * ||g_o(z_lossy) - stopgrad(z_identity)||^2

w 可由以下因素组合：
- retained channel ratio
- retained bandwidth ratio
- teacher predictive confidence
- teacher-student agreement
- 可配置 clipping

备选方法：
- 只预测 teacher latent 的 operator-visible mask；
- 或预测低频/可见通道对应的 token subset。

如果实现 visible-mask 方案困难，先实现 confidence-weighted latent prediction，但必须清楚记录其局限。

## 4. 防止坍塌

至少加入：
- variance loss
- covariance loss 或 feature decorrelation
- stop-gradient / EMA teacher

## 5. Operator leakage regularization

可选实现：
- gradient reversal operator classifier
- 或 HSIC penalty

默认权重很小，并单独消融。

# 六、训练资源限制

- CPU-only；
- 默认单次 smoke test 10–20 分钟以内；
- epoch 数量、样本数、模型尺寸可配置；
- 支持 mixed precision 的 CPU 不作为必要条件；
- 默认 batch size 小；
- 使用 early stopping；
- 数据和 operator views 缓存；
- 保存最佳 checkpoint；
- 日志包含每个 loss 分量。

# 七、下游评估

冻结 encoder，分别训练：

1. task linear probe；
2. operator linear probe；
3. subject linear probe；
4. 可选 full fine-tuning 的 tiny classifier。

评估：
- balanced accuracy
- macro F1
- subject-level bootstrap CI
- cross-operator matrix
- in-operator vs unseen-operator gap
- representation variance
- paired-view distance
- calibration，可选

# 八、关键负对照

必须构造故意违反标签保持性的强 lossy operator，例如：

1. motor imagery 中删除 central channels；
2. 只保留与任务无关或信息不足的频段；
3. 极高 channel dropout；
4. 删除关键时间窗，可选。

目的：
- 检查 hard consistency 是否迫使完整视图向残缺视图退化；
- 检查 COPA 的 confidence weight 是否下降；
- 检查 COPA 是否比 hard consistency 更能保留任务信息。

# 九、消融

至少运行：

1. equivalence only
2. lossy only
3. all hard consistency
4. split objective
5. split objective without confidence weight
6. split objective with fixed weight
7. gain as equivalence vs gain as lossy
8. without collapse prevention
9. without operator leakage regularization
10. vanilla attention vs Operator-Partial Attention，可选

# 十、结果判据

程序必须计算：

Cross-operator gap：
Delta_op = Acc_in_operator - Acc_cross_operator

Information retention：
R_task = Acc_task_method / Acc_task_identity_baseline

Operator leakage：
L_op = Acc_operator_probe - chance

Collapse indicators：
- per-dimension std
- effective rank
- covariance spectrum

需要重点比较：

COPA split objective
vs
All-view hard consistency

理想支持 H5 的结果：
- 在 equivalence operators 上，两者都能获得稳定性；
- 在强 lossy operators 上，COPA 的 task accuracy 更高；
- COPA 的 cross-operator gap 更小；
- COPA 不出现表示坍塌；
- confidence weight 随信息损失程度合理下降。

如果没有出现这些现象，报告必须明确 H5 未被支持。

# 十一、统计分析

- 多随机种子，默认 3 个，正式实验建议 5 个；
- subject-level bootstrap；
- paired comparison；
- Wilcoxon 或 permutation test；
- Holm correction；
- 报告 effect size；
- smoke test 不做过度显著性解释。

# 十二、项目结构

copa_eeg_h5/
├── configs/
│   ├── smoke.yaml
│   └── full.yaml
├── src/
│   ├── data.py
│   ├── operators.py
│   ├── model.py
│   ├── opa.py
│   ├── losses.py
│   ├── trainer.py
│   ├── probes.py
│   ├── evaluation.py
│   └── plotting.py
├── scripts/
│   ├── run_pretrain.py
│   ├── run_probes.py
│   ├── run_ablation.py
│   └── run_smoke_test.py
├── tests/
├── outputs/
├── requirements.txt
├── pyproject.toml
└── README.md

# 十三、输出

至少生成：

outputs/
├── checkpoints/
├── training_curves.csv
├── downstream_results.csv
├── operator_probe_results.csv
├── subject_probe_results.csv
├── cross_operator_matrix.csv
├── ablation_results.csv
├── statistical_tests.csv
├── figures/
│   ├── task_vs_operator_probe.png
│   ├── cross_operator_heatmap.png
│   ├── confidence_vs_information_loss.png
│   ├── representation_variance.png
│   └── ablation_summary.png
└── report.md

`report.md` 必须回答：
1. 统一强一致性是否在有损算子下伤害任务信息；
2. 分算子目标是否改善；
3. confidence weighting 是否必要；
4. 哪些算子应归入 equivalence，哪些不应；
5. operator leakage regularization 是否有收益；
6. 是否支持 H5；
7. 下一步是否值得扩展到完整 COPA-EEG。

# 十四、工程验收

完成后必须：
1. 运行单元测试；
2. 运行 synthetic smoke test；
3. 若可访问真实数据，运行最小真实数据实验；
4. 检查无 CUDA 依赖；
5. 检查无 train/test subject leakage；
6. 输出所有配置、随机种子和版本；
7. 列出新增文件；
8. 不得在实验未完成时编造结果。

该项目的目标是判断分算子训练目标是否成立，而不是强行得到正结果。
