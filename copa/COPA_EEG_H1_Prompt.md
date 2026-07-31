# H1——观测算子是否会系统性改变 EEG 表征与任务泛化

你是一名负责 EEG 表征学习与可复现实验工程的研究代码开发者。请在当前工作目录中创建一个完整、可运行的 CPU-only Python 项目，用于验证以下假设：

H1：reference、channel subset、filter、resampling 和 gain 等 EEG 观测算子，会产生稳定且可识别的分布变化，并导致跨算子任务泛化下降。

不要只给方案或伪代码，直接实现完整项目、运行脚本、配置文件、测试和 README。不要等待我确认；遇到非关键细节时自行采用合理默认值，并将其写入配置文件。

# 一、实验目标

需要完成两组实验：

1. Operator separability：
   给定 EEG epoch 的传统特征，预测该 epoch 使用了哪一种观测算子。
2. Cross-operator task generalization：
   在某一种算子视图上训练任务分类器，在另一种算子视图上测试，生成完整的 train-operator × test-operator 性能矩阵。

# 二、数据集

首选 MOABB 的 BNCI2014_001 数据集，任务先做 left hand vs right hand 二分类。

要求：
- 默认只使用前 3 名受试者做 smoke test；
- 配置中允许调整 subject 数量；
- 所有划分必须 subject-disjoint；
- 禁止随机打散 window 后再切分，否则会发生 subject leakage；
- 数据下载到项目内可配置 cache 目录；
- 如果 MOABB 数据获取失败，提供一个 synthetic EEG fallback，仅用于检查代码流程，不能把 synthetic 结果作为正式实验结果；
- 在 README 中明确区分 real-data mode 与 synthetic smoke-test mode。

# 三、观测算子

对每个原始 epoch 构造以下 operator views：

1. identity
2. common average reference, CAR
3. Cz reference
4. global gain，a ∈ {0.5, 2.0}
5. random channel dropout，drop ratio ∈ {0.25, 0.5}
6. region channel dropout，默认删除 central 区域或配置指定区域
7. band-pass filter，默认 4–30 Hz
8. band-pass filter，默认 8–30 Hz
9. resample down-and-up，例如 250 Hz → 125 Hz → 250 Hz

要求：
- 所有算子必须是独立、可组合、可单元测试的类或函数；
- 每个输出必须保留 shape；
- channel mask、sampling rate、reference type、filter range、gain 等 metadata 必须与样本一起保存；
- 对会改变信息量的算子标记 `operator_type = lossy`；
- 对近似等价算子标记 `operator_type = equivalence`；
- identity 单独标记。

# 四、特征

至少实现：

1. 每通道频带功率：
   delta、theta、alpha、beta，可配置频带；
2. 每通道均值、标准差、峰峰值、偏度、峰度；
3. 通道协方差矩阵上三角展开；
4. 通道相关矩阵上三角展开；
5. 协方差特征的前若干特征值；
6. 可选：pyriemann tangent-space 特征，如果依赖安装成功。

所有特征必须：
- 仅在训练折拟合 scaler；
- 对 NaN/Inf 做显式检查；
- 保存 feature schema；
- 支持通过 YAML 配置启用或禁用。

# 五、模型与评估

Operator classifier：
- LogisticRegression
- LinearSVC 或 SVC(kernel="linear")
- RandomForestClassifier

Task classifier：
- LogisticRegression
- LDA
- 可选 CSP + LDA
- 可选 pyriemann MDM / tangent space + LogisticRegression

评估必须使用 GroupKFold 或 LeaveOneGroupOut，group 为 subject ID。

报告：
- balanced accuracy
- macro F1
- confusion matrix
- 每类 precision / recall / F1
- bootstrap 95% confidence interval，bootstrap 单位必须是 subject，而不是 epoch

# 六、Cross-operator matrix

对每个 operator o_i：
- 只用 o_i 的训练受试者样本训练任务分类器；
- 分别在每个 o_j 的测试受试者样本上评估；
- 输出矩阵 M_ij；
- 同时计算：
  - diagonal mean
  - off-diagonal mean
  - average cross-operator drop
  - 每个 operator 的 outgoing drop 和 incoming drop

输出：
- CSV
- Markdown 表格
- heatmap PNG/PDF
- 带置信区间的结果 JSON

# 七、统计检验

至少实现：
- 对角线与非对角线性能差异的 paired test；
- 默认使用 Wilcoxon signed-rank test；
- 报告 effect size；
- 多重比较时使用 Holm correction；
- 若受试者数量过少，不强行报告显著性，在结果中明确标记 insufficient sample size。

# 八、项目结构

至少包含：

copa_eeg_h1/
├── configs/
│   └── default.yaml
├── src/
│   ├── data.py
│   ├── operators.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   ├── statistics.py
│   └── plotting.py
├── scripts/
│   ├── download_data.py
│   ├── run_operator_probe.py
│   ├── run_cross_operator.py
│   └── run_smoke_test.py
├── tests/
├── outputs/
├── requirements.txt
├── pyproject.toml
└── README.md

# 九、运行与资源约束

- CPU-only；
- 不允许代码默认调用 CUDA；
- 默认 `n_jobs` 要保守，避免占满机器；
- 提供 `--max-subjects`、`--max-epochs-per-subject` 和 `--seed`；
- 对中间特征做磁盘缓存，避免重复计算；
- 记录运行时间、峰值内存和软件版本；
- 提供一条 5–10 分钟内可完成的 smoke-test 命令；
- 提供一条正式实验命令。

# 十、验收标准

代码完成后，必须：
1. 自动运行单元测试；
2. 自动运行 synthetic smoke test；
3. 输出一个 operator confusion matrix；
4. 输出一个 cross-operator task matrix；
5. 输出一个 `report.md`，说明：
   - operator 是否可被跨受试者识别；
   - cross-operator drop 是否存在；
   - 哪类算子影响最大；
   - 结果是否支持 H1；
   - synthetic 结果不得被写成科学结论。
6. 最后列出所有新增文件及其用途。

所有关键结论必须由程序实际生成的结果支持，不允许在 README 中预设实验结论。
