# COPA EEG H2：观测算子效应的低秩结构

本项目在 H1 已观察到的稳定算子效应基础上，直接检验：

> 同一 EEG 样本施加算子前后的表示差分，是否集中在一个可跨受试者迁移的低秩子空间。

程序不会预设 H2 成立。`report.md` 根据本次运行的低秩集中度、对照差异、LOSO
迁移和 operator-specificity 自动给出“支持 / 仅样本内集中但不可迁移 / 不支持”的判断。

## 与 H1 的衔接

H1 的 9-subject BNCI2014_001 结果显示：观测算子跨受试者可识别，最佳 balanced
accuracy 为 0.7952；跨算子任务泛化相对对角线平均下降 0.0724。H2 复用相同
CAR、Cz、gain、dropout、band-pass 和 down/up-sampling 定义，但研究对象改为
`ΔE_o(X)=E(T_o(X))-E(X)` 的奇异谱及跨受试者右奇异子空间。

## 数据模式

正式模式默认使用 BNCI2014_001 前 5 名受试者、left/right motor imagery。
加载顺序为：

1. MOABB；
2. H1 已下载的官方 BNCI MAT 文件（仍是真实 BNCI2014_001，不是 synthetic）；
3. 只有配置显式设置 `fallback_to_synthetic: true` 才允许退回 synthetic。

synthetic 仅用于 smoke test，报告会明确标记其不能作为 H2 科学证据。

## 算子与表征

9 个算子逐一分析，不混成总体类别：

- CAR reference、Cz reference；
- gain 0.5、gain 2.0；
- random channel dropout 25%、50%；
- 4–30 Hz、8–30 Hz band-pass；
- 250→125→250 Hz downsample-and-upsample。

4 种表征都以 temporal patch 为 token，并输出 `[epoch, token, embedding_dim]`：

- `raw_patch`：`C × patch_samples` 直接 flatten；
- `pca`：每个 LOSO 折只在训练受试者 identity patches 上拟合；
- `random_projection`：固定种子的线性投影；
- `frozen_patch`：固定一层 patch linear + tanh（无训练、无 PyTorch 依赖）。

逐 token metadata 保存到 `representation_metadata.csv.gz`，包含 operator、subject、
sample ID、token ID、embedding type、维度和 PCA fit subjects。

## 匹配对照

每个 operator/embedding 同时计算：

1. paired operator difference；
2. unpaired difference（epoch derangement，保证 `i != j`）；
3. same-sample Gaussian-noise difference；
4. random orthogonal transform difference；
5. label-preserving epoch permutation（类内 derangement）。

所有对照拥有相同 epoch 数、token 数与 embedding dimension。

## 主要指标

- `ρ(r)`，`r ∈ {1,2,4,8,16,32}`；
- 达到 50/70/80/90% 能量的最小 rank；
- entropy effective rank 与 stable rank；
- paired 相对各 control 的 subject-bootstrap 95% CI；
- subject-level Wilcoxon、rank-biserial effect size、Holm correction；
- LOSO train/random/other-operator/oracle 子空间解释率与 residual；
- operator-specificity matrix、principal angles、subspace similarity；
- 按 label、训练 subject、信号功率四分位的局部子空间角度与迁移增益。

统计单位始终是 subject，不把 epoch 或 token 当独立样本。

## 安装与运行

基础环境：

```bash
python -m pip install -r requirements.txt
```

synthetic 流程检查：

```bash
python scripts/run_smoke_test.py --output-dir outputs/smoke
```

默认真实实验（前 5 名受试者）：

```bash
python scripts/run_full_experiment.py \
  --mode real \
  --max-subjects 5 \
  --max-epochs-per-subject 40 \
  --output-dir outputs/real_bnci2014_001
```

9 名受试者、全部 left/right epoch 的完整验证：

```bash
python scripts/run_full_experiment.py --config configs/full9.yaml
```

`full9.yaml` 不设置每受试者 epoch 上限。已核验运行实际加载 2,592 个 epoch
（9×288，左右手各 1,296），生成到 `outputs/real_bnci2014_001_full9/`。
本机该次 CPU-only 运行约 40.3 分钟，峰值 RSS 约 3.20 GiB；算子与表征缓存合计
约 2.7 GiB，可供相同配置复跑使用。

内存紧张时可先用每人 8–16 个 epoch；这不会改变 subject-disjoint 设计，但会降低
谱和统计估计精度。项目为 CPU-only，并把算子 view 与 paired representation
difference 以压缩 NPZ 缓存；全矩阵 SVD 通过较小 Gram 矩阵计算，迁移只估计所需
的 top-r 子空间。

## 测试

```bash
pytest
```

覆盖：

- 数据 shape、确定性与 subject grouping；
- 9 个算子 shape/有限值/随机 dropout 可复现；
- 4 种表示 shape；
- PCA 测试 sample 泄漏检测；
- paired relation 与控制样本匹配；
- 已知低秩矩阵和零矩阵的 SVD 数值稳定性；
- subject-level bootstrap、Wilcoxon 小样本保护与 Holm correction。

## 输出

```text
outputs/<run>/
├── explained_variance.csv
├── rank_thresholds.csv
├── effective_rank.csv
├── control_comparisons.csv
├── cross_subject_transfer.csv
├── operator_specificity.csv
├── principal_angles.csv
├── locality_analysis.csv
├── representation_metadata.csv.gz
├── operator_metadata.jsonl
├── dataset_summary.json
├── runtime.json
├── manifest.json
├── report.md
└── figures/
    ├── cumulative_spectrum_*.png
    ├── transfer_heatmap.png
    ├── operator_specificity_heatmap.png
    ├── principal_angle_heatmap.png
    └── subspace_similarity_heatmap.png
```

`report.md` 明确回答：哪些算子最接近低秩、rank 2/4/8 的解释率、是否跨受试者
迁移、是否 operator-specific、固定 operator subspace 是否成立，以及哪些证据
支持或反驳 H2。
