# COPA EEG H1

本项目验证观测算子（reference、channel subset、filter、resampling、gain）是否产生稳定可识别的 EEG 特征分布变化，以及是否降低跨算子任务泛化。项目为 CPU-only，所有评估以 subject ID 分组，绝不先随机打散 epoch/window 再切分。

项目不会在 README 中预设 H1 成立。每次运行的结论由该次输出目录中的 `report.md` 和 JSON 数值生成。

## 两种数据模式

### Real-data mode（正式实验）

正式模式使用 MOABB `BNCI2014_001` 的 left hand vs right hand 二分类。默认受试者为 1、2、3，可在 YAML 或 CLI 中扩大。数据下载到项目内 `cache/data/`。

安装：

```bash
python -m pip install -r requirements-real.txt
```

下载：

```bash
python scripts/download_data.py --max-subjects 3
```

正式运行示例：

```bash
python scripts/run_full_experiment.py \
  --mode real \
  --max-subjects 9 \
  --max-epochs-per-subject 200 \
  --output-dir outputs/real_bnci2014_001
```

配置默认不静默回退到 synthetic；MOABB 下载或依赖失败会明确报错。若确需允许回退，显式把 `data.fallback_to_synthetic` 改为 `true`，并检查输出的 `mode` 字段。

### Synthetic smoke-test mode（仅流程检查）

synthetic 数据只用于检查算子、特征、分组评估、缓存、绘图、统计和报告路径。它不是正式 EEG 数据，其结果不能作为支持或反驳 H1 的科学证据。

5–10 分钟内的默认 smoke test：

```bash
python scripts/run_smoke_test.py --output-dir outputs/smoke
```

资源更紧张时：

```bash
python scripts/run_smoke_test.py \
  --max-subjects 3 \
  --max-epochs-per-subject 8 \
  --output-dir outputs/smoke_tiny
```

## 实验与防泄漏设计

- Operator separability：LogisticRegression、LinearSVC、RandomForestClassifier；GroupKFold 的 group 为 subject ID。
- Cross-operator task generalization：每个 LeaveOneGroupOut 折内，仅从某一 train operator 的训练受试者拟合任务模型，再在所有 test operator 的留出受试者上测试。
- scaler 封装在 sklearn Pipeline 中，只在训练折 `.fit()`。
- bootstrap 以完整 subject 为抽样单位，而不是 epoch。
- 受试者少于 `evaluation.minimum_subjects_for_test`（默认 5）时，不强行报告显著性。
- 多算子 outgoing 比较使用 Holm correction。
- 默认 `n_jobs: 1`，不调用 CUDA。

## 观测算子与 metadata

默认构造 11 个独立 view：identity、CAR、Cz reference、gain 0.5、gain 2.0、random channel dropout 0.25、random channel dropout 0.5、central region dropout、4–30 Hz、8–30 Hz、250→125→250 Hz down/up resampling。

每个算子保持 `[channels, samples]` shape，并记录：

- `channel_mask`
- `sampling_rate` 与中间采样率（适用时）
- `reference_type`
- `filter_range`
- `gain`
- `operator_type`（`identity`、`equivalence` 或 `lossy`）

逐样本记录写入 `operator_metadata.jsonl`。`ComposeOperator` 可顺序组合任意算子并保留 trace。

## 特征

YAML 可分别开关：

- 每通道 delta/theta/alpha/beta band power
- 每通道 mean/std/peak-to-peak/skew/kurtosis
- covariance 上三角
- correlation 上三角
- covariance 前若干 eigenvalues

常量 dropout 通道引起的未定义 skew/kurtosis/correlation 会被显式转换为 0；最终特征矩阵再次检查 NaN/Inf。特征 schema 写入 `feature_schema.json`，中间特征按数据与配置签名缓存到 `cache/features/`。

`features.py` 保留传统特征的稳定默认路径。pyRiemann tangent-space、MDM，以及 CSP+LDA 属于可选扩展；安装 `pyproject.toml` 的 `riemann` extra 后可扩展模型工厂，默认 smoke test 不依赖它们。

## 输出

完整运行会产生：

- `operator_probe_results.json`
- `operator_confusion_matrix.png` 及各模型 confusion matrix
- `cross_operator_balanced_accuracy.csv`
- `cross_operator_balanced_accuracy.md`
- `cross_operator_heatmap.png`
- `cross_operator_heatmap.pdf`
- `cross_operator_results.json`（每个 cell 的 subject-bootstrap 95% CI）
- `statistics.json`
- `feature_schema.json`
- `operator_metadata.jsonl`
- `dataset_summary.json`
- `runtime.json`（时间、峰值内存、软件版本）
- `resolved_config.yaml`
- `report.md`

## 分开运行

```bash
python scripts/run_operator_probe.py --mode synthetic --output-dir outputs/probe
python scripts/run_cross_operator.py --mode synthetic --output-dir outputs/cross
```

两个脚本都支持 `--max-subjects`、`--max-epochs-per-subject` 和 `--seed`。

## 测试

```bash
pytest
```

测试覆盖 synthetic 数据确定性、所有算子的 shape/metadata、随机 dropout 确定性、算子组合、常量通道特征有限性、subject-disjoint split、bootstrap 路径和统计小样本保护。

## 项目结构

```text
copa_eeg_h1/
├── configs/default.yaml
├── src/
│   ├── config.py
│   ├── data.py
│   ├── operators.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   ├── statistics.py
│   ├── plotting.py
│   └── pipeline.py
├── scripts/
├── tests/
├── outputs/
├── requirements.txt
├── requirements-real.txt
├── pyproject.toml
└── README.md
```
