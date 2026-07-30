# COPA EEG H3：Transformer 内部的 operator 信息路径

本项目承接 H1（operator 可跨受试者识别）与 H2（operator 差分的低秩/迁移结构），
定位 operator 信息究竟进入 Q、K、V、attention、residual、MLP 还是最终表征。
程序不会预设 reference 一定污染 Q/K，也不会为了支持 H3 删除 V 或 residual。

## 模型身份

默认 `model.type: auto`，只检查本地 checkpoint，不自动下载大型权重：

1. 若本地存在并安装了相应官方 CBraMod 架构，可据明确适配映射接入；
2. 可用 `--checkpoint /absolute/path/to/checkpoint.pt` 加载与配置架构严格匹配的本地权重；
3. 无权重时运行显式 Q/K/V 的 Tiny EEG Transformer。

Tiny fallback 会在 `model_summary.json` 和 `report.md` 明确标为
`architecture smoke test`，不能冒充 pretrained foundation model 结论。
当前实现不会把不兼容的 CBraMod state dict 静默塞进 Tiny 模型。

## 数据与算子

真实命令读取 H1 已下载的官方 BNCI2014_001 MAT，使用 left/right hand，默认前 5 名
受试者、每人平衡抽样 24 个 epoch。每个 source epoch 构造 identity、CAR、Cz、
gain×2、channel dropout×2、band-pass×2、250→125→250 Hz resampling，共 10 views。
逐 view 元数据保留 source epoch、subject、task、operator、family、dataset ID 和参数。

## 提取与内存控制

自定义 attention 是非 fused 路径。`forward_with_internals` 返回 input embedding、逐层
Q/K/V、logits/probabilities、attention output、attention 前后 residual、MLP output、
final tokens 与 pooled representation。单元测试验证正常 forward 与捕获 forward
误差，并逐值验证 Q/K/V 就是本次前向实际投影。

全尺寸内部张量只存在于默认 batch size 4 的当前 CPU batch。落盘 `compressed_summaries/`
仅包含 NPZ 压缩摘要：

- token mean/std/max/first；
- covariance diagonal、off-diagonal mean/std；
- attention entropy、diagonal/off-diagonal mass、top-k concentration；
- 可通过配置限制 layer、head、subject 和 epoch 数。

## Probe 与负对照

主 operator/task probe 使用 leave-one-subject-out；每折 scaler 只拟合训练受试者。
模型为 LogisticRegression 与 LinearSVC，输出 balanced accuracy、macro F1 以及
subject-fold bootstrap CI。family 使用 one-vs-rest；attention 另输出 layer×head probe。

负对照包括 shuffled operator labels、shuffled subject split labels、random initialized
model、identity-only duplicate views、dataset/source ID probe、same operator/different
source，以及 same source/different operator。单数据集的 dataset-ID probe 会明确记为
不可估计，不伪造分数。

## 运行

```bash
python -m pip install -r requirements.txt
pytest -q
python scripts/run_smoke_test.py --output-dir outputs/smoke
python scripts/run_full_experiment.py \
  --mode real \
  --max-subjects 5 \
  --max-epochs-per-subject 24 \
  --output-dir outputs/real_bnci2014_001
```

smoke 只验证工程链路。真实数据加随机初始化 fallback 仍然只是 architecture smoke test；
要做 foundation-model 科学结论，必须提供经过适配并在 `model_summary.json` 中确认为
pretrained 的 checkpoint。

## 输出

```text
outputs/<run>/
├── probes/
│   ├── operator_probe_by_layer.csv
│   ├── task_probe_by_layer.csv
│   ├── operator_family_probe.csv
│   ├── operator_one_vs_rest.csv
│   ├── evidence_table.csv
│   └── controls.csv
├── attention/
│   ├── attention_distance.csv
│   └── attention_statistics.csv
├── figures/
│   ├── operator_leakage_heatmap.png
│   ├── task_information_heatmap.png
│   ├── qkv_comparison.png
│   ├── operator_family_paths.png
│   └── operator_head_heatmap.png
├── compressed_summaries/
├── operator_metadata.jsonl
├── dataset_summary.json
├── model_summary.json
├── runtime.json
├── resolved_config.yaml
├── manifest.json
└── report.md
```
