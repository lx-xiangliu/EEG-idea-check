# H3——Operator 信息主要泄漏在 Q、K、V、Attention 还是最终表征

你是一名熟悉 PyTorch Transformer internals、forward hooks 和 EEG Foundation Model 的研究工程师。请创建一个完整的 CPU-only 项目，用于验证：

H3：不同 EEG 观测算子的可识别信息分别存在于 Transformer 的 Q、K、V、attention map、residual stream 和最终 representation 中；reference/montage 是否主要污染 attention affinity，需要通过数据而不是假设判断。

不要只写分析方案。直接实现完整代码、模型适配层、hooks、probe、图表、测试和 README。

# 一、实验原则

不训练大型模型。优先执行冻结模型推理和轻量线性 probe。

模型优先级：
1. 尝试适配官方 CBraMod 预训练模型，在 CPU 上加载；
2. 如果预训练权重不可获得，支持用户通过配置提供 checkpoint 路径；
3. 仍不可用时，使用一个结构明确的 Tiny EEG Transformer 作为 fallback；
4. fallback 结果必须明确标为 architecture smoke test，不能冒充 pretrained foundation model 结论。

# 二、数据

首选 BNCI2014_001：
- left hand vs right hand；
- 默认 3–5 名受试者；
- 只抽取足够完成 CPU 实验的样本；
- subject-disjoint；
- 对每个原始 epoch 构造 identity、CAR、Cz reference、gain、filter、channel dropout、resampling views。

每个 operator view 必须保留：
- same source epoch id
- subject id
- task label
- operator label
- operator family
- operator metadata

# 三、模型适配

建立统一接口：

class EEGTransformerAdapter:
    forward_with_internals(x, metadata) -> dict

输出至少包含：
- input_embedding
- per-layer Q
- per-layer K
- per-layer V
- per-layer attention_logits
- per-layer attention_probs
- per-layer attention_output
- per-layer residual_before_attention
- per-layer residual_after_attention
- per-layer mlp_output
- final_tokens
- pooled_representation

如果模型内部使用 fused attention，必须：
- 尝试关闭 fused/flash attention；
- 或重写对应 attention forward；
- 确保提取的 Q/K/V 数值与原 forward 一致；
- 写单元测试验证 hook 前后模型输出误差小于给定容差。

# 四、内存控制

CPU 环境下不得保存全部大张量。

对每个 layer/head/token 表示，至少实现以下压缩方式：
- mean pooling
- std pooling
- max pooling
- first/CLS token
- covariance summary
- attention entropy
- attention diagonal mass
- attention off-diagonal mass
- top-k attention concentration

配置允许：
- 只分析指定层；
- 只分析指定 head；
- 限制 subject 和 epoch 数量；
- 将中间结果分批保存为 parquet/npz。

# 五、Probe 任务

对以下 representation 分别训练 operator classifier：

1. input embedding
2. Q
3. K
4. V
5. QK attention logits summary
6. attention probability summary
7. attention output
8. residual stream
9. MLP output
10. final pooled representation

同时训练 task-label classifier。

模型：
- LogisticRegression
- LinearSVC
- 可选 RandomForest

划分：
- 必须 subject-disjoint；
- scaler 只在训练 fold 拟合；
- 输出 balanced accuracy、macro F1 和 subject-level bootstrap CI。

# 六、层与 head 分析

输出：

1. layer × representation operator-decoding heatmap；
2. layer × representation task-decoding heatmap；
3. 对每种 operator family 单独做 one-vs-rest probe；
4. 如果可能，输出 layer × head 的 operator probe heatmap；
5. 输出 attention entropy 和 operator label 的关系；
6. 输出不同 operator 下 attention map 的 paired distance：
   - Frobenius distance
   - Jensen-Shannon divergence
   - centered kernel alignment，可选

# 七、泄漏路径判断

程序需要自动总结以下量：

- Acc(operator | Q)
- Acc(operator | K)
- Acc(operator | V)
- Acc(operator | attention)
- Acc(operator | residual)
- Acc(operator | final representation)

并按 operator family 输出：

- reference
- channel/montage
- filter
- resampling
- gain

不要预设 reference 一定主要出现在 Q/K。

需要输出一个 evidence table：

| Operator family | Q | K | V | Attention | Residual | Final | Dominant path |
|---|---:|---:|---:|---:|---:|---:|---|

Dominant path 必须由规则计算，例如：
- 最高 probe score；
- 相对 chance level 的增量；
- bootstrap CI 是否重叠。

# 八、负对照

必须包含：

1. shuffled operator labels；
2. shuffled subject labels；
3. random initialized model；
4. identity-only duplicate views；
5. dataset/source ID probe；
6. same operator but different source epoch；
7. same source epoch but different operator。

这些对照用于区分：
- operator signal；
- subject signal；
- task signal；
- sample identity signal；
- dataset shortcut。

# 九、关键输出

至少生成：

outputs/
├── probes/
│   ├── operator_probe_by_layer.csv
│   ├── task_probe_by_layer.csv
│   ├── operator_family_probe.csv
│   └── controls.csv
├── attention/
│   ├── attention_distance.csv
│   └── attention_statistics.csv
├── figures/
│   ├── operator_leakage_heatmap.png
│   ├── task_information_heatmap.png
│   ├── qkv_comparison.png
│   └── operator_family_paths.png
└── report.md

`report.md` 必须回答：
1. 哪些 operator 在 Q/K 中最可分；
2. 哪些 operator 在 V 中最可分；
3. final representation 是否仍保留大量 operator 信息；
4. Q/K-only projection 是否有理论上的实证依据；
5. 是否需要处理 V、residual 或 MLP；
6. 结果支持、部分支持还是反驳 H3；
7. 当前结果来自 pretrained model 还是 fallback model。

# 十、工程要求

- CPU-only；
- PyTorch inference_mode；
- batch size 默认很小；
- 支持 checkpoint 本地路径；
- 不自动下载巨大 checkpoint，除非用户显式开启；
- 所有随机种子固定；
- 提供 adapter 单元测试；
- 提供 hook 输出一致性测试；
- 提供 synthetic smoke test；
- 提供一条真实数据命令；
- 自动记录模型参数量、运行时间、峰值 RSS 内存和版本；
- 运行结束后列出所有生成文件。

不要为了得到支持 H3 的结果而选择性删除 V 或 residual 的分析。
