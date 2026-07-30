# H4——无需训练，离线投影 operator subspace 是否能减少泄漏并保留任务信息

你是一名负责表示去偏、低秩投影和 Transformer 机制验证的研究工程师。请在当前工作目录创建一个完整的 CPU-only 实验项目，用于验证：

H4：从 Q、K、V 或最终 representation 的 paired operator differences 中估计低秩 operator subspace，并进行离线软投影，能否降低 operator 可识别性，同时保留或改善任务标签信息和跨算子一致性。

不训练大型模型。直接实现完整代码、alpha/rank sweep、交叉验证、随机子空间对照、Pareto 分析、图表和报告。

# 一、输入

项目应支持两种输入方式：

1. 读取 Prompt 3 生成的缓存特征；
2. 若缓存不存在，调用统一 Transformer adapter 重新提取：
   - Q
   - K
   - V
   - attention output
   - final representation

样本 metadata 必须包括：
- subject id
- source epoch id
- task label
- operator label
- operator family
- paired identity/operator view mapping

# 二、估计 operator subspace

对 representation R ∈ {Q, K, V, final}：

D_o^R = R(T_o(X)) - R(X)

仅使用训练受试者估计 SVD：

D_train = U Σ V^T

取前 r 个右奇异向量作为 operator subspace：

U_o^R = V[:, :r]

严格禁止使用测试受试者的数据估计投影方向。

rank sweep：
r ∈ {1, 2, 4, 8, 16}

如果维度不足，自动裁剪。

# 三、软投影

实现：

R' = R (I - alpha U U^T)

alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}

需要比较：

1. Q-only
2. K-only
3. QK
4. V-only
5. QKV
6. final-representation-only
7. QK + final
8. random orthogonal subspace
9. PCA top-variance removal
10. PCA bottom-variance removal
11. shuffled operator-pair subspace

对于 Q/K 投影：
- 必须重新计算 attention logits；
- 重新计算 softmax attention；
- 使用原始 V 或投影后的 V；
- 重新得到 attention output；
- 如果无法继续通过后续完整模型层，至少在当前层 attention output 上评估；
- 在报告中明确评估层级。

# 四、评估指标

每个 projection configuration 都需要计算：

1. Operator probe：
   balanced accuracy、macro F1
2. Task probe：
   balanced accuracy、macro F1
3. Paired-view representation distance：
   Euclidean / cosine distance
4. Cross-operator task transfer：
   train on identity, test on each operator
5. Subject probe：
   检查投影是否只是删除 subject identity
6. Representation variance：
   防止投影导致 collapse
7. Effective dimensionality
8. Reconstruction energy retained：
   ||R'||_F^2 / ||R||_F^2

所有 probe 都必须 subject-disjoint，scaler 仅在训练 fold 拟合。

# 五、Pareto 分析

定义：
- x 轴：operator probe balanced accuracy
- y 轴：task probe balanced accuracy

为每个 alpha、rank 和 projection target 画点。

需要：
- 标出 baseline alpha=0；
- 标出 random projection；
- 计算 Pareto frontier；
- 计算 task-retention / operator-removal trade-off；
- 给出一个内部选择规则，例如：
  在 task accuracy 下降不超过 1 个百分点的条件下，使 operator accuracy 最低。

不要把该规则当作普遍理论，只用于实验筛选。

# 六、跨受试者与跨算子验证

使用 Leave-One-Subject-Out 或 GroupKFold：

每一折：
1. 训练受试者估计 operator subspace；
2. 训练受试者训练 probes；
3. 测试受试者应用固定投影；
4. 测试 operator 和 task 信息。

同时做：
- operator-specific subspace；
- pooled operator-family subspace；
- 使用 operator A 的 subspace 投影 operator B；
- unseen composition，例如 filter + channel dropout。

# 七、随机与错误子空间对照

必须包含：

1. 同 rank 随机正交子空间，至少重复 20 次；
2. 从 shuffled pairs 估计的子空间；
3. 从另一个 operator 学到的子空间；
4. 从 subject differences 学到的子空间；
5. 从 task-label differences 学到的子空间。

需要检验 learned operator subspace 是否显著优于这些对照。

# 八、统计分析

- 以 subject 为分析单位；
- learned projection vs random projection；
- QK vs QKV；
- QK vs final-only；
- operator-specific vs pooled；
- Wilcoxon 或 permutation test；
- Holm correction；
- bootstrap 95% CI；
- 报告 effect size。

# 九、关键结论规则

`report.md` 需要根据结果回答：

1. 投影是否降低 operator probe；
2. task probe 是否基本保留；
3. QK 是否优于 QKV；
4. V leakage 是否导致 QK-only 失败；
5. operator subspace 是否优于随机降维；
6. 是否存在可接受的 alpha/rank 区间；
7. 是否支持 H4；
8. 如果失败，失败原因最可能是：
   - subspace 不低秩；
   - subspace 不跨受试者；
   - operator 与 task 强纠缠；
   - V/residual leakage；
   - 单纯降维效应；
   - paired samples 构造错误。

# 十、输出

至少生成：

outputs/
├── sweep_results.csv
├── pareto_frontier.csv
├── cross_operator_results.csv
├── control_comparisons.csv
├── statistical_tests.csv
├── figures/
│   ├── pareto_qk_qkv.png
│   ├── alpha_rank_heatmap.png
│   ├── operator_vs_task_tradeoff.png
│   ├── paired_distance.png
│   └── random_projection_comparison.png
└── report.md

# 十一、工程要求

- CPU-only；
- 大矩阵使用 randomized_svd；
- 分 fold 缓存投影矩阵；
- 不得测试数据泄漏；
- 所有 projection matrix 保存到磁盘并带 metadata；
- 单元测试包括：
  - U^T U ≈ I
  - alpha=0 输出等于原始表示
  - alpha=1 时目标子空间分量接近 0
  - train/test subject 隔离
  - random subspace rank 正确
- 提供 smoke-test 和正式运行命令；
- 运行结束后列出新增文件和结果摘要。

不要预设 QK 一定优于 QKV。该实验的目标就是证伪或支持这一设计选择。
