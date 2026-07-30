# H2——观测算子效应在特征空间中是否近似低秩

你是一名负责表示空间分析、线性代数实验和 EEG 信号处理的研究工程师。请在当前工作目录创建一个完整的 CPU-only Python 项目，用于验证：

H2：同一 EEG 样本在施加某一观测算子前后的表征差异，是否集中在一个可跨受试者迁移的低秩子空间中。

不要只输出思路。请直接实现数据加载、算子构造、特征提取、SVD 分析、跨受试者子空间迁移、统计对照、可视化、测试和 README。

# 一、核心定义

对同一 EEG epoch X 和观测算子 T_o，定义：

ΔE_o(X) = E(T_o(X)) - E(X)

将多个样本的差分堆叠为矩阵：

D_o = [ΔE_o(X_1); ΔE_o(X_2); ...; ΔE_o(X_n)]

对 D_o 做 SVD：

D_o = U Σ V^T

计算前 r 个奇异方向解释的能量比例：

rho_o(r) = sum_{i=1}^r sigma_i^2 / sum_i sigma_i^2

需要验证：
1. 某些 operator effect 是否由较小 r 解释；
2. 这种低秩结构是否显著强于非配对差分和随机噪声；
3. 在训练受试者上学习的 operator subspace 是否能解释未见受试者上的差分。

# 二、数据

首选：
- MOABB BNCI2014_001；
- 默认前 5 名受试者；
- left/right motor imagery epochs；
- subject-disjoint split。

同时实现 synthetic EEG fallback，仅用于 smoke test。

# 三、观测算子

至少包含：
- CAR reference
- Cz reference
- global gain 0.5 与 2.0
- channel dropout 25% 与 50%
- 4–30 Hz band-pass
- 8–30 Hz band-pass
- downsample-and-upsample

每个算子单独分析，不要先混合成一个总体类别。

# 四、表示函数 E

至少实现四种表征：

1. Raw-patch flatten：
   将 C×T epoch 切成固定时长 temporal patches，再 flatten。
2. PCA embedding：
   只在训练受试者的 identity view 上拟合 PCA。
3. Fixed random projection：
   使用固定随机种子生成线性映射。
4. Tiny frozen patch encoder：
   一层 Conv1d 或 Linear patch embedding，不训练或仅在 identity reconstruction 上做极轻量训练；必须可选。

所有表示最终统一到二维矩阵：
- 行：sample-token；
- 列：embedding dimension d。

必须保存 representation metadata，包括：
- operator
- subject
- sample id
- token id
- embedding type

# 五、必须实现的对照

1. Paired operator difference：
   E(T_o(X_i)) - E(X_i)
2. Unpaired difference：
   E(T_o(X_i)) - E(X_j), i != j
3. Same-sample Gaussian noise difference：
   E(X_i + epsilon_i) - E(X_i)
4. Random orthogonal transform difference
5. Label-preserving epoch permutation control

所有对照必须匹配样本数量，并使用相同 embedding dimension。

# 六、主要分析

对每种 operator、每种 embedding：

1. 画 cumulative explained variance：
   r ∈ {1, 2, 4, 8, 16, 32}
2. 输出达到以下阈值所需的最小 rank：
   - 50%
   - 70%
   - 80%
   - 90%
3. 输出 effective rank：
   基于奇异值归一化熵计算。
4. 输出 stable rank：
   ||D||_F^2 / ||D||_2^2
5. 输出与对照的差异及 bootstrap CI。

# 七、跨受试者子空间迁移

使用 Leave-One-Subject-Out：

1. 在训练受试者上估计前 r 个右奇异向量 V_r；
2. 对测试受试者差分 D_test 计算：
   explained_test(r) = ||D_test V_r V_r^T||_F^2 / ||D_test||_F^2
3. 计算 residual ratio：
   residual(r) = ||D_test - D_test V_r V_r^T||_F^2 / ||D_test||_F^2
4. 与以下基线比较：
   - 同 rank 随机正交子空间；
   - 从另一个 operator 学到的子空间；
   - 测试受试者 oracle SVD 上界。

需要输出：
- train-subspace vs random-subspace 的差异；
- operator-specificity matrix：
  用 operator A 的子空间解释 operator B 的差分；
- principal angle matrix；
- 子空间相似性热图。

# 八、局部性分析

需要进一步检查低秩结构是否依赖 signal content：

1. 按任务标签分组估计子空间；
2. 按受试者分组估计子空间；
3. 按信号功率四分位分组；
4. 比较不同分组的 principal angles；
5. 如果全局固定子空间迁移很差，但局部分组子空间更好，在报告中指出固定 U_o 假设可能不成立。

# 九、统计检验

- 以 subject 为 bootstrap 单位；
- paired operator difference 与每个 control 比较；
- 默认 Wilcoxon；
- Holm correction；
- 报告 effect size；
- 不要把 epoch 当独立样本计算虚假的超小 p 值。

# 十、输出

至少生成：

outputs/
├── explained_variance.csv
├── rank_thresholds.csv
├── effective_rank.csv
├── cross_subject_transfer.csv
├── operator_specificity.csv
├── principal_angles.csv
├── figures/
│   ├── cumulative_spectrum_*.png
│   ├── transfer_heatmap.png
│   ├── operator_specificity_heatmap.png
│   └── principal_angle_heatmap.png
└── report.md

`report.md` 必须回答：
1. 哪些算子最接近低秩；
2. rank r=2/4/8 时可解释多少差异；
3. 子空间是否跨受试者迁移；
4. 子空间是否 operator-specific；
5. 固定 operator subspace 是否成立；
6. 哪些结果支持或反驳 H2。

# 十一、工程要求

- CPU-only；
- numpy/scipy/sklearn/mne/moabb/matplotlib；
- 可选依赖缺失时优雅降级；
- 中间表示使用 numpy memmap 或压缩 npz 缓存；
- 设置内存上限意识，避免一次性堆叠全部 token；
- 对大矩阵支持 IncrementalPCA 或 randomized_svd；
- 提供 smoke-test 和 full-run 命令；
- 添加 shape、配对关系、SVD 数值稳定性和数据泄漏单元测试；
- 运行结束后列出新增文件与实际生成结果。

不要预设“低秩假设成立”。程序必须允许得出 H2 不成立的结论。
