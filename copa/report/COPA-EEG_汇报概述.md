# COPA-EEG 汇报概述

## 0. 汇报核心

**COPA-EEG** 面向 EEG Foundation Model 在跨设备、跨参考方式、跨通道配置和跨预处理流程下的迁移问题。

现有前沿工作已经开始解决 EEG 数据之间的**格式异构性**，例如不同通道数、电极位置、记录长度和空间拓扑。但是，能够接收不同格式的 EEG，并不等于模型已经消除了不同观测过程造成的表征偏差。即使两段信号来自相同或近似相同的神经活动，参考方式、montage、滤波器、采样率和设备响应仍可能改变通道幅值、频谱结构以及 token 之间的相关性。

COPA-EEG 的核心观点是：

> 普通 Transformer attention 学习的是受观测过程混杂的 EEG token 边际相关性；COPA-EEG 通过 Counterfactual Operator Pretraining 和 Operator-Partial Attention，学习在不同观测算子下仍然可识别、可迁移的神经表征。

---

# 1. 研究背景

## 1.1 为什么需要 EEG Foundation Model

传统 EEG 深度学习模型通常针对单一数据集、任务和设备设计，模型容易依赖固定的数据分布。EEG Foundation Model 希望通过大规模自监督预训练，在不同任务、受试者和数据集之间复用表征。

LaBraM（ICLR 2024）指出，EEG 数据规模有限且格式差异显著，传统面向单一数据集的方法难以获得通用感知能力。该工作通过大规模 EEG 预训练建立通用表示，是 EEG Foundation Model 的代表性起点之一。

但是，EEG 与图像、文本存在明显差异。EEG 数据的输入形式并不统一，常见差异包括：

- 电极数量不同；
- 电极空间位置不同；
- montage 不同；
- 参考方式不同；
- 采样率和记录长度不同；
- 滤波与预处理流程不同；
- 放大器、模数转换和噪声特性不同。

因此，跨数据集预训练不仅是数据规模问题，也是**观测系统不一致问题**。

---

## 1.2 前沿会议明确提出的局限

### LaBraM：数据格式差异限制通用预训练

LaBraM 指出，EEG 数据集规模通常较小，并且不同数据集的格式差异很大。其主要贡献是利用大规模自监督预训练学习通用 EEG 表征。

**仍然存在的问题：**

LaBraM 主要解决大规模预训练与通用表示问题，但没有显式建模 reference、montage、filter 和 device 对观测信号的影响。模型可能把数据集或设备特征作为稳定模式学习下来。

---

### CBraMod：不同 EEG 格式限制下游泛化

CBraMod（ICLR 2025）明确指出两个局限：

1. 现有模型通常将全部 EEG patches 一起建模，忽略空间依赖和时间依赖的异质性；
2. EEG 数据格式多样，使 Foundation Model 难以适配广泛的下游任务。

CBraMod 使用 criss-cross Transformer 分别建模空间与时间依赖，并使用条件位置编码兼容不同格式。

**仍然存在的问题：**

CBraMod 解决了“空间和时间关系如何建模”以及“不同输入格式如何编码”，但输入通道间的关系本身仍可能受到参考方式影响。例如，同一脑活动在 average reference 和 linked-mastoid reference 下会产生不同的通道相关矩阵。普通 attention 仍可能把这种参考方式诱导的关系作为神经关系学习。

---

### REVE：设备和电极配置异构导致线性探测泛化不足

REVE（NeurIPS 2025）指出，公共 EEG 数据来自不同协议、设备和电极配置，现有 EEG Foundation Model 往往局限于单一 setup，并且在 linear probing 条件下泛化不足。

REVE 使用 4D positional encoding，使模型可以处理任意记录长度和电极排列，并在 92 个数据集、超过 60,000 小时 EEG 上预训练。

**仍然存在的问题：**

REVE 主要解决不同电极排列和信号长度的表示兼容性。它使模型能够接收不同 setup 的输入，但没有显式区分：

\[
\text{神经活动差异}
\quad\text{与}\quad
\text{设备、参考和滤波造成的观测差异}.
\]

因此，“支持任意 setup”与“对 setup 不敏感”是两个不同问题。

---

### NeurIPT：跨受试者、跨任务、跨条件和跨电极配置仍然困难

NeurIPT（NeurIPS 2025）指出，EEG Foundation Model 仍受到以下异构性的影响：

- inter-subject variability；
- inter-task variability；
- inter-condition variability；
- 不同记录系统的电极配置差异。

NeurIPT 使用电极三维坐标、幅值感知掩码预训练和渐进式 Mixture-of-Experts 处理这些变化。

**仍然存在的问题：**

NeurIPT 主要通过更强的模型容量和专家分工吸收异构性，但没有显式要求模型区分“神经状态变化”和“观测算子变化”。当设备、数据集和任务高度相关时，MoE 还可能形成数据集或设备特定专家。

---

### LUNA：电极拓扑异构限制模型扩展

LUNA（NeurIPS 2025）将 topological heterogeneity 视为大规模 EEG 建模的核心障碍。不同数据集具有不同电极布局，因此模型很难在多个 montage 间统一迁移。

LUNA 使用 learned queries 和 cross-attention，将不同通道配置压缩为固定大小的 topology-agnostic latent representation。

**仍然存在的问题：**

拓扑无关表示主要处理电极数量和几何布局变化，但同一拓扑下仍可能存在不同 reference、filter 和 device response。仅消除 topology dependency，不能保证 attention relation 不受观测过程影响。

---

### CSBrain：统一的密集建模忽略跨尺度结构

CSBrain（NeurIPS 2025）指出，现有 EEG Foundation Model 沿用 NLP 和视觉中的 scale-agnostic dense modeling，忽视不同 EEG 任务具有不同的时间和空间尺度。

其方法通过 cross-scale tokenization 和 structured sparse attention 建模跨尺度结构。

**与 COPA-EEG 的关系：**

CSBrain 说明 attention 中的 token relation 需要加入 EEG 特定归纳偏置。COPA-EEG进一步提出：除了空间、时间和尺度结构，attention relation 还受到观测算子的系统性影响，因此需要控制 operator-induced affinity。

---

## 1.3 当前方法留下的共同空缺

现有前沿方法主要集中于：

| 方法方向 | 解决的问题 |
|---|---|
| 大规模 masked pretraining | 数据规模与通用表征 |
| 电极位置编码 | 不同电极坐标 |
| topology-agnostic latent | 不同通道数和 montage |
| 空间—时间解耦 attention | EEG 时空结构异质性 |
| 多尺度 tokenization | 不同任务的时空尺度 |
| Mixture-of-Experts | 跨任务和跨条件异构性 |

它们主要回答：

> 不同格式的 EEG 如何输入同一个模型？

COPA-EEG希望进一步回答：

> 当不同格式已经能够输入模型后，如何避免模型把参考方式、滤波器和设备响应诱导的相关性误认为神经活动规律？

这构成 COPA-EEG 的研究位置。

---

# 2. 问题定义

## 2.1 EEG 是潜在神经活动经过观测系统后的结果

将 EEG 观测过程简化为：

\[
X_o
=
M_o R_o G_o
\left(
H_o * (LZ)
\right)
+
\epsilon_o,
\]

其中：

- \(Z\)：潜在神经源活动；
- \(L\)：脑源到头皮电极的传导过程；
- \(R_o\)：参考算子；
- \(M_o\)：montage 与通道选择；
- \(H_o\)：滤波器和设备频率响应；
- \(G_o\)：增益与单位变换；
- \(\epsilon_o\)：设备噪声和伪影；
- \(o\)：当前观测条件。

模型实际输入的是 \(X_o\)，而不是潜在神经活动 \(Z\)。

不同观测算子可能使同一神经活动产生明显不同的信号：

\[
Z
\xrightarrow{o_1}
X_{o_1},
\qquad
Z
\xrightarrow{o_2}
X_{o_2}.
\]

如果模型直接学习 \(X_o\) 中的相关性，它可能同时编码：

\[
\text{neural information}
+
\text{operator information}
+
\text{dataset identity}.
\]

---

## 2.2 普通 attention 的问题

标准 self-attention 为：

\[
Q=XW_Q,\qquad
K=XW_K,\qquad
V=XW_V,
\]

\[
A=
\operatorname{Softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right),
\qquad
Y=AV.
\]

其中 \(QK^\top\) 决定 token 之间的关联强度。

但是，在 EEG 中，token 相关性不仅由神经活动决定，还会受到以下因素影响：

- 重参考改变通道之间的线性关系；
- montage 改变可见空间范围；
- filter 改变不同频段的相关结构；
- gain 改变幅值尺度；
- device response 改变频率和噪声特性。

因此，普通 attention 学到的更接近：

\[
p(\text{token}_i,\text{token}_j\mid X_o),
\]

即在特定观测条件下的边际相关性，而不一定是跨观测条件稳定的关系。

---

# 3. COPA-EEG 的总体设计

COPA-EEG 包含两个核心组件：

1. **Counterfactual Operator Pretraining（COP）**  
   对同一 EEG 构造不同观测算子视图，使模型学习跨观测条件仍然可识别的信息。

2. **Operator-Partial Attention（OPA）**  
   在 self-attention 内部抑制由当前观测算子解释的 Query–Key 特征方向，减少 operator-induced token affinity。

整体框架不增加新的 EEG 编码分支，而是在单一 Transformer backbone 中完成。

---

# 4. Counterfactual Operator Pretraining

## 4.1 构造 operator-paired views

对同一段 EEG \(X\)，采样两个观测算子：

\[
X^{(1)}=T_{o_1}(X),
\qquad
X^{(2)}=T_{o_2}(X).
\]

候选算子包括：

- 重参考；
- 通道子集或通道缺失；
- 重采样；
- 带通、低通或高通滤波；
- 增益和单位变化；
- 设备频率响应的近似模拟。

这两个视图来自同一原始记录，因此共享神经内容，但可能具有不同的信息完整度。

---

## 4.2 区分等价算子和信息有损算子

不能假设所有变换都保持完整语义不变。COPA-EEG 将算子分成两类。

### 第一类：近似等价算子

例如：

- 合法重参考；
- 通道顺序置换；
- 单位换算；
- 不发生截断的全局增益变化。

对于这些算子，可以要求较强的一致性：

\[
\mathcal L_{\mathrm{eq}}
=
\left\|
z(X)-z(T_oX)
\right\|_2^2.
\]

其目标是使表示不依赖等价的观测表达方式。

### 第二类：信息有损算子

例如：

- 通道删除；
- 降采样；
- 限制带宽；
- 强滤波；
- 局部信号损坏。

这些算子会删除部分任务相关信息，因此不能强制两个视图的完整表示完全一致。

对于有损视图，模型只预测跨视图仍然可识别的 latent information：

\[
\mathcal L_{\mathrm{pred}}
=
w(X,o)
\left\|
g_o(z_s)-\operatorname{sg}(z_t)
\right\|_2^2,
\]

其中：

- \(z_t\)：完整视图的 teacher representation；
- \(z_s\)：退化视图的 student representation；
- \(g_o\)：operator-conditioned predictor；
- \(w(X,o)\)：根据保留通道、带宽和 teacher uncertainty 决定的可信权重；
- \(\operatorname{sg}\)：stop-gradient。

该设计避免模型为了匹配严重退化的视图而丢弃完整视图中的有效神经信息。

---

# 5. Operator-Partial Attention

## 5.1 Operator embedding

首先将当前观测条件编码成 operator embedding：

\[
e_o=E_{\mathrm{op}}(o).
\]

可使用的信息包括：

- reference 类型；
- channel mask；
- 电极三维坐标；
- sampling rate；
- filter cutoff；
- signal unit 和 gain；
- 已知的 device metadata。

---

## 5.2 学习 operator-related low-rank subspace

在第 \(\ell\) 层、第 \(h\) 个 attention head 中，根据 \(e_o\) 生成低秩基：

\[
U_{\ell,h}(o)
\in
\mathbb R^{d_h\times r},
\qquad
r\ll d_h.
\]

对应的投影矩阵为：

\[
P_{\ell,h}(o)
=
U_{\ell,h}(o)U_{\ell,h}(o)^\top.
\]

该子空间表示当前观测算子在 attention feature space 中主要影响的方向。

---

## 5.3 对 Query 和 Key 做软部分投影

COPA-EEG 不直接完全删除 operator subspace，而是使用逐层、逐 head 的软投影：

\[
\widetilde Q_{\ell,h}
=
Q_{\ell,h}
\left(
I-\alpha_{\ell,h}(o)P_{\ell,h}(o)
\right),
\]

\[
\widetilde K_{\ell,h}
=
K_{\ell,h}
\left(
I-\alpha_{\ell,h}(o)P_{\ell,h}(o)
\right),
\]

其中：

\[
0\leq\alpha_{\ell,h}(o)\leq1.
\]

新的 attention 为：

\[
A_{\ell,h}^{\mathrm{OPA}}
=
\operatorname{Softmax}
\left(
\frac{
\widetilde Q_{\ell,h}
\widetilde K_{\ell,h}^{\top}
}{
\sqrt{d_h}
}
\right).
\]

参数 \(\alpha_{\ell,h}(o)\) 控制投影强度：

- \(\alpha=0\)：退化为普通 attention；
- \(\alpha=1\)：完全移除当前估计的 operator subspace；
- \(0<\alpha<1\)：只抑制部分 operator effect。

这允许不同层和不同 attention head 自主决定是否需要 operator correction。

---

## 5.4 为什么优先修改 Query 和 Key

Query 和 Key 决定 token 之间如何建立连接：

\[
QK^\top
\longrightarrow
\text{attention affinity}.
\]

reference 和 montage 最直接改变的是通道之间的相关关系，因此先在 \(Q,K\) 路径控制 operator-induced affinity。

Value 保留 token 内容：

\[
Y=A^{\mathrm{OPA}}V.
\]

但是，Value、残差连接和 MLP 仍然可能携带 operator 信息。因此，COPA-EEG 还需要输出层的 operator leakage regularization。

---

# 6. Operator Leakage Regularization

在最终表示 \(z\) 上训练 operator classifier：

\[
\hat o=C_{\mathrm{op}}(z).
\]

通过 gradient reversal 或独立性约束，使最终表示减少对观测条件的可预测性。

例如：

\[
\mathcal L_{\mathrm{op}}
=
\operatorname{HSIC}(z,o),
\]

或使用 adversarial operator classification。

该约束用于控制以下路径重新引入 operator 信息：

\[
V,
\qquad
\text{residual connection},
\qquad
\text{MLP},
\qquad
\text{later layers}.
\]

因此，OPA 的目标是修正 attention relation，而 leakage regularization 负责约束最终表征。

---

# 7. 总体训练目标

COPA-EEG 的训练目标可写为：

\[
\mathcal L
=
\mathcal L_{\mathrm{pred}}
+
\lambda_{\mathrm{eq}}\mathcal L_{\mathrm{eq}}
+
\lambda_{\mathrm{op}}\mathcal L_{\mathrm{op}}
+
\lambda_{\mathrm{var}}\mathcal L_{\mathrm{var}}
+
\lambda_{\mathrm{orth}}\mathcal L_{\mathrm{orth}}.
\]

其中：

- \(\mathcal L_{\mathrm{pred}}\)：有损 operator views 之间的可识别信息预测；
- \(\mathcal L_{\mathrm{eq}}\)：等价 operator views 的表示一致性；
- \(\mathcal L_{\mathrm{op}}\)：最终表示的 operator leakage 控制；
- \(\mathcal L_{\mathrm{var}}\)：防止表示坍塌；
- \(\mathcal L_{\mathrm{orth}}\)：约束低秩基近似正交。

---

# 8. COPA-EEG 与现有工作的差异

| 方法 | 主要解决内容 | 未显式解决内容 |
|---|---|---|
| LaBraM | 大规模 EEG 自监督预训练 | 观测算子混杂 |
| CBraMod | 空间—时间依赖分离、不同格式适配 | reference/device 对 attention relation 的影响 |
| REVE | 任意长度和电极排列 | setup compatibility 与 setup invariance 的区别 |
| NeurIPT | 跨任务、跨条件和电极配置建模 | 神经变化与 operator 变化的显式分离 |
| LUNA | topology-agnostic latent space | 同一拓扑下的参考、滤波和设备差异 |
| CSBrain | 多尺度时空结构 | operator-induced token affinity |
| EEG-X（预印本） | device-agnostic 与 noise-aware reconstruction | 基于同一记录的 operator-paired latent prediction 和 attention 层局部修正 |
| SingLEM（预印本） | 通过单通道规避固定 montage | 多通道空间关系及其参考依赖 |

COPA-EEG 的主要差异可以概括为：

\[
\boxed{
\text{现有方法解决输入格式兼容，COPA-EEG解决观测算子对表征关系的混杂}
}
\]

---

# 9. 核心科学假设

COPA-EEG建立在以下可验证假设上：

### 假设一

同一 EEG 经过不同物理有效观测算子后，存在跨视图仍可识别的共享神经信息。

### 假设二

reference、montage、filter 和 device 会在 Transformer 的 Query–Key feature space 中形成可压缩的 operator-related directions。

### 假设三

抑制 operator-related Query–Key directions，可以减少 attention map 对设备和参考方式的依赖，同时保留下游任务相关关系。

### 假设四

只修改 \(Q,K\) 不足以保证最终表示稳定，因此需要额外控制 Value、残差和 MLP 路径中的 operator leakage。

这些假设都需要通过实验验证，不能作为未经检验的事实直接宣称。

---

# 10. 汇报中应强调的核心主张

COPA-EEG 不主张恢复唯一、真实的脑源活动，也不主张完全消除所有设备差异。

更稳妥的核心主张是：

> COPA-EEG 将 EEG 观测变换区分为近似等价算子和信息有损算子。对于近似等价算子，模型学习稳定表示；对于信息有损算子，模型只预测跨视图仍可识别的信息。架构上，Operator-Partial Attention 通过逐层、逐 head 的低秩软投影，抑制 reference、montage、filter 和 device 引起的 token-affinity 偏差，并通过 operator leakage regularization 控制其他网络路径重新引入观测条件信息。

一句话版本：

> **COPA-EEG 不是简单让模型兼容不同 EEG 格式，而是显式控制不同观测系统对 attention relation 和最终表征的影响。**

---

# 11. 建议的汇报结构

## 第一部分：背景与问题

1. EEG Foundation Model 的目标；
2. EEG 数据的设备、通道和预处理异构性；
3. LaBraM、CBraMod、REVE、NeurIPT、LUNA 和 CSBrain 分别解决了什么；
4. 现有方法主要解决 format compatibility；
5. format compatibility 不等于 operator robustness；
6. 引出观测算子混杂问题。

## 第二部分：方法

1. EEG 观测模型；
2. 构造同一 EEG 的 operator-paired views；
3. 区分等价算子与信息有损算子；
4. Counterfactual Operator Pretraining；
5. 普通 attention 为什么受到 operator 影响；
6. Operator-Partial Attention；
7. operator leakage regularization；
8. 总体训练目标。

## 第三部分：验证思路

1. 跨设备、跨 reference、跨 montage 和跨数据集测试；
2. operator decoding probe；
3. \(Q\)、\(K\)、\(V\)、attention map 和 final representation 的泄漏分析；
4. 普通 attention、Q-only、K-only、QK、QKV 的消融；
5. synthetic operator shift 与真实设备 shift 的区分；
6. frozen probing、few-shot fine-tuning 和 full fine-tuning；
7. 与 specialist models 和随机初始化模型比较。

---

# 12. 参考文献

## 正式会议论文

1. Jiang, W.-B., Zhao, L., & Lu, B.-L. **Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI.** ICLR 2024.  
   https://proceedings.iclr.cc/paper_files/paper/2024/hash/47393e8594c82ce8fd83adc672cf9872-Abstract-Conference.html

2. Wang, J., et al. **CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding.** ICLR 2025.  
   https://proceedings.iclr.cc/paper_files/paper/2025/hash/bbbd6d915cb90be21c1254a82d45cedd-Abstract-Conference.html

3. El Ouahidi, Y., et al. **REVE: A Foundation Model for EEG — Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects.** NeurIPS 2025.  
   https://proceedings.neurips.cc/paper_files/paper/2025/hash/20a917f77773ac0fa8bea2bdd6606b66-Abstract-Conference.html

4. Fang, Z., et al. **NeurIPT: Foundation Model for Neural Interfaces.** NeurIPS 2025.  
   https://proceedings.neurips.cc/paper_files/paper/2025/hash/dd9c5ce8803e1898d438e636fbae0236-Abstract-Conference.html

5. Döner, B., et al. **LUNA: Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis.** NeurIPS 2025.  
   https://proceedings.neurips.cc/paper_files/paper/2025/hash/66969a9e6bd7a26dfeccea7227178ca7-Abstract-Conference.html

6. Zhou, Y., et al. **CSBrain: A Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding.** NeurIPS 2025.  
   https://proceedings.neurips.cc/paper_files/paper/2025/hash/7e199ad8ae40eb19b2980f61f659cb07-Abstract-Conference.html

## 近期预印本与基准

7. Mohammadi Foumani, N., et al. **EEG-X: Device-Agnostic and Noise-Robust Foundation Model for EEG.** arXiv, 2025.  
   https://arxiv.org/abs/2511.08861

8. Sukhbaatar, J., et al. **SingLEM: Single-Channel Large EEG Model.** arXiv, 2025.  
   https://arxiv.org/abs/2509.17920

9. Liu, D., et al. **EEG Foundation Models: Progresses, Benchmarking, and Open Problems.** arXiv, 2026.  
   https://arxiv.org/abs/2601.17883

10. Kastrati, A., et al. **EEG-Bench: A Benchmark for EEG Foundation Models in Clinical Applications.** arXiv, 2025.  
    https://arxiv.org/abs/2512.08959

---

# 13. 最终概述

COPA-EEG 的研究动机来自 EEG Foundation Model 当前仍未完全解决的观测异构问题。LaBraM、CBraMod、REVE、NeurIPT、LUNA 和 CSBrain 分别推进了大规模预训练、时空结构建模、电极位置适配、跨条件建模、拓扑无关表示和多尺度建模，但这些方法主要关注不同 EEG 格式如何被统一模型接收和编码。它们没有直接控制 reference、montage、filter 和 device response 对 token correlation 的影响。

COPA-EEG首先通过 Counterfactual Operator Pretraining，对同一 EEG 施加物理有效观测变换，构造 operator-paired views。对于合法重参考和单位变化等近似等价算子，模型学习稳定表示；对于通道缺失、滤波和降采样等信息有损算子，模型只预测跨视图仍然可识别的神经信息。随后，Operator-Partial Attention 根据当前观测算子生成低秩子空间，并从 Query 和 Key 中进行逐层、逐 head 的软投影，以抑制 operator-induced token affinity。最终表示再通过 operator leakage regularization 控制 Value、残差连接和 MLP 路径携带的观测条件信息。

该方法的核心创新不在于增加新的编码分支，而在于重新定义 EEG Foundation Model 的预训练视图关系，并对 Transformer self-attention 做局部、可解释和可证伪的修改。
