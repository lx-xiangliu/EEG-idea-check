# Codex 全流程研究验证 Prompt：TRF-Partial Attention for EEG–Audio Foundation Models

> 使用方式：将本文件完整交给 Codex。  
> 任务性质：严格研究验证，不是单纯实现。  
> 最终目标：判断 TRF-Partial Attention 是否具有足够的新颖性、理论合理性、可实现性、机制证据与顶会投稿潜力。  
> 核心原则：不得默认方法有效；必须优先寻找反例、近邻工作、伪改进来源、数据泄漏和任务 shortcut。

---

# 0. 研究 Idea

研究主题：

# TRF-Partial Attention

一句话定义：

> 在 EEG–Audio 对齐前，先从 EEG Query 和 Audio Key 中解析去除由低级声学变量及其神经时间延迟所解释的成分，再计算跨模态 attention，使模型学习控制低级声学 tracking 之后仍然存在的 EEG–Audio correspondence。

现有 EEG–Audio 方法通常直接优化：

\[
I(E;A)
\]

或者使用标准 cross-attention：

\[
\operatorname{Attn}(Q_E,K_A,V_A)
=
\operatorname{softmax}
\left(
\frac{Q_EK_A^\top}{\sqrt d}
\right)V_A.
\]

但是 EEG 与音频之间最强、最容易利用的关系往往来自：

- speech envelope；
- onset；
- energy；
- pitch；
- duration；
- spectral flux；
- stimulus timing。

这些信号可能使模型在 match–mismatch、retrieval 或 alignment 任务上取得较高分数，却没有学习 phonetic、lexical 或 semantic information。

本研究希望把目标从：

\[
I(E;A)
\]

改写为近似的：

\[
I(E;A\mid C),
\]

其中 \(C\) 是低级声学驱动及其 lag-expanded representation。

---

# 1. 核心数学定义

## 1.1 低级声学特征

从音频中提取：

\[
c(t)
=
[
\text{envelope},
\text{onset},
F_0,
\text{energy},
\text{spectral flux},
\text{optional spectral centroid}
].
\]

必须允许对特征集合进行消融，不得默认所有特征都必要。

## 1.2 EEG 侧 lag-expanded acoustic design matrix

构造：

\[
C_E(t)
=
[
c(t-\tau_1),
c(t-\tau_2),
\ldots,
c(t-\tau_K)
],
\]

其中：

\[
\tau_k \in [\tau_{\min},\tau_{\max}].
\]

推荐首轮使用：

\[
\tau_{\min}=-100\text{ ms},
\qquad
\tau_{\max}=500\text{ ms}.
\]

必须测试：

- 是否允许负 lag；
- lag 间隔；
- lag range；
- subject-specific lag；
- 固定 lag 与可学习 lag。

## 1.3 Audio 侧 acoustic design matrix

基础版本：

\[
C_A(t)=c(t).
\]

可选版本：

\[
C_A(t)
=
[
c(t-\omega_1),
\ldots,
c(t+\omega_J)
].
\]

首轮必须保持简单，不能同时引入复杂音频上下文和复杂 EEG lag。

## 1.4 Residual-maker matrix

定义 ridge-stabilized projection：

\[
P_C
=
C(C^\top C+\lambda I)^{-1}C^\top,
\]

\[
M_C=I-P_C.
\]

则：

\[
Q_E^\perp=M_{C_E}Q_E,
\]

\[
K_A^\perp=M_{C_A}K_A.
\]

TRF-Partial Attention：

\[
\boxed{
\operatorname{TPA}(Q_E,K_A,V_A)
=
\operatorname{softmax}
\left(
\frac{
Q_E^\perp {K_A^\perp}^{\top}
}{
\sqrt d
}
\right)V_A
}
\]

或者写为：

\[
\boxed{
\operatorname{TPA}
=
\operatorname{softmax}
\left(
\frac{
(M_{C_E}Q_E)
(M_{C_A}K_A)^\top
}{
\sqrt d
}
\right)V_A
}
\]

---

# 2. 你需要回答的核心问题

## Q1：Novelty

是否已经存在：

- EEG–Audio 中的 partial correlation attention；
- TRF-based residualized cross-attention；
- acoustic nuisance projection；
- conditional contrastive learning for EEG–speech；
- lag-expanded covariate removal before attention；
- Frisch–Waugh–Lovell residualization in multimodal attention；
- low-level acoustic shortcut removal in brain–speech alignment？

## Q2：理论有效性

该方法是否真的近似：

\[
I(E;A\mid C)?
\]

或者它只是：

- 线性去相关；
- nuisance regression；
- 一般正交投影；
- attention 前的预处理；
- residualization regularization？

必须明确该方法可以声称什么、不能声称什么。

## Q3：机制有效性

控制 \(C\) 后，EEG 与高级 audio representation 是否仍有可检测关系？

若：

\[
I(E;A\mid C)\approx 0,
\]

则该方法不应继续。

## Q4：架构必要性

为什么必须修改 attention？

是否简单地在输入 embedding 上 residualize 就足够？

必须比较：

- input residualization；
- pooled embedding residualization；
- loss-level partial correlation；
- attention-level residualization。

若 attention-level 方法没有额外收益，则不应声称架构创新。

## Q5：是否真的控制了低级声学 shortcut

必须证明：

- acoustic leakage 降低；
- acoustically matched negative 下性能提升；
- shuffled acoustic covariate 后效果消失；
- random subspace 不能产生同样收益；
- performance gain 不只是额外正则化。

## Q6：是否保留有用声学信息

TPA 只 residualize \(Q,K\)，保留 \(V\)。

必须验证该设计是否：

- 保留任务需要的声学内容；
- 避免完全去除 envelope；
- 比对 \(Q,K,V\) 全部 residualize 更合理。


---

# 3. 工作原则

## 3.1 不得默认 idea 成立

必须同时进行：

- 支持性实验；
- 反例实验；
- 伪改进排查；
- 机制控制；
- 负结果记录；
- 明确停止条件。

## 3.2 不得伪造

所有文献、数据、下载链接、实验结果、命令输出必须真实。

若网络不可用，必须明确写：

> 当前环境无法完成在线文献核验。

若数据不可用，必须明确写：

> 未运行真实数据实验。

不得使用假数据冒充真实结果。

## 3.3 先诊断，后训练

严格按顺序：

1. novelty audit；
2. dataset audit；
3. zero-training diagnostic；
4. synthetic falsification；
5. CPU smoke test；
6. 单数据集最小实验；
7. 强基线；
8. 多数据集；
9. 表示分析；
10. 最终结论。

## 3.4 优先证伪

以下任何一项失败，都必须暂停扩展：

- residualized audio 与 EEG 无显著关联；
- TPA 不优于普通 residualization；
- shuffled \(C\) 与真实 \(C\) 效果相同；
- random subspace 与 acoustic subspace 效果相同；
- acoustically matched negatives 无提升；
- semantic/phonetic probing 无提升；
- only random-negative benchmark 提升；
- held-out subject/stimulus 无提升。

---

# 4. 第一阶段：文献查重与 Novelty Audit

## 4.1 检索范围

检索 2018 年至当前日期。

### A. EEG–Audio / Brain–Speech

```text
EEG audio alignment
EEG speech representation learning
EEG audio foundation model
EEG to speech
brain to speech
listened speech EEG
auditory EEG decoding
speech envelope tracking EEG
neural speech decoding
EEG audio contrastive learning
EEG speech retrieval
EEG speech semantic decoding
```

### B. TRF / mTRF / temporal encoding model

```text
temporal response function EEG speech
mTRF speech EEG
lagged regression EEG audio
dynamic TRF speech EEG
speech envelope EEG latency
auditory encoding model EEG
lag-resolved neural encoding speech
```

### C. Partial correlation / conditional dependence

```text
partial correlation attention
conditional mutual information multimodal
partial CCA multimodal
conditional contrastive learning
nuisance-conditioned representation learning
residualized contrastive learning
partialling out nuisance variables neural network
Frisch Waugh Lovell deep learning
```

### D. Orthogonal projection / nuisance removal

```text
orthogonal projection nuisance representation
nuisance attribute projection
remove speaker information representation
remove acoustic information embedding
orthogonal subspace projection multimodal
residual maker neural network
confound regression deep learning
```

### E. Attention-level conditioning

```text
attention residualization
partial attention
conditional attention nuisance
projected query key attention
orthogonal query key attention
deconfounded attention
causal attention confounder removal
covariate adjusted attention
```

### F. Brain–model alignment shortcut

```text
brain model alignment acoustic confounds
speech model EEG acoustic confound
neural encoding semantic beyond acoustics
speech brain alignment low-level acoustic features
variance partitioning speech EEG semantics
acoustic matched controls EEG language
```

## 4.2 最低文献数量

至少精读：

- 10 篇 TRF / mTRF / speech EEG；
- 10 篇 EEG–Audio / brain-to-speech；
- 10 篇 nuisance removal / orthogonal projection；
- 10 篇 conditional multimodal learning；
- 5 篇 attention deconfounding；
- 5 篇 brain–model variance partitioning。

## 4.3 Novelty Matrix

创建：

```text
reports/novelty_matrix.csv
```

字段：

```csv
paper_id,title,authors,year,venue,task,modalities,trf_or_lagged_design,partial_correlation,residualize_input,residualize_embedding,residualize_q,residualize_k,residualize_v,condition_attention,acoustic_covariates,brain_data,teacher_model,hard_negative_control,code_available,closest_overlap,novelty_risk,url,notes
```

## 4.4 必须回答的 novelty 问题

1. 是否已有论文显式使用：

\[
M_CQ,\quad M_CK
\]

后再计算 attention？

2. 是否已有 EEG–Audio 工作将 lag-expanded acoustic design matrix 作为 nuisance subspace？

3. 是否已有工作使用 Frisch–Waugh–Lovell residualization 近似条件跨模态依赖？

4. 是否已有脑—模型 alignment 论文证明控制 envelope 后仍存在 semantic alignment？

5. 是否已有方法使用 acoustically matched negatives 检验高级语言信息？

6. 该 idea 的核心贡献属于：

- 新问题；
- 新目标；
- 新 attention operator；
- 已有统计方法迁移；
- EEG–Audio 特定组合；
- 评测协议创新；
- 仅工程组合。

## 4.5 Novelty 结论等级

```text
A：核心算子与问题定义均未发现直接重叠
B：统计思想已有，但 attention 形式和 EEG–Audio 场景有实质新意
C：主要是 TRF + residualization + attention 的组合
D：已有直接方法，不建议继续
```

必须给证据链。


---

# 5. 第二阶段：理论审查

创建：

```text
reports/theory_review.md
```

## 5.1 TPA 是否近似条件互信息

分析：

\[
I(E;A\mid C)
\]

与 residualization 后相似性之间的关系。

必须说明：

- 在线性高斯条件下，partial correlation 与 conditional dependence 的关系；
- 非线性情况下不成立；
- attention score 不是 mutual information estimator；
- 只能声称“conditioning-inspired”或“partialled alignment”，除非有更强理论证明；
- 不得把普通线性 residualization 直接称为 causal adjustment。

## 5.2 Frisch–Waugh–Lovell 条件

分析 FWL 定理适用条件：

- 线性回归；
- 相同样本空间；
- covariate matrix rank；
- ridge regularization 后是否仍严格等价；
- batch-wise residualization 的偏差；
- mini-batch 内估计是否稳定。

## 5.3 时间维度与特征维度

必须明确 \(C\) 的 shape。

例如：

\[
Q_E\in\mathbb R^{B\times T_E\times d},
\]

\[
C_E\in\mathbb R^{B\times T_E\times p}.
\]

投影发生在时间轴。

检查：

- 计算复杂度；
- memory；
- variable-length mask；
- padding；
- rank deficiency；
- 短窗口下 \(p>T\)；
- batch 中每个样本独立 QR。

## 5.4 是否应该 residualize Q/K/V

分别分析：

### Q only

\[
Q_E^\perp=M_{C_E}Q_E.
\]

### K only

\[
K_A^\perp=M_{C_A}K_A.
\]

### Q + K

\[
Q_E^\perp,\quad K_A^\perp.
\]

### Q + K + V

\[
V_A^\perp=M_{C_A}V_A.
\]

必须给出：

- 信息学解释；
- 任务适配；
- 风险；
- 建议主版本。

## 5.5 线性 residualization 的局限

分析：

- audio teacher 可能非线性编码 envelope；
- EEG 中 envelope tracking 可能非线性；
- residualized representation 仍可能泄漏低级声学；
- 过度 residualization 可能删除 phonetic 信息；
- envelope、phoneme、semantics 不严格独立；
- pitch 与 speaker identity 混杂；
- onset 与 phoneme boundary 混杂。

## 5.6 低级声学变量不是纯 nuisance

必须区分任务：

### 更适合 TPA

- semantic retrieval；
- unseen-story retrieval；
- phoneme/word probing；
- cross-stimulus transfer；
- foundation pretraining；
- language-relevant EEG representation。

### 不一定适合 TPA

- speech envelope reconstruction；
- auditory attention decoding；
- speech tracking；
- low-level acoustic decoding。

不得在后者上强行要求提升。


---

# 6. 第三阶段：数据集审计

创建：

```text
reports/dataset_audit.md
```

## 6.1 候选数据集

至少审计：

- SparrKULee；
- ICASSP 2023 Auditory EEG Challenge；
- ICASSP 2024 Auditory EEG Challenge；
- KUL auditory attention dataset；
- DTU auditory attention dataset；
- ChineseEEG-2；
- UGR-MINDVOICE；
- BCI Competition 2020 Track 3；
- 其他同步 EEG–speech 数据集。

## 6.2 每个数据集记录

```markdown
- 数据集名称：
- 官方链接：
- License：
- 受试者数：
- 录音时长：
- EEG 通道：
- EEG 采样率：
- Audio 采样率：
- 是否有 transcript：
- 是否有 phoneme timestamp：
- 是否有 word timestamp：
- 是否有 speaker ID：
- 是否有 story ID：
- listened/overt/covert/imagined：
- 是否支持 held-out subject：
- 是否支持 held-out stimulus：
- 是否支持 acoustically matched negatives：
- 主要限制：
```

## 6.3 推荐实验优先级

### Tier 1

有同步 audio、subject split、story split、足够自然语音长度。

### Tier 2

有同步 audio，但 transcript 或 phoneme annotation 不完整。

### Tier 3

只有分类标签，不适合验证条件依赖。

---

# 7. 第四阶段：Zero-Training Diagnostic

这是整个项目的第一道核心 gate。

不得先训练 TPA。

## 7.1 Audio teacher 选择

至少选 3 类：

1. speech SSL：
   - wav2vec 2.0；
   - HuBERT；
   - WavLM。

2. ASR encoder：
   - Whisper encoder。

3. general audio：
   - BEATs；
   - AudioMAE；
   - CLAP。

记录：

- 参数量；
- 层数；
- stride；
- sampling rate；
- hidden dimension；
- 是否语义监督；
- 是否冻结。

## 7.2 Variance partitioning

对每个 audio layer \(H_A^m\)，拟合：

\[
H_A^m
=
C_{\mathrm{low}}B_{\mathrm{low}}
+
C_{\mathrm{phonetic}}B_{\mathrm{phonetic}}
+
C_{\mathrm{semantic}}B_{\mathrm{semantic}}
+
\epsilon.
\]

其中：

- \(C_{\mathrm{low}}\)：envelope、onset、pitch、energy；
- \(C_{\mathrm{phonetic}}\)：phoneme、syllable；
- \(C_{\mathrm{semantic}}\)：word embedding、sentence embedding。

输出：

- 各层 \(R^2\)；
- unique variance；
- shared variance；
- acoustic residual norm；
- residualized representation rank。

## 7.3 EEG–Audio alignment before/after residualization

比较：

\[
\operatorname{RSA}(E,H_A^m),
\]

\[
\operatorname{RSA}(E,M_C H_A^m),
\]

或者：

- CKA；
- CCA；
- cross-validated encoding score；
- ridge regression；
- partial correlation。

必须扫描：

- audio layer；
- EEG channel；
- EEG region；
- lag；
- time window；
- subject。

## 7.4 Gate 1

只有满足以下至少一个条件，才进入训练：

1. residualized audio representation 与 EEG 仍有显著对齐；
2. 控制低级声学后，部分 audio layer 仍能预测 EEG；
3. residualized alignment 与 phoneme/word/semantic label 有关联；
4. residualized alignment 在 held-out story 上稳定。

若所有条件均不成立：

```text
Decision: STOP
Reason: No measurable EEG–audio dependence remains after acoustic adjustment.
```


---

# 8. 第五阶段：合成数据验证

创建：

```text
reports/synthetic_validation.md
```

## 8.1 合成生成过程

构造：

\[
z_a=\text{low-level acoustic latent},
\]

\[
z_p=\text{phonetic latent},
\]

\[
z_s=\text{semantic latent}.
\]

Audio：

\[
A=g_A(z_a,z_p,z_s)+\epsilon_A.
\]

EEG：

\[
E=
g_E(z_a(t-\delta),z_p,z_s,u_{\text{subject}})
+\epsilon_E.
\]

其中：

- \(z_a\) 强；
- \(z_p,z_s\) 弱；
- subject nuisance 可调；
- lag 可调；
- SNR 可调。

## 8.2 合成条件

至少设置：

### Condition A：只有 acoustic dependence

\[
E\leftarrow z_a.
\]

预期：

- TPA 不应产生高级任务提升；
- residualization 后对齐接近零。

### Condition B：acoustic + semantic dependence

\[
E\leftarrow z_a+z_s.
\]

预期：

- TPA 能恢复 \(z_s\)；
- standard alignment 更偏 \(z_a\)。

### Condition C：semantic 极弱

测试 TPA 的检测边界。

### Condition D：nonlinear acoustic leakage

测试线性 residualization 是否不足。

### Condition E：错位 lag

测试 lag expansion 的必要性。

### Condition F：random nuisance subspace

测试一般投影正则化解释。

## 8.3 合成基线

至少比较：

1. standard attention；
2. standard contrastive alignment；
3. input residualization；
4. pooled embedding residualization；
5. loss-level partial correlation；
6. Q-only TPA；
7. K-only TPA；
8. Q+K TPA；
9. Q+K+V TPA；
10. random subspace projection；
11. shuffled acoustic design；
12. no lag expansion；
13. perfect oracle residualization；
14. nonlinear residualization upper bound。

## 8.4 合成指标

- acoustic latent probing；
- phonetic latent probing；
- semantic latent probing；
- subject leakage；
- match–mismatch；
- hard-negative retrieval；
- residualized alignment；
- calibration；
- seed variance。

## 8.5 Gate 2

必须满足：

1. 在 Condition B 中，TPA 提高 \(z_s\) 恢复；
2. 在 Condition A 中，不制造虚假高级信息；
3. shuffled \(C\) 明显失效；
4. random subspace 不等价；
5. Q+K 版本优于至少一个简单 residualization baseline；
6. no-lag 版本弱于 lag-expanded 版本。

否则：

```text
Decision: PIVOT or STOP
```

---

# 9. 第六阶段：代码仓库

创建：

```text
trf_partial_attention/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── data/
│   ├── models/
│   ├── experiments/
│   └── sweeps/
├── src/
│   ├── data/
│   ├── acoustics/
│   ├── trf/
│   ├── models/
│   ├── attention/
│   ├── losses/
│   ├── metrics/
│   ├── analysis/
│   ├── training/
│   └── utils/
├── scripts/
│   ├── prepare_data.py
│   ├── extract_acoustic_covariates.py
│   ├── extract_audio_teacher.py
│   ├── run_zero_training_diagnostic.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_smoke_test.sh
│   ├── run_ablation.sh
│   └── run_full_benchmark.sh
├── tests/
│   ├── test_acoustic_features.py
│   ├── test_lagged_design.py
│   ├── test_residual_maker.py
│   ├── test_partial_attention.py
│   ├── test_masking.py
│   ├── test_no_leakage.py
│   ├── test_determinism.py
│   └── test_gradients.py
├── reports/
└── outputs/
```


---

# 10. 工程要求

必须：

- Python 3.10+；
- PyTorch；
- 类型注解；
- structured config；
- 单 GPU；
- CPU smoke test；
- deterministic mode；
- mixed precision 可选；
- gradient clipping；
- NaN detection；
- logging；
- checkpoint；
- resumable training；
- wandb 可选但不得强依赖；
- 所有路径可配置；
- 不得静默 fallback；
- 不得自动更换数据或模型；
- 不得在 test set 调参。

---

# 11. 核心模块接口

## 11.1 Acoustic feature extractor

```python
class AcousticFeatureExtractor:
    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> dict[str, torch.Tensor]:
        ...
```

返回：

```python
{
    "envelope": ...,
    "onset": ...,
    "f0": ...,
    "energy": ...,
    "spectral_flux": ...,
}
```

## 11.2 Lagged design matrix

```python
class LaggedDesignBuilder(nn.Module):
    def forward(
        self,
        acoustic_features: torch.Tensor,
        lags_in_samples: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

## 11.3 Residual maker

```python
class ResidualMaker(nn.Module):
    def __init__(
        self,
        ridge: float = 1e-4,
        method: str = "qr",
    ):
        ...

    def forward(
        self,
        x: torch.Tensor,
        covariates: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
```

必须支持：

- QR；
- ridge solve；
- mask；
- batch；
- rank-deficient covariates；
- gradient flow；
- no-gradient covariates。

## 11.4 Partial attention

```python
class TRFPartialAttention(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        eeg_covariates: torch.Tensor,
        audio_covariates: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ...
```

---

# 12. 计算复杂度审查

标准 residual-maker：

\[
M_C=I-C(C^\top C+\lambda I)^{-1}C^\top
\]

若直接构造 \(T\times T\) 矩阵，复杂度和显存可能过高。

必须优先实现：

\[
X^\perp
=
X-C(C^\top C+\lambda I)^{-1}C^\top X
\]

而不显式构造 \(M_C\)。

复杂度：

\[
O(Tp^2+p^3+Tpd)
\]

而不是：

\[
O(T^2d).
\]

必须记录：

- T；
- p；
- d；
- wall-clock；
- 显存；
- 与标准 attention 的额外开销。


---

# 13. 第七阶段：真实任务

## 13.1 Primary Task：Acoustically Matched EEG–Audio Retrieval

给定 EEG 片段，从候选 audio 中检索正确片段。

负样本必须满足：

- envelope 相似；
- duration 相似；
- RMS energy 相似；
- pitch distribution 相似；
- preferably same speaker；
- transcript 内容不同。

指标：

- Recall@1；
- Recall@5；
- median rank；
- MRR；
- subject-averaged score；
- held-out story；
- held-out subject。

## 13.2 Secondary Task：Match–Mismatch

必须同时报告：

- random negatives；
- nearby negatives；
- acoustically matched negatives；
- same-speaker negatives；
- same-story negatives。

## 13.3 Probing Tasks

从 EEG representation 预测：

### 低级

- envelope；
- onset；
- pitch；
- energy。

### 中级

- phoneme；
- syllable；
- word boundary。

### 高级

- word identity；
- semantic category；
- sentence embedding；
- story identity；
- comprehension-related label。

### Nuisance

- subject ID；
- session；
- device；
- speaker ID。

## 13.4 Foundation Model Transfer

至少测试：

- frozen linear probe；
- LoRA；
- full fine-tuning；
- few-shot 1%、5%、10%、25%。

---

# 14. 强基线

必须比较：

1. no audio pretraining；
2. EEG masked modeling；
3. standard EEG–audio contrastive learning；
4. standard cross-attention；
5. lag-aware local attention；
6. input-level acoustic residualization；
7. pooled embedding residualization；
8. loss-level partial correlation；
9. adversarial acoustic removal；
10. Q-only TPA；
11. K-only TPA；
12. Q+K TPA；
13. Q+K+V TPA；
14. random subspace projection；
15. shuffled covariate projection；
16. no-lag projection；
17. oracle acoustic residualization；
18. matched parameter control。

---

# 15. 必须完成的消融

## 15.1 Acoustic feature set

比较：

- envelope only；
- envelope + onset；
- + pitch；
- + energy；
- + spectral flux；
- all features。

## 15.2 Lag range

比较：

- 0–200 ms；
- 0–300 ms；
- 0–500 ms；
- -100–500 ms；
- subject-specific；
- learned scalar lag；
- no lag。

## 15.3 Ridge strength

\[
\lambda\in
\{0,10^{-6},10^{-4},10^{-3},10^{-2},10^{-1}\}.
\]

## 15.4 Projection location

比较：

- EEG input；
- EEG hidden；
- Q；
- K；
- Q+K；
- pooled output；
- loss space。

## 15.5 Projection frequency

比较：

- 每层；
- 仅第一层；
- 仅 cross-modal block；
- 仅预训练 head；
- 训练时；
- 推理时保留或移除。

## 15.6 Audio teacher

至少 3 个：

- WavLM/HuBERT；
- Whisper；
- BEATs/CLAP。

## 15.7 EEG encoder

至少：

- EEGNet；
- EEGConformer；
- small Transformer；
- available EEG foundation model。

## 15.8 Data scale

- 1%；
- 5%；
- 10%；
- 25%；
- 50%；
- 100%。

## 15.9 Channel count

- full；
- 32；
- 16；
- 8；
- 4。


---

# 16. 伪改进排查

## 16.1 参数量

所有 baseline 使用 matched projector capacity。

## 16.2 训练计算

报告：

- FLOPs；
- wall-clock；
- GPU memory；
- number of trainable parameters。

## 16.3 Covariate quality

比较：

- true acoustic covariates；
- shuffled covariates；
- random Gaussian covariates；
- random orthogonal subspace；
- wrong audio segment covariates；
- speaker-only covariates。

## 16.4 多层监督

若 TPA 使用多个层，baseline 必须也得到同等层数的 teacher supervision。

## 16.5 Hard negative quality

必须记录 hard-negative matching quality：

- envelope distance；
- pitch distance；
- energy distance；
- duration difference；
- semantic difference。

## 16.6 任务难度

若 random-negative accuracy 接近天花板，不能作为主要结果。

---

# 17. 数据泄漏控制

必须自动检测：

- 同一 subject 跨 train/test；
- 相邻 epoch 跨 split；
- 同一 story 跨 split；
- 同一 audio segment 的 overlap；
- feature normalization 使用全数据；
- hard negative 来自 test set；
- teacher extraction cache 混用；
- train/test covariate statistics 混用；
- duplicate segments；
- window overlap leakage。

创建：

```text
tests/test_no_leakage.py
```

---

# 18. 统计计划

每个主实验：

\[
N_{\text{seed}}\ge5.
\]

主要推断单位：

- subject；
- 不得使用 window 作为独立样本进行显著性检验。

报告：

- mean；
- standard deviation；
- 95% CI；
- subject-level bootstrap；
- paired permutation；
- Wilcoxon signed-rank；
- effect size；
- multiple comparison correction。

预注册 primary endpoint：

\[
\text{held-out-subject Recall@1 on acoustically matched negatives}.
\]

Secondary endpoint：

\[
\text{held-out-story semantic probing score}.
\]


---

# 19. 机制假设

## H1：低级声学 shortcut 存在

标准 alignment 的表示能较好预测：

- envelope；
- onset；
- pitch；
- energy。

## H2：控制声学后仍存在 EEG–Audio dependence

Zero-training diagnostic 中：

\[
\operatorname{Align}(E,H_A^\perp)>0.
\]

## H3：TPA 降低 acoustic leakage

TPA representation 对 envelope 等的 probe score 下降，但高级 probe 不下降。

## H4：TPA 改善 acoustically matched retrieval

\[
\text{TPA}>\text{Standard}
\]

在 matched negatives 上成立。

## H5：TPA 提高高级信息

至少一个：

- phoneme；
- word；
- semantic；
- unseen-story retrieval。

## H6：真实 acoustic covariates 必须优于随机 covariates

\[
\text{True }C
>
\text{Shuffled }C
\approx
\text{Random }C.
\]

## H7：lag-expanded covariates 必须优于 no-lag

否则 TRF 部分缺乏必要性。

## H8：attention-level residualization 必须优于简单输入 residualization

否则不能将贡献定义为 attention architecture。

---

# 20. 表示分析

必须输出：

## 20.1 Acoustic leakage curve

各层对：

- envelope；
- pitch；
- onset；
- energy。

## 20.2 Linguistic probing curve

各层对：

- phoneme；
- word；
- semantics。

## 20.3 Subject leakage

各层 subject ID accuracy。

## 20.4 CKA / RSA

比较：

\[
\operatorname{CKA}(E,H_A)
\]

和：

\[
\operatorname{CKA}(E,H_A^\perp).
\]

## 20.5 Attention map

比较：

- standard attention；
- TPA；
- shuffled covariate；
- random subspace。

## 20.6 TRF coefficient visualization

显示：

- lag；
- channel；
- feature；
- subject average；
- confidence interval。

## 20.7 Residual energy

\[
\frac{\|M_CX\|_F^2}{\|X\|_F^2}.
\]

防止过度删除。


---

# 21. 失败条件

## Failure A：Novelty failure

发现已有直接 TRF-residualized EEG–Audio attention。

## Failure B：No residual dependence

控制低级声学后，EEG 与 audio teacher representation 无关联。

## Failure C：Synthetic failure

TPA 无法恢复已知高级潜变量。

## Failure D：Mechanism failure

shuffled/random covariates 与真实 covariates 同样有效。

## Failure E：Architecture failure

简单输入 residualization 与 TPA 等价。

## Failure F：Task failure

只在 random negatives 上提升。

## Failure G：Generalization failure

held-out subject 或 held-out story 无提升。

## Failure H：Information destruction

高级 probe 与低级 probe 一起下降。

## Failure I：Compute failure

额外训练开销过大，不再属于最小改动。

---

# 22. Reviewer Simulation

创建：

```text
reports/reviewer_critique.md
```

模拟三类审稿人。

## Reviewer A：Multimodal Representation Reviewer

攻击：

- 这是否只是 partial correlation；
- 是否只是 nuisance projection；
- 是否真正近似 conditional mutual information；
- attention 修改是否必要；
- 与 conditional contrastive learning 的差异。

## Reviewer B：EEG / Neuroscience Reviewer

攻击：

- envelope 是否真的是 nuisance；
- lag range 是否合理；
- subject-specific latency；
- TRF 线性假设；
- phoneme/semantic EEG SNR；
- volume conduction；
- 神经解释是否过度。

## Reviewer C：Experimental Reviewer

攻击：

- split；
- hard-negative construction；
- covariate leakage；
- teacher choice；
- parameter fairness；
- statistical unit；
- multiple comparisons；
- reproducibility。

每位 reviewer 输出：

- summary；
- strengths；
- weaknesses；
- questions；
- score 1–10；
- confidence 1–5；
- accept/reject tendency。

---

# 23. 输出文件

必须生成：

```text
reports/
├── literature_review.md
├── novelty_matrix.csv
├── novelty_audit.md
├── theory_review.md
├── dataset_audit.md
├── zero_training_diagnostic.md
├── synthetic_validation.md
├── smoke_test_report.md
├── benchmark_results.csv
├── ablation_results.csv
├── statistical_analysis.md
├── representation_analysis.md
├── failure_analysis.md
├── reviewer_critique.md
└── final_recommendation.md
```


---

# 24. final_recommendation.md 固定结构

```markdown
# Final Recommendation

## 1. Idea summary

## 2. Closest prior work

## 3. What is genuinely new

## 4. What is not new

## 5. Mathematical validity

## 6. Zero-training diagnostic

## 7. Synthetic validation

## 8. Real-data results

## 9. Strongest positive evidence

## 10. Strongest negative evidence

## 11. Alternative explanations

## 12. Key failure modes

## 13. Required additional experiments

## 14. Scores

| Dimension | Score / 6 | Evidence |
|---|---:|---|
| Novelty | | |
| Motivation | | |
| Minimality | | |
| Empirical strength | | |
| Mechanistic evidence | | |
| Venue potential | | |

## 15. Decision

只能选择：

- GO
- CONDITIONAL GO
- PIVOT
- STOP

## 16. Exact next steps
```

---

# 25. 评分标准

## Novelty

- 0：已有直接方法；
- 3：组合创新；
- 6：新问题与新算子。

## Motivation

- 0：无证据；
- 3：合理；
- 6：数据、TRF、shortcut 证据共同支持。

## Minimality

- 0：堆叠；
- 3：中等复杂；
- 6：局部 attention 改动。

## Empirical Strength

- 0：无提升；
- 3：单数据集；
- 6：多数据集、OOD、显著。

## Mechanistic Evidence

- 0：只有最终指标；
- 3：部分控制；
- 6：shuffled/random/matched-negative 全部通过。

## Venue Potential

分别评估：

- ICLR；
- NeurIPS；
- ICML；
- CVPR；
- ACL；
- Interspeech；
- ICASSP。

---

# 26. 执行顺序

## Phase A：文献和理论

1. novelty audit；
2. theory review；
3. dataset audit。

## Phase B：无训练诊断

1. audio layer variance partitioning；
2. EEG–audio partial alignment；
3. lag scan；
4. Gate 1。

## Phase C：合成验证

1. acoustic-only；
2. acoustic+semantic；
3. nonlinear leakage；
4. random subspace；
5. Gate 2。

## Phase D：最小实现

1. repository；
2. unit tests；
3. CPU smoke test；
4. single-GPU run。

## Phase E：真实任务

1. matched retrieval；
2. match–mismatch；
3. probing；
4. cross-subject；
5. cross-story。

## Phase F：机制实验

1. shuffled \(C\)；
2. random \(C\)；
3. no-lag；
4. simple residualization；
5. leakage analysis。

## Phase G：最终审计

1. statistics；
2. reviewer simulation；
3. GO/PIVOT/STOP。

---

# 27. 阶段状态模板

每完成一阶段，输出：

```markdown
## Stage status

- Completed:
- Failed:
- Missing:
- Key findings:
- Blocking issues:
- Decision:
```

---

# 28. 首轮资源预算

首轮不得超过：

- 1 张 24–40 GB GPU；
- EEG encoder 少于 50M trainable parameters；
- 每个主配置 5 seeds；
- smoke test 少于 30 分钟；
- 单配置少于 12 小时；
- 首轮主配置不超过 24 个；
- 不做 waveform generation；
- 不做大规模 multi-teacher ensemble。

---

# 29. 最终必须回答的问题

最终必须明确回答：

1. 控制低级声学后，EEG 与 audio representation 是否仍有关联？
2. 该关联是否包含 phonetic、lexical 或 semantic 信息？
3. TRF lag expansion 是否必要？
4. Q/K residualization 是否比简单输入 residualization更有效？
5. 真实 acoustic covariates 是否优于 shuffled/random covariates？
6. hard-negative retrieval 是否提升？
7. held-out subject 和 held-out story 是否提升？
8. 该方法是否只是 nuisance projection 的应用？
9. 新颖性是否足以支撑顶会？
10. 哪个实验可以直接推翻该 idea？

最终不得使用模糊结论。必须给出：

- 支持证据；
- 反对证据；
- 替代解释；
- 明确决策。

---

# 30. 最终核心判据

该项目只有在以下条件同时成立时，才可判定为 GO：

\[
\boxed{
\operatorname{Align}(E,A\mid C)>0
}
\]

并且：

\[
\boxed{
\text{TPA}>\text{Standard Alignment}
}
\]

在：

- acoustically matched negatives；
- held-out subjects；
- held-out stimuli；

上同时成立。

此外必须满足：

\[
\boxed{
\text{True acoustic }C
>
\text{Shuffled }C
}
\]

以及：

\[
\boxed{
\text{Attention-level residualization}
>
\text{Simple residualization}
}
\]

否则：

- 若条件依赖存在但架构无优势：PIVOT；
- 若条件依赖不存在：STOP；
- 若 novelty 高但证据不足：CONDITIONAL GO；
- 若多项机制和泛化均通过：GO。
