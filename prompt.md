# Codex 全流程研究验证 Prompt：Depth-Derivative EEG–Audio Foundation Model

> 使用方式：将本文件完整复制给 Codex。  
> 目标：要求 Codex 以“研究工程师 + 严格审稿人 + 复现实验负责人”的角色，对本研究 idea 进行从文献、理论、实现到实验的完整验证。  
> 核心要求：不得默认 idea 成立；必须主动寻找反例、已有重叠、不可辨识性、数据泄漏和伪改进来源。

---

## 0. 总体任务

你需要对下面的研究 idea 进行一次完整、可复现、可审稿的验证。

研究主题：

**Depth-Derivative EEG–Audio Pretraining**

核心思想：

现有 EEG–Audio 跨模态方法通常对齐两个 encoder 的完整隐藏状态：

\[
H_E^l \leftrightarrow H_A^m.
\]

但完整隐藏状态是逐层累积的：

\[
H^l = H^0 + \sum_{i=0}^{l-1}\Delta H^i.
\]

因此，高层表示中仍混合了低层声学信息、模态特异信息和早期处理历史。直接对齐完整 hidden states，可能要求两个模态的全部累计信息都相似，这对 EEG 与音频并不合理。

本研究改为对齐表示沿网络深度的变化：

\[
\Delta H_E^l = H_E^{l+1}-H_E^l,
\]

\[
\Delta H_A^m = H_A^{m+1}-H_A^m.
\]

目标是：

\[
\boxed{
H_E^{l+1}-H_E^l
\;\longleftrightarrow\;
H_A^{\phi(l)+1}-H_A^{\phi(l)}
}
\]

其中：

- \(H_E^l\)：EEG encoder 第 \(l\) 层输出；
- \(H_A^m\)：冻结 audio foundation model 第 \(m\) 层输出；
- \(\phi(l)\)：EEG 层到 audio 层的软深度映射；
- \(\phi\) 应大致满足单调性，但不能强制为严格一一对应。

核心范式表述：

> Align how EEG and audio representations evolve, rather than forcing their accumulated representations to coincide.

你需要验证：

1. 该 idea 是否真正新颖；
2. 数学和表示学习假设是否成立；
3. residual/depth derivative 是否真的比 hidden-state alignment 更适合 EEG–Audio；
4. 该方法是否能以极小改动带来稳定提升；
5. 提升是否来自真正的层级对齐，而不是参数量、额外监督、正则化或 teacher quality；
6. 是否存在比该方法更简单的等价解释；
7. 是否值得继续投入为 ICLR、NeurIPS、ICML 或 CVPR 级别工作。

---

# 1. 工作原则

你必须遵守以下原则。

## 1.1 不得默认方法有效

必须同时进行：

- 支持性验证；
- 反例搜索；
- 失败条件分析；
- 替代解释排查；
- 负结果记录。

不得为了完成任务而强行得到正面结论。

## 1.2 不得伪造文献、数据、结果或命令输出

所有文献必须记录：

- 标题；
- 作者；
- 年份；
- venue 或 arXiv 编号；
- 官方链接；
- 与本 idea 的具体重叠点；
- 是否真的阅读了正文或代码。

如果网络不可用，必须明确写出：

> 当前环境无法完成在线文献核验。

不得使用记忆补全不存在的论文。

## 1.3 先做最小可证伪实验，再扩大规模

必须按以下顺序推进：

1. 文献查重；
2. 理论与张量审查；
3. 合成数据验证；
4. 单数据集 smoke test；
5. 强基线比较；
6. 多数据集验证；
7. 跨受试者与跨设备测试；
8. 完整消融；
9. 统计检验；
10. 最终结论。

不得在 hypothesis 尚未通过 smoke test 时直接启动大规模训练。

## 1.4 所有实验必须可复现

必须固定并记录：

- random seed；
- 数据版本；
- subject split；
- stimulus split；
- preprocessing；
- sampling rate；
- reference；
- channels；
- window length；
- optimizer；
- learning rate；
- batch size；
- epoch；
- checkpoint；
- hardware；
- wall-clock time；
- software versions。

---

# 2. 第一阶段：文献查重与 novelty 审计

## 2.1 检索范围

检索 2020 年至当前日期的以下方向：

### A. EEG–Audio / Brain–Speech

关键词至少包括：

```text
EEG audio alignment
EEG speech foundation model
EEG to speech
brain to speech
neural speech decoding
listened speech EEG
imagined speech EEG
covert speech EEG
auditory attention decoding
EEG audio contrastive learning
EEG speech representation learning
EEG audio codec
speech neural decoding
brain audio latent alignment
```

### B. 跨模态层级对齐

```text
hierarchical multimodal alignment
layer-wise multimodal alignment
cross-modal layer alignment
multi-level representation alignment
hierarchy-aware contrastive learning
audio visual hierarchical alignment
vision language layer alignment
speech text hierarchical alignment
```

### C. residual / depth derivative / trajectory matching

```text
residual representation alignment
residual distillation
representation trajectory matching
hidden state difference matching
layer transition alignment
feature evolution alignment
depth derivative representation
neural ODE feature alignment
flow matching hidden states
teacher student residual matching
incremental representation distillation
```

### D. Matryoshka、ordered representation、nested embedding

```text
Matryoshka representation learning
nested representation learning
ordered representation
coarse to fine embedding
hierarchical embedding dimensions
```

### E. Speech model layer hierarchy 与脑区对应

```text
speech model layers brain hierarchy
audio model layers cortical hierarchy
HuBERT layer brain alignment
Whisper layer neural encoding
acoustic phonetic semantic hierarchy brain
speech representation cortical hierarchy
```

---

## 2.2 文献筛选规则

至少筛选并精读：

- 10 篇 EEG Foundation Model；
- 10 篇 EEG–Audio、brain-to-speech 或 auditory decoding；
- 10 篇跨模态层级对齐；
- 10 篇 residual、trajectory、layer transition 或 distillation；
- 5 篇 speech model layer 与脑层级对应研究。

优先级：

1. 顶会正式论文；
2. 顶刊；
3. 官方 proceedings；
4. 最新 arXiv；
5. 有公开代码的工作。

---

## 2.3 建立 novelty matrix

创建：

```text
reports/novelty_matrix.csv
```

字段：

```csv
paper_id,title,year,venue,task,modalities,align_full_hidden,align_layerwise,align_residual_updates,learn_layer_mapping,monotonic_constraint,multi_branch,extra_inference_cost,teacher_frozen,code_available,closest_overlap,novelty_risk,notes,url
```

必须回答：

1. 是否已有工作明确对齐：

\[
H^{l+1}-H^l
\]

而不是 \(H^l\)？

2. 是否已有工作使用 layer-transition matching、residual distillation 或 representation trajectory alignment？

3. 是否已有工作在 EEG–Audio 中学习单调或软单调的层映射？

4. 是否已有工作在其他模态中将网络深度解释为抽象层级轨迹？

5. 该 idea 的 novelty 属于：

- 新问题；
- 新范式；
- 新训练目标；
- 新架构；
- 现有方法迁移；
- 组合创新；
- 仅实现差异。

---

## 2.4 文献审计输出

创建：

```text
reports/literature_review.md
reports/novelty_audit.md
```

`novelty_audit.md` 必须包含：

### 最接近的 10 篇工作

每篇按以下格式：

```markdown
## Paper X

- 标题：
- 年份与 venue：
- 官方链接：
- 任务：
- 核心方法：
- 与本 idea 的重叠：
- 与本 idea 的关键区别：
- novelty 风险：
- 是否需要在论文中主动讨论：
```

### 最终 novelty 判断

使用以下等级：

```text
A：当前未发现直接重叠，核心问题和方法定义均有明显新意
B：存在相近思想，但应用对象、数学定义或验证方式有实质区别
C：主要是已有方法迁移，需要额外理论或任务贡献
D：与已有方法高度重叠，不建议继续
```

不得只给等级，必须给出证据链。

---

# 3. 第二阶段：理论与数学可行性审查

## 3.1 明确 residual 的定义

验证以下几种定义是否等价：

### 定义 A：层输出差分

\[
R^l = H^{l+1}-H^l.
\]

### 定义 B：Transformer block residual branch 输出

若：

\[
H^{l+1}=H^l+F_l(H^l),
\]

则：

\[
R^l=F_l(H^l).
\]

### 定义 C：attention 与 FFN 分别差分

\[
R_{\mathrm{attn}}^l
=
H_{\mathrm{post-attn}}^l-H_{\mathrm{pre-attn}}^l,
\]

\[
R_{\mathrm{ffn}}^l
=
H_{\mathrm{post-ffn}}^l-H_{\mathrm{post-attn}}^l.
\]

### 定义 D：归一化后的差分

\[
\widetilde R^l
=
\operatorname{LN}(H^{l+1})
-
\operatorname{LN}(H^l).
\]

检查：

- pre-LN 与 post-LN Transformer 下的差异；
- residual magnitude 随层深变化；
- 是否存在尺度爆炸；
- 是否需要 normalization；
- 是否只是近似等于 block output；
- 对卷积、SSM、Mamba、Conformer 是否仍适用。

---

## 3.2 可辨识性问题

分析：

1. 由于 hidden representation 对任意可逆线性变换不唯一，直接比较 residual 是否有意义？
2. 若 audio encoder 与 EEG encoder 的 feature basis 不同，cosine alignment 是否合理？
3. projection head 是否会吸收全部差异，使 residual alignment 退化成普通 distillation？
4. 是否需要：

\[
R_E^l W_E
\approx
R_A^m W_A
\]

而不是直接比较？
5. 是否应该比较：

- direction；
- subspace；
- covariance；
- pairwise geometry；
- CKA；
- cross-covariance；
- canonical correlations？

---

## 3.3 深度映射 \(\phi\) 的定义

比较以下方案。

### 方案 1：固定线性比例

\[
\phi(l)=\left\lfloor \frac{lM}{L}\right\rfloor.
\]

优点：

- 无额外参数；
- 易复现。

缺点：

- 假设抽象速度一致。

### 方案 2：每层一个可学习标量

\[
\mu_l\in[0,M-1].
\]

通过线性插值：

\[
\widetilde R_A^l
=
(1-\alpha_l)R_A^{\lfloor \mu_l\rfloor}
+
\alpha_lR_A^{\lceil\mu_l\rceil}.
\]

### 方案 3：softmax 深度权重

\[
w_{lm}
=
\operatorname{softmax}_m(a_{lm}),
\]

\[
\widetilde R_A^l
=
\sum_m w_{lm}R_A^m.
\]

### 方案 4：Sinkhorn / optimal transport

学习 EEG 层与 audio 层之间的软耦合矩阵。

### 方案 5：单调参数化

例如：

\[
\mu_l=
M\cdot
\frac{
\sum_{i\le l}\operatorname{softplus}(u_i)
}{
\sum_{i\le L}\operatorname{softplus}(u_i)
}.
\]

该参数化天然保证：

\[
\mu_{l+1}\ge \mu_l.
\]

必须比较：

- 参数量；
- 稳定性；
- 可解释性；
- 是否容易 collapse；
- 是否真的需要单调性；
- 是否过度约束真实脑处理中的反馈和并行路径。

---

## 3.4 理论风险列表

至少分析以下风险：

1. residual update 并不等于“新增语义信息”；
2. residual 可能主要反映优化尺度；
3. 不同层 residual norm 不可比较；
4. audio teacher 层级未必严格 acoustic→semantic；
5. EEG encoder 深度未必对应生理处理层级；
6. 单调 mapping 可能错误；
7. 同一 EEG 层可能对应多个 audio 层；
8. 同一 audio 层可能被多个 EEG 层重复匹配；
9. projection head 可能成为主要容量来源；
10. loss 可能迫使 EEG encoder 模仿 audio teacher 的模态偏差。

创建：

```text
reports/theory_review.md
```

其中必须包含：

- 支持该 idea 的理论理由；
- 反对该 idea 的理论理由；
- 最低成立条件；
- 明确失败条件；
- 可检验预测。

---

# 4. 第三阶段：代码仓库与工程设计

创建一个完整研究仓库：

```text
depth_derivative_eeg_audio/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── data/
│   ├── model/
│   ├── experiment/
│   └── sweeps/
├── src/
│   ├── data/
│   ├── models/
│   ├── losses/
│   ├── metrics/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── extract_audio_teacher.py
│   ├── run_smoke_test.sh
│   ├── run_ablation.sh
│   └── run_full_benchmark.sh
├── tests/
│   ├── test_shapes.py
│   ├── test_residual_extraction.py
│   ├── test_monotonic_mapping.py
│   ├── test_loss.py
│   ├── test_no_leakage.py
│   └── test_determinism.py
├── notebooks/
├── reports/
└── outputs/
```

---

## 4.1 代码要求

必须：

- Python 3.10+；
- PyTorch；
- 类型注解；
- dataclass 或 structured config；
- 清晰错误信息；
- 日志；
- checkpoint；
- mixed precision 可选；
- 单 GPU 可运行；
- CPU smoke test 可运行；
- seed 可控；
- 所有路径可配置；
- 不得把本地绝对路径写死；
- 不得 silently fallback；
- 遇到缺失数据必须报错；
- 遇到 NaN 必须中止并记录。

---

## 4.2 实现统一接口

EEG encoder：

```python
class EEGEncoder(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        channel_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        ...
```

Audio teacher：

```python
class AudioTeacher(nn.Module):
    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = True,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        ...
```

Depth derivative extractor：

```python
class DepthDerivativeExtractor(nn.Module):
    def forward(
        self,
        hidden_states: list[torch.Tensor],
        normalize: bool = True,
    ) -> list[torch.Tensor]:
        ...
```

Layer mapper：

```python
class MonotonicDepthMapper(nn.Module):
    def forward(
        self,
        eeg_residuals: list[torch.Tensor],
        audio_residuals: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        ...
```

---

# 5. 第四阶段：合成数据验证

正式 EEG 实验前，必须先构建合成任务验证核心机制。

## 5.1 合成层级生成过程

构造三个潜变量：

\[
z_1=\text{low-level acoustic},
\]

\[
z_2=\text{phonetic abstraction},
\]

\[
z_3=\text{semantic abstraction}.
\]

生成 audio：

\[
A=g_A(z_1,z_2,z_3)+\epsilon_A.
\]

生成 EEG：

\[
E=g_E(z_1,z_2,z_3,s,o,\delta)+\epsilon_E,
\]

其中：

- \(s\)：subject nuisance；
- \(o\)：operator nuisance；
- \(\delta\)：时间延迟；
- \(\epsilon\)：噪声。

设计 teacher encoder，使：

- 浅层主要恢复 \(z_1\)；
- 中层主要恢复 \(z_2\)；
- 深层主要恢复 \(z_3\)。

---

## 5.2 合成实验对比

至少比较：

1. final-state alignment；
2. all-hidden-state alignment；
3. fixed layer-wise alignment；
4. Matryoshka alignment；
5. residual/depth-derivative alignment；
6. residual alignment + monotonic mapping；
7. random teacher layers；
8. shuffled residual order；
9. no audio supervision。

评测：

- \(z_1,z_2,z_3\) probing accuracy；
- nuisance leakage；
- learned \(\phi\) 与真实层级关系；
- sample efficiency；
- noise robustness；
- delay robustness；
- residual alignment 是否只在有层级结构时有效。

---

## 5.3 合成实验的关键反例

必须构建以下反例：

### 反例 A：无层级数据

所有层表示同一潜变量。

若 DDA 仍显著提升，说明可能只是正则化。

### 反例 B：非单调层级

高层信息在早期出现，低层信息在后期再次出现。

检验单调约束是否有害。

### 反例 C：并行层级

两个潜变量同时在多个层出现。

检验一对一深度映射是否错误。

### 反例 D：teacher 层级错误

打乱 audio teacher 层序。

若性能不下降，说明“层级”解释不成立。

输出：

```text
reports/synthetic_validation.md
```

---

# 6. 第五阶段：真实数据与 benchmark 选择

优先选择公开、可下载、同步 EEG–Audio 数据。

候选包括：

- SparrKULee；
- KUL auditory attention；
- DTU auditory attention；
- ICASSP 2023 Auditory EEG Challenge；
- ICASSP 2024 Auditory EEG Challenge；
- UGR-MINDVOICE；
- BCI Competition 2020 Track 3；
- ChineseEEG-2；
- 其他有同步 speech/audio 与 EEG 的公开数据。

不得默认数据许可允许再分发。必须记录：

- 官方来源；
- license；
- 下载方式；
- 数据规模；
- 受试者数量；
- 通道数；
- 采样率；
- 是否同步 audio；
- 是否有 overt/covert/listened/imagined 条件；
- 是否允许商业或学术使用。

创建：

```text
reports/dataset_audit.md
```

---

## 6.1 第一阶段推荐任务：match–mismatch

输入：

\[
(E,A^+,A^-_1,\ldots,A^-_K).
\]

模型计算：

\[
s_k=\operatorname{sim}(f_E(E),f_A(A_k)).
\]

输出：

\[
\hat k=\arg\max_k s_k.
\]

优先使用该任务的原因：

- 不需要复杂生成器；
- 可以直接测试 EEG–Audio 表征；
- 不容易被 speech prior 掩盖；
- 适合进行 unseen-subject 和 unseen-stimulus 评测。

---

## 6.2 第二阶段任务：speech envelope reconstruction

预测：

\[
\hat a(t)=g(E).
\]

指标：

- Pearson correlation；
- subject-averaged correlation；
- unseen-subject performance；
- bootstrap 95% CI。

---

## 6.3 第三阶段任务：audio retrieval

给定 EEG，从候选音频库中检索对应音频。

指标：

- Recall@1；
- Recall@5；
- Recall@10；
- median rank；
- subject-wise retrieval；
- stimulus-wise retrieval。

---

## 6.4 第四阶段任务：phoneme / word / semantic probing

使用 EEG encoder 不同层输出，训练轻量 probe：

- envelope；
- phoneme；
- syllable；
- word identity；
- semantic category；
- speaker identity；
- subject identity。

这一步用于验证层级结构，而不仅是最终性能。

---

# 7. 第六阶段：模型配置

## 7.1 EEG encoder 基线

至少实现或调用：

1. EEGNet；
2. ShallowFBCSPNet；
3. EEGConformer；
4. 小型 Transformer；
5. 可用时加入 LaBraM、CBraMod、REVE 或其他公开 EEG-FM。

必须区分：

- random initialization；
- generic EEG pretraining；
- proposed EEG–Audio pretraining。

---

## 7.2 Audio teacher 候选

至少比较：

- wav2vec 2.0；
- HuBERT；
- WavLM；
- Whisper encoder；
- BEATs；
- CLAP；
- AudioMAE；
- 其他可公开调用模型。

不得只选择对 proposed method 最有利的 teacher。

对于每个 teacher，记录：

- 参数量；
- 层数；
-输入形式；
- temporal stride；
- 是否偏 speech；
- 是否含语义监督；
- 是否冻结；
- hidden state 维度。

---

# 8. 第七阶段：核心方法实现

## 8.1 Baseline A：Final-State Alignment

\[
\mathcal L_{\mathrm{final}}
=
d(P_E(H_E^L),P_A(H_A^M)).
\]

---

## 8.2 Baseline B：Layer-Wise Hidden Alignment

\[
\mathcal L_{\mathrm{hidden}}
=
\frac{1}{L}
\sum_l
d(P_E(H_E^l),P_A(H_A^{\phi(l)})).
\]

---

## 8.3 Proposed：Depth-Derivative Alignment

\[
R_E^l=H_E^{l+1}-H_E^l,
\]

\[
R_A^m=H_A^{m+1}-H_A^m.
\]

\[
\mathcal L_{\mathrm{DDA}}
=
\frac{1}{L}
\sum_l
d(P_E(R_E^l),P_A(\widetilde R_A^l)).
\]

---

## 8.4 总目标

\[
\mathcal L
=
\mathcal L_{\mathrm{task}}
+
\lambda_f\mathcal L_{\mathrm{final}}
+
\lambda_d\mathcal L_{\mathrm{DDA}}
+
\lambda_o\mathcal L_{\mathrm{order}}.
\]

必须进行：

- \(\lambda_f\) sweep；
- \(\lambda_d\) sweep；
- \(\lambda_o\) sweep；
- 单独 DDA；
- final + DDA；
- hidden + DDA；
- 无 task loss 的纯预训练；
- 下游 linear probe；
- full fine-tuning。

---

# 9. 第八阶段：强基线与公平性

所有方法必须满足：

- 相同 EEG encoder；
- 相同 audio teacher；
- 相同 preprocessing；
- 相同训练数据；
- 相同 batch size；
- 相同 epoch；
- 相同 optimizer；
- 相同 random seeds；
- projection head 参数量尽量匹配；
- 训练 FLOPs 尽量接近。

至少比较：

1. no pretraining；
2. masked EEG modeling；
3. final-state contrastive alignment；
4. final-state regression；
5. all-layer hidden alignment；
6. random layer alignment；
7. Matryoshka representation；
8. multi-branch acoustic/phonetic/semantic；
9. DDA fixed mapping；
10. DDA learned mapping；
11. DDA monotonic mapping；
12. shuffled DDA；
13. norm-only matching；
14. gradient matching；
15. CKA-based layer matching；
16. optimal-transport layer matching。

---

# 10. 第九阶段：必须完成的消融实验

## 10.1 residual 定义消融

比较：

- raw difference；
- pre-LN difference；
- post-LN difference；
- normalized difference；
- block output；
- attention branch only；
- FFN branch only；
- direction only；
- magnitude only。

---

## 10.2 layer mapping 消融

比较：

- fixed linear；
- learnable scalar；
- monotonic scalar；
- softmax over all layers；
- Sinkhorn；
- no order constraint；
- reversed order；
- random order；
- uniform average。

---

## 10.3 teacher 消融

比较至少 3 个 audio teacher：

- speech SSL；
- audio SSL；
- speech recognition encoder。

确认提升不是某个 teacher 特例。

---

## 10.4 模型深度消融

EEG encoder 层数：

- 4；
- 6；
- 8；
- 12。

检验 DDA 是否只在深网络有效。

---

## 10.5 数据规模消融

使用：

- 1%；
- 5%；
- 10%；
- 25%；
- 50%；
- 100%。

绘制 sample-efficiency curve。

---

## 10.6 EEG 通道消融

使用：

- 全通道；
- 32 通道；
- 16 通道；
- 8 通道；
- 4 通道。

检验是否适用于低通道设置。

---

## 10.7 时间窗口消融

使用：

- 0.5 s；
- 1 s；
- 3 s；
- 5 s；
- 10 s。

---

## 10.8 时间延迟消融

人为加入：

\[
\delta\in\{-500,-250,0,250,500\}\text{ ms}.
\]

检验方法是否对同步误差敏感。

---

# 11. 第十阶段：排查伪改进

必须进行以下控制。

## 11.1 参数量控制

确保 proposed method 的 projection heads 不比 baseline 明显更大。

## 11.2 额外监督控制

若 proposed 使用多层 teacher，而 baseline 只用最后一层，则需要增加：

- multi-layer hidden alignment baseline；
- multi-layer feature averaging baseline。

否则无法证明改进来自 residual，而不是更多 teacher supervision。

## 11.3 normalization 控制

加入：

- normalized hidden-state baseline；
- hidden-state difference + random mapping；
- residual norm regression。

确认提升不是 LayerNorm 或尺度修正造成。

## 11.4 teacher bypass 控制

对 audio teacher residual：

- 打乱层顺序；
- 跨样本打乱；
- 时间打乱；
- 使用随机冻结 teacher。

## 11.5 数据泄漏控制

必须检查：

- 同一受试者是否跨 train/test；
- 相邻时间窗是否跨 split；
- 同一音频 stimulus 是否跨 split；
- 同一故事不同片段是否跨 split；
- 标准化是否在全数据拟合；
- 预训练数据是否包含下游测试受试者。

创建自动化测试：

```text
tests/test_no_leakage.py
```

---

# 12. 第十一阶段：统计检验

所有主要实验至少运行：

\[
N_{\mathrm{seed}}\ge 5.
\]

报告：

- mean；
- standard deviation；
- 95% confidence interval；
- per-subject score；
- subject-level bootstrap；
- paired test。

优先使用：

- paired permutation test；
- Wilcoxon signed-rank；
- subject-level bootstrap。

不得把大量 epoch/window 当成独立样本进行显著性检验。

多重比较时使用：

- Holm correction；
- Benjamini–Hochberg。

必须报告 effect size，不得只报告 \(p\)-value。

---

# 13. 第十二阶段：核心假设验证

该论文不能只证明最终 accuracy 上升。

必须直接验证以下假设。

## H1：audio encoder 的 residual updates 具有层级结构

测试不同 audio layers 对：

- envelope；
- pitch；
- phoneme；
- word；
- semantics；
- speaker identity；

的 probing performance。

预期：

- 浅层 residual 更偏声学；
- 中层 residual 更偏音素；
- 深层 residual 更偏语义。

若不存在该趋势，DDA 的核心动机受损。

---

## H2：EEG encoder 在 DDA 训练后形成相似层级

对 EEG 各层和 residual update 做同样 probing。

预期：

\[
\text{acoustic}
\rightarrow
\text{phonetic}
\rightarrow
\text{semantic}.
\]

---

## H3：对齐 residual 比对齐 full hidden state 更能减少模态特异历史干扰

比较：

- subject identity leakage；
- device leakage；
- speaker leakage；
- audio-only nuisance leakage。

---

## H4：学习到的深度映射具有稳定性

检查：

- 不同 seeds；
- 不同 datasets；
- 不同 teachers；
- 不同 EEG encoders。

绘制：

\[
l\mapsto\phi(l).
\]

如果 mapping 在不同运行间完全不稳定，则解释性不足。

---

## H5：性能提升依赖正确的层级顺序

将 audio residual 层序打乱后，性能应明显下降。

如果不下降，则 proposed method 可能只是一般多层正则。

---

# 14. 第十三阶段：表示分析

必须生成以下分析。

## 14.1 CKA layer similarity matrix

绘制：

\[
\operatorname{CKA}(H_E^l,H_A^m)
\]

以及：

\[
\operatorname{CKA}(R_E^l,R_A^m).
\]

比较训练前后。

## 14.2 RSA

比较 EEG 与 audio 表征的 representational similarity matrix。

## 14.3 Probing curves

横轴为 layer，纵轴为：

- acoustic score；
- phonetic score；
- semantic score；
- subject leakage；
- speaker leakage。

## 14.4 Residual norm

绘制每层：

\[
\|R_E^l\|_2,
\qquad
\|R_A^m\|_2.
\]

防止某些层因 norm 较大主导 loss。

## 14.5 Mapping visualization

绘制 soft layer mapping matrix：

\[
W\in\mathbb R^{L\times M}.
\]

## 14.6 Temporal alignment

如数据允许，分析不同 EEG latency 下的匹配。

---

# 15. 第十四阶段：失败判据

出现以下任一情况时，必须暂停扩大实验，并明确标记风险。

## 15.1 Novelty failure

发现已有论文直接提出：

- residual layer transition alignment；
- EEG–Audio depth derivative matching；
- 相同单调 layer mapping；
- 相同核心公式与动机。

## 15.2 Synthetic failure

DDA 在存在真实层级结构的合成任务中不优于 final-state 或 hidden-state alignment。

## 15.3 Real-data failure

在至少两个真实数据集上：

- 平均提升小于 1 个百分点；
- 置信区间大量重叠；
- seed 方差大于提升；
- 仅单一 teacher 有效；
- 仅单一 split 有效。

## 15.4 Explanation failure

打乱 audio layer order 后性能不下降。

## 15.5 Control failure

加入参数量和多层监督控制后，DDA 优势消失。

## 15.6 Generalization failure

只在 within-subject 有效，在 unseen-subject 无效。

---

# 16. 第十五阶段：最终评价标准

最终给出以下六项评分，每项 0–6 分。

## 16.1 Novelty

- 0：已有直接工作；
- 3：已有相近方法迁移；
- 6：问题定义与方法均明显新颖。

## 16.2 Motivation

- 0：无证据；
- 3：直觉合理；
- 6：文献、神经科学和实验均支持。

## 16.3 Minimality

- 0：大量组件堆叠；
- 3：中等复杂度；
- 6：极小修改，推理无额外成本。

## 16.4 Empirical strength

- 0：无提升；
- 3：单数据集有效；
- 6：多数据集、多 split、统计显著。

## 16.5 Mechanistic evidence

- 0：只有最终性能；
- 3：部分 probing；
- 6：层级、映射、打乱和反事实证据完整。

## 16.6 Venue potential

分别评价：

- ICLR；
- NeurIPS；
- ICML；
- CVPR；
- ACL；
- Interspeech；
- ICASSP。

说明适配原因。

---

# 17. 最终输出文件

必须生成：

```text
reports/
├── literature_review.md
├── novelty_matrix.csv
├── novelty_audit.md
├── theory_review.md
├── dataset_audit.md
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

# 18. final_recommendation.md 的固定结构

```markdown
# Final Recommendation

## 1. Idea summary

## 2. Closest prior work

## 3. What is genuinely new

## 4. What is not new

## 5. Mathematical validity

## 6. Synthetic validation

## 7. Real-data results

## 8. Strongest positive evidence

## 9. Strongest negative evidence

## 10. Alternative explanations

## 11. Key failure modes

## 12. Required additional experiments

## 13. Scores

| Dimension | Score / 6 | Evidence |
|---|---:|---|
| Novelty | | |
| Motivation | | |
| Minimality | | |
| Empirical strength | | |
| Mechanistic evidence | | |
| Venue potential | | |

## 14. Decision

只能从以下选择一个：

- GO：可以进入完整论文阶段
- CONDITIONAL GO：需要完成指定关键实验
- PIVOT：核心问题有价值，但方法需要替换
- STOP：不建议继续

## 15. Exact next steps

按优先级给出不超过 10 项。
```

---

# 19. reviewer_critique.md 的固定要求

你需要模拟至少三种审稿人。

## Reviewer A：方法审稿人

重点攻击：

- 是否只是 residual distillation；
- 是否只是 multi-layer supervision；
- 是否只是 Matryoshka 或 layer matching 的变体；
- 单调 mapping 是否合理；
- 新增参数是否真实很少。

## Reviewer B：EEG / neuroscience 审稿人

重点攻击：

- network depth 是否能代表脑处理层级；
- EEG 是否真的包含 phonetic/semantic 信息；
- volume conduction；
- subject variability；
- temporal latency；
- preprocessing leakage；
- 生理解释是否过度。

## Reviewer C：实验审稿人

重点攻击：

- split；
- 数据泄漏；
- baseline 是否公平；
- teacher 是否选择性；
- 统计显著性；
- 多数据集泛化；
- 算力与参数量；
- 复现性。

每位 reviewer 给出：

- summary；
- strengths；
- weaknesses；
- questions；
- score 1–10；
- confidence 1–5；
- accept/reject 倾向。

---

# 20. 执行顺序

必须严格按以下顺序执行。

## Phase A：静态审查

1. 文献查重；
2. novelty matrix；
3. 理论风险；
4. 数据许可；
5. 实验设计。

## Phase B：最小实现

1. 建仓库；
2. 单元测试；
3. 合成数据；
4. CPU smoke test；
5. 单 GPU 训练。

## Phase C：核心比较

1. no pretraining；
2. final-state alignment；
3. hidden-state alignment；
4. DDA fixed；
5. DDA learned；
6. DDA monotonic。

## Phase D：机制验证

1. layer probing；
2. CKA；
3. shuffled layer order；
4. subject leakage；
5. teacher ablation。

## Phase E：扩展验证

1. 多数据集；
2. cross-subject；
3. cross-stimulus；
4. channel reduction；
5. data efficiency。

## Phase F：最终审计

1. 统计检验；
2. failure analysis；
3. reviewer simulation；
4. GO / PIVOT / STOP。

---

# 21. 每个阶段的停止与汇报机制

每完成一个阶段，必须输出：

```markdown
## Stage status

- Completed:
- Failed:
- Missing:
- Key findings:
- Blocking issues:
- Decision:
```

若遇到阻塞：

- 不得伪造结果；
- 不得跳过问题；
- 记录缺失依赖；
- 给出可执行修复方案；
- 对当前可得证据做阶段性判断。

---

# 22. 最小可运行实验要求

即使没有完整真实数据，也必须完成以下最小闭环：

1. 构造合成 EEG–Audio 数据；
2. 实现 4 层 EEG Transformer；
3. 实现 6 层固定 audio teacher；
4. 实现 final-state alignment；
5. 实现 hidden-state alignment；
6. 实现 DDA fixed mapping；
7. 实现 DDA monotonic mapping；
8. 训练至少 5 个 seeds；
9. 输出 probing、mapping 和统计结果；
10. 判断核心假设是否在可控环境成立。

---

# 23. 推荐的首轮资源预算

首轮实验不得超过：

- 单卡 24–40 GB GPU；
- 每个模型少于 50M trainable parameters；
- 每个主配置 5 seeds；
- smoke test 低于 30 分钟；
- 单个完整配置低于 12 小时；
- 首轮总配置不超过 20 个。

先验证方向，不追求大模型规模。

---

# 24. 最终目标

本任务的最终目标不是“实现一个能跑的模型”，而是回答以下问题：

\[
\boxed{
\text{Does aligning representation evolution provide a better inductive bias}
}
\]

\[
\boxed{
\text{for EEG--audio learning than aligning cumulative hidden states?}
}
\]

最终必须清楚回答：

1. Depth derivative 是否有真实的信息层级意义？
2. 是否优于完整 hidden-state alignment？
3. 是否优于多层监督带来的普通正则化？
4. 是否在 unseen-subject 和 unseen-stimulus 下成立？
5. 是否能用极少参数实现？
6. 是否足够新颖？
7. 是否值得投稿顶会？
8. 哪些实验会直接推翻该 idea？

不得使用模糊表述。必须给出证据、反例和明确决策。
