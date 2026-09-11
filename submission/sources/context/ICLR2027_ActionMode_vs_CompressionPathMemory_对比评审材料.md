# ICLR 2027 候选课题对比评审材料

## Action-Mode Spectrum vs. Compression-Path Memory

**版本日期：2026-09-10**  
**用途：第三方独立评审与路线选择**  
**原则：不预设优胜者；区分“科学方案本身”与“当前完成度”；对未验证结论一律写为假设。**

---

## 0. 文档用途与比较边界

本材料比较两个候选 ICLR 2027 研究方向：

- **方案 A：Action-Mode Spectrum**：研究结构化表示中，同一个几何变换在不同构件支持集（support / coalition）上发生时，模型是否赋予了错误或不充分的语义；进一步研究这种响应如何由局部表示几何形成，以及能否通过任务条件化的表示路由加以控制。
- **方案 B：Compression-Path Memory**：研究 LLM/Agent 在固定最终记忆预算下，最终记忆质量是否不仅取决于压缩率，还取决于“经历了多少次、以什么路径压缩到该预算”；进一步区分完整长上下文中的**信息可访问性问题**与多轮有损压缩中的**不可逆信息损失问题**。

### 0.1 为避免不公平比较，必须注意两点

**第一，两个方案当前成熟度不对称。** Action-Mode 已经完成相当数量的诊断实验并获得部分机制证据；Compression-Path Memory 目前仍处于方案设计阶段，没有 pilot 结果。因此，第三方应分别评价：

1. 如果两个方案都从零开始，它们各自的科学问题、创新空间和理论/实验价值如何；
2. 截至 2026-09-10，哪一条路线更接近形成可投稿的证据闭环。

**第二，本文不把“已经投入更多工作”当作科学优势，也不把“尚未实验”当作概念劣势。** 当前证据成熟度会单独列出。

### 0.2 ICLR 评审参照

ICLR 2027 Reviewer Guide 的核心判断可压缩为四个问题：

1. 论文具体解决什么问题？
2. 方法/分析是否有充分动机并正确放置在既有文献中？
3. 证据是否真正支持主要 claim，且科学上严谨？
4. 是否为社区带来新的、重要的知识；这并不要求一定取得 SOTA？

因此，本材料重点比较：**问题重要性、创新来源、与近邻工作的区分度、可证伪性、证据门槛、方法必要性、跨域一般性、失败风险和当前可执行性**。

---

# 1. 一页式并列概览

| 维度 | 方案 A：Action-Mode Spectrum | 方案 B：Compression-Path Memory |
|---|---|---|
| 研究对象 | 结构化视觉/图表示中的变换语义与局部表示几何 | LLM/Agent 长期记忆、上下文压缩与多轮 compaction |
| 核心问题 | **同一个 operator 的语义是否取决于它作用于谁、作用在哪、服务什么任务？** | **在最终预算相同的情况下，memory quality 是否依赖到达该预算的 compression path？** |
| 关键自变量 | operator × support/coalition × task；图谱模式与节点支持 | final budget/rate × compression depth × intermediate schedule |
| 主要因变量 | embedding response、局部 Jacobian 几何、任务性能、nuisance robustness | 下游任务效用、事实/约束保留、更新正确性、provenance、压缩错误类型 |
| 理论起点 | invariance/equivariance、group action、graph spectral modes、local representation geometry | data-processing inequality、rate-distortion / information bottleneck、bounded memory |
| 最强潜在发现 | 变换本身不是 nuisance/semantic 的充分描述；support-conditioned geometry 才决定模型响应 | 同样的最终 token budget 可能对应显著不同的有效记忆；memory 是 path-dependent 的有损状态 |
| 论文形态 | 诊断 + 机制 +（若成功）新表示方法 AMR | benchmark/empirical law + mechanism decomposition；方法不是核心必需项 |
| 计划方法 | Action-Mode Routing（AMR）：context/mode 双通道 + task-conditioned spectral routing | 当前不预设主方法；先证明 path effect，再决定是否需要 mitigation |
| 主要实验资产 | 足球 tracking 为显式结构测试床；篮球低成本重复；书法跨域验证 | LongMemEval/BEAM 等真实长期对话 + 可控 memory histories；多模型、多压缩轨迹 |
| 当前证据 | 已有强诊断/局部几何结果；下游任务收益未建立；AMR 未启动 | 尚无 pilot；核心假设未验证 |
| 最大创新风险 | 被归入 partial equivariance / invariance-equivalence / “Jacobian 预测局部变化是显然的” | 被近期 repeated compaction / rate-distortion 工作视为增量 benchmark |
| 最大科学风险 | 诊断规律不能转化为独立任务价值，或跨域不成立 | 固定 final budget 后 path effect 很小，或完全可由普通信息损失解释 |
| 最小强成功 | 独立数据上证明 support/coalition 是 operator 之外的必要坐标，并展示至少一个可控的任务相关后果 | 在严格 budget-matched、query-blind 条件下发现稳定的 path effect，并解释其信息类型与错误机制 |
| 失败后的降级形态 | 可收缩为 representation diagnosis / architecture anatomy 论文 | 如果只有“压得越多忘得越多”，基本不构成足够新颖的主线 |

---

# 2. 方案 A：Action-Mode Spectrum

## 2.1 研究问题

### 2.1.1 原始动机

很多表示学习方法把“变换类型”当作定义不变性或等变性的基本单位。例如，rotation、translation、scaling 常被视为一个全局 operator。Action-Mode 的出发点是：**operator 本身不足以决定语义**。

同样是 translation：

- 足球中，全队整体前移、后防线整体前提、单一后卫脱离阵线，可以具有相同的位移幅度，却具有完全不同的战术意义；
- 书法中，页面整体倾斜、一个偏旁整体旋转、一笔相对其他笔画发生旋转，可以由同类几何算子产生，却对应 nuisance、结构性变化或错误。

因此，研究对象不应只有：

> transformation/operator

而应至少写成：

> **operator × support/coalition × task**

方案 A 的核心问题是：

> **结构化表示是否错误地把“变换是什么”当作主要语义，而没有正确编码“谁与谁一起发生这个变换”？这种 support-conditioned sensitivity 能否被测量、解释和控制？**

## 2.2 形式化对象

设结构化输入由节点/部件组成：

\[
X = (x_1, x_2, \ldots, x_n),
\]

在构件关系图 \(G\) 上施加扰动场 \(\delta\)。同一个几何 operator 可以作用于不同的 support \(S\subseteq V\)。

原方案用图 Laplacian 谱对扰动模式进行分解，并定义结构传递函数：

\[
H_f(k;\varepsilon)
=
\mathbb E_X
\frac{d_z(f(X\oplus\delta_k),f(X))}{\varepsilon},
\qquad \|\delta_k\|=\varepsilon.
\]

它试图描述模型对不同“协同运动模式”的敏感性。

后续实验表明，仅使用频率/谱位置并不足以解释模型响应。当前更有支持的坐标是局部表示几何：

\[
q_f(X,\delta)=\frac12\|J_f(X)\delta\|^2,
\]

其中 \(J_f(X)\) 为模型表示对输入结构的局部 Jacobian。对 Jacobian 的节点块和跨节点项进行分解，可以研究某个 support/coalition 为什么被模型放大或压缩。

### 2.2.1 当前科学命题已发生的重要修正

最初更强的假设之一是存在统一的“中频凹陷/组织盲区”：模型可能对全局变化和尖锐单点异常敏感，却系统忽视有组织的子群协同。

**这一强版本目前没有被建立。** 当前仓库证据更支持：

> **图频率不是充分坐标；同一频段内，不同节点支持和跨节点局部几何会产生显著不同的表示响应，而且这种规律具有架构依赖性。**

因此，第三方评审应评价**当前版本**，而不是把“统一中频盲区”当作已经成立的 claim。

## 2.3 主要科学假设

### A-H1：Operator insufficiency

仅知道 transformation/operator 不能充分预测一个结构化扰动对任务的语义；support/coalition 和 task 是必要条件。

### A-H2：Frequency insufficiency

即使控制扰动总能量、support size、Rayleigh quotient / graph frequency 和位移方向分布，真实语义联盟与随机同频 support 仍可能导致不同表示响应。

### A-H3：Support-conditioned local geometry

局部 Jacobian 的节点与跨节点结构比单纯图频率更能解释同一 operator 在不同 support 上的表示响应。

### A-H4：Architecture anatomy

不同架构在 message passing、pooling、relational interaction 等阶段形成不同的跨节点敏感度结构，因此“谁与谁一起变化”的编码方式并不统一。

### A-H5：Task-conditioned routing（尚未验证）

如果显式拆分 context 与 intrinsic mode，并让不同任务选择不同 action modes，模型可以在不牺牲必要绝对上下文的前提下，提高对内部结构的敏感性和 nuisance robustness。

## 2.4 计划的方法贡献：AMR

**Action-Mode Routing (AMR)** 的目标不是简单做一个 centered embedding，而是让任务决定哪些 action modes 应当被压掉、哪些应当被保留。

核心结构：

\[
z=(z_{ctx},z_{mode})
\]

- \(z_{ctx}\)：保存球场绝对位置、页面姿态、场景背景等；
- \(z_{mode}\)：保存内部阵型、部件关系和 support-conditioned structural modes。

进一步以 Chebyshev 图谱滤波器构造稳定频段 \(P_b(L)\)，并用等能量谱干预训练。任务 \(\tau\) 对频段 \(b\) 学习门 \(\alpha_{\tau,b}\)：

\[
\mathcal L_{route}=
\sum_b[
\alpha_{\tau,b}\mathcal L_{inv}^{(b)}
+(1-\alpha_{\tau,b})\mathcal L_{eqv}^{(b)}].
\]

原方案设计了四级实现：

- M0：CAP / centered-action prediction，仅 baseline；
- M1：AMR-Fixed，固定频段路由；
- M2：AMR-Learned，可学习 task-conditioned gate；
- M3：AMR-Implicit，从图像 token 推断隐式构件图，高风险扩展。

**截至当前，AMR 仍是拟议方法，不是已经得到实验支持的贡献。**

## 2.5 实验体系

### 2.5.1 足球：显式结构测试床

作用：

- 有明确节点坐标和关系；
- 可以构造严格等能量的 counterfactual interventions；
- 可以定义真实战术 coalition 与 matched random support；
- 可以定位 message passing / pooling 等架构组件的因果作用。

### 2.5.2 篮球：低成本重复验证

目的不是扩展应用，而是用更少节点、更不同的协同结构判断足球结果是否是 11 人图结构的特例。

### 2.5.3 中国书法：隐式结构跨域验证

书法中的 page / component / stroke 层级提供了另一种结构体系。原方案要求体育端先冻结弱点预测，再在书法中测试对应谱位置或 support 规律，避免“看完结果再解释”的跨域包装。

### 2.5.4 关键实验控制

必须至少包括：

- 严格等扰动能量；
- 同 support size；
- 同/近 graph frequency；
- 真实 coalition vs matched random coalition；
- canonicalization、Procrustes/Kendall shape、frame averaging 等强基线；
- augmentation / pooling / relational interaction 的因果开关；
- 新比赛/新结构作为真正独立确认集。

## 2.6 当前已有证据（截至 2026-09-10）

以下是项目内部目前最重要的实证结果，不应与未来 claim 混淆：

### 2.6.1 局部几何预测表示响应：强支持

在当前 9 个模型与既有数据上：

| 预测量 | 回顾性 Spearman | 新干预 Spearman |
|---|---:|---:|
| 图频率/残差类普通基线 | 0.1222 | 0.1294 |
| diagonal geometry | 0.6528 | 0.6580 |
| full local geometry | **0.8466** | **0.8526** |
| off-diagonal geometry | 0.3886 | 0.3815 |

这说明当前实验中，support-conditioned local geometry 显著优于单纯频率描述模型响应。

### 2.6.2 层级形成过程：支持

geometry-response 关系沿网络逐步增强，内部报告给出的近似相关为：

- 输入层：~0.21；
- 第一层 message passing：~0.44；
- 第二层：~0.57；
- pooling 前：~0.76；
- pooled embedding：~0.84。

不同架构形成 cross-node terms 的阶段不同，为 architecture anatomy 提供了初步证据。

### 2.6.3 因果结构开关：只能证明“改变表示”，不能证明“修复任务”

将 `team_mean` 改为 `relational_pairwise` 后，off-diagonal/full 比例和 geometry-response correlation 都发生系统改变。但目前没有稳定的 downstream task gain。

因此，当前严谨表述应是：

> pooling/relational design 可以塑造局部表示几何；尚未证明这种塑形改善了有意义任务。

### 2.6.4 当前任务评价存在语义冲突

现有 phase classification 同时包含 high/medium/low block 等依赖绝对位置的类别，而优化指标又奖励对 global translation 更不敏感。这使 context signal 与 nuisance robustness 被不恰当地压在一个目标里。

当前建议是把：

- context/deployment task；
- intrinsic formation/structure task

分开评价，再测试 context/mode 双读出。

### 2.6.5 确认性证据边界

当前 prospective intervention 仍来自同一批样本/模型，只更换 support draws，因此不是新比赛、新模型或新领域泛化。

此外，已有 candidate search 曾在候选排序流程中暴露 heldout 指标。虽然排序公式并未直接使用 heldout，但该 heldout 已不再能作为严格的最终确认集。后续若需要强 claim，应使用真正未见比赛或重新设计 nested match-level evaluation。

## 2.7 方案 A 的创新点来源

### 创新点 A1：把变换语义从“operator”提升为“operator × support/coalition × task”

这不是简单说“有些变换应 invariant、有些不应 invariant”，而是强调：**同一个 operator 的语义会因作用 support 改变。**

### 创新点 A2：Action-Mode Probe / support-matched counterfactual

通过严格等能量、同频率、同 support size 的干预，直接测量模型对“谁一起动”的结构响应，而不是仅观察自然数据中的相关性。

### 创新点 A3：从 graph frequency 转向 support-conditioned local representation geometry

若独立验证成立，核心贡献不是又一个 spectral bias，而是证明频率只描述“扰动长什么样”，不足以描述“模型在什么节点组合上如何响应”；Jacobian block structure 提供更接近模型内部机制的坐标。

### 创新点 A4：Architecture-specific representation anatomy

将跨节点敏感度定位到 message passing / pooling / relational interaction 的形成阶段，可能形成机制性而非纯 benchmark 的贡献。

### 创新点 A5：AMR（条件性）

若双通道和 task-conditioned routing 能带来真实任务收益，方法贡献是：不把全局姿态统一删除，而是让任务选择 context 与 internal mode。

## 2.8 最近邻工作与创新碰撞风险

### 2.8.1 Soft Equivariance Regularization（ICLR 2026）

SER 已明确指出强 invariance 会损失 transformation-dependent structure，并通过“最终 embedding 保持 invariance、中间空间 token 保持 equivariance”来解耦两者。

**与 Action-Mode 的重叠：** 都反对“统一强 invariance”。  
**拟议区分：** Action-Mode 的基本单位是 operator × support/coalition × task，而不是仅按 layer 或 transformation type 决定 invariance/equivariance。

### 2.8.2 PooDLe（ICLR 2025）

PooDLe 已把 pooled invariance 与 dense equivariance 联合使用。

**风险：** 如果 Action-Mode 最终只表现为“全局不变、局部等变”，容易被认为概念上过近。  
**必须区分：** 同一几何算子在不同 coalition 上具有不同语义，并且这一差异可通过 matched support interventions 和局部几何实证。

### 2.8.3 Partial equivariance / symmetry breaking

ICLR 2025-2026 已有 probabilistic symmetry breaking、partially equivariant RL 等研究，说明“symmetry 不是处处成立”不是新命题。

**风险：** 若文章只写“局部需要关闭 equivariance”，新颖性不足。  
**必须区分：** Action-Mode 关注的是多构件系统内同一 operator 的 coalition-dependent action semantics，以及该语义如何在表示几何中形成。

### 2.8.4 “Jacobian 预测局部变化是显然的”风险

\(\|J\delta\|\) 本身就是局部响应的一阶近似，因此 0.85 的相关性不能单独成为标题级发现。

需要证明至少一项额外新知识：

- Jacobian block structure 揭示了既有频率/等变理论无法解释的可重复 failure mode；
- 可以预测独立数据/独立任务；
- 操作该结构导致正确的任务变化。

## 2.9 方案 A 的最低充分成功证据

如果目标是 ICLR 主会场，而非单纯项目报告，我认为最低充分证据应包括：

1. **独立确认：** 新比赛/新结构上重复 support-conditioned geometry 规律；
2. **强对照：** 同能量、同 support size、同频率下真实 coalition 与随机 coalition 存在稳定差异；
3. **语义正确的任务评价：** context task 与 intrinsic structure task 分开；
4. **至少一个因果机制：** 架构开关可重复改变目标表示属性；
5. **若以方法论文投稿：** 至少一个 intervention 能同时改善相关 task 和目标 robustness；否则应降级为 diagnosis/mechanistic paper；
6. **跨域：** 书法可显著增强一般性，但若时间不足，不应以薄弱书法结果充当强 claim。

## 2.10 方案 A 的致命失败条件

以下情况会显著削弱甚至终止强版本：

- 固定能量/频率/support 后，语义 coalition 与随机 coalition 无稳定差异；
- local geometry 只能解释训练集内小扰动，不能迁移到独立比赛/结构；
- architecture switch 改变几何却始终没有任何任务相关后果；
- 主要规律可被已有 partial equivariance / canonicalization 直接解释或修复；
- 最终只能说“Jacobian 能预测小扰动响应”。

---

# 3. 方案 B：Compression-Path Memory

## 3.1 研究问题

### 3.1.1 不采用的原始版本

以下问题本身不够新：

> “完整长 context、一次 summary、多次 summary，哪个记忆更好？”

原因是已有长期记忆 benchmark 已比较 long-context 与 memory systems，而 2026 年的近期工作也已经直接观察到 repeated compaction 的累积退化。

### 3.1.2 优化后的核心问题

优化版把自变量从“是否压缩/压缩多少”改为：

> **在完全相同的最终 memory budget 下，最终能力是否取决于压缩路径本身？**

设原始历史为 \(H\)，压缩器 \(C_B\) 把输入转为预算不超过 \(B\) 的 memory。

**直接压缩：**

\[
M_{direct}=C_B(H)
\]

**多阶段压缩：**

\[
M_{path}=C_B(C_{b_{K-1}}(\cdots C_{b_1}(H)))
\]

其中所有路线最终都满足同一预算 \(B\)，但压缩深度 \(K\) 与中间预算路径 \((b_1,\ldots,b_{K-1})\) 不同。

核心待检验命题：

\[
U(M_{direct}) \stackrel{?}{=} U(M_{path})
\]

如果不相等，则 memory 不仅是“剩余多少 token”的函数，还具有**compression-history dependence**。

## 3.2 关键概念：Retention 与 Accessibility 必须分开

完整长 context 理论上包含最多原始信息；有损压缩依据 data-processing inequality 不可能凭空增加与未来任务有关的真实信息。

但实际 LLM 并不是无限能力的最优解码器。很长的 raw context 可能因为 retrieval interference、attention dilution、position effects 等原因而使**信息存在但不易被利用**。

因此需要区分两个维度：

### 3.2.1 Information retention

压缩后的 memory 中，回答未来问题所需的事实、关系、时间更新、约束和 provenance 是否仍然存在。

### 3.2.2 Information accessibility

即使证据仍存在，给定 evaluator LLM 是否能够从当前 memory state 中正确调用它。

于是可能出现四种情况：

| Memory 状态 | 下游回答 | 解释 |
|---|---|---|
| 证据保留 | 正确 | 成功保留且可访问 |
| 证据保留 | 错误 | accessibility failure |
| 证据丢失 | 错误 | irreversible information loss |
| 证据丢失 | 正确 | 推理/猜测/外部知识命中，不能算真正 retention |

这个拆分使研究不退化为普通 QA leaderboard。

## 3.3 主要科学假设

### B-H1：Fixed-budget path dependence

在最终预算 \(B\) 相同、压缩器与 prompt 相同的条件下，直接压缩与多阶段压缩产生稳定不同的下游效用：

\[
\Delta_{path}(B,K,S)
=U(M_{direct})-U(M_{path})\neq 0,
\]

其中 \(S\) 表示 intermediate budget schedule。

### B-H2：Depth 与 final rate 是可分离因素

过去很多多轮 compaction 结果同时改变 compaction round 与剩余信息量。本方案要求构造二维/三维实验：

> **final budget/rate × compression depth × path schedule**

以检验 \(K\) 是否在控制 \(B\) 后仍有独立作用。

### B-H3：信息类型对 path 的敏感性不同

预计最容易被 repeated compression 破坏的不是所有信息，而是具有特定逻辑结构的内容，例如：

- exact atomic facts / numbers / identifiers；
- negation 与 hard constraints；
- temporal updates（旧事实被新事实覆盖）；
- cross-session relational / multi-hop information；
- provenance / source attribution；
- global gist（可能相对耐受）。

如果不同信息类型表现出稳定不同的 depth-response curve，贡献会显著强于“总体 accuracy 下降”。

### B-H4：可能存在 access-loss crossover

在某些任务和模型上，适度直接压缩可能删除 distractors，使：

\[
U(C_B(H))>U(H)
\]

尽管真实信息量下降。

但随着 repeated lossy transforms 增多，不可逆损失可能占主导：

\[
U(M_{K=0})
< U(M_{K=1})
> U(M_{K\gg1})
\]

如果出现稳定的非单调 crossover，它比“压得越多越差”更有科学价值。

### B-H5：Repeated compression 产生结构化 memory drift

多轮 summary 可能不是独立删除事实，而会产生特定错误动力学：

- stale fact resurrection：旧事实重新成为当前事实；
- temporal collapse：多个时间状态被压成无时间顺序的概括；
- entity merging：不同实体属性被合并；
- negation/constraint weakening；
- provenance disappearance；
- confidence amplification：早期的轻微错误被后续 summary 当作事实继续压缩。

这些是可测量、可分类的机制性现象。

## 3.4 核心 benchmark 设计

### 3.4.1 最重要的公平性约束：最终预算严格相同

例如原始 history 为 128K tokens，所有路径最终都得到 8K memory：

- 128K → 8K；
- 128K → 32K → 8K；
- 128K → 64K → 32K → 16K → 8K。

不能把“第五次 compaction 后只剩 4K”与“一次 compaction 后仍剩 32K”直接比较，否则 depth 与 rate 完全混杂。

### 3.4.2 Query-blind compression

压缩时不得知道最终问题。正确顺序为：

1. 给定 history \(H\)；
2. 按预定路径生成 memory \(M\) 并冻结；
3. 再抽取/揭示 query \(Q\)；
4. evaluator 基于 frozen memory 回答。

否则 compressor 只需保留答案相关证据，问题会退化为 query-aware retrieval/compression。

### 3.4.3 Factorial grid

最低实验矩阵建议：

- final budget/rate：例如 50%、25%、12.5%、6.25%；
- compression depth \(K\)：1、2、4、8；
- intermediate schedule：geometric、front-loaded、back-loaded；
- compressor：至少 2-3 个模型/方法；
- evaluator：至少 3 个不同 family/scale；
- task type：5 类以上。

真正分析对象不是一个 leaderboard，而是：

\[
U(B,K,S,T,C,E)
\]

其中 \(T\) 为 information/task type，\(C\) 为 compressor，\(E\) 为 evaluator。

### 3.4.4 必须增加的机制对照

**Control 1：Iterative rewrite at fixed final budget**  
把同一个 8K summary 连续改写 K 次，每次仍保持 8K，区分“反复生成/重写的漂移”与“逐级缩小信息瓶颈”的贡献。

**Control 2：Matched single-shot compression**  
最终 token 数、输出格式和 compressor 相同，只改变是否经历中间 memory states。

**Control 3：Order/schedule control**  
相同 depth 下改变中间预算路径，测试是否存在 schedule effect，而不是只有 compaction count。

**Control 4：Evidence-presence audit**  
用可控 canonical facts / structured annotations 判断所需信息是否仍存在，避免把 evaluator 的读取失败误写为压缩丢失。

**Control 5：Model-role separation**  
尽量不要只用同一 LLM 同时生成 summary 和评价 summary；需要交叉 compressor × evaluator，避免自偏好。

### 3.4.5 数据与任务

可以采用“受控测试床 + 真实长期对话 benchmark”两层结构。

**受控层：**

- 明确的 atomic facts；
- 多实体属性；
- 多轮 temporal updates；
- 正/负约束；
- multi-hop dependency；
- provenance tags。

受控层负责精确定位什么信息在哪一轮被破坏。

**真实层：**

- LongMemEval：information extraction、multi-session reasoning、temporal reasoning、knowledge updates、abstention；
- BEAM：更长、更连贯、可扩展到百万/千万 token 的对话；
- 可选择其他成熟长上下文/长期对话数据作为 robustness，而不是重新建设巨大数据集。

## 3.5 主要指标

### B-M1：Downstream utility

- QA / task accuracy；
- multi-hop exactness；
- temporal update accuracy；
- constraint compliance。

### B-M2：Retention / evidence coverage

对具有 canonical evidence 的项目，直接判断必要事实是否仍存在于 memory。

### B-M3：Provenance fidelity

事实与来源/时间/实体绑定是否保留。

### B-M4：Contradiction / stale-state rate

压缩后是否同时保留互相冲突的历史版本，或错误恢复过期事实。

### B-M5：Path effect size

在相同 final budget 下：

\[
\Delta_{path}=U(M_{direct})-U(M_{path})
\]

并用 paired design 对同一 history / query 进行统计比较。

### B-M6：Access-vs-loss decomposition

把失败分成：

- evidence present but unread；
- evidence absent；
- evidence corrupted；
- evidence conflated with stale/conflicting state。

## 3.6 方法贡献的定位

**本方案当前不应预先承诺一个“新 memory algorithm”。**

核心 paper 可以在以下条件下成立：

1. fixed-budget path dependence 是稳定、跨模型的；
2. 它不能被普通 final compression rate 解释；
3. 不同 information types 有可重复的 degradation law；
4. retention 与 accessibility 分解揭示了 previously conflated mechanisms。

在这种情况下，benchmark + empirical law + mechanism analysis 本身可以构成主要贡献，类似 LongMemEval/BEAM 这类被 ICLR 接受的 benchmark/analysis paper。

只有在诊断结果明确后，再考虑 mitigation。当前不建议把 type-aware compaction 作为原创主方法，因为 2026 年已有非常接近的 Knowledge Triage / TypeCompact 工作。

## 3.7 方案 B 的创新点来源

### 创新点 B1：把“压缩程度”与“压缩历史”正交化

核心 novelty candidate 不是 repeated compaction 本身，而是：

> **在同一 final budget 下，系统操纵 compression depth/path。**

如果 \(K\) 在固定 \(B\) 后仍有独立影响，就说明 memory state 不能仅由当前 token footprint 描述。

### 创新点 B2：二维/三维 compaction response surface

从单一 accuracy-vs-context-length 曲线升级为：

> final rate × depth × schedule × information type

其目标是建立 compaction dynamics，而不仅是模型排名。

### 创新点 B3：Retention 与 Accessibility 解耦

压缩可能减少真实信息却提高可访问性；长 context 可能保留信息却读取失败。直接测两者，可以定位“什么时候应该压缩、什么时候压缩已经越界”。

### 创新点 B4：Memory drift taxonomy / dynamics

如果能稳定刻画 stale fact resurrection、temporal collapse、entity merging、negation weakening、provenance loss 等随 depth 的增长规律，可形成比总 accuracy 更具机制性的贡献。

### 创新点 B5：Query-blind future utility

压缩时不知道未来 query，使 benchmark 更接近 agent memory consolidation，而不是 query-aware prompt compression。

## 3.8 最近邻工作与创新碰撞风险

### 3.8.1 LongMemEval（ICLR 2025）

已系统评估长期交互记忆中的 extraction、multi-session reasoning、temporal reasoning、knowledge updates、abstention，并比较 long-context 和 memory pipeline。

**结论：** “长期对话记忆很难”不是新 claim。  
**本方案必须超越：** LongMemEval 是任务基础设施，不是创新本身。

### 3.8.2 BEAM / Beyond a Million Tokens（ICLR 2026）

已把长期对话扩展到最高约 10M tokens，并提出多记忆组件框架。

**结论：** “更长 context 仍会掉性能”“多种 memory system 有帮助”也不是新 claim。

### 3.8.3 What to Keep, What to Forget: A Rate-Distortion View...（arXiv 2026-07）

这是方案 B 最重要的邻近工作之一。该工作已经：

- 用 rate-distortion 统一 KV、prompt、architectural state、agent memory；
- 明确指出 repeated compaction 长期缺乏系统评价；
- 提出 COMPACT-Bench；
- 做了 repeated irreversible summarization vs reversible retrieval 的 reference experiment；
- 在相近平均预算下观察到 repeated summary 的累积误差，并理论预测 repeated irreversible summarization 的错误增长。

**因此，本方案不能声称：**

> “我们首次发现 repeated compaction 会累积错误。”

**可能的剩余区分空间：**

- 同一个 irreversible summary operator 内，严格匹配最终 budget，比 one-shot vs multi-stage path；
- 显式正交化 final rate、depth、schedule；
- retention-accessibility decomposition；
- 针对 temporal update / provenance / entity binding 等 information type 建立 path-specific dynamics。

这一区分需要进一步系统查重才能确定是否足够新。

### 3.8.4 The Compaction Cliff in Long-Running AI Agent Memory（arXiv 2026-08）

该工作已经直接测量 sequential compaction：例如固定每轮 50% 压缩，观察多轮后 safety constraint retention 显著下降；并提出 typed Knowledge Triage / TypeCompact 等解决办法。

**因此，本方案不能把“多轮 summary 让规则越来越丢”作为主创新。**

潜在区分仍然是：

- 固定最终 budget 而不是每轮固定 ratio；
- one-shot vs multi-stage 的 path counterfactual；
- 广泛 information classes，而不仅是 safety/agent configuration；
- 将 retention 与 evaluator accessibility 分离；
- 测试中间 schedule 是否产生 hysteresis/path dependence。

### 3.8.5 Decision-centric rate-distortion memory（arXiv 2026-05）

已有工作把 agent memory 价值定义为对未来 decision quality 的保持，而不是描述保真。

**结论：** rate-distortion 或“memory 应保留任务相关信息”不是方案 B 的独占理论创新。理论框架应作为解释工具，而不能作为 novelty claim 本身。

## 3.9 方案 B 的最低充分成功证据

最低需要同时满足：

1. **Budget-matched path effect：** 同 final budget、同 compressor/prompt、同 history/query，multi-stage 与 direct 的差异显著且有稳定 effect size；
2. **不是 trivial round effect：** iterative rewrite、schedule、token-length 等控制后仍存在；
3. **跨模型：** 至少 3 个 evaluator 或 2-3 个 compressor family 复现主要规律；
4. **跨任务类型：** 不仅是单一 safety rule / needle recall；
5. **机制拆分：** 能区分 evidence loss 与 evidence-present-but-inaccessible；
6. **至少一个非平凡结果：** 例如明显的 access-loss crossover、特定 information type 的相变/非线性、schedule-dependent hysteresis，或 one-shot 与 repeated compaction 的排序反转。

如果只能得到：

> K 越大，accuracy 越低

而且这种下降完全可以由每轮信息删除自然解释，则科学贡献偏弱。

## 3.10 方案 B 的致命失败条件

- 严格控制 final budget 后，compression depth 的独立效应很小或不稳定；
- 所有效应主要由输出 token 数、不同 compressor compute 或 prompt 格式造成；
- evidence retention 与 downstream accuracy 几乎完全重合，没有新的 accessibility mechanism；
- 近期工作已经做过完全相同的 one-shot vs same-final-budget multi-stage 设计；
- 结果只在一个 proprietary model 或一个安全规则数据集上成立；
- 最终贡献只能表述为“反复总结会越总结越丢”。

---

# 4. 两个方案的逐维度对比

## 4.1 科学问题的抽象层次

### Action-Mode

试图改变我们描述结构化 transformation semantics 的基本坐标：operator 不足以定义 nuisance/semantic，需要加入 support/coalition 和 task。它属于**representation learning / inductive bias / geometry**问题。

### Compression-Path Memory

试图改变我们描述 agent memory state 的基本坐标：final token budget 不足以定义 memory quality，需要加入 compression history。它属于**long-context / memory / information dynamics**问题。

### 可供第三方判断的关键差异

- Action-Mode 的抽象对象更基础、更接近 representation theory，但也更难通过实验把“新概念”与已有 equivariance 体系真正区分开。
- Memory 的问题更直观、社区关注度更高，但 2026 年相关工作更新极快，创新窗口更窄、更容易被 contemporaneous preprint 接近。

## 4.2 创新来源

### Action-Mode 的创新主要依赖“新问题坐标 + 机制”

如果成立，创新不是数据集，而是：

1. operator × support × task；
2. matched coalition interventions；
3. support-conditioned local geometry；
4. architecture anatomy；
5. 可选 AMR 方法。

### Memory 的创新主要依赖“实验正交化 + 新经验规律”

如果成立，创新不是新的 summarizer，而是：

1. fixed final budget 下的 path counterfactual；
2. rate × depth × schedule response surface；
3. retention vs accessibility decomposition；
4. information-type-specific memory drift。

## 4.3 对新方法的依赖程度

**Action-Mode：较高。**  
即使 diagnosis 可以单独成文，原始强版本明显期待“机制可控 + AMR + task gain”。当前最大 blocker 也正是 representation change 尚未转化为稳定任务收益。

**Memory：较低。**  
如果 benchmark 揭示稳定、非平凡、跨模型的新规律，方法不是必需项。LongMemEval、BEAM 已说明高质量 benchmark/analysis 可以获得 ICLR 认可。但“规律必须真的是新的”，否则没有方法兜底时更容易被判增量。

## 4.4 理论深度

### Action-Mode

可接入：

- group action / invariance / equivariance；
- graph spectral analysis；
- local differential geometry / Jacobian blocks；
- task-conditioned quotient / information routing。

潜在理论深度较高，但容易过度形式化一个实际只由局部 Jacobian 近似解释的现象。

### Memory

可接入：

- data-processing inequality；
- rate-distortion / information bottleneck；
- repeated lossy channel / error accumulation；
- sufficient statistics / future-task utility。

理论非常自然，但 2026 年 rate-distortion memory 工作已经存在，因此理论更适合作为解释，不宜作为主要原创标签。

## 4.5 实验因果识别难度

### Action-Mode

优点：

- 可以人工构造严格等能量 intervention；
- support/coalition 是明确可操作变量；
- 可直接开关 pooling/message passing 组件。

难点：

- 需要定义真正语义正确的下游任务；
- 频率、support size、关系图、位置语义容易相互混杂；
- 足球→书法的一般性验证成本较高。

### Memory

优点：

- history 相同，可对同一个样本做 paired counterfactual；
- final budget、depth、schedule 都可以程序化操纵；
- 大部分 benchmark 已存在。

难点：

- compactor stochasticity；
- 多轮路径带来不同总 compute；
- compressor/evaluator model self-bias；
- “memory 中证据还在不在”本身需要可靠 annotation/判定；
- API 模型版本变化会影响可复现性。

## 4.6 最容易产生强 Figure 1 的方式

### Action-Mode

需要一个非常直观的 matched counterfactual：

> 相同 operator、相同总扰动，但全局/真实 coalition/随机 coalition 的任务语义与 embedding response 出现排序反转。

难点是需要证明反转不是简单 frequency/support-size artifact。

### Memory

可以直接画：

> 同一 history、同一最终 8K memory budget，one-shot 与 2/4/8-stage compaction 的 utility/retention 曲线分离。

视觉叙事更直接，但如果只是单调下降，容易被认为“意料之中”。真正强的 Figure 1 应包含：

- 同 final budget；
- clear depth effect；
- 至少一种 information type 或 access-loss crossover 显示非平凡结构。

## 4.7 与现有工作的碰撞方式

### Action-Mode

风险来自一个相对稳定的成熟文献树：invariance/equivariance、partial symmetry、dense equivariance。主要挑战是**概念区分是否足够深**。

### Memory

风险来自一个高速增长、非常新的文献树。2026 年 7-8 月已经出现 repeated compaction 和 rate-distortion 直接近邻。主要挑战是**时间上是否已经有人做了同样的 factorial control**。

因此两者风险性质不同：

- Action-Mode：较少“明天突然被抢”，但可能被评为已有理论的特殊情况；
- Memory：问题更热点，但 prior-art collision 风险更动态。

## 4.8 当前执行成熟度

### Action-Mode

已经完成的部分：

- Action-Mode / geometry probe；
- Jacobian/JVP 验证；
- retrospective + prospective support intervention；
- layer anatomy；
- pooling causal switch；
- candidate search。

尚未完成的关键部分：

- 语义正确的双任务/双读出确认；
- 真正独立确认集；
- AMR-Fixed/Learned；
- 体育→书法的确认性跨域验证。

### Memory

目前完成：

- 核心科学问题重构；
- 初步近邻文献查重；
- fixed-budget × depth × schedule 的设计框架；
- retention vs accessibility 的机制拆分。

尚未完成：

- 任何 pilot；
- exact benchmark matrix；
- model/API selection；
- effect size 估计；
- 与 2026 年近邻工作更系统的 exhaustive novelty audit。

## 4.9 工程量与证据链长度

### Action-Mode

需要结构数据处理、图模型/视觉模型、干预生成、Jacobian 分析、任务构建、独立赛事验证，以及可能的书法跨域。工程链较长，且当前 downstream task 定义还需要修正。

### Memory

核心 benchmark 可以主要依赖现有对话数据和 API/开源 LLM，工程链更短；但若需要多模型 × 多预算 × 多深度 × 多随机种子，推理成本会迅速增大。最大的资源消耗不是训练，而是**大规模长上下文推理与多轮 compaction 调用**。

## 4.10 结果阴性时的研究价值

### Action-Mode

如果 AMR 失败但局部几何规律独立复现，仍可收缩成 diagnosis / representation anatomy paper；已有实验证据使其存在一定下限。

### Memory

如果固定 budget 后 path effect 很弱，核心论点基本消失。除非阴性结果本身能明确否定业界普遍假设，否则较难形成同等强度的 paper。

---

# 5. 对第三方评审最重要的“不要混淆”事项

## 5.1 不要把 Action-Mode 的已完成度直接算作 idea quality

Action-Mode 有更多现成结果，但当前最关键的方法/任务闭环尚未完成。评审时建议同时给出：

- **conceptual score**：假设两个项目都从零开始；
- **execution-adjusted score**：考虑当前已有证据和剩余时间。

## 5.2 不要把 Memory 评价成“long context vs summary”

优化版 Memory 的唯一值得认真比较的版本是：

> **同 final budget 的 one-shot vs multi-stage compression path dependence。**

若评审对象退化成“summary 多次后忘得更多”，则应直接视为较弱版本。

## 5.3 不要把 Action-Mode 评价成“global invariance vs local equivariance”

这一表述过于接近现有工作。当前更准确的问题是：

> **同一 operator 在不同 support/coalition 上具有不同 task semantics，且模型响应由 support-conditioned local geometry 决定。**

## 5.4 两个项目都不能依赖“我们是第一个”

两条路线都有非常接近的现有工作。真正需要判断的是：

> **新增的控制变量/形式化/实验证据，是否揭示了近邻工作没有回答的新问题。**

---

# 6. 建议第三方独立评审回答的问题

建议将以下问题分别对 A、B 打分或写文字意见，而不是只给一个总分。

1. **问题新颖性：** 核心问题是否已被近邻工作实质回答？如果没有，剩余 gap 是否足够清楚？
2. **问题重要性：** 即使方法最终不 SOTA，这个发现是否会改变社区如何理解或评估相关模型？
3. **“意料之中”风险：** 最可能出现的主结果是否只是常识的量化？什么结果才算真正 surprising？
4. **最小可发表单位：** 不做所有扩展时，最小哪组实验已经足以形成完整论文？
5. **方法必要性：** 是否必须有新 algorithm 才能达到 ICLR 的贡献强度？
6. **因果识别：** 核心自变量能否与 confounders 正交化？
7. **理论价值：** 理论框架是在解释结果，还是只是给已有直觉换数学写法？
8. **跨域一般性：** claim 是否天然超出当前数据域？需要多少跨域证据才能支持？
9. **复现性：** 第三方能否在合理资源下复现实验？是否依赖易变化的闭源 API？
10. **近邻工作碰撞：** 哪一篇 prior work 最可能成为 reviewer 的第一条拒稿理由？作者目前的区分是否足够？
11. **失败风险：** 哪个单一 kill test 如果阴性，应立即停止该方向？
12. **当前成熟度：** 考虑已完成工作后，哪条路线更可能在当前投稿窗口形成证据闭环？

### 建议第三方输出格式

对每个方案分别给出：

- Overall scientific interest：1-10；
- Novelty after nearest-prior-work adjustment：1-10；
- Evidence tractability：1-10；
- ICLR fit：1-10；
- Fatal concern（最多 2 条）；
- Must-have experiment（最多 3 条）；
- 如果只能选一条路线：A / B / 都不选；
- 选择理由：不超过 300 字。

---

# 7. 最关键的公平比较摘要

## 方案 A：Action-Mode Spectrum

**研究本体：** structured representation 中 transformation semantics 的定义问题。  
**最强创新候选：** operator × support/coalition × task；support-conditioned local geometry；architecture anatomy。  
**当前最强证据：** local geometry 比 graph frequency 明显更能预测表示响应，并可定位到网络层级。  
**当前最弱环节：** 尚未证明表示几何的改变带来语义正确的 downstream gain；AMR 与跨域确认未完成。  
**最大 prior-art 风险：** partial equivariance / invariance-equivalence 已很成熟，必须证明 coalition/support 是真正新增的必要坐标。  
**若成功的论文形态：** 机制型 representation-learning paper，理论与方法上限较高。

## 方案 B：Compression-Path Memory

**研究本体：** fixed-budget agent memory 的 path dependence 与 information dynamics。  
**最强创新候选：** 严格固定 final budget，正交化 compression depth/path；retention-accessibility decomposition；information-type-specific drift。  
**当前最强优势：** 问题直观、benchmark 容易形成 paired counterfactual、方法不是必须项。  
**当前最弱环节：** 尚无 pilot；2026 年已有非常接近的 repeated compaction / rate-distortion preprints，创新空间必须精确守住。  
**最大 prior-art 风险：** 如果结果只是“反复压缩会累计错误”，几乎肯定不够；必须展示 fixed-budget path effect 及其非平凡机制。  
**若成功的论文形态：** benchmark/empirical-law paper，传播性和即时相关性较强。

---

# 8. 来源与近邻文献

## 8.1 项目内部材料

**[S1]** 《ICLR课题选择分析》：Action-Mode Spectrum 的正式项目方案、核心形式化、AMR 设计、体育/书法角色和关键 gate。  
**[S2]** 《仓库进度与下一步方案》：截至 P3 的当前结果、局部几何证据、任务 blocker、heldout 协议问题和后续决策门。

## 8.2 ICLR 评审标准

**[R1]** ICLR 2027 Reviewer Guide. https://iclr.cc/Conferences/2027/ReviewerGuidelines

## 8.3 Action-Mode 近邻工作

**[A-R1]** Lee et al. *Soft Equivariance Regularization for Invariant Self-Supervised Learning.* ICLR 2026.  
https://proceedings.iclr.cc/paper_files/paper/2026/hash/3be6511c8f56d0dca4b5ed59fdf9b2f4-Abstract-Conference.html

**[A-R2]** Wang et al. *PooDLe: Pooled and dense self-supervised learning from naturalistic videos.* ICLR 2025.  
https://proceedings.iclr.cc/paper_files/paper/2025/hash/d0528732673f7535f457e8109b0cbf06-Abstract-Conference.html

**[A-R3]** Lawrence et al. *Improving Equivariant Networks with Probabilistic Symmetry Breaking.* ICLR 2025.  
https://proceedings.iclr.cc/paper_files/paper/2025/hash/c7138635035501eb71b0adf6ddc319d6-Abstract-Conference.html

**[A-R4]** Chang et al. *Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments.* ICLR 2026.  
https://proceedings.iclr.cc/paper_files/paper/2026/hash/951f87360544eeda24b5e72cf725da1d-Abstract-Conference.html

## 8.4 Compression-Path Memory 近邻工作

**[B-R1]** Wu et al. *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.* ICLR 2025.  
https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html

**[B-R2]** Tavakoli et al. *Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs.* ICLR 2026.  
https://proceedings.iclr.cc/paper_files/paper/2026/hash/d7f0cfa0fe759b033d5262e1bb7d4065-Abstract-Conference.html

**[B-R3]** Colaco & Lahjouji. *What to Keep, What to Forget: A Rate-Distortion View of Memory Compaction in LLMs and Agents.* arXiv:2607.08032, 2026-07.  
https://arxiv.org/abs/2607.08032

**[B-R4]** Zerhoudi, Mitrovic & Granitzer. *The Compaction Cliff in Long-Running AI Agent Memory.* arXiv:2608.22752, 2026-08.  
https://arxiv.org/abs/2608.22752

**[B-R5]** Zou et al. *Remember the Decision, Not the Description: A Rate-Distortion Framework for Agent Memory.* arXiv:2605.10870, 2026-05.  
https://arxiv.org/abs/2605.10870

---

# 9. 最终说明

本文故意**不提供“哪个方案应该胜出”的结论**，因为用途是让第三方评审独立判断。两个方案真正应被比较的是：

> **哪一个问题在扣除最近邻工作后仍然留下足够明确的新知识增量，并且能用当前可获得的数据、模型和实验设计达到足以支撑主要 claim 的最低充分证据。**

对 Action-Mode，关键不是已有工作量，而是 support-conditioned geometry 是否能跨独立数据并产生任务相关后果；对 Compression-Path Memory，关键不是 LLM memory 热度，而是 fixed-final-budget 的 path effect 是否真实、非平凡、且没有被近期 repeated-compaction 工作实质覆盖。
