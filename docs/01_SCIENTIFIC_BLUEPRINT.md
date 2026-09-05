# 科学蓝图：从位移到组织

> 一句话摘要：模型也许能看见每个元素移动了多少，却未必表示了哪些元素是协同变化的；本文把“协同方式”定义为表征学习的一等变量。

## 1. 项目的核心科学问题

这篇论文不研究“如何识别足球阵型”，也不研究“如何给书法打分”。它研究：

> 对一个由多个构件组成的系统，同一个几何算子作用于不同构件联盟时，现代表示是否给出了正确、可访问且任务可选择的响应？

三个例子：

- 足球：整队前移、后卫线前移、单个后卫前移；
- 篮球：全队同步收缩、弱侧三人轮转、单人漏防；
- 书法：页面整体偏移、一个偏旁整体偏移、单笔偏移。

变换幅度可以完全相同，但语义不同。问题的索引应当是：

\[
\text{operator} \times \text{coalition/support} \times \text{task},
\]

而不是单独的 operator。

## 2.4 P2 证据如何修正问题

P2 rigid formal v2、独立 fracture continuity 和 P2-H1 异质性诊断均已完成。P2 的冻结事实是：exact-spectrum 对照说明普通谱功率不能解释全部表示响应；fracture 在部分条件下跨比赛稳定，但方向和幅度依赖 architecture、graph、role 和 energy；当前结果是 representation response，不是下游任务性能。36 个条件中有 20 个总体正向、16 个总体负向，LOMO 同号比例中位数为 1.0，但 seed 方向完全一致的条件为 83.3%。

因此，`mixed_or_graph_specific` 不是把 P2 降级为“没有效应”，而是把一维平均问题升级为条件化机制问题。历史候选“统一中频凹陷”和“角色联盟普遍更特殊”不再作为默认事实；它们保留为待检验或已收缩的历史假设。当前中心问题变为：

> 在总能量、谱功率或位移向量 multiset 受控时，频率、支持分配和跨构件耦合如何共同决定表示响应？

Action-Mode Spectrum 仍是总体的边缘汇总。P3 将其机制化为：

\[
H_f(b,S,\tau;\varepsilon)=
\mathbb E\left[
\frac{d(f(X\oplus\delta),f(X))}{\varepsilon}
\mid \delta\in b,\operatorname{support}(\delta)=S,\tau
\right].
\]

对归一化表示 \(\hat z(X)=z(X)/\|z(X)\|_2\)，定义

\[
J_f(X)=\frac{\partial \hat z(X)}{\partial\operatorname{vec}(X)},
\qquad G_f(X)=J_f(X)^\top J_f(X).
\]

小扰动下：

\[
1-\cos(z(X),z(X+\delta))
\approx \frac12\delta^\top G_f(X)\delta.
\]

当每个节点为二维坐标时，\(G_f\) 由 \(2\times2\) block 构成：对角 block 表示节点自身敏感度，非对角 block 表示跨节点共同或相对变化的耦合。P2 fracture 保持非零位移向量 multiset、支持大小和合法性条件，只改变向量被分配到哪些端点，因此是检验这些 block 的天然仪器。图频率只约束图基底中的功率，不能唯一决定支持落在哪些 block 上。

P3 是 `P3_CAUSAL_MECHANISM` 内的 post-P2 mechanistic follow-up，不新增阶段节点，也不先训练 AMR。强成功版本必须依次具备“预测—定位—因果开关—任务意义”；未满足时按条件支持或不支持收缩 claim。

## 2. 形式化对象

### 2.1 多构件状态与关系图

设结构包含 \(n\) 个构件，状态为：

\[
X=[x_1,\ldots,x_n]^\top\in\mathbb R^{n\times d}.
\]

构件关系由图 \(\mathcal G=(V,E,W)\) 表示，图拉普拉斯为：

\[
L=D-W=U\Lambda U^\top.
\]

一个变换场或扰动场写成：

\[
\delta=[\delta_1,\ldots,\delta_n]^\top.
\]

它在图谱基中的系数为：

\[
c=U^\top\delta.
\]

- 零频/DC 模式：所有构件共同变化；
- 低至中频：大范围或有组织的子群协同；
- 高频：个体或很小局部偏离。

“共同/子群/局部”只是直觉说明；论文的中心对象是连续频谱或稳定频段，而不是人工三档。

### 2.2 作用模式谱

给定表示 \(f\)，对第 \(k\) 个等能量作用模式定义结构传递函数：

\[
H_f(k;\varepsilon)
=
\mathbb E_X
\frac{
 d_z\left(f(X\oplus\delta_k),f(X)\right)
}{\varepsilon},
\qquad
\|\delta_k\|_F=\varepsilon.
\]

这里 \(\oplus\) 表示在合法状态空间中施加扰动；\(d_z\) 可以是余弦距离、欧氏距离或任务指定度量。

它回答：

> 模型对每种“谁和谁一起动”的模式听得有多清楚？

对于图像编码器，先把坐标状态渲染为相同外观的 minimap，或在矢量笔画层施加无接缝干预，再测图像表示。

### 2.3 任务语义谱

表示的高敏感度本身不等于正确。定义任务对作用模式的真实敏感度：

\[
S_\tau(k;\varepsilon)
=
\mathbb E_X
\frac{
 \Delta \ell_\tau(X, X\oplus\delta_k)
}{\varepsilon},
\]

其中 \(\Delta\ell_\tau\) 可由：

- 客观状态变化；
- oracle 几何量；
- 任务标签改变概率；
- 结构匹配距离；

定义。

核心问题不是让 \(H_f\) 全部变大，而是让它与 \(S_\tau\) 对齐，同时不永久删除其他任务可能需要的模式。

## 3. 我们要发现什么

### 3.1 首要候选发现：等能量排序反转

在相同扰动能量下：

\[
d(f(X),f(X+\delta_{common}))
>
d(f(X),f(X+\delta_{semantic})),
\]

但任务语义恰好相反。直观说：

> 模型觉得全队搬家比防线破裂更重要；或者觉得页面倾斜比关键笔画错位更重要。

这是 Figure 1 级假设，不是预设事实。

### 3.2 更强候选发现：中频凹陷

普通低通/谱偏置会预测敏感度随图频率大体单调下降。我们押注更强的现象：

> 对有意义的子群协同，作用谱可能出现非单调的中频凹陷；模型既能感知全局大变化，也能感知尖锐局部异常，却忽略组织化的中尺度关系变化。

中频凹陷比“高频先丢”更有新颖性，因为它不能被普通过平滑一句话解释。

### 3.3 组织性超越频率

即使两个扰动具有：

- 相同总能量；
- 相近 Rayleigh quotient；
- 相同支持大小；
- 相同位移分布；

“真实战术子群/部件”与“随机挑选的同频子集”仍可能引起不同的表示响应。这说明模型遗漏的不是简单频率，而是组织或联盟结构。

### 3.4 可控性

若改变增强分布、聚合机制或模式约束类型，凹陷的位置和深度应按预测变化。最有价值的反直觉预测是：

> 把缺失的中频扰动作为“不变性增强”加入，可能使中频凹陷更深；正确做法不是增加不变性，而是在该频段保留等变性或可恢复性。

## 4. 核心假设

### H1：Matched-energy reversal

至少两类现代表征在等能量条件下出现共同模式优先于语义结构模式的距离排序反转。

### H2：Mid-frequency notch

作用谱存在稳定的非单调中频凹陷，且不是图拉普拉斯频率单调衰减的简单实例。

### H3：Semantic coalition effect

在相同频率带与能量下，真实子群和随机子集之间存在稳定差异。

### H4：Augmentation-spectrum predictability

增强分布在作用模式坐标中的能量谱，能够预测表示敏感度的凹陷位置或压缩顺序。

### H5：Constraint-type causality

同一中频扰动被施加 invariance 或 equivariance 约束时，会以相反方向改变该频段的 \(H_f\)。

### H6：Pooling/interaction mechanism

只使用 unary features 和可分聚合的模型更容易丢失关系相位；加入跨构件关系交互后，谱响应发生可预测改变。

### H7：Scale is not a cure

模型增大并不必然消除作用模式错配；缺陷可能稳定或因更强的捷径学习而加深。

### H8：Cross-domain prediction

体育端发现的凹陷频段或支持域边界，可用于引导对书法中偏旁/笔画关系的探索，并与自然和受控样本进行比较。

### H9：AMR improves the frontier

AMR 在相同全局鲁棒性水平下，提高语义模式可访问性和自然任务性能，并优于 CAP、普通关系池化和 canonicalization 基线。

## 4.1 P2 后的假设状态

下表保留 H1–H9 的原始含义，同时标出 P2 后的证据状态。`supported` 仅指当前范围内的部分证据，不等于普遍因果结论。

| 原假设 | P2 后状态 | 当前解释和下一检验 |
|---|---|---|
| H1 Matched-energy reversal | `mixed` | exact-spectrum 下存在空间组织响应差异，但统一 common-over-semantic 排序反转未建立；改检验 support-conditioned geometry 是否预测 pair effect。 |
| H2 Mid-frequency notch | `not tested` | P2 没有给出稳定的统一 notch；“统一中频盲区”降为历史候选。 |
| H3 Semantic coalition effect | `mixed` | exact-spectrum 组织差异较稳定，但 role-defined fracture 相对 topology/frequency control 依赖 architecture、graph、role 和 matching residual。 |
| H4 Augmentation-spectrum predictability | `not tested` | 需要 P3 之后、且不与 P2 response 混用的训练机制证据。 |
| H5 Constraint-type causality | `not tested` | 只有在 P3 选择出明确机制量后，才允许测试一个约束开关。 |
| H6 Pooling/interaction mechanism | `reformulated` | 不再先验宣称 unary 或 interaction 是根因；由 layer-wise block 证据选择 pooling 或 interaction 路由。 |
| H7 Scale is not a cure | `not tested` | 当前没有尺度轴的机制或任务证据。 |
| H8 Cross-domain prediction | `not tested` | 书法不用于选择体育机制；仅在体育端预测冻结后另行验证。 |
| H9 AMR improves the frontier | `not tested` | AMR 训练尚未启动；P4 只能由 P3 gate 触发。 |

这些状态保留失败、混合和未测试假设，不把工程完成写成科学支持。

## 5. 两个域为什么不是拼盘

### 5.1 体育：受控仪器域

体育 tracking 提供：

- 明确构件；
- 精确二维坐标；
- 可构造等能量反事实；
- 自然发生的阵型和子群变化；
- 低算力实验环境。

它负责建立定量规律和因果机制。

### 5.2 书法：跨域预测与隐式结构域

书法提供：

- 构件关系高度密集；
- 参考系不显式画出；
- 小幅关系变化可能决定结构；
- 多种全局姿态、风格与内容共存。

它负责验证：在没有球场线、角色标签和显式图时，规律是否仍能预测真实像素表征的失效。

正确叙述是：

> 一台仪器建立规律，一个现场实验检验预测。

不是“方法在两个应用上都有效”。

## 6. 学界空缺

现有研究覆盖了许多邻近问题：

- invariant 与 equivariant representation；
- pooled invariance 与 dense equivariance；
- soft/partial equivariance；
- canonicalization 与 frame averaging；
- GNN oversmoothing 和 spectral bias；
- 组合性与关系理解失败。

但仍缺少一个统一对象来回答：

1. 同一个算子经由不同构件联盟作用时，表征的响应如何变化；
2. 表征的作用模式谱是否与任务语义谱对齐；
3. 错配是否具有非单调规律和可控机制；
4. 应如何在不删除全局上下文的前提下，按任务路由 invariant/equivariant 约束。

本文的空缺不是“没人做过不变性”，而是：

> 不变性通常按算子家族索引，而多构件语义要求按作用联盟和任务索引。

## 7. 工业空缺

工业系统通常已有检测、跟踪、OCR 或 embedding，但缺少一个稳定的“组织表示层”：

- 体育：相似阵型检索、阵线变化、局部空当、绝对部署位置常由独立规则或任务模型分别处理；
- 书法：字符识别能够回答“是什么字”，却不能同时忽略拍摄姿态、保留整体风格姿态，并定位笔画关系偏差；
- 通用多实体系统：模型容易编码单体特征和全局背景，却缺少可迁移的联盟响应坐标。

论文不需要证明完整商业系统，只需证明这一表示层确实缺失且可被统一方法改善。

## 8. 难点

1. **等量扰动定义**：全体移动和单体移动的总能量天然不同，应尽量配平并报告偏差。
2. **频率与组织混淆**：任何结果都可能只是低通偏置，需要同频随机联盟对照。
3. **全局信息不一定是 nuisance**：通用 backbone 不应被惩罚为“泄漏姿态”，而应让任务选择模式。
4. **节点身份与排列**：跨比赛球员身份不同，方法需要考虑集合/图置换鲁棒性。
5. **书法结构隐式**：笔画或部件图不可无成本获得，需要区分诊断支持、弱监督支持与真正无支持推理。
6. **图谱不稳定**：逐样本 eigenvector 存在符号、重根和拓扑漂移问题；方法应优先使用稳定频段滤波器。
7. **编辑伪影**：书法优先在矢量笔画层重新渲染，避免把拼接修图作为主证据。
8. **方法近邻多**：仅加 centered-action prediction 不足以形成方法贡献。

## 9. 预期贡献

强成功版本的贡献堆栈：

1. 定义和实现等能量 Action-Mode Probe；
2. 发现非单调、可预测且可控制的模式响应规律；
3. 将规律定位到增强谱、聚合或跨构件交互机制；
4. 提出 AMR：任务条件下的频段不变/等变路由；
5. 体育端建立定律，书法端成功兑现跨域预测；
6. 在相同鲁棒性下，AMR 提高内部构型保真、检索和变化定位。

## 10. 结果解释与方向更新

以下结果对应不同的解释和后续探索方向，不构成机械的淘汰门槛：

- 若等能量后没有稳定的排序反转，可转向研究模型正确响应结构模式的条件；
- 若敏感度主要随频率单调变化，或语义联盟与随机联盟差异很小，可聚焦关系频谱偏置或 pooling accessibility；
- 若凹陷对训练机制不敏感，可把它作为描述性现象，并继续寻找其他可解释机制；
- 若中心化或 canonicalization 已解决大部分问题，可收缩 AMR 的方法主张，研究其仍有价值的边界；
- 若书法结果受编辑边缘或主观标签影响，应改进渲染与客观结构任务；
- 若 AMR 不优于 CAP、关系池化或 canonicalization，仍可保留诊断工具，并如实说明方法局限；
- 若体育与书法的对应关系较弱，可降低跨域主张，探索更合适的映射或把书法作为独立检验域。

若只得到单调频率偏置，可以把研究方向调整为“多构件表示中的关系频谱偏置”，而不必保留组织性或中频盲区表述。

### 10.1 P3 最小机制路线

P3 不默认执行原先的 augmentation × pooling × constraint 全矩阵。任务编号属于本阶段内部施工协议：

1. `P3-T0`：在同输入、同权重、`eval()`、无 dropout、固定 dtype/device 下，复现 P2 baseline embedding，并暴露输入节点编码、message passing 后、pooling 前和 pooled embedding；验证 normalized-embedding JVP、有限差分和二阶近似。
2. `P3-T1`：以 \(q_f(X,\delta)=\frac12\|J_f(X)\delta\|_2^2\) 预测 raw response、anchor-control pair effect、方向和 match 聚合；与频率、Rayleigh、support size、role/graph、residual、diagonal-only 等低自由度基线比较，按 match 分组交叉验证。
3. `P3-T2`：冻结公式、层、指标和统计单位后，用同一 250 samples、9 models、新 seed/合法 support reallocation 生成 response-blind prospective set；失败时收缩机制 claim。
4. `P3-T3`：分解 diagonal/off-diagonal、team 内外、support 内外、edge/non-edge，并按 layer-wise 定位信息形成或丢失的位置；不直接比较不可比层的绝对 Frobenius norm。
5. `P3-T4`：只选择一个由 T3 证据支持的 causal switch（pooling accessibility、interaction coupling 或 constraint placement 三者之一），完成 matched-capacity、3 seeds 或明确稳定性分析。
6. `P3-T5`：先做 perturbation localization，再选择一个已有且非 intervention-label 的客观任务，冻结任务定义后检验 geometry-task alignment。

P3 的工作假设如下，均不得提前写成结论：

- `P3-G1`：\(q_f\) 跨 match 预测 fracture response 或 pair effect，并优于非几何基线；
- `P3-G2`：diagonal 与 off-diagonal 贡献解释部分 role × architecture 差异；若 off-diagonal 没有额外解释力，收缩为节点敏感度各向异性；
- `P3-G3`：异质性在节点编码、message passing、pooling 前后中的一个可定位层形成、丢失或被放大；
- `P3-G4`：一个 matched-capacity 开关按预测改变 geometry、fracture response 和至少一个客观任务指标；
- `P3-G5`：geometry 与任务语义越对齐，结构任务越好，而不是单纯把 response 推大。

## 11. 项目边界

- 研究对象：二维多构件状态及其栅格/像素表示。
- 核心算子：二维平移；书法可补充旋转和尺度。
- 不声称覆盖所有 Lie groups 或非交换群。
- 不声称所有 foundation model 都删除全局姿态。
- 不声称书法审美有单一客观标尺。
- 不把已知构件支持描述为无监督发现。
- 不把 VLM 关系盲作为已证实的因果后果，只作为可选影响力桥梁。

## 12. P3 证据回写与当前科学边界（2026-09-01）

P3-T0–T5 已完成。冻结的 normalized-embedding local geometry 在当前 250 samples、10 个 source matches 和 9 个冻结模型上，对 P2 fracture pair effect 的 retrospective mean Spearman 为 `0.8466`，同资产 response-blind prospective draw 为 `0.8526`；频率/residual baseline 分别为 `0.1222` 和 `0.1294`。这使 P3-G1 获得条件性支持，但不构成新比赛或新数据域泛化。

T3 表明 diagonal sensitivity 是主要贡献，off-diagonal coupling 提供额外但 architecture/layer-dependent 的信息：GAT/Phase-GAT 的非对角贡献在 message passing 中形成，DeepSets 的非对角贡献主要在 pooling 后出现。该结果支持局部几何解剖，不支持统一的 endpoint、role 或关系因果理论。P3-G2/G3 以 `SUPPORTED_CONDITIONALLY` 记录。

依据 T3 只执行一个 Phase-GAT pooling-accessibility switch。`relational_pairwise` 在 3 个 seed 上均改变 off-diagonal/full geometry 比例和 prospective response，但 context response 均增加，heldout macro-F1 均值略降；nodewise support localization 也只略高于 uniform baseline。因此 P3-G4 只能写为 `RESPONSE_SHAPING_ONLY`，P3-G5 不支持作为方法修复；P4 AMR gate 未满足。

Action-Mode Spectrum 不被撤回：它是 support、relationship 和 task 条件化响应的边缘汇总，允许条件效应异号。统一中频盲区仍只是历史候选。原 T0–T5 的停止边界仍然有效；截至 2026-09-02，补丁包在同一 P3 内授权有限的 T5R task-semantic repair，结果未出前不启动 AMR。完整旧 P3 证据见 `reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md` 和 `artifacts/phase3/`。

## 13. P3-T5R：任务语义修复与独立数据确认（2026-09-02）

补丁包将原 T5 阴性结果限定为旧的 conflated task gate：它把需要保留 absolute deployment 的 context task 与要求对 global translation 稳健的 intrinsic task 放进同一个表示优劣判据。因此旧结论保留为 `NOT_SUPPORTED_UNDER_OLD_CONFLATED_TASK_GATE`，不被扩写为所有 task-conditioned repair 均失败。

新的中心问题仍属于本阶段：

```text
operator × frequency × support/relationship × task semantics
```

P3-T5R 先固定两个互补读出：raw positions 进入 `z_ctx`，globally centered positions 进入 `z_mode`。`z_ctx` 负责 phase/deployment（显示名：match-half/period context probe，非战术阶段，见 semantic addendum）和 absolute field context；`z_mode` 负责 natural pair ranking 或 intrinsic formation retrieval，并要求 global translation robustness。两个任务分别评价，cross-readout 只作 leakage audit。

当前固定分支暂以 IDSSE 替代 SNGAR 的开发数据角色：IDSSE 原始数据用于 T5R1–T5R4 的 conversion、任务构造、候选开发和固定双通道检验；7 场数据必须按 source match 划分，不能伪装成 SNGAR 的 45/9/10 场。已用于开发的 IDSSE 不再同时承担独立 external confirmation，SNGAR 恢复或其他 provider/source 保留为独立确认分支。所有 support 规则必须 response-blind、可审计且按 source match 统计。P3-T5R 是有界 post-P2 follow-up，不新增 Phase，不先训练 AMR；geometry 可预测不等于任务语义已被理解。
