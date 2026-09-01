# Action-Mode Spectrum 项目方案包（合并版）

> 本文件便于连续阅读；执行时以各独立文档、配置和脚本为准。

## 当前同步状态（2026-09-02）

本节是本合并版的当前入口，优先于下方保留的历史方案文字；独立 canonical source docs 和 machine-readable configs 是执行真源。

- P2 rigid formal v2、independent fracture continuity、P2-H1 heterogeneity diagnosis 已完成并冻结。
- P2 的冻结 route 为 mixed_or_graph_specific；P2 是 representation response，不是下游任务性能。P2 reports、manifest、checksum、artifacts 和历史 claim 只读。
- 当前 active phase 为既有 P3_CAUSAL_MECHANISM，不增加 P2.5/P3a/P3b；原 P3-T0–T5 已完成，当前进入有界 P3-T5R task-semantic repair，旧 task gate 仍未支持，P4 gate 未满足。
- P3 的一句话目标是用 support-conditioned local geometry 预测、定位并干预 P2 异质性。机制对象为归一化 embedding 的局部 Jacobian 和 G_f=J_f^T J_f；Action-Mode Spectrum 保留为条件化对象的边缘汇总。
- P3 原顺序为 T0 接口 → T1 retrospective grouped prediction → T2 response-blind prospective → T3 layer/block localization → T4 一个 evidence-selected causal switch → T5 objective task alignment；当前在同一 P3 内追加 T5R0–T5R6：协议修复、独立数据、任务拆分、fixed dual-channel、有限 autoresearch、candidate lock/test 和外部确认。完整 causal matrix 仍为 gate 后 optional extension；P4 AMR 在 gate 前不运行。
- 当前 P3 结果为：full local geometry 在 retrospective/prospective 同资产评估中优于频率/residual 基线，T3 选择了 Phase-GAT pooling-accessibility 开关；nodewise support localization 仅接近 uniform，客观任务 macro-F1 未改善且 context robustness 变差，因此不进入 P4、不训练 AMR。入口配置为 `configs/phase3_support_geometry_v1.yaml`，主报告为 `reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md`，T5R0 迁移验收为 `reports/P3_T5R_DOCUMENT_MIGRATION.md`。
- P3-T5R 可按数据 manifest 使用独立体育数据，但不访问、修改或重建 infra/bioinf-data-index；书法结果保持 exploratory，不用于选择体育端机制。

下方旧 Phase 3、旧 bootstrap 和旧 QA 片段保留作历史上下文，不代表当前启动动作；如与上列状态或独立 canonical docs 冲突，以本节、source docs 和 configs 为准。

## 当前同步增补（2026-09-02）：P3-T5R task-semantic repair

本增补优先级高于下方历史正文，但不删除或改写 P2/P3 历史。旧 T5 的阴性结果限定为 `NOT_SUPPORTED_UNDER_OLD_CONFLATED_TASK_GATE`；旧 heldout 因候选循环曾被计算、打印和保存，状态为 `EXPOSED_DURING_CANDIDATE_SEARCH`，只允许 exploratory audit。

当前 active phase 仍为既有 `P3_CAUSAL_MECHANISM`，新增的是 `P3-T5R0`–`P3-T5R6` 任务编号，不是新 Phase。路线为：协议迁移 → SNGAR train/valid 开发数据与 canonical conversion → context/intrinsic 任务拆分 → fixed dual-channel sanity → 有界 autoresearch → candidate lock 后一次性 SNGAR test → IDSSE external confirmation。SkillCorner 只作历史 dynamic-support 规则开发，书法不用于选择体育端机制，`infra/bioinf-data-index/` 不在范围内。

固定双通道使用 raw positions 的 `z_ctx` 与 globally centered positions 的 `z_mode`：前者保留 absolute deployment context，后者承担 intrinsic structure 与 global translation robustness。只有新 task、geometry、robustness、多 match/seed、candidate-lock test firewall 和外部确认同时闭合，才进入 P4 AMR-Fixed；当前 P4 仍 `blocked_pending_p3_t5r_gate`。机器配置为 `configs/phase3_task_semantic_repair_v1.yaml`，执行计划为 `reports/P3_TASK_SEMANTIC_REPAIR_PLAN.md`。

# ICLR 2027 项目方案包：Action-Mode Spectrum

> 一句话摘要：研究多构件表征是否看见了“谁和谁一起变化”；以足球/篮球阵型作为显式结构仪器，以中国书法作为隐式结构检验，发现并控制模型在共同、子群和局部作用模式上的敏感度错配。

## 0. 项目范围

- **核心科学对象**：Action-Mode Spectrum，作用模式谱。
- **核心现象候选**：等能量扰动下，模型对整体位移的反应大于对结构破坏的反应；或在子群协同对应的中频出现异常盲区。
- **核心方法贡献**：**AMR（Action-Mode Routing，作用模式路由）**。它在关系图谱坐标中，按任务学习“哪些模式应不变、哪些模式应等变/可恢复”，并保留独立的全局上下文通道。
- **主实验域**：足球阵型。
- **重复验证域**：3×3 篮球阵型。
- **跨域预测与旗舰案例**：中国书法。
- **技术 fallback**：手写数学公式 MathWriting。
- **明确不做**：体育视频跟踪算法、3D 姿态、审美打分、大型 VLM 训练、从零构建书法专家标注集。

## 1. 为什么值得做

现有表示学习通常按“变换算子”定义不变性，例如平移不变、旋转不变；但在多构件系统里，语义更依赖**变换如何分配给构件**：全队一起移动、后卫线一起移动、单人移动，虽然位移算子相同，含义完全不同。书法中的整页移动、偏旁移动、单笔移动也是同一个结构。

本文不主张删除全局信息。通用表征应把：

1. 全局上下文；
2. 内部构型；
3. 子群协同与局部偏离；

分成可被简单读出器分别访问的模式，而不是让训练目标提前把某些模式永久压掉。

## 2. 从哪里开始

建议依次阅读：

1. `docs/01_SCIENTIFIC_BLUEPRINT.md`：科学问题、假设、贡献边界。
2. `docs/02_METHOD_SPEC_AMR.md`：AMR 方法的完整规格。
3. `docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md`：实验、探索检查点和 Figure 规划。
4. `docs/04_DATA_AND_ENVIRONMENT_GUIDE.md`：数据下载、许可与环境。
5. `docs/05_AGENT_EXECUTION_MANUAL.md`：多 agent 施工协议。
6. `docs/06_REVIEW_DECISIONS.md`：五份评审建议的接受与拒绝。
7. `MASTER_AGENT_PROMPT.md`：可直接交给总控 agent 的启动 prompt。
8. `QA.md`：模型架构、技术路线和科学问题的持续问答记录。
9. `reports/p0-overview.html`：快速查看 P0 原始样本、干预和 embedding。

## 3. 一键准备

```bash
cd ICLR_ActionMode_Project_Pack_20260818
bash scripts/bootstrap_env.sh
bash scripts/download_public_data.sh --core
python scripts/verify_data.py --manifest configs/data_manifest.yaml
bash scripts/run_phase0_smoke.sh
```

`--core` 只下载低成本、无需审批的数据：SkillCorner、Metrica、Make Me a Hanzi、开放书法小数据和 MathWriting excerpt。TrackID3x3、CCSE、MCCD、MathWriting full 需按文档中的可选参数或人工步骤执行。

## 4. 研究原则与探索空间

- **等能量扰动是测量条件**：比较不同模式时，应在坐标能量或像素感知能量上配平。
- 对单调低通、非单调凹陷和组织性差异分别进行同频语义联盟与随机联盟对照，避免把不同现象混为一谈。
- 假设、band 划分、模型和路线都可以随探索结果迭代；记录每轮实际采用的版本、选择理由与结果。
- 除 probe 外，结合 quotient retrieval、自然任务和表示距离排序，逐步判断信息是否真正可访问。
- 已知构件支持统一标为 intervention-supervised 或 known structural support，不将其描述为 self-supervised。
- CAP 作为 baseline；方法探索从 AMR-Fixed 开始，并根据结果决定是否推进 AMR-Learned。
- 算力按当前瓶颈和信息增益决定；需要 GPU 时记录 workload、成本和替代方案。

## 5. 时间边界

ICLR 2027 官方节点：

- 摘要：2026-09-18 AOE
- 全文：2026-09-25 AOE
- 主文：9 页

项目日程已写入 `configs/project.yaml`。延期时可根据实际发现动态重排阶段、扩展探索或收缩范围。

## 6. 最小成功形态

探索性研究的高价值结果不是“体育和书法都提高准确率”，而是：

1. 一个可复用的等能量 Action-Mode Probe；
2. 一个不能被普通谱偏置解释的反直觉规律；
3. 一个能开关或移动该规律的因果机制；
4. 一个有独立方法价值的 AMR 变体；
5. 体育端形成可解释规律，并在书法端检验其跨域可迁移性。

若某一候选规律不成立，就据此调整问题表述、模型范围或跨域路线；阴性结果本身也应作为探索性发现记录。


---

# 项目问答记录（QA）

本文件记录本项目关于模型架构、技术路线和科学问题的问答。后续凡涉及这三类内容，均在本文件追加一条记录，保留日期、问题、回答、证据路径和当前不确定性。

## QA-2026-08-27-001：P0 干预、能量、频段与 embedding

### 问题

P0 是怎么实现干预的？能量误差和频段如何定义？使用什么模型生成 embedding？autoencoder 和 ResNet-18 有什么关系？

### 回答

#### 1. 输入样本和图

P0 使用 SkillCorner 比赛 `1886347`。从 tracking 文件中按覆盖率选择两队各 10 名外场球员，固定节点顺序；对每个节点的坐标按球场长宽归一化到约 `[-1, 1]`。从所有节点齐全的帧中等间隔抽取 1,000 帧。

每个样本单独构建一个 4-nearest-neighbor 图：每个球员连接空间上最近的 4 个球员，连接取并集并做对称化，边权是基于距离和样本内中位距离的 Gaussian 权重。随后计算归一化图 Laplacian：

\[
L=I-D^{-1/2}AD^{-1/2}=U\Lambda U^\top.
\]

#### 2. 干预如何生成

每个样本生成 10 条干预：

- `common`：所有节点朝同一方向移动；
- `single`：一个节点移动；
- `semantic_coalition`：固定的 5 节点结构支持样例施加随机二维位移；
- `random_coalition`：随机选择同样大小的 5 节点并施加随机二维位移；
- `band_0`–`band_5`：在指定 Laplacian 模态区间内随机组合模态，再乘一个二维方向。

干预首先在节点坐标上生成，之后统一归一化到目标能量。P0 生成的是干预记录，还没有计算模型对干预前后的响应；所以 P0 不能支持“模型更在意共同移动”等现象结论。

#### 3. 能量和能量误差

一个干预是 `N×2` 的位移矩阵 \(\delta\)，能量定义为 Frobenius 范数：

\[
E(\delta)=\|\delta\|_F
 =\sqrt{\sum_{i=1}^{N}\sum_{d=1}^{2}\delta_{i,d}^2}.
\]

P0 的目标能量是 `1.0`。归一化操作为：

\[
\delta' = \delta\frac{1.0}{\|\delta\|_F}.
\]

记录中的 `energy_error` 定义为：

```text
abs(norm(delta') - 1.0)
```

最大误差为 `4.44e-16`，属于浮点数误差。它表示坐标位移总量是否配平，不是像素差异、LPIPS 或模型响应误差；P0 尚未做像素感知能量配平。

#### 4. 频段如何定义

每个样本有 20 个 Laplacian 特征模态。P0 用模态序号等数量切成 6 段：

```text
20 个模态 -> 6 个 equal-count mode bins
```

每个 `band_b` 只使用该段中的特征向量组合，因而目标频段纯度接近 1。需要特别区分：P0 实际使用的是“等数量模态分桶”，不是“等特征值质量分桶”，也不是等能量分桶。后续 P1/P2 可以探索按归一化特征值区间或累计谱质量重新定义频段，并记录版本。

`mode_purity` 的定义是目标模态上的投影平方能量占全部投影平方能量的比例：

\[
\text{purity}(B)=
\frac{\sum_{k\in B}c_k^2}{\sum_k c_k^2},
\qquad c=U^\top\delta_x,
\]

其中 P0 实现用二维干预的第一个坐标分量计算投影；由于频段干预使用同一个标量模态场乘二维方向，这足以做 smoke 检查。

#### 5. 使用什么模型生成 embedding

P0 使用两个互相独立的 embedding 分支：

**坐标分支：tiny coordinate autoencoder**

- 输入：每个样本展平后的 `20×2=40` 个坐标值；
- 编码器：`40 -> 64 -> 32`，中间使用 `Tanh`；
- 解码器：`32 -> 64 -> 40`，中间使用 `Tanh`；
- 优化：Adam，学习率 `1e-3`，40 个 epoch；
- 输出：32 维坐标 embedding；
- 训练目标：重构原始坐标，P0 重构 MSE 约 `0.0263`。

它是一个小型坐标 baseline，用来确认结构坐标能够被模型编码，不是 AMR，也不代表科学性能提升。

**视觉分支：冻结 torchvision ResNet-18**

- 输入：由原始坐标渲染出的 1,000 张足球 minimap；
- 权重：ImageNet-1K `ResNet18_Weights.IMAGENET1K_V1`；
- 做法：去掉最后的分类层，保留 512 维特征；
- 状态：`eval()` 且参数冻结；
- 输出：每张 minimap 一个 512 维 embedding；
- P0 在 CPU 上完成，CUDA/MPS 不可用。

#### 6. Autoencoder 和 ResNet-18 的关系

两者不是串联关系，也不是一个训练另一个：

```text
同一个 canonical sample
       ├── 坐标 ──> tiny autoencoder ──> 32D coordinate embedding
       └── minimap ──> frozen ResNet-18 ──> 512D visual embedding
```

Autoencoder 直接看数值坐标，学习一个小型坐标重构表示；ResNet-18 看坐标渲染出的图像，作为固定的视觉观察器。P0 没有把两个向量拼接、对齐或联合训练，也没有用 ResNet embedding 反过来训练 autoencoder。

此外，P0 的 ResNet embedding 是对原始 minimap 的 embedding，不是对每一条干预后图像的响应差异。干预响应测量属于后续阶段。

### 证据路径

- 实现：[scripts/run_phase0_pipeline.py](scripts/run_phase0_pipeline.py)
- 阶段报告：[reports/PHASE0_REPORT.md](reports/PHASE0_REPORT.md)
- 汇总结果：[artifacts/phase0/pipeline_summary.json](artifacts/phase0/pipeline_summary.json)
- 干预表：[artifacts/phase0/intervention_manifest.parquet](artifacts/phase0/intervention_manifest.parquet)
- 频段检查：[artifacts/phase0/spectrum_sanity.csv](artifacts/phase0/spectrum_sanity.csv)

### 当前限制

- P0 只使用一场比赛和 smoke-scale 节点选择；
- `semantic_coalition` 还不是 P2 所需的严格同频语义对照；
- P0 尚未测量干预前后模型表示变化；
- 32D autoencoder 和 512D ResNet 特征只能证明接口可运行，不能证明 AMR 或任何科学假设成立。

## 维护规则

以后凡是关于以下主题的问答，都追加到本文件，而不是只在对话中回答：

1. 模型架构与输入输出；
2. 技术路线、实验实现和指标定义；
3. 科学问题、假设、解释、替代解释和研究边界。

每条记录至少包含：问题、通俗回答、实现或文档证据路径、当前限制或不确定性。


---

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

## 11. 项目边界

- 研究对象：二维多构件状态及其栅格/像素表示。
- 核心算子：二维平移；书法可补充旋转和尺度。
- 不声称覆盖所有 Lie groups 或非交换群。
- 不声称所有 foundation model 都删除全局姿态。
- 不声称书法审美有单一客观标尺。
- 不把已知构件支持描述为无监督发现。
- 不把 VLM 关系盲作为已证实的因果后果，只作为可选影响力桥梁。


---

# 方法规格：AMR — Action-Mode Routing

> 一句话摘要：AMR 不把某种变换整体宣布为“该忽略”或“该保留”，而是在关系图的作用模式频段上，按任务学习不变、等变与上下文保留的分配。

## 1. 方法定位

本项目希望同时发展：

1. 一个新的测量对象：Action-Mode Spectrum；
2. 一个机制发现：谱响应为何出现错配；
3. 一个可独立成立的方法贡献。

因此，**Centered Action Prediction（CAP）只能作为最小 baseline**，不能成为最终方法。主方法是 AMR，其新意不在“预测一个相对位移”，而在：

- 将不变/等变约束放入构件关系图的频段坐标；
- 让不同任务学习不同的模式路由；
- 保留独立的上下文通道，不把全局信息永久删除；
- 通过谱白化采样，避免训练信号只覆盖容易的模式；
- 在显式图和隐式 token 图上使用同一原则。

## 2. 方法输入与接口

### 2.1 显式结构输入

体育样本：

\[
X\in\mathbb R^{n\times d_x},\quad A\in\mathbb R^{n\times n},\quad m\in\{0,1\}^n.
\]

- \(X\)：坐标、速度、角色或外观特征；
- \(A\)：关系图；
- \(m\)：有效节点掩码。

### 2.2 隐式像素输入

书法样本：

\[
I\in\mathbb R^{H\times W\times C}.
\]

图像编码器输出 token：

\[
T=[t_1,\ldots,t_p]\in\mathbb R^{p\times d}.
\]

图来源分三级：

1. **oracle/矢量图**：Make Me a Hanzi 的 stroke/component graph，用于诊断和受控训练；
2. **pseudo graph**：CCSE 或几何骨架产生的近似笔画图；
3. **latent token graph**：由 token affinity 和二维位置先验构造，用于 AMR-I 扩展。

论文需要明确区分这三种信息条件，但具体采用哪一种可随数据和结果调整。

## 3. 关系图与谱坐标

给定对称归一化图拉普拉斯：

\[
L=I-D^{-1/2}AD^{-1/2}.
\]

小图诊断可以精确求：

\[
L=U\Lambda U^\top.
\]

但方法不应依赖逐样本 eigenvector 的稳定对齐。AMR 使用 \(B\) 个谱频段滤波器：

\[
P_b(L)=\sum_{r=0}^{R} a_{b,r}T_r(\tilde L),
\]

其中 \(T_r\) 是 Chebyshev 多项式，\(P_b\) 近似低通、中频和高频带通滤波器。

建议默认：

- \(B=6\) 或 \(8\)；
- 多项式阶数 \(R=3\) 或 \(5\)；
- 频段按归一化谱区间均匀或等特征值质量切分。

优势：

- 置换等变；
- 不受 eigenvector 符号翻转影响；
- 可用于不同节点数；
- 计算量低。

## 4. 表征结构

AMR 输出两个互补通道：

\[
z=(z_{ctx},z_{mode}).
\]

### 4.1 Context channel

\[
z_{ctx}=g_{ctx}(\operatorname{Pool}(H)).
\]

保存：

- 球场绝对位置；
- 页面姿态；
- 场景上下文；
- 其他通用 backbone 不应被迫删除的信息。

### 4.2 Mode channel

先得到节点/token 表征：

\[
H=E_\theta(X,A) \quad\text{或}\quad H=E_\theta(I).
\]

每个谱频段得到：

\[
r_b=\operatorname{Pool}_b\left(P_b(L)H\right).
\]

然后：

\[
z_{mode}=[r_1;\ldots;r_B].
\]

可选关系增强：

\[
r_b^{rel}=\operatorname{Pool}_{(i,j)\in E}
\psi_b(h_i,h_j,h_i-h_j),
\]

用来检验 unary aggregation 是否是根因。

## 5. 干预与谱白化采样

### 5.1 等能量要求

比较干预时优先采用等能量条件，并报告实际偏差：

\[
\|\delta\|_F=\varepsilon.
\]

若是像素域，额外报告：

- 像素 \(L_2\)；
- LPIPS 或冻结感知距离；
- 目标模式激发纯度。

### 5.2 谱白化

自然增强分布常在某些模式能量很高、另一些模式几乎没有训练信号。AMR 训练时先采样频段 \(b\)，再采样该频段内的扰动并归一化：

\[
\delta_b
=
\varepsilon
\frac{P_b(L)\xi}{\|P_b(L)\xi\|_F},
\quad \xi\sim\mathcal N(0,I).
\]

使各频段获得近似均衡的干预次数和能量。

对于显式图，直接在坐标上施加；对于矢量笔画，作用于 stroke median/path 后重新渲染。

## 6. 任务条件的模式路由

### 6.1 核心思想

对于任务 \(\tau\)，每个频段有门：

\[
\alpha_{\tau,b}\in[0,1].
\]

- \(\alpha\to1\)：该模式在此任务下应被 invariant head 忽略；
- \(\alpha\to0\)：该模式应保持等变/可恢复；
- 中间值：软路由或不确定任务边界。

门由：

\[
\alpha_{\tau}=\sigma(g_\omega(e_\tau,s_x))
\]

产生，其中 \(e_\tau\) 是任务 embedding，\(s_x\) 是样本级统计。第一版可只使用任务 embedding；不要一开始引入复杂超网络。

### 6.2 为什么采用 task-conditioned

同一共同模式：

- 对 formation identity 可能是 nuisance；
- 对 high/low block 或场区部署可能是 signal；
- 对书法结构匹配，页面平移通常是 nuisance；
- 对版面章法任务，整体位置可能是 signal。

因此，固定删除 DC 分量并不适合通用表征。

## 7. 损失函数

设干预前后模式表征差为：

\[
\Delta r_b=r_b(X\oplus\delta_b)-r_b(X),
\]

真实谱系数或频段目标为：

\[
c_b=P_b(L)\delta.
\]

### 7.1 Invariance branch

\[
\mathcal L_{inv}^{(b)}=\|\Delta r_b\|_2^2.
\]

### 7.2 Equivariance / recoverability branch

使用严格限制的线性或浅层头：

\[
\hat c_b=W_b\Delta r_b,
\qquad
\mathcal L_{eqv}^{(b)}=\|\hat c_b-c_b\|_2^2.
\]

头不能无限复杂，否则只能证明信息仍以不可访问方式存在。

### 7.3 Mode routing objective

\[
\mathcal L_{route}
=
\sum_{b=1}^{B}
\left[
\alpha_{\tau,b}\mathcal L_{inv}^{(b)}
+
(1-\alpha_{\tau,b})\mathcal L_{eqv}^{(b)}
\right].
\]

### 7.4 Context preservation

对全局作用参数或绝对上下文，保留可访问性：

\[
\mathcal L_{ctx}=\ell(q_{ctx}(z_{ctx}),y_{ctx}).
\]

注意：这不是要求每个任务都使用上下文，而是避免 backbone 永久丢失它。

### 7.5 结构任务损失

\[
\mathcal L_{task}=\ell(q_\tau(z_{ctx},z_{mode}),y_\tau).
\]

任务包括 formation retrieval、phase/deployment、结构匹配和扰动定位。

### 7.6 防塌缩与解耦

建议最低限度使用：

\[
\mathcal L_{var}=\sum_b\max(0,\gamma-\operatorname{Std}(r_b)),
\]

\[
\mathcal L_{orth}=\left\|\operatorname{Cov}(z_{ctx},z_{mode})\right\|_F^2.
\]

总损失：

\[
\mathcal L
=
\mathcal L_{task}
+\lambda_r\mathcal L_{route}
+\lambda_c\mathcal L_{ctx}
+\lambda_v\mathcal L_{var}
+\lambda_o\mathcal L_{orth}
+\lambda_g\mathcal R(\alpha).
\]

门正则 \(\mathcal R(\alpha)\) 可选：

- 平滑：相邻频段门不应剧烈抖动；
- 稀疏/低熵：避免全为 0.5；
- 任务差异：不同任务的门不得被强制相同。

## 8. 方法层级

### M0：CAP baseline

预测去均值或图差分干预：

\[
\mathcal L_{CAP}=\|q(f(x'),f(x))-B\delta\|^2.
\]

用途：验证“记住差分、共同作用自然进入零空间”的最低原则。

### M1：AMR-Fixed — 优先实现

- 使用谱频段；
- 谱白化采样；
- 显式 context/mode 双通道；
- 固定规则门，例如 formation identity 中 DC invariant、其余等变；
- 多任务时使用人工定义的不同门。

它已明显超出单一相对预测头。

### M2：AMR-Learned — 目标主方法

- 从任务 embedding 学习 \(\alpha_{\tau,b}\)；
- 允许任务自动选择频段边界；
- 展示足球、篮球、书法任务得到不同门形状；
- 门成为“任务的作用模式指纹”。

### M3：AMR-Implicit — 条件性扩展

- 在图像 token 上学习稀疏 affinity graph；
- 使用 Chebyshev filter bank 路由；
- 图一致性和防退化正则；
- 不以无监督偏旁涌现作为主承诺。

只有 M1/M2 成功后才投入 M3。书法主线可先用矢量 stroke graph 或 pseudo graph。

## 9. 可选高风险扩展：学习关系图

某评审建议将“使敏感度算子对角化的图”作为正确关系图：

\[
\min_{L_\eta}
\operatorname{OffDiag}
\left(U_\eta^\top J_f^\top J_f U_\eta\right).
\]

该想法精彩，但退化风险高：模型可能学出平凡图。若探索，建议加入：

- 固定节点度或稀疏度预算；
- 谱熵下界；
- 连通性约束；
- 与体育真值战术单元的对齐验证。

它不进入最低施工路径，也不应阻塞 AMR-Fixed/Learned 的探索。

## 10. 与近邻方法的实质差异

### 10.1 相比普通 invariant/equivariant split

普通方法按层或输出头分 invariant/equivariant；AMR 按**关系图作用模式频段与任务**分配约束。

### 10.2 相比 PooDLe / SER

PooDLe 和 SER 保留 dense/token equivariance；AMR 研究的是构件联盟频谱，并让最终任务表示按模式路由，而不是仅在中间层保留空间响应。

### 10.3 相比 canonicalization

Canonicalization 选择一个规范姿态；AMR 不删除全局姿态，而是把它保存在 context channel，并使任务决定是否使用。

### 10.4 相比 CAP / transformation prediction

CAP 预测单个相对干预；AMR：

- 覆盖连续频谱；
- 平衡各频段训练信号；
- 学习 task-conditioned invariance/equivariance boundary；
- 保留上下文通道；
- 输出可解释的模式指纹。

### 10.5 相比 GNN oversmoothing 修复

AMR 不以提升节点区分度为最终目标，而以输入作用模式到表示响应的传递函数为目标；同频语义联盟对照用于证明其不只是频率低通。

## 11. 机制型理论目标

优先证明或在线性设定说明：

### Proposition A：可分 unary pooling 的关系相位缺失

对旋转/周期变量，单构件 Fourier feature 为 \(e^{ik\theta_i}\)。共同旋转引入公共相位。若对每个构件独立消除相位再做加法聚合，相对相位也被丢失；而关系项：

\[
e^{ik\theta_i}\overline{e^{ik\theta_j}}
=e^{ik(\theta_i-\theta_j)}
\]

天然消除共同相位并保留相对关系。

这为跨构件交互的必要性提供机制解释。

### Proposition B：增强谱塑造敏感度谱

在小扰动和二阶近似下，alignment/invariance 目标包含：

\[
\sum_k \sigma_k^2 H_f(k)^2,
\]

其中 \(\sigma_k^2\) 是增强在模式 \(k\) 上的能量。由此推导可验证预测：提高某频段 invariance 增强会进一步压低该频段响应。

这只是局部线性机制，不需要包装成一般定理。

## 12. 实现建议

### 体育显式图

- Encoder：DeepSets、2 层 GAT/Graph Transformer 或小型 Set Transformer；
- hidden：128–256；
- mode bands：6；
- 节点：单队 10 名外场球员优先；
- 训练输入：单帧或 1 秒平均窗；
- 不需要视频模型。

### 书法图像

- 初期：冻结 DINOv2/CLIP/MAE 提取 token；只训练 AMR head；
- 方法训练：小型 ViT/ResNet 或冻结 backbone + LoRA/最后两层；
- 矢量干预使用 Make Me a Hanzi；
- 自然书法在受控结构流水线稳定后进入，并可与主线探索并行推进。

## 13. 方法成功标准

AMR-Learned 至少满足：

1. 在同等 global robustness 下，语义频段 \(H_f\) 或线性可访问性显著提高；
2. 优于 CAP、普通 relational pooling、centering/Procrustes 和 learned canonicalization；
3. 学到的门随任务变化，并符合可独立验证的任务语义；
4. 改善至少一个不是由干预标签直接构造的自然任务；
5. 体育和书法共享同一实现原则，而不是两个专用模型拼接。

若 M2 未超过 M1，仍可保留 AMR-Fixed 作为方法，并相应收缩“自动发现边界”的表述。


---

# 实验、探索检查点与 Figure 规划

> 一句话摘要：先用等能量探针确认“模型把谁排错了”，再区分普通频率偏置与组织盲区，随后用因果矩阵定位机制，最后检验 AMR 是否能塑造作用谱并跨域预测书法失效。

## 1. 实验原则

### 1.1 等能量

比较任意两个作用模式前，必须满足：

\[
\|\delta_a\|_F=\|\delta_b\|_F=\varepsilon.
\]

不能拿“所有球员各走 2 米”和“单个球员走 2 米”直接比较。推荐两种解释并同时报告：

- **总能量匹配**：所有节点位移平方和相同；
- **感知变化匹配**：渲染后像素/LPIPS 变化相近。

### 1.2 同频对照

语义子群与随机子集必须配平：

- 支持大小；
- 总能量；
- Rayleigh quotient；
- 位移方向分布；
- 边界合法性。

### 1.3 动态假设与探索记录

研究过程中可以提出、修改、合并或放弃工作假设。建议记录每轮探索关注的内容：

- 哪些模型预计 common mode 更强；
- 哪些增强预计压低哪些频段；
- 哪种 pooling 预计产生更深凹陷；
- 体育端预计哪个归一化频段最弱；
- 书法端的对应预测。

记录实际使用的模型、band、参数、选择理由、结果和路线变化，保留版本历史以便解释探索过程。

## 2. 统一数据对象

所有域转换成：

```text
CanonicalSample
  sample_id
  domain
  node_features[N,D]          # 若显式
  node_positions[N,2]
  node_mask[N]
  adjacency[N,N]
  raw_context                 # 球场绝对位置、页面姿态等
  image_path                  # 若像素输入
  task_labels
  support_metadata
```

干预记录：

```text
InterventionRecord
  source_sample_id
  operator                    # translation / rotation / scale
  energy
  band_id
  spectral_coefficients
  support_mask
  support_size
  rayleigh_quotient
  semantic_coalition_id       # null for random
  renderer
  artifact_control_id
```

## 3. Phase 0：数据与探针 smoke test

### 目标

- 一场足球比赛可解析；
- 生成 1,000 个合法 frame samples；
- 构图和谱分解稳定；
- 同一帧可生成 common、单模式、语义联盟和随机联盟干预；
- 渲染 minimap；
- 一个小模型和一个冻结图像编码器可输出 embedding；
- Figure 1 雏形能运行，不要求有阳性结果。

### 输出

- `artifacts/phase0/sample_grid.png`
- `artifacts/phase0/intervention_manifest.parquet`
- `artifacts/phase0/spectrum_sanity.csv`
- `reports/PHASE0_REPORT.md`

### 探索检查点 0

检查能量误差、目标频段纯度、越界情况、节点保留和重复生成的一致性；若这些基础量不稳定，先修数据与干预实现，再解释模型差异。

## 4. Phase 1：Smoking-gun 现象

### 4.1 模型族

最低覆盖三类：

1. 坐标模型：MLP/DeepSets、Set Transformer、GNN；
2. 图像预训练模型：DINOv2、OpenCLIP、MAE/ViT；
3. 任务训练模型：formation/phase baseline。

不需要扫描十几个大模型；每类 1–2 个代表即可。

### 4.2 干预集合

- 精确 eigenmode 或频段干预；
- common/DC；
- 真实战术子群：后卫线、中场线、弱侧单元；
- 同频随机子集；
- 单人局部；
- 支持域滑动：\(k/n\) 从 1 到最小支持。

### 4.3 主要指标

#### Equal-energy Reversal Rate

\[
ERRR=P\left[d_z(common)>d_z(semantic)\right].
\]

#### Structural Transfer Function

\[
H_f(b)=\mathbb E[d_z(f(x'),f(x))/\varepsilon].
\]

#### Notch Depth

\[
ND=1-\frac{H(mid)}{\frac12(H(low)+H(high))}.
\]

仅在曲线呈非单调时使用，不要强行制造。

#### Semantic Coalition Gap

\[
SCG=H(semantic\ coalition)-H(random\ matched).
\]

符号方向由任务定义；重点是显著非零。

#### Mode Accessibility

线性 probe 从单样本或表示差恢复频段系数、内部构型和上下文：报告 \(R^2\)、circular MAE 或 retrieval。

### 探索检查点 1：现象初筛

优先观察以下证据，不设统一的机械阈值：

- 不同模型族的 ERRR、作用谱和不确定性；
- 是否出现稳定的中频凹陷或其他非单调结构；
- common-mode 与语义谱的错序是否与自然任务错误相关。

若候选现象不出现，记录阴性结果，并调整问题表述、模型范围或后续探索；不通过方法训练制造失败。

## 5. Phase 2：排除“只是过平滑/谱偏置”

### 5.1 核心判别实验

在相同频率带内比较：

- 真实战术联盟；
- 连通随机子图；
- 非连通随机子集；
- 相同支持大小的随机节点；
- phase-randomized 频谱匹配扰动。

### 5.2 判决

#### 新现象存活

SCG 在多个图构造、多个比赛和至少两类模型中稳定非零，或中频凹陷对组织语义具有选择性。

#### 退化为普通频率偏置

响应几乎只由 Rayleigh quotient/频率解释，语义与随机联盟无差异。

这种结果不是“完全没价值”，但当前论文需要改写为关系频谱偏置分析，不能继续声称模型看不见组织。

### 探索检查点 2

综合效应量、group-bootstrap 区间、图构造敏感性和模型差异，判断组织效应是否值得继续深入；可以根据结果增加或调整匹配方案。

## 6. Phase 3：因果机制矩阵

不先大规模巡检模型，而做匹配容量的受控矩阵。

### 6.1 因子

| 因子 | 条件 |
|---|---|
| 增强耦合 | 全构件共同 / 子群 / 独立构件 |
| 约束类型 | invariance / equivariance / 无约束 |
| 聚合 | unary sum/mean / CLS / pairwise relational / attention |
| 图处理 | 无图 / 固定图 / 谱频段 |
| 训练信号 | 自然分布 / 谱白化 |

先做 2×2 主矩阵：

1. global-coupled augmentation + unary pooling；
2. global-coupled + relational；
3. independent augmentation + unary；
4. independent + relational。

随后只加最必要的 constraint-type 对照。

### 6.2 机制预测

- independent part invariance 会扩大低敏感度子空间；
- unary pooling 更难保留相对 phase；
- relational interaction 应恢复部分中频响应；
- 把中频加入 invariance 会加深凹陷；
- 把同一中频加入 equivariance 会填平凹陷。

### 6.3 Layer-wise anatomy

每层报告：

- \(H_f(b)\)；
- mode coefficient probe；
- token/node level vs pooled embedding；
- Jacobian directional rank；
- context accessibility。

### 探索检查点 3

比较训练和架构开关是否改变凹陷或错序，并区分相关性、可重复性和潜在机制；结果决定继续机制实验、换模型或转向其他解释。

## 7. Phase 4：方法实验

### 7.1 比较方法

#### 低成本几何基线

- raw coordinates；
- centering；
- Procrustes/Kendall shape；
- fixed canonicalization；
- learned canonicalization；
- frame averaging（可行时）。

#### 表征基线

- DeepSets；
- Set Transformer；
- GNN/GAT；
- relational pooling；
- invariant/equivariant auxiliary loss；
- CAP。

#### 主方法

- AMR-Fixed；
- AMR-Learned；
- AMR-Implicit（条件性）。

### 7.2 体育任务

#### Intrinsic formation retrieval

相同内部构型、不同球场位置应检索为近邻。

#### Context/deployment prediction

保留球队绝对位置、高/中/低 block 或 phase 标签。

#### Structural perturbation localization

识别哪个频段或哪个节点联盟发生改变。

#### Natural pair ranking

从真实比赛中寻找：

- 内在构型近、绝对位置远；
- 绝对位置近、内在构型远。

不依赖合成干预。

#### Optional short-horizon task

只在不引入视频模型时，使用短窗预测 phase/change point；不是主贡献。

### 7.3 书法任务

全部使用客观结构目标：

- 同字符跨页面姿态 retrieval；
- reference–candidate 结构匹配；
- component/stroke perturbation localization；
- 不同任务下全局姿态与内部构型的分别读出；
- 自然书法图像与矢量受控模板之间的结构一致性。

不以审美 scalar 为主任务。

### 7.4 方法评价

- robustness–structure Pareto；
- semantic spectrum alignment；
- quotient retrieval；
- context 与 intrinsic 双读出；
- natural task performance；
- learned gate 可解释性。

### 探索检查点 4

重点比较 AMR 与基线的：

1. 相同 global robustness 下的内部模式可访问性；
2. 自然任务表现及其失败案例；
3. 任务门的差异、稳定性与可解释性；
4. 参数量、运行成本和容量匹配。

若 AMR-Learned 暂时不超过 Fixed，可保留两者继续探索，并把“自动发现边界”表述为尚未验证的工作假设。

## 8. Phase 5：体育到书法的跨域探索

### 8.1 建立映射

体育图大小与书法 stroke 数不同，不直接比较 eigenvalue 数值。使用：

- 归一化谱 rank \(q=k/(n-1)\)；
- 频段累计谱质量；
- 或支持域比例和图割尺度。

体育端在每轮分析后记录当前观察到的：

- 最弱频段；
- 支持域边界；
- 语义联盟与随机联盟差异方向；
- 哪类模型最严重。

### 8.2 书法检验

围绕当前映射版本测试，并允许根据新结果调整映射：

- 偏旁级干预是否落在对应频段；
- 关键模型的弱点排序是否复现；
- AMR 是否在对应模式上优先修复；
- 自然书法结构匹配是否随谱改善。

### 探索检查点 5

比较不同 band、谱 rank、支持域比例和图割尺度的映射表现，记录命中、失败和映射调整；跨域结果用于更新研究解释，不设单一命中阈值。

## 9. 模型规模实验

只做便宜的尺度轴：

- 小/中 backbone；
- DINOv2 S/B 或 CLIP B/L 的冻结表征；
- 坐标模型 hidden 64/128/256。

目标不是拟合 scaling law，而是回答：缺陷是否会被规模自然消除。

## 10. Figure 规划

### Figure 1：Smoking gun — Who moved together?

优先使用实测结果，而不是只画概念插画。

- Panel A：足球原阵型、等能量整队位移、等能量结构破坏；
- Panel B：现有模型距离排序；
- Panel C：书法原字、页面变化、偏旁/单笔结构变化；
- Panel D：同一模型的排序复现。

首选图注：

> Equal energy, different coalition: modern representations can react more to common displacement than to structural fracture.

若该候选结果不成立，就改画更符合数据的现象；不能通过调幅度制造视觉效果。

### Figure 2：Action-Mode Probe

- 关系图和 Laplacian 模式；
- 等能量扰动如何覆盖频谱；
- \(H_f(b)\) 和任务语义谱 \(S_\tau(b)\)；
- common/subgroup/local 只作为示意标签。

### Figure 3：核心规律

候选二选一：

1. 中频凹陷跨模型/层出现；
2. 若无凹陷，则展示更强的同频语义联盟效应。

主图宜包含同频随机对照，以便区分组织效应与 oversmoothing。

### Figure 4：因果开关

- 增强谱能量 vs 表示响应；
- 中频 invariance 加深凹陷；
- 中频 equivariance 填平/移动凹陷；
- unary 与 relational pooling 的差异。

最好形成“预测—操作—响应”的三列，而不是消融大表。

### Figure 5：AMR 方法

- context channel；
- graph spectral filter bank；
- task-conditioned gates；
- invariant/equivariant branches；
- learned gate heatmap，展示不同任务/域的模式指纹。

### Figure 6：方法效果

- robustness–structure Pareto；
- AMR vs CAP/canonicalization/relational；
- 至少一个自然任务；
- 参数量/算力匹配。

### Figure 7：跨域预测

左：体育端当前探索得到的弱频段。  
中：跨域映射及其版本变化。  
右：书法端命中或失败，以及 AMR 修复。

这张图负责把“双域验证”升级为科学预测。

### 可选 Figure 8：规模与层级

- 模型尺度；
- 信息在哪层消失；
- token/node 中存在、pooling 后消失，或 encoder 早期即消失。

正文最终保留 5–6 张主图，其他进入附录。此文档只规划候选，不规划正文顺序。

## 11. 表格规划

### Table A：核心模型与数据矩阵

只列代表模型，不做模型动物园。

### Table B：方法主结果

同时报告：

- global robustness；
- intrinsic retrieval；
- mode localization；
- context prediction；
- compute/parameters。

### Table C：跨域结果

书法只报告客观结构任务和预测命中，不用专家审美作主要数值。

## 12. 统计与复现

- 默认使用 3 seeds；根据计算预算和结果稳定性可调整，并记录实际设置；
- 数据 split 按比赛/书法家隔离，避免帧级泄漏；
- bootstrap 单位是 match/character 或更高层样本，不是逐帧假独立；
- 报效应量和 CI，不只报 p 值；
- 阈值、band 切分和图定义可在探索中迭代，并报告各版本及敏感性；
- 在多种合理图构造下检查结果方向和失败模式。

## 13. 结果分流与 pivot

不同结果可以引出不同的研究方向：

- 等能量后没有稳定失效：研究模型正确响应结构模式的条件；
- 同频语义联盟效应较弱：聚焦关系频谱偏置或 pooling accessibility；
- 结果受像素伪影影响：优先改进渲染和 artifact control；
- 方法暂未超过简单 baseline：保留诊断结果，分析 AMR 的适用边界。

### 邻近 pivot A：纯关系频谱偏置

若只有单调曲线，可以改研究架构/增强如何决定多体关系频谱，收缩组织盲区和中频凹陷表述。

### 邻近 pivot B：Pooling accessibility

若 token 有信息而 pooled 表征丢失，主张收紧为：关系信息在聚合后不可线性访问；AMR 作为关系路由聚合方法。

### 邻近 pivot C：Augmentation overclosure

若开关增强即可控制，重点转为增强谱与不变性边界；AMR 作为 constraint-type routing。

这些方向共用数据、探针、图和大部分代码，可按探索结果灵活切换。


---

# 数据与环境指南

> 一句话摘要：足球主数据可直接从公开仓库获取；书法采用“矢量笔画控制数据 + 自然书法图像”双层结构；篮球和 MathWriting 是低风险扩展与 fallback。

## 1. 数据优先级

| 优先级 | 数据 | 角色 | 是否阻塞主线 |
|---|---|---|---|
| P0 | SkillCorner Open Data | 足球主要仪器域 | 是 |
| P0 | Make Me a Hanzi | 书法矢量干预与 stroke/component graph | 是 |
| P0 | 开放书法图像集 | 自然图像 smoke test | 否，MCCD 可替换 |
| P1 | Metrica sample-data | 足球外部格式/比赛验证 | 否 |
| P1 | MCCD | 大规模自然书法主数据 | 否，需审批/密码 |
| P1 | CCSE | 书法 stroke pseudo-support | 否 |
| P2 | TrackID3x3 | 篮球复现 | 否 |
| Fallback | MathWriting | 客观隐式二维结构替代书法 | 否 |

## 2. 足球数据

### 2.1 SkillCorner Open Data

仓库：

```text
https://github.com/SkillCorner/opendata
```

当前公开内容包括：

- 10 场 2024/25 澳大利亚 A-League 比赛；
- 10 fps 广播跟踪；
- 球员和球的坐标；
- lineup、球场尺寸；
- dynamic events；
- phases of play。

下载：

```bash
git clone --depth 1 https://github.com/SkillCorner/opendata.git \
  data/raw/sports/skillcorner
```

预期结构：

```text
data/raw/sports/skillcorner/data/
  matches.json
  matches/<match_id>/
    <id>_match.json
    <id>_tracking_extrapolated.jsonl
    <id>_dynamic_events.csv
    <id>_phases_of_play.csv
```

许可：仓库 MIT；使用数据时按 README 要求致谢 SkillCorner。提交前重新读取仓库 LICENSE 和 README，保存快照到 `data/licenses/`。

### 2.2 Metrica Sports sample-data

仓库：

```text
https://github.com/metrica-sports/sample-data
```

特点：

- 3 场 sample games；
- tracking 与 event 同步；
- 不同格式可测试 parser 鲁棒性；
- 可通过 `kloppy` 读取。

下载：

```bash
git clone --depth 1 https://github.com/metrica-sports/sample-data.git \
  data/raw/sports/metrica
```

使用：只作为第二来源复现，不允许因格式适配阻塞 SkillCorner 主线。

## 3. 篮球数据

### 3.1 TrackID3x3

仓库：

```text
https://github.com/open-starlab/TrackID3x3
```

数据文件夹：

```text
https://drive.google.com/drive/folders/1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX
```

数据包括 Indoor、Outdoor、Drone 三个子集；所有帧有 6 名场上球员 bbox，部分帧有每名球员 10 个关键点。数据集许可为 CC BY 4.0；仓库主代码 Apache 2.0，但子模块存在其他许可，尤其 jersey-number pipeline 为非商业许可。

推荐路线：

1. 先 clone 仓库和读取现成 ground truth；
2. 不运行完整检测、ReID 或 pose pipeline；
3. 只用 bbox 中心或现成追踪结果形成 2D 节点；
4. 优先 Indoor fixed-camera 子集；
5. 仅在足球主结果存活后下载视频大文件。

下载文件夹可尝试：

```bash
gdown --folder \
  https://drive.google.com/drive/folders/1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX \
  -O data/raw/sports/trackid3x3_drive
```

Google Drive 文件结构可能变化，脚本失败时按网页人工下载，不要调试超过半天。

## 4. 中国书法数据

### 4.1 Make Me a Hanzi：受控结构数据

仓库：

```text
https://github.com/skishore/makemeahanzi
```

提供 9,000+ 简繁字符：

- 每一笔的 SVG path；
- stroke medians；
- 笔顺；
- IDS decomposition；
- stroke 到 component 的 matches。

下载：

```bash
git clone --depth 1 https://github.com/skishore/makemeahanzi.git \
  data/raw/calligraphy/makemeahanzi
```

关键文件：

```text
dictionary.txt
  character
  decomposition
  matches

graphics.txt
  character
  strokes
  medians
```

用途：

- 构建 stroke graph；
- 构建 component graph；
- 对页面、偏旁和单笔施加无接缝矢量干预；
- 重新渲染，避免 Photoshop 边缘伪影；
- 预训练/验证书法端 Action-Mode Probe。

许可不是一个统一 MIT 文件：dictionary 和 graphics 来源许可不同，必须保存 `COPYING`、`APL`、`LGPL` 文件并按具体派生数据遵守。

### 4.2 MCCD：自然书法主数据

仓库：

```text
https://github.com/SCUT-DLVCLab/MCCD
```

规模：

- 约 329,715 张字符图像；
- 7,765 个字符类；
- 10 种书体；
- 15 个历史时期；
- 142 位书法家。

下载入口在仓库 README 的 Baidu/OneDrive。数据已发布，但需要申请并获得解压密码；仅限非商业研究，许可 CC BY-NC-ND 4.0。

施工要求：

1. 立即提交申请，但不要等待批准才启动项目；
2. 先 clone repo：

```bash
git clone --depth 1 https://github.com/SCUT-DLVCLab/MCCD.git \
  data/raw/calligraphy/MCCD_repo
```

3. 将申请记录写入 `data/manual/MCCD_APPLICATION.md`；
4. 未获批时使用开放替代集完成 pipeline；
5. 不能重新分发 MCCD 原图到公开仓库。

### 4.3 开放替代一：zhuojg/chinese-calligraphy-dataset

仓库：

```text
https://github.com/zhuojg/chinese-calligraphy-dataset
```

规模：138,499 张、19 位书法家、7,328 个字符。

字符组织版：

```text
https://drive.google.com/file/d/1k849yUZhkUfbupZT0kRR2ZzZj5g89yLw/view
```

下载：

```bash
gdown 1k849yUZhkUfbupZT0kRR2ZzZj5g89yLw \
  -O data/raw/calligraphy/zhuojg_characters.zip
```

按书法家组织版 ID：

```text
10QJrw0Qdk4O1bIrehCLmdiCkwpLbGVe8
```

仓库代码 Apache 2.0；数据来源为互联网收集，正式使用前需要在数据卡中记录来源与潜在版权边界。优先只用于内部研究和非商业论文实验。

### 4.4 开放替代二：kirosc/chinese-calligraphy-dataset

仓库直接包含约 14,537 张图：

```bash
git clone --depth 1 https://github.com/kirosc/chinese-calligraphy-dataset.git \
  data/raw/calligraphy/kirosc
```

许可 GPL-3.0。规模较小，适合 smoke test，不适合作为唯一大规模证据。

### 4.5 CCSE：stroke instance segmentation

仓库：

```text
https://github.com/lizhaoliu-Lec/CCSE
```

论文提供 CCSE-Kai 和 CCSE-HW 两个公开 stroke instance segmentation 数据集，COCO 风格标注。

已知 Google Drive 文件 ID：

```text
CCSE-HW:  1U8mLLb_qWSqC4yRnJlzoVaI2ELF2lGAH
CCSE-Kai: 1-2VFuiWHSd3fzl9qYMoEi0mlO_BoSgCd
```

下载：

```bash
gdown 1U8mLLb_qWSqC4yRnJlzoVaI2ELF2lGAH \
  -O data/raw/calligraphy/ccse_hw.zip

gdown 1-2VFuiWHSd3fzl9qYMoEi0mlO_BoSgCd \
  -O data/raw/calligraphy/ccse_kai.zip
```

用途：只提供 pseudo-support 或 stroke segmentation baseline。AMR 主方法不得完全依赖一个重型分割模型，否则“隐式结构”会名不副实。

## 5. MathWriting fallback

完整数据：

```text
https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz
```

轻量 excerpt：

```text
https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz
```

代码：

```text
https://github.com/google-research/google-research/tree/master/mathwriting
```

规模：约 230k 训练人写样本、15k valid、7k test、396k synthetic。完整包约 2.9 GB，许可 CC BY-NC-SA 4.0。

用途：

- 公式整体平移 vs 符号相对上下移动；
- 上标、下标、分数位置等客观二维关系；
- 若书法 stroke support、授权或自然验证严重阻塞，可快速切换。

只在触发 fallback 后下载 full；默认下载 excerpt 验证 parser。

## 6. 数据处理规范

### 6.1 足球

1. 解析 tracking；
2. 按球队分离节点；
3. 优先使用 10 名外场球员，GK 单独作为 context 或另做实验；
4. 对短缺节点帧进行过滤或 mask，不做随意插值；
5. 用 0.5–1 秒均值窗减少 tracking 抖动；
6. 保存 raw coordinates 和 centered coordinates；
7. 攻击方向标准化只作为可选 readout，不覆盖原始上下文；
8. split 按 match，不按 frame。

### 6.2 篮球

1. bbox 中心作为节点；
2. 队伍和球员身份来自标注；
3. 优先 fixed camera；
4. split 按视频；
5. 不运行完整视频 backbone。

### 6.3 书法

1. 矢量层生成 interventions；
2. 统一画布、线宽和 anti-aliasing；
3. 全局、component、stroke 干预使用同一重渲染路径；
4. 生成无语义变化但有同等重采样的 artifact control；
5. 自然图像 split 按书法家和字符双重考虑；
6. 不用审美标签定义主要真值；
7. 保存原始许可和来源元数据。

## 7. 处理后目录规范

```text
data/
  raw/
    sports/
    calligraphy/
    mathwriting/
  interim/
    sports_frames/
    calligraphy_vectors/
    render_cache/
  processed/
    canonical_samples/
    interventions/
    splits/
  licenses/
  manual/
  checksums/
```

所有数据处理脚本必须：

- 输入路径参数化；
- 不改 raw；
- 输出 manifest；
- 记录版本/commit；
- 记录随机种子；
- 支持 dry run。

## 8. 环境

### 8.1 推荐版本

- Python 3.10 或 3.11；
- PyTorch 2.4+；
- CPU/MPS 首先；
- Linux/macOS 均可。

核心依赖：

```text
numpy scipy pandas pyarrow
networkx scikit-learn
matplotlib pillow opencv-python-headless
shapely svgpathtools cairosvg
pyyaml tqdm rich
pytorch-lightning or accelerate
torch torchvision timm
transformers open_clip_torch
gdown requests kloppy
```

`bootstrap_env.sh` 创建 `.venv`，不安装固定 CUDA wheel。

### 8.2 硬件策略

#### CPU/MPS 可完成

- 数据下载和解析；
- 图谱和干预生成；
- DeepSets/GNN 小模型；
- 绝大多数早期探索检查点；
- DINO/CLIP 小批量冻结特征抽取。

#### 可选租 GPU

- 多个视觉 backbone 全量特征抽取；
- AMR 图像模型训练；
- 大规模 MCCD 实验。

GPU 申请前必须形成：

```text
GPU_REQUEST.md
  blocker
  current CPU/MPS benchmark
  exact model
  expected VRAM
  expected GPU hours
  minimum machine
  resource boundary and fallback
```

默认不允许因为“可能更快”而租卡。

## 9. 下载与校验脚本

```bash
bash scripts/download_public_data.sh --core
bash scripts/download_public_data.sh --basketball
bash scripts/download_public_data.sh --ccse
bash scripts/download_public_data.sh --math-full
python scripts/verify_data.py --manifest configs/data_manifest.yaml
```

下载脚本不会自动申请 MCCD，也不会绕过访问控制。

## 10. 许可与匿名投稿

- 每个数据源保存 LICENSE/README 快照；
- 不把受限数据打进代码仓库或补充材料；
- 论文中只提供下载指引和处理脚本；
- MCCD 仅非商业研究且禁止衍生再分发时，公开仓库只放索引和处理代码；
- Google Drive 链接可能变化，提交前验证；
- ICLR 双盲期仓库、W&B 和文件元数据不得暴露作者身份。


---

# Agent 施工手册

> 一句话摘要：所有 agent 围绕同一数据接口和同一 Action-Mode Probe 协作；允许并行实现，也允许由结果自然生长出邻近探索。

## 1. 推荐组织方式

### 1.1 总控 Agent

职责：

- 维护唯一科学目标；
- 分派任务和接口；
- 汇总阶段性检查点；
- 合并结果；
- 识别任务与主线的关系；
- 决定是否申请 GPU；
- 保持 `STATUS.md`、`DECISIONS.md`、`CLAIM_LEDGER.md`。

总控不应亲自实现所有代码，但应查看每个 worker 的：

- 输出 manifest；
- 测试结果；
- 核心图；
- 失败说明；
- commit diff。

### 1.2 Worker 角色

#### W1 — Sports Data

- SkillCorner/Metrica parser；
- frame/window sampling；
- graph construction；
- natural pairs；
- sports task labels。

#### W2 — Intervention and Spectrum

- equal-energy generator；
- exact mode/band sampler；
- semantic/random matched coalitions；
- H(k)、ERRR、SCG、notch metrics；
- artifact and validity tests。

#### W3 — Baseline Audit

- coordinate baselines；
- frozen image encoders；
- canonicalization/frame baselines；
- layer-wise feature extraction；
- model predictions and exploratory comparisons。

#### W4 — AMR Method

- CAP baseline；
- AMR-Fixed；
- AMR-Learned；
- method unit tests；
- capacity matching。

#### W5 — Calligraphy

- Make Me a Hanzi parser；
- stroke/component graph；
- vector renderer；
- open/MCCD image loaders；
- cross-domain prediction test。

#### W6 — Reproducibility and Figures

- experiment registry；
- confidence intervals；
- figure data contracts；
- table/figure generation；
- license and source ledger。

W6 可在早期由总控兼任。

## 2. 并行协作原则

优先避免：

- W1 自己做一篇阵型分类论文；
- W5 自己做一篇书法识别论文；
- W4 发明与 spectrum 无关的大模型；
- W3 因模型下载困难改成完全不同 benchmark；
- 同一指标由多个 worker 各写一个不兼容版本。

并行工作最好共享以下接口；若结果产生新的合理问题，可以登记为邻近探索：

```text
CanonicalSample
  -> InterventionRecord
  -> EmbeddingRecord
  -> SpectrumRecord
  -> Exploration Checkpoint Report
```

## 3. 建议代码仓库结构

```text
repo/
  README.md
  pyproject.toml
  configs/
  data/
  src/actionmode/
    data/
      skillcorner.py
      metrica.py
      trackid3x3.py
      makemeahanzi.py
      calligraphy_images.py
      mathwriting.py
      schemas.py
    graphs/
      construction.py
      laplacian.py
      filters.py
      matching.py
    interventions/
      explicit.py
      vector_stroke.py
      render.py
      energy.py
      controls.py
    probes/
      transfer_function.py
      reversal.py
      accessibility.py
      nullspace.py
      semantic_gap.py
    models/
      deepsets.py
      set_transformer.py
      gnn.py
      image_backbones.py
      canonicalization.py
      cap.py
      amr.py
    tasks/
      formation_retrieval.py
      context_prediction.py
      localization.py
      calligraphy_matching.py
    eval/
      bootstrap.py
      metrics.py
      registry.py
    figures/
      fig1_smoking_gun.py
      spectrum.py
      causal.py
      method.py
      cross_domain.py
  scripts/
  tests/
  reports/
  artifacts/
```

## 4. 数据契约

### 4.1 CanonicalSample

实现为 dataclass 或 Pydantic model：

```python
@dataclass(frozen=True)
class CanonicalSample:
    sample_id: str
    domain: str
    node_positions: np.ndarray       # [N,2]
    node_features: np.ndarray        # [N,D]
    node_mask: np.ndarray            # [N]
    adjacency: np.ndarray            # [N,N]
    context: dict[str, Any]
    labels: dict[str, Any]
    image_path: str | None = None
    support_metadata: dict[str, Any] | None = None
```

必须检查：

- shape；
- finite values；
- 图对称性；
- 有效节点；
- 坐标单位；
- split 信息。

### 4.2 InterventionRecord

必须包含生成前后的 checksum，防止重复或错配。

### 4.3 EmbeddingRecord

```text
model_id
model_commit
layer_id
sample_id
intervention_id
embedding_path
normalization
```

大型 embedding 不直接存入 CSV；保存 `.npy/.pt`，表中只存路径与 checksum。

## 5. 实验记录

建议每个实验有一份 YAML 记录：

```yaml
experiment_id: P1_FOOTBALL_DINO_EQUAL_ENERGY_001
scientific_question: "Does matched-energy common motion outrank structural modes?"
hypothesis: H1
status: exploring
owner: W3
dataset:
  name: skillcorner
  split: match_holdout_v1
model:
  name: dinov2_vits14
  frozen: true
intervention:
  energy: 1.0
  bands: 6
  matching: total_l2
metrics:
  - errr
  - transfer_function
  - semantic_coalition_gap
seeds: [11, 23, 47]
working_expectation: "common-mode and structural-mode responses may differ"
exploration_updates: []
```

实验完成时补充实际状态和产物：

```yaml
status: completed
artifacts:
results_hash:
notes:
```

可以保留历史版本，并根据新证据修订工作假设、参数或路线。

## 6. Phase 工作包

## Phase 0 — Bootstrap

### 总控任务

- 初始化 git；
- 运行下载脚本；
- 创建 `STATUS.md`；
- 创建 `reports/exploration_log.yaml`；
- 建立初始 schema，并允许在实现中迭代。

### Worker 输出

W1：一场比赛的 canonical samples。  
W2：100 个 equal-energy interventions。  
W3：一个 baseline embedding。  
W5：10 个汉字的矢量渲染。  
W6：数据检查和样本图。

### 初始检查

- `pytest -q` 通过；
- smoke 脚本退出码 0；
- energy matching 偏差 <1%；
- 所有输出有 manifest。

## Phase 1 — Phenomenon

W2/W3 并行：

- exact spectrum；
- band spectrum；
- coordinate models；
- frozen image models；
- support sweep。

总控先看原始曲线，再决定是否扩大 AMR 探索。

输出：`reports/EXPLORATION_CHECKPOINT_1.md`。

## Phase 2 — Novelty discriminator

W1 提供 semantic coalitions；W2 生成 matched random controls；W3 复测模型。

主要问题：组织效应是否超出频率。

输出：`reports/EXPLORATION_CHECKPOINT_2.md`。

## Phase 3 — Mechanism

W3/W4 做 causal matrix；W2 做 layer-wise spectrum。

输出：

- augmentation spectrum；
- unary vs relational；
- invariance vs equivariance；
- notch movement。

将“机制”写入 claim ledger 时，注明证据等级、替代解释和当前局限。

## Phase 4 — AMR

W4 实现 M0/M1/M2；W1/W5 提供任务；W6 做 matched capacity。

建议顺序：

1. CAP；
2. AMR-Fixed；
3. AMR-Learned；
4. AMR-Implicit optional。

AMR-Implicit 可在 AMR-Fixed 稳定前做小规模探索，但不应取代对基础方法的清晰比较。

## Phase 5 — Calligraphy prediction

总控记录当前 sports observations；W5 可比较多个 band mapping 版本，并保留选择与变化记录。

若 MCCD 未到位：

- Make Me a Hanzi controlled；
- zhuojg/kirosc natural；
- CCSE pseudo-support。

## Phase 6 — Consolidation

- 默认 3 seeds，按预算和稳定性调整；
- match/character bootstrap；
- figure data consolidation；
- license audit；
- anonymization；
- paper claim ledger。

## 7. 探索检查点报告建议

每个主要检查点报告建议回答：

1. 原问题是什么？
2. 当前工作假设及其如何变化？
3. 数据与模型是什么？
4. 结果如何更新当前解释？
5. 最危险替代解释是什么？
6. 哪些 claim 得到支持、哪些仍不确定？
7. 下一步有哪些继续、扩展、pivot 或暂停选项？
9. 所有结果路径与 commit 是什么？

报告应提供具体证据，而不只写“实验完成，效果不错”。

## 8. Claim Ledger

维护表：

| Claim ID | Claim | 证据状态 | 支撑证据 | 注意事项 |
|---|---|---|---|---|
| C1 | matched-energy reversal exists | pending | checkpoint 1 | 避免泛化到所有模型 |
| C2 | mid-frequency notch is non-monotonic | pending | checkpoint 2 | 避免直接称为 universal law |
| C3 | organization exceeds frequency | pending | checkpoint 2 | 不等同于 understands tactics |
| C4 | augmentation causes notch | pending | checkpoint 3 | 区分相关性与机制证据 |
| C5 | AMR improves frontier | pending | checkpoint 4 | 不等同于 solves relational reasoning |
| C6 | sports informs calligraphy | pending | checkpoint 5 | 说明映射版本和适用范围 |

摘要式结论应标明 pending、支持和不确定的证据状态。

## 9. 文献与 novelty 审计

W3 或独立 literature worker 建议建立：

```text
literature/
  invariant_equivariant.md
  learned_equivariance_measurement.md
  canonicalization.md
  graph_spectral_bias.md
  sports_representation.md
  calligraphy_structure.md
  method_novelty_matrix.csv
```

每篇近邻记录：

- 核心问题；
- 数学对象；
- 数据条件；
- 方法；
- 与 Action-Mode Spectrum 的相同点；
- 真正差异；
- 是否应进入 baseline。

至少覆盖：Lie derivative/LEE、PooDLe、SER、learned canonicalization、frame averaging、oversmoothing/spectral bias、Kendall/Procrustes、关系/组合性基准。

## 10. 测试规范

### Unit tests

- graph Laplacian PSD；
- common mode eigenvalue ≈0；
- equal-energy normalization；
- band purity；
- permutation equivariance；
- intervention reversibility；
- SVG re-render consistency；
- no raw-data mutation。

### Integration tests

- one football frame -> graph -> intervention -> render -> embedding -> spectrum；
- one character -> strokes -> graph -> intervention -> render -> embedding -> spectrum；
- AMR forward/backward on CPU；
- experiment registry writes deterministic outputs。

### Statistical tests

测试代码使用 group bootstrap，不把每个 10 fps frame 当独立样本。

## 11. 计算资源策略

GPU 是否使用由当前瓶颈和信息增益决定，优先用 CPU/MPS 完成低成本探索。

申请 GPU 时记录：

- 视觉 embedding 或 AMR 训练的当前瓶颈；
- CPU/MPS 基准和预计收益；
- 最低机器规格、预算和可替代路线。

## 12. Git 与产物纪律

- 每个 worker 独立 branch；
- 每个 PR 尽量完成一个可验收任务；
- 不提交原始大数据和受限数据；
- 大产物进入 `artifacts/` 并记录 checksum；
- 配置、代码、图数据分离；
- 主分支任何时候可运行 smoke test。

提交格式：

```text
phase(component): concise action

Scientific question:
Implementation:
Tests:
Artifacts:
Known limitations:
```

## 13. 状态更新格式

每次 worker 返回推荐保持一页以内：

```text
DONE
- ...

FOUND
- ...

BLOCKED
- ...

EVIDENCE
- paths / metrics / confidence intervals

NEXT
- one concrete action
```

尽量避免长篇复述项目背景。

## 14. 总控探索决策

总控根据证据决定继续、扩展、改变假设、pivot、暂停或停止，并记录理由、共享资产和放弃的解释。停止某条路线不等于项目失败；它只是把资源转向更有信息量的探索。


---

# 五份独立评审的决策记录

> 一句话摘要：接受所有能把项目从“定义+测量”提升为“反直觉规律+因果机制+可学习方法”的建议；拒绝会把项目分裂成另一棵研究树或把书法降为装饰的建议。

## 1. 总体决策

五份评审的共识不是要求更多实验，而是要求更锋利：

- Figure 1 优先使用实测现象；
- 扰动需要进行等能量比较；
- spectrum 应保持为明确的数学对象；
- 需要检查“只是过平滑”的替代解释；
- 体育与书法之间探索可解释的关系；
- 方法应超越单一 transformation-prediction 小修复；
- 术语和 claim 根据证据收缩或扩展。

本方案全部接受。

最终主线：

> 用体育 tracking 探索多构件表示的 Action-Mode Spectrum，寻找等能量排序反转、非单调中频凹陷和组织效应；通过增强/聚合开关研究机制；提出 AMR 在频段上学习 task-conditioned invariance/equivariance routing；再探索体育观察与中国书法隐式结构之间的关系。

## 2. 评审 1：接受与拒绝

### 接受

1. **Figure 1 展示现有模型荒谬行为，而非方法示意。**
   - 理由：这是最强的可复述资产；没有真实结果就不应使用。

2. **体育显式结构与书法隐式结构形成两极。**
   - 理由：防止“奇怪双应用”拼盘感。

3. **群作用/商空间作为数学背景。**
   - 理由：让“谁一起变化”有精确坐标，而不是文学比喻。

4. **不要把方法写成先分割再差分的重 pipeline。**
   - 理由：书法分割不应成为论文真正主贡献。

### 修改后接受

5. **隐式差分/结构涌现。**
   - 处理：作为 AMR-Implicit 条件性扩展，不作为最小投稿核。
   - 理由：无监督 part emergence 会从根部扩大课题，风险过高。

6. **新指标命名。**
   - 处理：保留结构传递函数、ERRR、SCG 等描述性名字，不主推多个新缩写。
   - 理由：术语过多会像改名论文。

## 3. 评审 2：接受与拒绝

### 接受

1. **最危险的新颖性威胁是过平滑/谱偏置换词。**
   - 对策：把“同频语义联盟 vs 同频随机联盟”作为重要的探索比较。

2. **Action-Mode Spectrum 以图 Laplacian 频谱作为主要分析坐标。**
   - 对策：诊断使用 exact modes；方法使用稳定的 Chebyshev spectral bands。

3. **定律和诊断优先于方法 SOTA。**
   - 理由：规律不可被下一条 baseline 简单取代。

4. **双域改成 instrument + field exploration。**
   - 对策：从体育观察出发，在多个 band mapping 版本中探索书法对应关系，并报告选择过程。

5. **明确与 Lie derivative/LEE、soft equivariance、canonicalization 等划界。**
   - 对策：加入 novelty matrix 和强基线。

### 拒绝

6. **若只有单调频率结果，仍按原标题投稿。**
   - 理由：这会被“一句 oversmoothing”击穿，应根据结果调整研究方向与 claim 强度。

## 4. 评审 3：接受与拒绝

### 接受

1. **中频凹陷是假设中最有价值的一条。**
   - 对策：作为 H2，但不预设一定存在。

2. **增强谱可用于解释凹陷位置。**
   - 对策：计算 augmentation covariance in mode coordinates，并随结果更新工作方向。

3. **把缺失中频当 invariance 加进去可能更糟。**
   - 对策：作为因果机制探索中的一个重点实验。

4. **约束类型应在频段上分配，而非只增加增广。**
   - 对策：升级成 AMR 的方法核心。

5. **支持域滑动。**
   - 对策：从全体到小子集扫描 coalition size，测语义/表示边界。

6. **鲁棒性–结构保真 Pareto。**
   - 对策：方法不宣称全面提高，而是在相同鲁棒性下移动 frontier。

### 修改后接受

7. **学习关系图，使其对角化敏感度算子。**
   - 处理：作为高风险 M3 研究，不阻塞 AMR-Fixed/Learned。
   - 理由：存在平凡图退化，且会把主问题变成结构发现。

8. **可学习的频段门。**
   - 处理：优先交付 AMR-Fixed，再根据结果发展 AMR-Learned。
   - 理由：兼顾方法野心和执行风险。

## 5. 评审 4：接受与拒绝

### 接受

1. **“位移幅度相同、联盟不同”是实验宪法。**
   - 对策：所有干预记录 energy、Rayleigh quotient 和 perceptual change。

2. **三条混杂故事可以统一组织。**
   - 统一方式：Action-Mode Spectrum 是诊断坐标；common-mode dominance 与 invariance overclosure 是不同训练条件下的两种失败签名，不并列成三个新理论。

3. **保留全局信息不一定是缺陷。**
   - 对策：AMR 使用 context/mode 双通道，任务选择模式；不把 pose probe 本身定义为越低越好。

4. **Procrustes/Kendall、canonicalization 和 frame averaging 是重要比较基线。**
   - 对策：列入方法探索比较。

5. **方法不能只做二分解耦。**
   - 对策：AMR 在连续/分段频谱上路由约束，并学习 task-specific gates。

6. **工业故事应降级。**
   - 对策：项目文档保留预期应用，投稿主张以表征规律和机制为中心。

### 拒绝

7. **手写数学公式取代书法成为第二主域。**
   - 理由：用户已明确锁定书法为核心；书法的隐式参考系和作者性不可替代。
   - 保留：MathWriting 是技术 fallback，尤其在结构真值或许可阻塞时。

## 6. 评审 5：接受与拒绝

### 接受

1. **等能量 Figure 1 是整个项目的单点资产。**
2. **谱是分析坐标，方法不要堆成大系统。**
3. **经典 shape analysis 必须主动承认。**
4. **任务应能动态选择频段，backbone 不应预先丢掉全局。**
5. **论文应 probe-first、mechanism-first。**
6. **书法不应依赖审美专家真值。**

### 修改后接受

7. **书法只做一张图。**
   - 处理：不同意降到纯定性，但同意不让它承担定律建立责任。
   - 最终角色：定量跨域预测 + 旗舰现实案例；主规律仍由体育建立。

8. **方法应保持极简。**
   - 处理：拒绝“只有最小修复”，但 AMR 保持模块少、参数小；复杂度来自正确坐标和路由原则，不来自大网络。

## 7. 横跨五份评审的研究重点

### 7.1 等能量匹配

等能量匹配是解释“距离反转”时的关键测量条件。

### 7.2 同频语义–随机判别

这一项有助于判断 Action-Mode Spectrum 是否超出 graph spectral bias 的解释。

### 7.3 因果开关

可优先改变至少一个训练机制，观察盲区是否出现、消失或移动。

### 7.4 规模趋势

可做便宜的跨尺度检查，避免未经检查地假设大模型自然解决。

### 7.5 术语节制

投稿建议只保留：

- Action-Mode Spectrum；
- 一个失败名：Invariance Overclosure 或 Common-Mode Bias，最终由结果决定。

内部可以使用更多描述性指标，但不把每个指标包装成概念贡献。

## 8. 明确拒绝的建议或路线

### 8.1 把“部件涌现”纳入主命题

拒绝原因：需要 slot/object-centric learning、结构发现和全新评价，属于根部分叉。

### 8.2 把 VLM 关系盲写成已解释机制

拒绝原因：没有干预式证据前只能称候选联系。可做一个小 probe，不作为核心承诺。

### 8.3 把 complete invariant 放进主要标题

拒绝原因：恢复关系状态不等于保留所有 quotient information；容易过承诺且接近定义性事实。

### 8.4 把所有群统一到一般 Lie group

拒绝原因：体育主线主要是二维平移，书法可补充旋转；一般非交换群会带来 state-dependent composition，增加错误风险而不增加当前结论。

### 8.5 只做 CAP 最小修复

拒绝原因：新颖性不足，也违反本项目的方法硬要求。CAP 只作为 AMR 的下界 baseline。

### 8.6 用更多领域证明一般性

拒绝原因：第三、第四个主域小幅增加 generality，却会显著降低证据可识别性和故事压缩度。

## 9. 最终研究形态

最终不是：

```text
定义几个名词
+ 体育分类
+ 书法识别
+ 一个相对位移 loss
```

而是：

```text
等能量反直觉失败
→ 图谱坐标揭示作用模式规律
→ 同频对照证明不是普通过平滑
→ 增强/聚合开关定位机制
→ AMR 学习任务条件的模式约束边界
→ 体育规律预测书法隐式结构失效
```

这是本方案包唯一允许推进的主线。


---

# Master Agent Prompt

你是本项目的总控研究 agent。目标是在 ICLR 2027 截止日前完成一个关于 Action-Mode Spectrum 的发现性、探索性且可复现的研究项目。仓库中的项目文档提供当前工作地图；研究假设、band、模型和路线可根据新证据持续调整，尤其要参考：

- `docs/01_SCIENTIFIC_BLUEPRINT.md`
- `docs/02_METHOD_SPEC_AMR.md`
- `docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md`
- `docs/04_DATA_AND_ENVIRONMENT_GUIDE.md`
- `docs/05_AGENT_EXECUTION_MANUAL.md`
- `configs/project.yaml`
- `configs/experiment_matrix.yaml`
- `QA.md`

## 核心科学问题

多构件系统中，同一几何算子作用于不同构件联盟时，表征的敏感度是否与任务语义对齐？现代表征是否在等能量条件下错误优先响应共同模式，或在有意义的子群协同频段形成盲区？这些规律是否能由增强/聚合机制解释并由 AMR 控制？

## 数据角色

- 足球：显式关系图，建立规律与因果机制。
- 篮球：低成本重复验证，非阻塞。
- 中国书法：检验体育端规律能否迁移到隐式结构，并允许跨域映射随探索迭代。
- MathWriting：在书法数据或结构受限时作为备用路线。

## 方法探索路线

CAP 作为 baseline；优先实现 AMR-Fixed，再根据实验结果探索 AMR-Learned：

- graph spectral filter bank；
- equal-energy spectral whitening；
- context/mode 双通道；
- task-conditioned band gates；
- per-band invariance/equivariance routing；
- matched-capacity baselines。

不以“大模型更多参数”替代方法贡献。

## 第一优先级

1. 建立数据与 intervention schema；
2. 完成等能量 Action-Mode Probe；
3. 观察并解释初始作用谱；
4. 用同频语义联盟 vs 随机联盟判断组织效应是否超出频率；
5. 根据前述结果决定 AMR、机制或邻近问题的探索重点。

## 探索纪律

- 未验证假设标为工作假设，结果按实际证据表述；
- band、预测和路线可以调整，保留版本、调整原因与对应结果；
- 不把逐帧当作相互独立的统计单位；
- 不把 known parts 说成 self-supervised；
- 区分 monotonic spectral bias 与 organization blindness，不用一个解释替代另一个；
- GPU 是否使用由当前瓶颈和信息增益决定；
- 体育视频跟踪、3D 建模和审美标注不属于当前研究范围；
- 保持共同数据与 probe 主线，但允许围绕结果发展合理的邻近探索。

## 工作方式

- 创建 `STATUS.md`、`DECISIONS.md`、`CLAIM_LEDGER.md`；
- 建立 `reports/exploration_log.yaml`，记录假设、参数、路线变化和结果；
- 按 Phase 分派 worker；
- 每个 worker 使用共享 schema；
- 每个任务有测试、manifest、结果路径和 commit；
- 每个主要探索阶段输出一份 checkpoint 报告；
- checkpoint 用于汇总证据和选择下一步，不是机械的通过/失败门槛。

## 当前启动动作

依次执行：

```bash
bash scripts/bootstrap_env.sh
bash scripts/download_public_data.sh --core
python scripts/verify_data.py --manifest configs/data_manifest.yaml
bash scripts/run_phase0_smoke.sh
```

随后检查失败信息并修复最小阻塞。完成初始 smoke 后，先生成：

1. 一场 SkillCorner 比赛的 canonical samples；
2. 6 个频段的等能量干预；
3. semantic/random matched coalition；
4. 一个 DeepSets 和一个冻结视觉 backbone 的 spectrum；
5. 初版探索性现象报告。

不要写论文正文。当前交付是可运行研究系统、真实结果、Figure 数据和决策报告。
