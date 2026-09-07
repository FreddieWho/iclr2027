# ICLR 2027 发表胜算综合评审（内部评估＋外部检索）

日期：2026-09-05
评估对象：项目当前全部成果（P0→P2→P3→P3-T5R→T5R5→T5R6）
评估目的：判断投稿 ICLR 2027 的胜算，给出调整/改进清单
状态：内部评估与外部检索均已完成，结论见第三部分

---

## 第一部分：内部评估（基于仓库全部证据）

### 1.1 当前可投稿的核心资产

1. **测量对象**：Action-Mode Spectrum——把"位移算子 × 支持联盟 × 任务"索引的
   表征响应变为一等测量对象；等能量扰动重新分配到不同 support 作为仪器。
2. **诚实的负结果与异质性刻画**（P2）：fracture 响应 `mixed_or_graph_specific`，
   依赖 architecture/graph/role/energy；exact-spectrum 对照排除平凡解释。
3. **机制预测器**（P3）：归一化 Jacobian Gram + 图拉普拉斯谱特征预测干预响应，
   retrospective/prospective Spearman ≈ 0.85，远超频率/residual 基线（≈0.12）；
   异质性可定位到网络阶段；证据选择的因果开关（仅 response shaping，已如实记录）。
4. **任务语义修复**（T5R）：context/intrinsic 双通道＋共享编码器 2:1 更新配比
   （两轮有界 autoresearch、冻结规则选出、非后验）；单次保留场 J03WQQ 确认
   PASS_STRONG。
5. **独立外部确认**（T5R6）：SoccerTrack-v2 8 场同方向（任务 8/8 占优、
   干预 full 8/8 高于谱基线、方向 0.66–0.73）。
6. **流程资产**（可转化为论文的 reproducibility 卖点）：claim ledger、split/baseline/
   candidate/intervention/shadow 五层锁、单次读取纪律、审计脚本、全链路 SHA 清单。

### 1.2 强项（内部视角）

- **证据链完整度罕见**：预测→定位→因果开关→任务意义→外部确认，逐环有锁、有
  审计、有边界措辞。这在 ML 会议投稿中极稀缺，可作为 differentiator。
- **问题形式化清晰**：G_f 的 2×2 block 分解（对角=自敏感度，非对角=耦合）与
  fracture 仪器的对应关系，是论文最干净的理论锚点。
- **负结果处理诚实**：P2 的 mixed 结论、P3-C6 的 NOT_SUPPORTED 均如实冻结——
  审稿人最恨 overclaim，本项目天然免疫。
- **外部确认是独立 provider**，非同质数据切分，回应了最常见的"单数据集"质疑。

### 1.3 弱项与风险（内部视角，按严重度排序）

1. **【致命级】时间线**：今日 2026-09-05，ICLR 2027 截稿大概率在 2026 年 9 月中
   下旬（待外部检索确认）。手稿尚未开始撰写。剩余时间以"周"计，任何新增支柱
   （AMR、书法、篮球）都不可行；可行范围=把已有证据写成论文＋极少量高价值
   低成本实验。
2. **【高危】方法贡献缺旗舰**：AMR（P4）NOT_STARTED。没有 AMR，论文的
   "方法贡献"是双通道＋2:1 配比——审稿人可能视为"一个简单的设计选择"而非方法。
   对策：把论文定位改为"仪器＋现象＋机制＋原则性修复"，AMR 作为 future work
   并用 gate 故事说明为什么不贸然做。
3. **【高危】规模与基线**：模型仅 ~140k 参数、单一自监督家族、GAT/Phase-GAT。
   审稿人必问"现代/大模型上是否成立""与标准 SSL 基线（SimSiam/BYOL/DINO 类）
   比如何"。当前无对照。这是**最高价值的可补实验**（同训练管线、低成本）。
4. **【中危】跨域主张未兑现**：README 声称篮球复现域＋书法旗舰案例，实际均
   blocked。要么删，要么放 future work 并明确；写在摘要里而未兑现会招拒稿。
5. **【中危】下游收益幅度小**：任务指标改善幅度有限（pair −0.0026 即 gate 以
   "非劣"通过）。论文卖点必须是机制理解而非任务提升，措辞要严格对齐 ledger。
6. **【中危】故事复杂度高**：P2→P3→T5R→T5R5→T5R6 五段弧，9–10 页正文极难
   装下。需要大幅压缩 P2 为一节＋把协议细节全部放附录。
7. **【低危】2:1 是单个标量**：须表述为"update allocation 是实质变量"这一原则
   的一个实例，而非调出来的超参数。

### 1.4 建议的论文形状（草案，待外部检索修订）

**标题方向**：机制先行（mechanism-first）——"Do learned multi-agent
representations see who moves together? A support-conditioned local-geometry
account, with a task-semantic repair and independent external confirmation."

**结构**（9 页正文）：
1. Intro：operator×support×task 问题（1 页，含 spectrum 概念图）
2. Instrument：等能量 support-reallocation＋formalization（1 页）
3. Phenomenon：P2 压缩版（heterogeneity is real, graph/architecture-dependent）（1 页）
4. Mechanism：G_f block 分解＋预测（0.85）＋定位（2 页，论文核心）
5. Repair：双通道＋update allocation 原则＋单次保留场确认（1.5 页）
6. External confirmation：SoccerTrack-v2 8 场（1 页）
7. Related work＋Limitations＋Reproducibility statement（1.5 页）
附录：全部锁、审计、hash、per-match 表。

**AMR**：future work 一段，明确 gate 逻辑（机制闭合才启动方法训练）——
这本身是有原则的研究叙事，审稿人通常买账。

### 1.5 时间约束下的增减建议（草案）

增加（仅高价值低成本）：
- [ ] 标准 SSL 基线对照（SimSiam/BYOL 类，同数据同 harness）——直接回应规模质疑
- [ ] 一张"概览 Figure 1"（spectrum＋仪器＋预测散点）
- [ ] Reproducibility statement（把锁＋审计＋hash 写成一段差异化卖点）

删减/降级：
- [ ] 篮球 3×3、书法、AMR、MathWriting 全部移出正文，进 future work/附录
- [ ] P2 压缩为一节；T5R4 两轮 autoresearch 细节进附录

---

## 第二部分：外部检索结果（4 个研究 subagent，2026-09-05 完成）

完整简报归档于 `reports/research_briefs_20260905/`（4 份，含全部引用链接）。

### 2.1 ICLR 2027 官方时间线与规则（全部官方来源核实）

- **摘要截止 2026-09-18 23:59 AoE（距今 13 天）；全文截止 2026-09-25 23:59 AoE（距今 20 天）**。
  逾期无任何通融；**摘要截止后不得新增作者**；作者顺序可改到全文截止。
- 评审 10/1–10/21，评审意见 11/5 放出，rebuttal 11/5–11/18，最终决定 12/16。
  会议 2027-04-26~30，旧金山 Moscone。
- 正文 9 页（rebuttal/终版 10 页），参考文献不限页；严格双盲（正文或补充材料泄露身份→desk reject）。
- 2027 新规：每作者最多 20 篇；无合格互审审稿人的团队最多 1 篇；**强制 AI 使用披露章节**（不计页数）；作者审稿义务与 desk-reject 挂钩（≥1 篇→至少审 3 篇）。
- **最紧急的行政风险**：所有作者需 OpenReview profile；无机构邮箱的 profile 审核可长达 2 周——等于现在剩余的全部时间。今天就必须建。
- ICLR 2026 背景：27.4% 接收率，779 篇 desk reject；2027 政策明显收紧。

### 2.2 新颖性冲突排查（结论：无直接撞车）

- **等能量扰动跨 support 重分配作为探测仪器：检索五个相邻领域（体育分析、MARL 可解释性、对抗图攻击、点云等变、经典灵敏度分析）均无先例**。最接近的概念祖先是 Morris elementary-effects（1991）、Nettack 预算化攻击（2018）、COMA 反事实（2018）——论文应主动引用并把新颖性声明精确限定为"固定预算跨支持重分配的受控仪器"（新组合，而非新扰动）。
- 最近的三条碰撞线及区分口径：
  1. Gruver et al.（ICLR 2023，Lie 导数测等变）：他们测命名群作用的标量误差；我们按位移在构件上的分布模式分解，且绑定预测任务。
  2. Geiger et al. 因果抽象/interchange（NeurIPS 2021→JMLR 2025）：他们在内部激活上做对齐干预；我们的输入空间干预是**被预测对象**，不做因果模型对齐声明。需预防 ICLR 2026 divergent-representations 的 OOD 批评。
  3. TacticAI（Nature Comms 2024）＋Partial G-CNN（NeurIPS 2022）：双通道须表述为"按诊断结果设计的、有预算的部分不变性"，不是等变方法。**AMR 若写，必须显式区分 Partial G-CNN**。
- 两个理论陷阱（审稿人必提，必须在文中主动声明关系）：
  - Gram(∂embedding/∂positions) = 经验输入空间 NTK（Du et al. 2019）；
  - Rayleigh 商 = Dirichlet 能量，谱特征家族来自 over-smoothing/GSP 文献。
  - **新颖性必须落在"协议＋预测用途"上，而不是特征本身**。

### 2.3 ICLR 审稿品味（结论：本文形状是主赛道可行体裁）

- 官方审稿指南明确：**缺 SOTA/规模本身不构成拒稿理由**（可在 rebuttal 引用）。
- 直接先例：小模型表征机制分析拿过 ICLR 2025 **Spotlight**（Formation of Representations）；概念性机制可解释性 2025 Poster；几何框架（Geometry of Reasoning）ICLR 2026 Poster；"诊断→修复"弧有 *Test-Time Training Done Right*（ICLR 2026）；诊断-only 的 RankMe 在 ICML 2023。
- 反面模式：**纯描述性谱/几何分析在 ICLR 吃力**（两篇类似工作最终落在 NeurIPS）——本项目有预测器＋修复，正好补上这个缺口。
- 实证最常见弱点类别（arXiv 2511.15462）：小规模/缺真实验证、数据集太少、缺消融、增益边缘。**预期必收到一条规模/通用性差评，提前写 rebuttal**。
- 体育多智能体是已接受的主赛道域（Sports-Traj 2025、JointDiff 2026、Solving Football 2026），但写法必须是"通用方法/科学对象＋足球作测试床"，不能读成 sports analytics。
- 五把致命刀及本项目现状：
  1. 机制是相关而非因果 → 已被因果开关＋prospective 0.85 预防 ✅
  2. 规模/玩具模型 → 部分可预防（强调受控仪器定位＋官方条款）⚠️
  3. 通用性（一项运动、两个 provider）→ provider-shift 已预防大半 ⚠️
  4. **修复没有下游增益** → **当前最大软肋**：必须把任务指标增益做成 headline 数字 ❌→待做
  5. autoresearch 是未说明的搜索 → 把冻结规则＋两轮边界写成协议特性 ✅（文档已备）

### 2.4 审稿人会要求的基线与消融（按优先级）

1. 同数据 SSL 基线（SimSiam/BYOL 类）＋至少一个等变基线（EGNN/Set Transformer）——回答"病理是你的训练配方特有还是 SSL 通用"。
2. 修复的通道消融与配比扫掠（1:0、1:1、2:1、3:1）——注意与"无 Round 3"冻结承诺的张力，见 3.3。
3. 预测器特征家族消融（Jacobian-only / Gram-only / spectral-only）＋简单预测基线（原始坐标统计、attention rollout、CKA）。
4. 干预协议消融：剂量-响应线性、support 大小伸缩、随机 support 对照、非重分配单 support 对照。
5. 统计严谨性：8 场外部确认的 match-level bootstrap CI（n=8 必被挑）；所有训练组件 ≥3 seeds（已有）。
6. 用自家仪器回测修复后的模型（loop closure）——T5R5/T5R6 干预确认已做 ✅。

### 2.5 数据与可复现性定位

- 审稿人期待 SoccerNet 生态认知、Wyscout/StatsBomb、SkillCorner、Metrica、GRF 的意识。
- IDSSE 受限访问是可复现性负债；SoccerTrack-v2 独立确认是真实优势，应前置。
- 建议（时间允许时）：GRF 模拟臂可彻底消除 OOD-干预批评——本期大概率来不及，列入 rebuttal 预案/future work。

---

## 第三部分：胜算评估与改进清单

### 3.1 胜算评估（主观校准，非精确数字）

基准接收率 27.4%。本项目的科学资产、协议严谨度、外部确认明显强于中位投稿；但手稿未启动＋基线缺口＋下游增益未做成 headline 是实质减分。

| 情景 | 估计胜算 | 条件 |
|---|---|---|
| 维持现状、仓促裸投 | 20–30% | 20 天内赶出 9 页、不补任何实验 |
| 执行 P0 清单 | 35–45% | 写作质量过关＋SSL 基线＋headline 下游数字＋定位好 related work |
| 错过 9/18 摘要截止或 OpenReview profile 卡审 | 0% | 行政性死亡，今天就必须处理 |

诚实的判断：**科学内容已经够主赛道门槛，瓶颈完全在执行层（写作＋少量关键实验＋行政合规）**。

### 3.2 P0 必做（9/18 前不动手稿就归零）

1. **【今天】创建/核验所有作者的 OpenReview profile（用机构邮箱）；确认互审资格（决定能否投＋投几篇）**。
2. **【今天起】冻结论文形状**：机制先行 9 页结构（见 1.4），AMR/书法/篮球/MathWriting 全部移出正文进 future work。开始写。
3. **【3 天内】把修复的下游增益整理成 headline 表**：2:1 vs fixed-dual 在 8 场外部的 zone/pair/geometry/centroid 四指标全部占优（已有数字），算 effect size＋逐场胜场计数＋match-level bootstrap CI。这是回应"so what"的唯一弹药。
4. **【1 周内】SSL 基线**：SimSiam/BYOL 类在同数据同 harness 训练（仅开发集评估，**不新增任何保留场/外部读取**——协议红线），展示病理非我们模型独有。
5. **【写作中】related work 显式区分**：Gruver 2023、Geiger causal abstraction、Partial G-CNN、TacticAI、SIE/EquiMod；主动声明 Gram(Jacobian)=经验 NTK、Rayleigh=Dirichlet 能量的关系。
6. **【写作中】Reproducibility statement**：把五层锁＋单次读取＋审计＋hash 写成方法论贡献（评审环境对 AI slop 敏感，严谨协议是差异化卖点）。
7. **【合规】AI 使用披露章节；匿名性自查；IDSSE 许可/引用条款确认**。

### 3.3 P1 高价值但有协议张力，需用户裁决

- **配比扫掠补齐（1:0、1:1）**：审稿人会要；但与"bounded autoresearch 已关闭、无 Round 3"的冻结承诺冲突。科学上可行且合法的口径：**作为 post-hoc 消融而非候选选择，只在开发集评估，不做任何新的保留场/外部读取**。需要用户在"协议纯洁性 vs 审稿完备性"之间拍板。
- 剂量-响应／support 大小消融：纯推理，无协议张力，建议做（便宜）。
- EGNN/Set Transformer 等变基线：有训练成本，时间紧可降级为 related-work 讨论＋limitation。

### 3.4 明确不做（本期）

AMR 训练与 gate（未来论文）；书法/篮球跨域（future work）；GRF 模拟臂（rebuttal 预案）；任何新的保留场或外部数据读取（协议红线，T5R5/T5R6 的单次读取已消耗）。

### 3.5 对 PLAN 假设的影响

本评审不改变任何科学假设；它把项目目标从"继续产生证据"切换为"把已冻结证据在 20 天内转化为合规手稿"。P3-R5 的 P4 门维持"条件满足、待决策"——**本期明确建议不启动 P4**。
