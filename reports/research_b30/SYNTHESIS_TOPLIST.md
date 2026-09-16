# B30 高预期优化方案 list（综合 30 份简报：W1-M1–M10 / W2-A1–A10 / W3-S1–S10）

- 日期：2026-09-16；目标 mu3wl782-yxuns1，task-b30syn
- 分级：预期收益 × 成本 × 证据强度；只收录"可执行且有主来源"的条目，
  wild 项单独列并标注风险；每条含机制依据、预期数字区间、成本与风险

## Tier 1（高预期收益，低–中成本，证据强）——推荐开工顺序即此序

### T1-1 标题/摘要机制先行重构（S3 方案 A，零训练成本）
- 机制依据：Othello（ICLR 2023）/ROME（NeurIPS 2022）先例——机制名词作标题主语，
  testbed 后置；5-move 摘要结构；探针必须写成"带 dose-response 的干预"而非解码器。
- 预期：评审第一印象从"足球应用"转为"机制论文"； novelty 异议率下降（定性，无数字）。
- 成本：写作半天。风险：低。证据：medium-high（S3）。

### T1-2 Related-work 柔道四段（S4，零训练成本）
- 机制依据：四组 differentiator＋conversion 已逐字就绪（Gruver ICLR23 证据 A；
  Geiger NeurIPS21 证据 A−；TacticAI Nature24 证据 B＋，D2 细节待复核 PDF；
  Partial G-CNN NeurIPS22 证据 A）。
- 预期：novelty 异议转为"站在前人肩上"；related-work 章节一次写对。
- 成本：写作半天＋TacticAI PDF 复核 1 小时。风险：低（B＋项须先复核）。

### T1-3 Rebuttal 预演十大异议＋现成答案（S5，零训练成本）
- 机制依据：circularity（#1）/单域（#2）/n=8（#3）/增量（#4）已有冻结证据对位；
  小效应＋种子方差（#5/#7）走"承认＋成本重构"路线；file-drawer（#9）是我们的最强诚实信号。
- 预期：rebuttal 期响应速度＋命中率；写作期提前堵住 #1/#4/#6（基线表须先冻结核对）。
- 成本：写作 1 天（需 source-check 五处冻结数：split/外部集/基线/种子/算力）。
- 风险：低。证据：B+（S5，待 source-check 升 A−）。

### T1-4 Claim 分级制＋误差棒规范（M5 P1–P3，零训练成本）
- 机制依据：Colas 2018 / Henderson AAAI18 / Agarwal NeurIPS21 / Dodge EMNLP19 /
  Bouthillier ICML19 / NeurIPS checklist Q7——3 seeds 只许描述性 claim（min–max，
  禁 mean±std、禁 winner 语言）；volatile 指标标 exploratory；比较性 claim 需 ≥10 seeds。
- 预期：方法论 reviewer 的信任票；Q7 合规一次通过。
- 成本：写作半天＋全表数字改 min–max 格式。风险：低。

### T1-5 证伪设计 K1–K3 预注册（M9，零训练成本＋可选小算力验证）
- 机制依据：Hewitt & Liang EMNLP19（selectivity gap≥0.2）/Locatello ICML19/
  Allen & Hospedales ICML19（partial-ρ<0.2 则杀"geometry adds info"）。
- 预期：把"同义反复"质疑从被动 rebuttal 变成主动 severe test；K1 最便宜先跑。
- 成本：K1 permutation null 为 CPU 小任务（重算已有 artifact）；K2/K3 需预注册先行。
- 风险：低（K1）；中（K3 若 partial-ρ 真塌则被迫降级 claim——但这是诚实必须付的代价）。

### T1-6 证伪 ledger 表＋validity-regime 图＋机制 panel（S2 Spec A/B/C，零训练成本）
- 机制依据：Kaplan/Belkin/Grokking 的 regime-map 说服模式；claim×test×阈值×verdict
  表 30 秒建立信任；schematic→signature→causal-test 三段式（Circuits/Distill）。
- 预期：评审阅读体验质变；null 结果读作边界而非失败。
- 成本：作图 1–2 天（Spec A 轴范围须从 epsilon/support 日志锁定）。风险：低。

## Tier 2（中高预期收益，需训练轮，证据中强）

### T2-1 Margin–SupCon 混合＋τ sweep（S1 Play 1 × A8 D1，重合推荐）
- 机制依据：ArcFace（CVPR19）margin-in-softmax ＋ SupCon（NeurIPS20）base；
  τ=0.1×m=0.2 起步 sweep τ∈{0.07,0.1,0.2}×m∈{0.1,0.2,0.3}；保 InfoNCE 稳定性＋补 margin 锐度。
- 预期数字：margin 0.01–0.17 → 0.25–0.45（仍低于 0.64 frozen；低置信）；pair/zone ±50% 副作用须同报。
- 成本：1 训练轮（loss-only 变更，短 retrain＋eval）。风险：中（r@1 可能反降，arXiv 2510.02161）。
- 证据：medium-high（A8）。**训练预算首选开火项。**

### T2-2 Attention readout（A6 #1，1–2k 参数）
- 机制依据：PMA（Set Transformer ICML19）＋JK-Net 多尺度读出；mean＋att＋max＋energy
  concat，α-entropy 正则防塌缩到 mean。
- 预期数字：centroid −34% → −40~−48%；ρ 上探 0.88–0.93（低置信）。
- 成本：1 训练轮（单 flag 消融）。风险：中（dev 记忆化；须 T5R6 式外部门控才 claim）。
- 证据：medium-high（A6）。**次选开火项（与 T2-1 可并行，改动正交）。**

### T2-3 长 schedule（S1 Play 2，seed-23 triplet 未收敛）
- 机制依据：SimCLR 文献（800→3200 epochs 仍涨）；仅 seed-23 单跑 2–3× schedule＋cosine。
- 预期数字：pair ＋100–200% 相对（绝对仍小）；margin ＋0.02–0.10。
- 成本：1 训练轮（时间换，无新代码风险）。风险：叙事风险（"just trained longer" 须包装成收敛修正）。
- 证据：B+ 方向 / C 量级（S1）。**备选（T2-1 之后顺手做）。**

### T2-4 结构等变分离路线（A5 D1–D3，按序）
- 机制依据：EGNN（ICML21）不变/等变通道分离；Frame Averaging（ICLR22 Oral）质心帧；
  learned canonicalization（ICML23）。D1 冻结-context EGNN 默认；D2 固定质心帧备用；
  D3 学旋转最后。
- 预期：把"训练学分离"变成"结构保证分离"—— novelty 跳段（从 loss 技巧到架构主张）。
- 成本：2–3 训练轮（新架构＋审计）。风险：中高（allocation 比例仍须自定；对称输入不连续性）。
- 证据：medium-high（A5）。** novelty 野心最大的一项，适合占 1–2 轮。**

## Tier 3（wild/故事/低成本写作项，风险标注）

- W-1 书法同构 outlook（S6 O1，<1 人周 CPU）：只做定性图＋一段，不 eval。风险：低。
- W-2 应用故事三选一（S7 A/B/C）：broader impact 用 A（analyst 诊断）＋C（audit infra）；
  B（sandbox）降级半句。风险：hype（已标 non-claim）。
- W-3 对抗重构三档（S8）：只用 A（structured-corruption 口径，禁"adversarial/certificate"词）；
  B 需新实验，C 禁碰。风险：安全 reviewer（已标）。
- W-4 神经科学类比（S9）：只用 Rank 1（TEM Cell20）＋falsifiable 映射＋mixed-coding caveat；
  Rank 3 禁入正文。风险：装饰性质疑（已标）。
- W-5 Benchmark 提案（S10）：只收 B1（probe-discipline audit，无 leaderboard）进贡献；
  B2 pilot、B3 禁承诺。风险：维护负担（已标）。
- W-6 时间维度（M6）：SFA 辅助损失最便宜（无输入变更）；#2/#3 暂缓。风险：中（1Hz 稀疏迁移）。
- W-7 层级探针 E1–E3（M1）：零成本重算已有 checkpoint，机制深度证据。风险：低。
- W-8 耦合标度律 A1–A3（M8）：240 点表拟合，只许 2 参数饱和形＋hold-out 审计。风险：低。
- W-9 容量 pilot 设计（A9）：只设计不开跑（预算纪律）。风险：零。
- W-10 篮球解冻条件（A10）：数据可行性已列，解冻须用户点头＋1–2 周管线。风险：高（deadline 前禁碰）。

## 淘汰项（有依据地不做）
- X1 认证式鲁棒（S8 C）：无定理＋图即红旗，禁。
- X2 DiffPool/SAGPool（A6 #3）：n=10 反证据，禁。
- X3 Grid-cell metric 类比正文（S9 Rank 3）：装饰性，禁入正文。
- X4 B3 leaderboard 承诺（S10）：维护负担，禁承诺。
- X5 μ 阈值重做（GOAL-MU01 已证伪）：禁返工，除非分布外低 μ 新评估（未来工作）。
