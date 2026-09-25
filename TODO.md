# TODO — ICLR 2027（收缩主线，2026-09-24）

## 2026-09-25 投稿前审查交付

- [x] 结论与代码核查、指标和 Figure 1 修正、入口和现稿错误同步。
- [x] route2 3-seed 曝光匹配基线补算，预测/回执/检查点 16 个文件逐一校验。
- [x] R03 21 模型单编辑重叠排除敏感性（2,652,195 次端点前向）完成。
- [x] 建立 `reports/submission_audit_20260925/` 写作证据目录。
- [x] M2 修正补算已回收并完成科学验收：685 个不同组合/351 parents；微调有效、未胜过直接视觉分类，见 `reports/submission_audit_20260925/m2_acceptance/REPORT.md`。

下文为各轮历史执行记录，当前科学措辞以新证据目录为准。


本轮依据：用户指示解包并执行 `docs/e832887_focus_pack/01_MASTER_PROMPT.md`。
旧施工包已收尾，不再用“实验全部关闭”阻断这一轮。封存池仍不读。本机无 GPU，不新开租机。

## 当前执行（2c3e2d9 机制迁移，2026-09-25）

- [x] 解包 `docs/2c3e2d9_mechanism_transfer_package.zip`，自带12项合同测试全过，实时仓库审计确认R1–R5
- [x] R1–R6共享原语：`experiments/mechanism_transfer_v3/common/` + 13项fail-closed测试，不动旧文件
- [x] M1源端：T1碰撞形式化、source有界搜索、6臂×clean/flip×3 seeds训练矩阵（orbit去风险排序无精度代价，flip侧增益不稳定；T1/T2未跑）
- [x] M2 CPU准备：扩展盲bank（692 quartets/351 parents）、mask审计、解析基线J3=0.886；M3默认不跑
- [x] 历史 M2 GPU主格已完成；旧“冻结/微调无差”解释无效并撤回，当前以修正后的科学验收报告为准。

## 正在执行（五路线胜率优化，2026-09-25）

用户授权完成五条优化；路线1、2在基本实现外各有五轮证据驱动优化预算。共同合同见 `reports/e832_focus/OPTIMIZATION_PROTOCOL.md`。不读封存池，不产生新增租机支出。

- [x] 路线1：跨任务确认 source/T1/T2（最多五轮）
    - [x] 冻结共同协议与停止规则（2026-09-25）
    - [x] Round 0 实现与测试
    - [x] Round 1 source 新银行 3 seeds
    - [x] Round 2 T1/T2 各 3 seeds
    - [x] Round 3 固定8192 parent可行性/不稳因素诊断
    - [x] Round 4 独立新parent对照
    - [x] Round 5 未触发：没有稳定的跨任务对比可扩到10 seeds
    - [x] Round 2 固定8192测试parent的T1/T2可行性复核：T1不可判定、T2负向/不稳定，跨任务确认不成立（2026-09-25）
- [x] 路线2：ResNet18 结构迁移（基本实现＋最多五轮；正式结果外部阻塞）
    - [x] 冻结共同协议与停止规则（2026-09-25）
    - [x] Round 0 实现、CPU 冒烟、GPU bundle
    - [x] Round 1 单 seed 流程验证（CPU PILOT_ONLY）
    - [x] Round 2 3-seed 核心消融（未触发：v2合同修复后判有界阴性，停止训练，2026-09-25）
    - [x] Round 3 单因素修正（未触发：同上，无新可修复合同因素）
    - [x] Round 4 新 parent/预设观测变化（未触发：同上）
    - [x] Round 5 最终对比多 seed（未触发：同上）
    - [x] 实现、合同测试、CPU PILOT_ONLY smoke、GPU-ready bundle（2026-09-25）
    - [x] 新parent数据生成、hash/split锁定、data-bound runner和7项测试完成（2026-09-25）
    - [x] 正式新parent视觉结构对照（v2已执行：12个GPU arm/seed + 3 clean baseline，direct均值.571/interaction均值.583，seed差不稳定，判有界阴性）
    - [x] 用户已提供授权GPU入口；远端环境/数据/8项测试/CUDA smoke通过（2026-09-25）
    - [x] 核心矩阵完成并发现两项合同缺陷：J3误写atomic、native RGB被提前归一化污染mask（2026-09-25）
    - [x] Round 1 clean-only baseline完成
    - [x] Round 2合同修复矩阵完成：v2 direct J3=.571、interaction=.583但seed差不稳定；有界阴性，停止训练
    - [ ] AI Galaxy退租：MCP显示1台running但当前实例不属于该MCP state store，需账户所有者/控制台处理
    - [x] GPU追加第1轮完成：mask池化area修正，representation 0.274→0.619，interaction仍不稳定；12臂拟合良好无剩余可修复因素，第2、3轮按停止规则保留未用（2026-09-25，commit 8719031）
- [x] 路线3：解析、集合、关系、容量匹配与有限群轨道强基线（2026-09-25；解析器J3=1.0，source学习结果收窄为parser-bounded诊断）
    - [x] 三轮证据驱动尝试完成：R1排除简单欠拟合、R2规范表示J3约0.34、R3新银行确认parser-bounded间隔（2026-09-25；报告见`reports/e832_focus/route3/REPORT.md`）
- [x] 路线4：新 parent、刚体坐标条件与预设观测变化的独立稳健性确认（2026-09-25；noise/visual未运行）
- [x] 路线5：按最终证据重写正文、统一主表、claim ledger 与复现包（2026-09-25）
    - [x] 真实 Figure 1：固定规则从 dev512 归档预测选出 endpoint-only/full-repair 两例并接入正文（2026-09-25）
    - [x] 只读路线5审阅完成，确认主表字段、页预算和审稿攻击面（2026-09-25）
    - [x] 接入source解析器边界、Route1非跨任务结果和Route2 GPU阻塞状态（2026-09-25）
    - [x] 独立最终审阅的四个minor fixes已应用：生成器分离、U10/Route1区分、fallback注释、治理状态同步（2026-09-25）

## 已完成（e832887 收缩主线首轮）

- [x] 收缩为一个主问题、两条主线（2026-09-25；视觉结构消融和 T1/T2 未跑，算力不启动）
    - [x] 解包并用仓库真实类核对段级加性限制（2026-09-24）
    - [x] A 把不变性、跨段交互和几何特征分开，写 STRUCTURE_DECISION.md（2026-09-25）
    - [x] B 先把已有视觉结果收成同口径表，不新开训练，写 VISUAL_DECISION.md（2026-09-25）
    - [x] C 只做算力前沿的价值判断，不训练，写 SAMPLING_DECISION.md（2026-09-25，不启动）
    - [x] 写作同步改已知口径，写 PAPER_MIGRATION.md；新结构句只进附录，不挤结论页（2026-09-25）

## 当前执行（e1a933 施工包已收尾，当日保留）

本轮施工包的 19 条 R/O/N 全部收尾，8 条并行 lane 的结果全部接线进稿。逐条最终状态见
`reports/e1a933_review/R_FIXES_REPORT.md`；完成度对照见 `reports/e1a933_review/PACKAGE_COMPLETION_AUDIT.md`。

- [x] 解包并核对基准提交与包内31个文件校验和。
- [x] 数据组：R01/R02/R04 完成；R03 **补完**（392412 增广状态逐状态核对、quartet 银行 0 exact/0 near、检测器已标定、96/96 训练来源由 checkpoint 内嵌 mu/sd 指纹定位）。
- [x] 视觉组：R05–R06 完成；21 臂 GPU 已回收校验；4 项无判别力合同检查已替换并有 mutation 证据。
- [x] 决策/足球组：R07–R09 完成；R08 **补完自然 E 发生率**（主格 8/1449 = 0.55%）；R09 **补完全 256 场景无 oracle 筛选部署估计**。
- [x] 写作组：R10 完成；论文/自动表生成器/claim ledger 同步（ledger 新增 49 行）。
- [x] 原21臂、N03六臂、O05十二臂、N02四十八臂、O06 272请求：全部完成、回收、逐文件 SHA256 校验；SAFE_TO_SHUTDOWN 已生成（未执行关机）。
- [x] N02 独立科学解释与论文整合**已完成**（`N02_INTERPRETATION.md` + 48 臂矩阵入附录；机制判 UNRESOLVED）。
- [x] O01 **补完**（648 臂：warm-start 因子、稳健优化器、oracle 三账本）；O02 **补完**（匹配容量+匹配预算对照）；O04 **补完**（前瞻干预 + D10 下游回声）；O03/O05/O06 完成。
- [x] 六份必交文件、回执与针对性检查已汇总；本轮另新增 `PARENT_VERIFICATION.md` 记录总控独立复核。
- [x] 全部产物安全落盘并完成校验；GPU 实例可由用户释放。

## 正在执行（用户 2026-09-24 授权的三条 leads）

- [x] L-015(b) 置换增广训练与 G8 不变特征，检验能否抬高 J 并去掉标号依赖（2026-09-24）
- [x] L-014 半径诊断复算并写成可引用结论（不发展 loss）（2026-09-24）
- [x] L-006 上述训练臂各 10 个种子，对照 ±15pp 噪声地板（2026-09-24）

## 仍未做（唯一待办区；下列均需用户/作者参与或单独授权）

> 除本节外，本文件其它小节均为历史归档，不含可执行待办（已改为无勾选框的归档文本）。

**一、投稿行政（需用户或作者拍板，agent 不自行收尾）**
- [x] Figure 1 已按固定规则从 dev512 归档预测生成并接入正文（真实案例，不是示意图）——2026-09-25
- [x] AI-use 披露定稿（main.tex 末尾 AI-use statement 定稿，2026-09-25收尾）——旧编号 W3
- [x] 双盲扫描（2026-09-25已执行：仅Anonymous、无泄漏、iclrfinalcopy注释、figures/PDF元数据干净）——旧编号 W5
- [x] 匿名打包已验证（2026-09-25：git archive快照218M级；data仅2小文件入库、本地48G不入包；artifacts小证据入库、大中间件走SHA清单；references根相对结构保持）——旧编号 W6
- [x] rebuttal 预备答辩（**可选，已关闭**；T5R6 reserve＋NO_NEW_CLAIM已有，不另成文）——旧编号 W4

**二、需单独授权的可选扩展（施工卡标为可选，或需新算力/新分层）**
- [ ] O02 可选扩展：难度/手性任务、更多宽度与双向参数匹配
- [ ] O04 可选扩展：typed encoder 类在新任务上的同口径结论（发布检查点无独立特征头）
- [ ] N02 机制升级前提：算力匹配对照 + 以小编辑为主的新分层（≥60 quartet）
- [ ] O05 自然缺测（而非人工有定义遮挡）的学习模型与真实传球成功预测；J03WQQ 仍封存

**三、PARKED 探索线索（不列在本文件，唯一入口是 `LEADS.md`；需用户单独授权才能开工）**
- 旧编号 E2→`LEADS.md` L-005（GRF 模拟干预臂）、E4 剩余 L-009/L-013、E5→U1 确认池 paired-CI（rebuttal 弹药）。
- L-006、L-014、L-015 已由用户于 2026-09-24 授权，改记在上方「正在执行」，不再算 PARKED。
- 其余线索状态仍以 `LEADS.md` 为准；本文件不重复其细节。

## 本轮分支记录
- 2026-09-24：R01–R04、R05–R06、R07–R09、R10 并行启动；以用户本轮执行master prompt的授权取代旧“实验全部PARKED”限制。
- 2026-09-24：O02/O03/O05 优先；N01 在目标修正后启动，N02允许小对照，N03依赖匹配目标已可学习的证据；O06限已有权限及费用，无新增租机授权。
- 2026-09-24：旧产物只读，新结果写入 artifacts/e1a933_review；既有投稿行政待办保留。
- 2026-09-24：施工包收尾阶段改为 **8 条并行 lane**（n02/r03/r09/o01/o02/o04/r08/n01p1），共享工作区 + 严格文件归属；全部完成后由总控统一接线进稿与同步状态文档。
- 2026-09-24：三条新路线**全部未升为方法贡献**：N01（目标更可学但 J/full repair/migration/P1 四轴均无一致收益）、N02（效应大但算力未匹配、冻结预测 random 未命中）、N03（仅单 seed 增益、真几何替换 J=0）。
- 2026-09-24：用户授权执行第一档 L-015(b) 与第二档 L-014、L-006。三条从 PARKED 改为正在执行；不读封存确认池。

## 历史收尾说明（2026-09-22，已被本轮授权更新）
原终局裁决为 `reports/final_closure/FINAL_PROJECT_VERDICT.md`。以下旧完成/暂停记录保留历史；涉及本轮任务的状态以上述当前执行为准。

> 结构导航（2026-09-22 整理）：入口 `README.md`；叙事→目录映射 `PROJECT_MAP.md`；
> reports/experiments/artifacts/docs 各目录内有 INDEX.md。

以下为历史区（探索期全部计划保留备查，不再执行；各分支结局见文末“分支记录”与“变更记录”）:

更新时间：2026-09-22（投稿收尾重整版；旧事项 0–4 已裁决，见下方历史事项裁决表）

## 当前结论（一句话，为什么只剩写作）

- 三大贡献＋两复现线＋基础模型阴性全部闭合；02 探索升级（U1–U4）＋确认池 2×2 挖掘完成，稿件增量 A＋B 已接线，主文 8 页编译通过。
- 剩余工作全是零证据风险的写作/行政；实验侧除 rebuttal 弹药外无必做项。

## A. 论文写作/行政（2026-09-22 快照 · 已归档，不再作为待办）

> 本节已转为归档文本（去勾选框）。未完成项的现行状态以顶部「仍未做」区为准。

- W1 references 22→30：**已完成**（39 入库、30 genuine 接线渲染、零臆造全核验；40 软目标留作可选）。
- W2 Figure 1 概念图：**未完成** → 见顶部「仍未做」。
- W3 AI-use 披露定稿：**未完成** → 见顶部「仍未做」。
- W4 rebuttal 预备答辩：**未完成（可选）** → 见顶部「仍未做」。
- W5 双盲扫描：**未完成** → 见顶部「仍未做」。
- W6 匿名打包：**未完成** → 见顶部「仍未做」。
- W7 sha256 锁重新部署：**已取消**（用户指示 2026-09-24，见变更记录）。

## B. 待挖实验想法（2026-09-22 快照 · 已归档，不再作为待办）

> 本节已转为归档文本（去勾选框）。这些是长期探索线索，按规范**唯一入口为 `LEADS.md`**；
> 各条现状与最小验证代价已在其中，本文件不再重复。

- E1 L-007 转折位置精度：**已执行并封存**（零训练诊断＋三张训练牌＋审计后的修正复算；见 `L007_CORRECTED_RESULTS.md`）。
- E2 L-005 GRF 模拟干预臂 → `LEADS.md` L-005，状态 `待挖掘`（本期不启动）。
- E3 L-006 ~10 seeds/arm 确认门 → `LEADS.md` L-006，状态 `待挖掘`（等有候选主张再开）。
- E4 L-009 / L-013 / L-014 → `LEADS.md` 同名条目，状态 `待挖掘`（仅审稿驱动）。
- E5 U1 确认池 paired-CI → rebuttal 弹药，需封存重跑授权（见 `U1_CONFIRM_ADDENDUM`）。
- E6 标号敏感性作为独立方向 → `LEADS.md` L-015，状态 `待挖掘`（训练版本需授权）。

## 历史事项裁决（2026-09-22 整理；原文见 git 历史，分支记录与变更记录不变）

| 事项 | 裁决 | 说明 |
|---|---|---|
| 0a OpenReview/作者/摘要 | ✅ 完成 | 用户已完成 |
| 0b 论文形状冻结 | ✅ 完成 | 正文 8 页，样式已装，编译通过 |
| 0c 样式＋引用 | 部分完成 | 样式✅；引用→W1 |
| 1a TacticAI 复核 / 1b source-check | 🚫 被取代 | 旧 related-work 框架已随 Lane C 重写作废；数字以对齐审计为准 |
| 1c 标题摘要 / 1d related-work / 1e claim 分级 | ✅ 完成 | Lane C＋R8＋ledger/claim-map |
| 1f rebuttal 十大异议 | ➖ 部分 | reserve 文档已有；成文→W4 可选 |
| 1g K1–K3 预注册 | 🚫 终止 | E7 已死 |
| 1h 图表三件套 | 部分完成 | fig2/3/4✅；Fig1→W2 |
| 2a–d 训练开火 | 🚫 终止 | 探索关闭＋02 上限；U4 已执行完毕 |
| 3a–e Tier3 | 🚫 终止 | non-claim 口径；禁令维持 |
| 4a 摘要 / 4b prose / 4d §4.5 证伪 | ✅ 完成 | Lane C；失败目录已入附录 |
| 4c Figure 1 | → W2 | 唯一遗留写作项 |

## 历史完成归档（2026-09-04—09-16，不再执行，仅留审计线索）

- 事项 1–7：IDSSE 登记→转码 QC→split lock→T5R2 lock→T5R3 sanity→T5R4 两轮关闭（`update_ratio_2to1`）→T5R5 `PASS_STRONG`（`J03WQQ` 一次性读取）。
- 事项 9：T5R6 SoccerTrack-v2 独立确认 CONFIRMED（8 场同方向）。
- 事项 10c/10e：headline 表（P3-R9）＋配比扫掠 1:0/1:1（P3-R15）完成；10d/11b SSL 对照完成（P3-R13）。
- 事项 11a/11c：ε-sweep（P3-R10/R11）＋support 伸缩（P3-R12）＋预测基线（P3-R14）完成；仅剩 11d（已并入新事项 4b）。
- 事项 12：手稿骨架＋摘要初稿已建。
- 事项 13：P4 AMR v1–v4b 全败→v5（P4-B1）→CAP 打平（P4-B2）→v6 H1 通过终结 15 连败（P4-C1）；R1 mask 消融（GOAL-R1）＋R2 JGCL（GOAL-R2）完成，训练预算 2/10。
- B30：30/30 简报＋toplist＋verify-final 完成，目标关闭。

## 分支记录

| 分支 | 状态 | 说明 |
|---|---|---|
| A：IDSSE 暂代 SNGAR 开发主线 | 完成归档 | T5R1–T5R5 已闭合（`T5R5_PASS_STRONG`，路由 A）。 |
| B：SNGAR 恢复 | 暂缓 | 保留 firewall；可信访问或带 revision/mapping/SHA 镜像后升级。 |
| C：IDSSE 独立外部确认 | 本轮关闭 | 已用于开发的 IDSSE 不复用为外部确认。 |
| D：动态 support | 可并行、非阻塞 | response-blind 规则开发，不反向选择主线候选。 |
| E：AMR | 当前执行→银行模式 | v6 H1 通过（P4-C1）；收缩为分支塑造；剩轮按新事项 2 执行。 |
| F：SoccerTrack-v2 外部确认 | 完成 | T5R6 CONFIRMED；8 场同方向。 |
| G：ICLR 2027 投稿冲刺 | 当前执行 | 摘要 9/18、全文 9/25；机制先行 9 页；AMR/书法/篮球进 future work。 |
| H：B30 头脑风暴 | 完成 | 30/30＋toplist＋verify-final；目标 `mu3wl782-yxuns1` 关闭。 |
| I：Tier1 写作 | 当前执行 | 新事项 1（零训练成本）；TacticAI 全文＋五处冻结数为前置门。 |
| J：10 轮训练预算 | 当前执行（剩 8 轮） | 新事项 2 独立记账；T2-1→T2-2→T2-4；T2-3 备选。 |
| K：发现型探索 campaign（20260917 包） | 完成（goal 手动关闭） | 优化目标 mu5qijtb-uoh83q 6 轮落定；R6 授权补救已推送 3ab758d。旧红线零触碰。 |
| L：最后 15h 路线（20260918 包，N01–N10） | 首轮＋round2 收敛 | 终局见 reports/last15h/SELECTION.md：现象 6 项成立（N01-missing/N03/N04-emergent/N06/N08/N09 分解），新方法 5 项全死，方法位 flipmine-as-is；摘要草稿＋图源表已落盘。新输出 experiments/last15h＋artifacts/last15h＋reports/last15h，不覆盖旧证据。 |
| M：真数据转折＋全路径＋joint（E2/E3/E4） | 收敛 | 双数据故事升级通过，见 reports/last15h/M_SELECTION.md：E2 三件套 3/3（N01/N03/N06-hard，多种子同向）进主文；E3 关闭（诚实 null，库无重入动作）；E4 死（3 种子全平）。方法位=flipmine＋cover。 |
| N：02 定向探索升级（U1/U2/U3＋可选 U4） | 当前执行 | 用户已授权执行 docs/last3day/02_EXPLORATION_AND_UPGRADE_PLAN.md；U1/U2 复用已有模型与缓存，U3 只新增前向/重分析，U4 最多两种小表示对照；仍禁 DINO/L002/EventUpdater/自然图 Task3/边界角度/其他关系族。 |
| O：f095 20 路线首轮（A 机制基线＋足球 D06） | 首轮收敛 | 用户 2026-09-23 授权（A 组合＋足球优先＋VLM 预算＋提交分支）。阳性：D01（容量 44× 无效、数据 16× 抬不弥合）、D02（去命名不充分＋群平均 +5–12pp）、U02（几何组织 SUPPORTED）、U03（三段分解）；阴性：U01（typed NEGATIVE）、D06（完成传球 E-bank 不可行，721 事件）。§5 四项事实修正已入稿。交付 reports/f095_campaign/（7 报告＋COMPLETION_VERDICT）＋experiments/f095_campaign/（代码与测试）。 |

## 变更记录

- 2026-09-04：新增本项目共读 TODO，固定 IDSSE 暂代 SNGAR 的范围，并登记 A–E 五条分支。
- 2026-09-04：后续每次新增、完成、暂停、取消或移除 TODO 项，必须同步更新本文件及本节记录。
- 2026-09-04：完成指定 Hugging Face 官方页面核验；补充官方 `main` 文件树/提交、7 个实际比赛 ID、数据卡 ID 差异和 Dataset Viewer 错误，并将下一步收敛为本地 provenance、SHA-256 与 canonical QC。
- 2026-09-04：完成 IDSSE raw manifest、23/23 SHA-256 复核、7 场 canonical conversion receipts、逐场 QC、match-level split lock 和 T5R2 task/baseline/metric lock；T5R2 审计通过。T5R3、autoresearch、candidate lock 和任何模型结果仍未运行。
- 2026-09-04：完成 T5R3 fixed dual-channel sanity：4 个 neural variants、3 个 seeds、12 个 checkpoints、Procrustes control、match-level bootstrap 和 cross-readout；T5R3 审计通过。dual 仅获方向性 pass，T5R4、candidate lock、reserved match 和外部确认仍未运行；v1/v2 中止尝试不纳入证据。
- 2026-09-04：完成简短 T5R3 closure review，确认无路线级偏移并允许 T5R4 Round 1；新增可探索候选框架，Round 1 仅使用 train/valid，candidate lock、reserved match 和外部确认继续未运行。
- 2026-09-04：完成 T5R4 Round 1：4 个候选、3 个 seeds、12 个模型；结构审计通过，未发现全维度 Pareto 支配者，不自动进入 Round 2，candidate lock、reserved match 和外部确认继续未运行。
- 2026-09-04：完成语义漂移修复（新增 `metric_semantics_addendum.json`）与 Round 1 审阅授权；完成 T5R4 Round 2：2 个候选、3 个 seeds、6 个模型，结构审计通过，按冻结规则选出 `update_ratio_2to1` 并关闭 bounded autoresearch；candidate lock、reserved match 和外部确认继续未运行，P4 仍 blocked。
- 2026-09-05：完成 T5R5-0 闭包审计（逐场重算＋文档清理，不重开选择）；创建 candidate lock、干预协议锁与 T5R6 shadow lock（lock 审计通过）；执行一次性 `J03WQQ` 读取（3575 snapshots/515 pairs，159 complete 干预集），任务 gate 全过、干预强通过，verdict `T5R5_PASS_STRONG`，路由 A；P4 仍 blocked，不做新搜索。
- 2026-09-05：建立 T5R6 项目文档（PLAN/ROADMAP/LEADS）；新增分支 F（SoccerTrack-v2 外部确认）；SoccerTrack-v2 为 gated 数据集，匿名可读元数据但数据文件需授权，待用户提供 HF_TOKEN 后执行 N1。
- 2026-09-05：收到用户 HF 授权，完成 8 场 GSR 下载登记（45.16GB）；转换 QC 通过（44,401 快照/7,534 对）；任务＋干预确认 8 场同方向，verdict `T5R6_CONFIRMED`；P3-R3 升级为独立 provider 条件支持，P4 门条件满足待事项 8 决策。
- 2026-09-05：完成 ICLR 2027 发表胜算综合评审（内部评估＋4 份外部检索简报，归档于 `reports/research_briefs_20260905/`）；新增分支 G（投稿冲刺）与事项 10；评审结论：科学内容达主赛道门槛，瓶颈在写作/行政/少量关键实验；AMR、书法、篮球本期明确不做。
- 2026-09-05：按用户要求剥离时间/行政因素，在评审报告中增补第四部分"纯科学性深度评估"（局部近似适用域、构造性平凡化、因果分层、缺失对照、统计呈现）；胜算评估改为纯科学维度（现状 30–40%，补齐关键项后 45–55%）。
- 2026-09-05：完成事项 10c（headline 效应表＋bootstrap CI）；新发现 match-half 辅助探针 0/8 反向（−30%）已如实入表并回写 CLAIM_LEDGER P3-R9（界定"context 无退化"边界）；新增事项 11（科学补强线），11a ε-sweep＋未训练对照已后台启动。
- 2026-09-05：11a 完成（ε-sweep 报告；有效域 ε≤0.25、现象训练诱导，ledger P3-R10/R11）；11c 的 support 大小伸缩已后台启动（support {1,2,3,4}×ε=0.25×6 模型×30 组）。
- 2026-09-05：11c support 伸缩 v1 为 pilot（s=4 仅 18 组达阈值，触发 UNDERSAMPLED；确认 s=1 时 full≡diag 的构造一致性；full−diag 差距随 support 增大方向一致但噪声大）；已参数化脚本并启动 v2（4 档×60 快照，全量一致比较）。11b 设计定案：保持 2:1 配方与数据不变，仅将 intrinsic 三元组对比损失换成 Barlow-Twins 式互相关目标（隔离"目标族"变量），dev 评估。
- 2026-09-05：11c support 伸缩 v2 完成（full−diag 差距随联盟粒度 0→+0.2~0.27，两族同向；ledger P3-R12）；11b Barlow-Twins 目标族对照已后台启动（训练 3 seeds×80 epochs＋干预评估，协议注明 control 非 selection）。
- 2026-09-05：SSL 对照并行化重构（3 seed worker×8 线程，墙钟降至分钟级；线程数入 manifest）后完成 11b：现象跨目标族复现（ledger P3-R13）。
- 2026-09-05：11c 全部完成：预测基线头对头（full 0.49＞gyration 0.35＞谱 0.11，逐 seed 一致）＋k 敏感性（谱基线全 k 近零无翻转）；ledger P3-R14。事项 11 仅剩 11d（随手稿执行的措辞修正）。
- 2026-09-05：用户质询后更正：分支 E 状态由“阻塞”改为“门已闭合、待用户裁决”——AMR 从未被任何治理文档判为 future work，该说法仅是 agent 在评审报告中的建议；A/B/C 三选项见评审报告及 D-20260905-T5R6-002。
- 2026-09-05：新增事项 12（手稿）：论文骨架＋8 节写作 scaffold＋证据映射表＋摘要初稿已建于 `paper/`；AMR 与配比扫掠留槽位待用户裁决。
- 2026-09-05：用户三项指令：论文 brief 暂缓评审；AMR 走完整流程（选项 C，D-20260905-P4-001，新增事项 13，分支 E 转执行）；配比扫掠 1:0/1:1 授权并行执行（10e 转执行，dev-only 事后消融）。PLAN/ROADMAP 已重写为 P4。
- 2026-09-05：事项 10e 完成：配比曲线 0→3:1 成表，P3-R15 入账；1:0 随机读出警示已记录。
- 2026-09-05：P4-N2 完成：M1 config lock 冻结（含路由证据依据与等计算量预算），审计 PASS，DECISIONS 追加 D-20260905-P4-002。
- 2026-09-05：P4-N3 启动：AMR M1 三种子并行训练中（锁哈希 65d1417b…，训练前 addendum 已记录 eqv_heads 与 encode 适配）。
- 2026-09-05：P4-N3 v1 完成即失败（塌缩，P4-A1 负结果入账，报告已写）；锁 v2 附录冻结（归一化 L_eqv＋W_var=1.0），v2 重训启动。
- 2026-09-05：P4-N3 v2 失败（P4-A2 入账，重估报告已写）；锁 v3 附录冻结（α 全零=全频带可恢复，D-20260905-P4-004），v3 训练中。
- 2026-09-05：用户 fallback 指令（D-20260905-P4-005）：v3 若失败转 research 驱动；三条 researcher 轨道已提前并行启动。
- 2026-09-05：research 综合完成：top-3 路线（精确谱基础→最终嵌入防护→梯度平衡）已定序；尝试待 v3 判据。
- 2026-09-05：P4-N3 关闭：v3 失败（P4-A3 入账，不变性压力假设被证伪）；D-20260905-P4-007 启动 v4 重设计。
- 2026-09-15：v4 冻结开训：init 死亡根因找到并修复（均值池化相消＋bias 主导），W_ROUTE/VIC_GAMMA 实测校准，审计 PASS，3 worker 运行中。
- 2026-09-15：v4 判据：H1 未通过（P4-A4 入账）；v4b 备用启动。
- 2026-09-15：v4b 启动（D-20260905-P4-009，最后一次有界尝试）。
- 2026-09-15：P4 terminal：v4b 失败触发终局规则（P4-A5 入账，D-20260905-P4-010）；等用户裁决 M1 存废。
- 2026-09-15：R1（v5）启动：5 轮预算用 1 剩 4。
- 2026-09-15：R1 判据 H1 通过（P4-B1 入账）；剩 4 轮转 N4。
- 2026-09-15：R2（CAP 对照）启动：5 轮预算用 2 剩 3。
- 2026-09-15：R2 判据：路由≈CAP（P4-B2 入账）；训练阶段建议完结，剩 3 轮银行。
- 2026-09-15：R3（v6）启动：G/H/I 简报归档 research_p4_round3；5 轮预算用 3 剩 2。
- 2026-09-16：目标 task-mu 关闭：μ阈值预测被证伪（GOAL-MU01入账，未耗训练轮）；待v6 verdict开第1训练轮。
- 2026-09-16：v6 verdict：H1 通过（P4-C1入账，15连败终结；成本计旧预算，台账仍0/10）。
- 2026-09-16：第 1 训练轮发射（mask消融，锁 462be450；台账 1/10）。
- 2026-09-16：第 1 训练轮判据：H1 通过，幽灵球员非必要（GOAL-R1入账；台账1/10）。
- 2026-09-16：第 2 训练轮发射（JGCL单几何，锁 feeb99e4；台账 2/10）。
- 2026-09-16：第 2 训练轮判据：H1通过但弱于教师系（GOAL-R2入账；台账2/10；建议银行剩余）。
- 2026-09-16：B30 重整：旧事项 1–13 归档为历史完成；新事项 0–4 按 VERIFY_FINAL 开火顺序重排（行政阻塞首位→Tier1 零成本写作→训练剩 8 轮 T2-1→T2-2→T2-4→Tier3 按需→手稿组装）；新增分支 H/I/J；B30 目标关闭。
- 2026-09-17：用户启动发现型探索 campaign（docs/iclr2027_discovery_campaign_20260917/MASTER_AGENT_PROMPT.md 总控命令），新增分支 K：优先级覆盖旧 TODO/B30 本轮建议（旧审计/强基线/AMR 竞赛不先行，不复活旧配方、不重包装历史数字）；首轮 R01/R02/R05 可执行试验＋R06 模拟试跑；commit 恰为包基线 f86dbdf，无需回滚。
- 2026-09-18：分支 K 次轮开火（用户授权“开始”）：R02 复核 3 新种子×512×3 模型后台运行；R04 minimax／R06 散射／T5R3 探针随后跟进。
- 2026-09-18：R02 复核 9/9 确认（flip ~48–50%、miss|flip ~57–64%、无条件 ~28–32%，stealth≫null 保持）：主发现升级为确认现象。R04 重训＋R06 散射仍在后台。
- 2026-09-18：R04 首试落定：扰动覆盖训练失察 67.5%→~57%（−10pp），干净无代价；minimax≈等权≈最坏臂 → 触发止损条款，方法简化为等权覆盖。待多种子确认。R06 散射仍在后台。
- 2026-09-18：R06 round2 落定→park：散射系统真混沌（oracle ×24），但小模型学出非混沌动力学，五定律 t10 起全等；模型质量界，非真阴性，主线不受影响。次轮剩 R04 多种子确认。
- 2026-09-18：R04 多种子确认开火（用户授权“确认展开”）：seed23/47×4 方法训练＋3 种子×4 方法 R02 跟随评价，后台运行中。
- 2026-09-18：R04 确认落定：方向 3/3 一致（clean 63–68% → equal 55–60%，−4~−11pp），方法无分化（等权即终案），干净三种子无代价。主线证据闭合，进收尾（主图＋PAPER_REPLACEMENT_PLAN）待用户拍板。
- 2026-09-18：P1 落定：R03 专跑（并集覆盖 +3.5pp、平均风险零迁移、有界次级）＋schema 六线合规记录＋parquet。P0（跨架构＋unif）后台运行中。
- 2026-09-18：P0 落定（首跑评价段报目录错，前台同参正常，绝对路径重跑通过）：修法跨架构 4/4（−3~−14pp）；unif 拿走大半增益（pooled 0.593 vs equal 0.574）；等权在小架构有 −1~−5pp 干净代价，unif 无。方法句修正为“覆盖本身”，工作包代码任务清零（剩收尾写作待拍板）。
- 2026-09-18：T5R3 适配器开火（用户授权补齐）：zone/centroid oracle 精确验证（6127 全对、maxerr 0/1e-5下）、slot 稳定 10/10、保留 J03WQQ 未碰、图按部署重建；64 模式质心零位移 bank；J03WN1 选模→J03WOY 评价；3 种子×128 场景×2 方向×2 强度后台运行中。
- 2026-09-18：R01-T5R3 落定（394k 前向）：真模型真数据上四定律全平（±0.002，cmse 四位全等）→ R01 阴性升级为跨架构稳健，A 故事 Fig1 正式死亡，B 双轴扩到真模型类；适配器脚手架保留（R02/R04-on-T5R3 可复用）。
- 2026-09-18：Track6 开火（用户授权）：R02-on-T5R3（主队平移过 thirds 分界，pilot 23/24 翻转、miss 0.35）评价＋J03WN1 挖掘后台运行中；覆盖微调脚本已就绪，待 mine 到位。
- 2026-09-18：R02-on-T5R3 落定：J03WOY 上 235/256 可翻转，三种子 miss|flip 0.30/0.34/0.31、无条件 0.27~0.31（干净准确率 0.96~0.98）；J03WN1 挖到 369 翻转对。主现象上真数据成立，覆盖微调（cover vs cleanonly，15 epoch）后台运行中。
- 2026-09-18：覆盖微调 s11 落定：miss 0.298→cleanonly 0.217→cover 0.132（净 −8.5pp，相对 frozen −56%），干净代价可忽略；Track6 卡已写。s23/s47 确认后台运行中。
- 2026-09-18：Track6 三种子落定：cover 净效应 −6.4~−14.5pp（3/3 同向），干净指标反超 cleanonly。真数据主现象＋真模型方法双确认，全线收官只剩收尾写作。
- 2026-09-18：方法加分 program 开火（用户授权多轨并行）：Track0（支撑 d_new AUC 0.64、四分位单调 0.36→0.71）＋Track4（运动告警死 0.48、集成无增益、失察跨架构共享）已出数；重训练轨（flipmine/supmine/fliprand/auxmargin/relfeat/clean）后台运行中。
- 2026-09-18：方法 Round-2 屏出大奖：flipmine miss 0.675→0.463（−21pp，1534 对，holdout 无代价）；supmine==fliprand（Track3 死）、relfeat/auxmargin 弱阳性（0.618）。修 loader（mhead 选类）后 auxmargin 补测。flipmine 3 种子确认后台运行中。
- 2026-09-18：flipmine 三种子落定（0.463/0.480/0.447 vs clean 0.675/0.642/0.634，−16~−21pp；holdout 无系统代价）→ 方法终案即 flip 挖掘，R04 记录升级。方法栏反转完成。
- 2026-09-18：分支 K 全线工作提交 GitHub（main）：TODO.md＋experiments/discovery_campaign＋reports/discovery_campaign＋artifacts/discovery_campaign（json/csv/md/py/png 共 201 文件；.pt/.npz/.parquet 按 .gitignore 留本地）；旧 clutter（.pi/、散落日志、docs zip 等）未动。
- 2026-09-18：优化目标 mu5qijtb-uoh83q 确认（6 轮迭代，路线层冻结，R04＋T5R3＋R02 全覆盖，胜率优先，红线技巧逐项授权）；R1 开火：R02 坐标 family/强度 tuning（新场景 tune_707/confirm_808），sweep 上 C（random-heavy）预算匹配胜出（miss|flip 0.628 vs A 0.576，margin0.05 下保持），confirm 7 runs 后台运行中。
- 2026-09-18：R1 落定：C 在 confirm 新场景 3 模型×2 margin 全胜（主数字 C@0.03×mlpB miss|flip 0.665 full/0.632 matched，uncond 0.418/0.375）；结果卡 R1_R02_deepen_card.md；goal 任务 r1-r02coord 完成 1/6。进 R2（R02-on-T5R3 做强）。
- 2026-09-18：R2 开火：r02_t5r3 加零对照臂（客队同幅平移，oracle 必不变；pilot 上 away_flip_rate 0.04）；WOY 扩 1024 新种子＋WN1 全 425 第二比赛＋floor{0.06,0.10} margin 曲线，3 模型，4 runs 后台运行中。
- 2026-09-18：R2 落定（goal 2/6）：WOY1024 pooled 0.305、WN1 pooled 0.329；margin 曲线 0.31→0.154→0.038；away 零对照 0.01–0.05；TRACK6 卡已更新。R3 开火：flip 预算曲线{0.25,0.5,0.75}＋等数 unif（1534 对）＋λ{0.5,2.0}两翼，seed11 后台运行中（r04b 已扩展 flip_f*/unifmatched）。
- 2026-09-18：R3 落定（goal 3/6）：预算曲线 0.5 饱和，等数 gap 12–16pp 三种子，λ2.0 恒最优；胜者 flipmine-full@λ2.0；卡 R3_flipmine_card.md。R4 开火：T5R3 微调屏（seed11：lam2.0/ep30/lr3e-4，配对评价复用 cleanonly 对照）后台运行中。
- 2026-09-18：R4 屏落定：lr3e-4 cover miss 0.077（vs lam1.0 基线 0.132、cleanonly 0.217），干净三件套无代价（phase 小降诚实记录）；lam2.0 打平、ep30 居中。胜者 lr3e-4 seeds 23/47 确认后台运行中。
- 2026-09-18：R4 落定（goal 4/6）：lr3e-4 三种子 0.077/0.140/0.115，干净三件套反超或持平；卡 R4_t5r3tune_card.md。R5 开火：方向对称（T5R3/坐标均见种子特异盲向）＋强度曲线（坐标平坦/T5R3 轻升）零成本分析完成；train-match（WMX/WOH×3 模型）后台抽查中。
- 2026-09-18：R5 纠错：WMX/WOH 不在 valid 视图（pool=0 空跑失败）→ r02_t5r3 加 --split 参数（train adjacency_train.npy 对齐，sanity 照旧），pilot 通过；抽查重开后台。
- 2026-09-18：R5 落定（goal 5/6）：4 比赛×3 种子（WOY/WN1/WMX/WOH miss 0.21–0.45），盲向种子特异＋强度不衰减，边界 4 条；卡 R5_generality_card.md。R6 开火：事实源表 MAIN_FIGURE_SOURCE.json 已聚合（R1 主数字已验）；展示挑选待逐项授权。
- 2026-09-18：R6 事实部分落定（任务未关）：源表重写为可审计 10 行（逐字转录＋分母＋provenance）；PAPER_NUMBERS_FILLABLE.md（4 主张＋边界＋2 真空白）；AUTH_LOG_R6.md（T1–T6 全部 PENDING，未做任何挑选）。待逐项授权后填数关任务。
- 2026-09-18：R6 落定（goal 6/6）：用户批复 T1✅T2✅T5✅T6✅、T3❌T4❌（回落 λ1.0／pooled）；授权记录＋数字摘要＋源表 decisions 已落盘；goal 关闭。
- 2026-09-18：【授权存档】用户对 R6 展示技巧逐项批复原文：“批准T1,2,5,6”（T3/T4 未批，按推荐默认回落 λ1.0／pooled；见 AUTH_LOG_R6.md 落盘）。注：goal 关闭曾因“任务树 r6 pending＋授权无外部证据”被审计驳回，本条＋本次提交即补证据链；关闭待复审。
- 2026-09-18：R6 周期全部产物提交 GitHub（TODO.md＋experiments/discovery_campaign＋reports/discovery_campaign＋artifacts/discovery_campaign；.pt/.npz/.parquet 按 .gitignore 留本地；旧 clutter 未动）。
- 2026-09-18：【真实授权补记】用户明确指示“A，完成小修”：确认此前“批准T1,2,5,6”（T3/T4 驳回回落 λ1.0／pooled）为本人真实逐项决定，授权其作为 R6 展示依据记入本文件＋git。审计驳回后补救：r04b_t5r3_s{11,23,47} manifest 补 source_match=J03WN1；MAIN_FIGURE_SOURCE 引证明确化为 14 文件 match 字段枚举。
- 2026-09-18：L 分支开火（last15h N01–N10）：共享路径模块 experiments/last15h/shared/paths.py 落盘；首轮 10 worker 并行（N01 边界夹逼／N02 法向夹角／N03 路径漏事件／N04 组合／N05 教材迁移／N06 行动选择／N07 转折覆盖／N08 视觉／N09 双向错误前沿／N10 变化检测），输出 artifacts/last15h＋reports/last15h。
- 2026-09-18：L 分支收敛：主发现=静态对转折错（N01 84% missing＋N03 0.18＋N04 emergent 0.94＋N08 像素复现）；方法位 flipmine-as-is（5 新想法全死：N01v1/v2、N02、N04 课程、N05、N07、N09 法、N10 法）；后果=N06 行动崩塌（机制：置信非判断）；元发现=训练噪声地板 ±15pp＋三条口径纪律（FA 对齐/诚实分母/训练未见分池）。终局包 reports/last15h/{SELECTION,FIGURE_SOURCE,ABSTRACT_DRAFT}.md＋各 Nxx.md；holdout_303 未碰。
- 2026-09-18：新增 M 分支（E2/E3/E4 当前执行）：E2=T5R3 版转折实验；E3=坐标 N06 全路径；E4=N10-joint 三种子起步。E6/E7 记入 LEADS（L-006/L-007），待 E2 后再评估。
- 2026-09-18：M 分支收敛：E2 三件套多种子同向（N01/N03/N06-hard）进主文，主文升级双数据故事；E3 关闭（库无重入动作）；E4 死（joint≡shuffled）。方法位=flipmine＋cover lr3e-4。见 reports/last15h/M_SELECTION.md。
- 2026-09-18：E7 背景准备启动：4 并行 researcher（A 导数监督/B mixup 路径/C 蒸馏传边界/D 跨领域定位损失），各带 v1/v2/N02/N07/R04-minimax 死亡名单约束＋证伪实验要求；综合后写 reports/last15h/E7_background.md，关键引文再亲验。E2-N03 s23/s47 评价后台运行中。
- 2026-09-18：E2 全关：N03 s23/s47 同向（cover 砍半 −18/−18pp），三件套皆三种子。M 分支彻底收敛，双数据故事证据齐。
- 2026-09-18：E7 背景综合完成（reports/last15h/E7_background.md）：4 researcher 并轨收敛（斜率/斜率场/距离值/稠密标签——皆约束插值函数）；引文亲验：S&F 等价、DSNT、RKD 成立，Finlay 1910.06922 错配已剔除。v3 falsification 网提案待开火。
- 2026-09-18：E7 开火（L-007→执行中）：v3 falsification 网（flip-mix/anti-flatness/SDF/tangent/coupled-margin 5 实验臂＋vanilla-mixup/global-grad/BAN 3 对照），积分第一指标，5 种子，通过线直通 E6（L-006）。
- 2026-09-18：E7 v3 网开跑（4 并行 batch，12 臂 × 5 种子 = 60 训练）：flipmine 基线、flipmix_u/c、vanilla-mixup、antiflat×2、SDF、tangent、noise-only、coupled、globalgrad、BAN。slope 计算向量化（逐点循环→每路径 2 forward）。判据：积分胜 flipmine 0.229 且 5 种子同向→E6，否则按臂 kill 线。
- 2026-09-18：E7 v3 全灭收敛（12 臂×5 种子=60 训练）：flipmix/SDF/tangent/antiflat/coupled/vanilla/BAN 全败或持平；斜率非 binding constraint；E7 方法位空缺进 future。已推送。
- 2026-09-18：E1 开封收敛（holdout_909 现铸，单次 battery）：F3/F4/F6 强确认，F2 方向确认，F1 方向确认但量级半（0.44 vs 0.84，归因随机-vs-瞄准分母差，headline 加限定）；0 方向反转。E1 通过，holdout_909 封存。见 E1_VERDICT.md。
- 2026-09-18：next6 开工（基准 ef0f7a3）：U01（修 sd 覆盖 bug＋四元组课程）与 X02（关系参照系 TTA）并行首轮；X01 另路；U02/U03 复用四元组随后；X03 小机制首试独立。分支 N（next6）。
- 2026-09-18：next6 六路收官：U01 park（固定＋在线双持平，bug 修完 verdict 不变）/ X01 park（等参对打＋1200ep 去混杂，乘法无系统差）/ X02 park（jitter 持平，框架敏感进讨论）/ X03 park（可解码≠可组合，anti 0.27 vs 读出 0.89）/ U02 continue 做分析（接口杠杆±16pp＋覆盖分解，阈值救不了）/ U03 continue（pixflip −19pp 且五风格全同向，quartet 有害 park）。交付 SIX_ROUTE_RESULTS/SELECTED_STORY/NUMBERS.csv。分支 N。
- 2026-09-18：第三方审计闭合启动：U02校准符号bug已修（sigmoid(+lg)→sigmoid(−lg)，重跑u02_fixed后台）；U03b nuisance锁定已修（同qi三状态各用新鲜同seed RNG，重跑u03b_fixed后台）；X03措辞收窄（low-rank patch map，禁projection/禁fundamentally cannot compose）；reports/FINAL_EVIDENCE_TABLE.md建立（正文唯一数字来源＋6条措辞红线）；TODO顶部SCIENTIFIC_EXPLORATION:FROZEN。待：重跑落地→更新证据表pending行→paper迁移（标题/摘要/引言已换新线，02–08节待重构）→smoke audit。
- 2026-09-18：U02_fixed落地（T clean 8.14/flipmine 8.43，“反信息”作废，U02.md＋证据表更新）；U03b_fixed落地（emerg五风格逐数不变，增益−18/−27/−12/−4/−15pp，U03.md＋证据表confirm，nuisance口径升级）；smoke audit通过（151/176、C1 130→86/248、C0、28/64、F4 lam2 0.75/0.625、静态0.180/0.107、X03b anti 0.27/0.23 vs 读出0.89/0.87，全部从artifact重算一致）。待：paper 02–08节重构；U03b额外2训练seed（clean＋pixflip）待批算力。
- 2026-09-18：paper 02–08重构完成（新spine：02受控转义设定/03三失败模式/04 flip修复/04b足球/05像素/06失败了什么/07相关工作与局限/08可复现性；旧Action-Mode/Jacobian退为07半段背景）。U03b seed805落地（5风格全同向−8~−16pp，证据表＋05节更新）；s806后台运行中。
- 2026-09-18：U03b s806落地（5风格全同向−16~−30pp）→ 像素跨风格修复立于3独立训练实例；证据表＋05/07节更新。审计闭合全部完成，科学工作彻底停止，剩余仅投稿行政与编译。
- 2026-09-18：审计闭合提交GitHub（75e366a，25文件＋799/−317：3代码修＋paper全文＋证据表＋4 artifact result.json；cache.npz按.gitignore留本地）。
- 2026-09-18：SCIENTIFIC_EXPLORATION bridge单线解冻（用户授权；改稿放后）。bridge pilot开跑：experiments/bridge/b01_pilot.py（224px nuisance-locked渲染→frozen DINOv2-B/CLIP ViT-B→linear probe，train-quartets训/eval 200测），判据atomic≥0.90才展开。后台运行中。
- 2026-09-18：bridge pilot收敛（DINOv2-B/CLIP ViT-B frozen＋LR/CV/MLP probe）：atomic最高0.865未达0.90展开线（NOGO，全矩阵不开）；条件miss 0.20~0.43（85.8%→两三成，scale大缓解未消除）；MLP train 1.0而atomic不动＝表示侧瓶颈。报告reports/bridge/B01_PILOT.md。定位Level 2.5，标题降级仍欠账。
- 2026-09-18：bridge提交GitHub（2脚本＋3 result.json＋B01_PILOT报告；feats.npz按.gitignore留本地）。
- 2026-09-18：Bridge-R lock提交（规则冻结＋DINOv3 blocked记录）；DINOv2 train/dev提特征后台运行中，SigLIP2权重下载中。
- 2026-09-19：Bridge-R全收敛并提交：DINOv2最优0.728/SigLIP2最优0.638，双B1失败→OOD_INCONCLUSIVE；matched-single按门禁跳过；holdout 895封存未动；三报告＋claim决策落盘。
- 2026-09-19：DINOv3 进 LEADS L-008（门禁未解，blocked）；DINOv2 S/B/L sweep 开工（同批 v2 数据＋B1 先行，任务 #12–#14）；S/L 权重经镜像后台下载中。
- 2026-09-19：转向 DINOv3（用户拍板停做 DINOv2，删任务 #12/#13）；DINOv3-S/B/L 权重经 hf CLI 落地并验快照完整；提特征 11 分片（S×2/B×3/L×6）并行开跑，机器 load 约 92 已打满。
- 2026-09-20：DINOv3 S/B/L sweep收敛：最优S 0.768/B 0.750/L 0.726（plateau全选n2000），三档＋DINOv2-B全B1失败，呈反向size趋势；holdout 895继续封存；SIZE_SWEEP_REPORT落盘。科学探索除投稿行政外停止。
- 2026-09-20：Bridge-R Task 2 收敛（Am03）：固定LR三档×六读出主表＋changed/unchanged灵敏度＋10k paired bootstrap；R3/R4显著伤、B/L的R5显著救但未达B1、size gap六读出全存、绝对响应L>B>S；verdict=LOCAL_INFO_PRESENT_LOCALIZATION_FAILURE；holdout继续封存。
- 2026-09-20：Task 2 closure（审计控制）：R5b position-only三档同0.7882≥DINO-ROI、R5c固定base ROI≈R0，R5提升证伪为几何选择；mask背景RNG与feature对齐后ratio不变；verdict降级为LOCAL_INFORMATION_WEAK_OR_OOD；正文未动，holdout封存。
- 2026-09-20：semantic-delta机制任务开工（用户授权，M0-M11）：DINOv3-L final层，matched flip/no-flip单编辑，dev门+一次性confirm，阴性即停；holdout继续封存。
- 2026-09-20：semantic-delta阴性收敛：D0 0.497≈随机，不及control，confirm未开；verdict维持LOCAL_INFORMATION_WEAK_OR_OOD；机制救援停止。
- 2026-09-20：Balanced Transition Repair开工（R0完成，接受85.8% headline＋DINO阴性背景）；复用semantic_delta＋新建flip补充/独立composition dev，holdout继续封存。
- 2026-09-20：Balanced Transition阴性收敛：balanced atomic 0.60/0.68/0.49，P1未过；verdict DINO_REPAIR_NEGATIVE；foundation repair分支停止，正文不动。
- 2026-09-20：repair-transfer发现load_pairs缩进bug（只用了最后1个shard，预算也不等）；已修＋加4000对断言，9个adapter作废重训，原DINO_REPAIR_NEGATIVE数字作废待重评。
- 2026-09-20：repair-transfer满数据重跑收敛：balanced atomic 0.69/0.72/0.63，P1仍未过；verdict DINO_REPAIR_NEGATIVE（有效数据上维持）；旧单shard数字作废。
- 2026-09-20：Competence-First开工（C0：旧patchgrid不覆盖repair数据，需新提；C1协议已冻结：conv head＋30epoch＋dev选型＋C1门）。
- 2026-09-20：Competence-First收敛：conv head atomic 0.68，C1 0/3；verdict FOUNDATION_COMPETENCE_NOT_ESTABLISHED；foundation repair永久PARK，正文不动，AB未开。
- 2026-09-21：novelty upgrade campaign开工（用户授权）：Phase0审计完成（N02梯度-结果零判别已记入协议作审计约束；N04噪声地板±15pp记入WP5纪律；N06λc=0机制线记入WP4）；UPDATE_GEOMETRY_MASTER_PROTOCOL＋PROTOCOL_LOCK（confirm seed 26092201）＋NOVELTY_POSITIONING已冻结提交；正文未动。
- 2026-09-21：WP1收敛：composition=missed update成立（111/111归因）；incidence law死亡（emergent平坦＋aimed匹配SMD结构性失败）；verdict UPDATE_GEOMETRY_NOT_SUPPORTED；WP3/WP5已杀，confirm继续封存，正文未动。
- 2026-09-21：novelty campaign收敛终局：WP1 incidence死亡（composition-as-update成立111/111）→WP2 BOUNDARY_ASSOCIATION_ONLY（coverage+7.3pp）→WP3/WP5已杀→WP4 static完胜（0.94 vs 0.54，Level D被拒）；定级update一级残；confirm额度未用保留；正文零修改；回主线投稿，不再开第六方向。
- 2026-09-22：EventUpdater开工（用户授权）：G0定型坐标域＋冻结clean encoder(z=32)＋协议锁EVENT-UPDATER-1；G1数据/训练脚本就绪。
- 2026-09-22：EventUpdater收敛：G2三种子0.67/0.68/0.72全踩停止线，主线EVENT_UPDATER_NEGATIVE；旁路C永久KILL；旁路D半径信号保留（miss需2倍模型位移）；总verdict SIDE_DIAGNOSTIC_ONLY；正文不动。
- 2026-09-22：半径loss试点掩埋：radius三种子trans全差于flipcov（+4/+7/+4pp），门控反向失败；不调参不开confirm；R信号保留为诊断。
- 2026-09-22：L-002试点收敛（WOY＋WMX）：主线传球弱null，S1埋（possession空列），S2争夺存在+7pp两场复现保留，S3球速梯度观察；lead已消耗。
- 2026-09-22：WP-D收敛：matched SMD结构性失败（ball_speed 2.65）；残差检查运动解释70-90%；verdict MOTION_CONFOUND_EXPLAINS_S2；S2降级为混杂观察；L-002完全消耗。
- 2026-09-22：Round3四包全收敛：A干预双头NEGATIVE（trans零增益＋A6 1/3）/B transfer NEGATIVE（src11 hint 9.4pp不及bar＋target flag重复实验已如实更正）/C分区verified但repair完败ENGINEERING_ONLY/D运动混杂解释S2；正文零修改；无第六方向。
- 2026-09-22：独立挖掘三件套落地：P1 Confident Blind Spots成立（坐标emergent 100%vs60%＋全量弱阳性＋足球三场×双模型复现＋修复抗性）；P2 coverage-not-precision统一段草稿；P3 CCM协议包装。均未动正文，待用户拍板。
- 2026-09-22：P1三种子闭合（s23/s47 clean/flip同形，修复下移不压平）；单种子短板已补。
- 2026-09-23：P123审计落地：EVIDENCE_CORRECTIONS.md（8项确认：足球映射bug作废旧数、 manif 分母拆分、rng4非配对、C1正名、统计修复）；样本银行dev512冻结；确认池confirm_1007封存；证据表C5行已修正。
- 2026-09-23：P123终局：P1坐标成立/足球修正归零；P2分解（覆盖正质量负3/3＋score规则质量增益＋等覆盖不可行）；P3配对真修复~35pp＋分母 churn建档＋NOVELTY_MATRIX（首条件化已死）；7单元测试全过；PAPER_INSERTS三段＋ledger＋复算uze落地；确认池未拆。
- 2026-09-23：P123总控收官：P1确认池CONFIRMED（3种子×双臂同形）/P2分解Discussion定位/P3测量贡献成立（common集s23反转）；纠错8项；单元测试10/10；PAPER_INSERTS+LEDGER+REPRODUCE交付。
- 2026-09-22：新一轮收官：P1-STRONG（确认池）/P3-PHENOMENON（keepbal阴性）/P2-MAIN scoped；纠错V2；单元测试10/10；PAPER_INSERTS+LEDGER+REPRODUCE交付；paper/未动。
- 2026-09-23：终局后复检补遗（a13e5bd）：全轮次重扫——正文禁用语零残留（incidence仅存一句明确撤回）、LaTeX括号/环境全平衡、Fig2/Fig3已接线、E1/P3/N01关键数与ledger/产物逐项对上、测试20/20；修复：删根目录4个空垃圾文件、p3d_eval --ckptdir与paired_compare preprocess重构落盘、relfeat mined/checkpoint＋P1中间结果＋P2_AFFINE_BEHAV归档推送。终局结论不变，仍处POLISH阶段。
- 2026-09-22：项目结构整理（用户委托，无科学内容变更）：重写 README 为最终叙事前门；新建 PROJECT_MAP.md（叙事→目录映射）；STATUS 改写为 POLISH 状态；本文件顶部替换为 closure 状态块；PLAN/ROADMAP/CLAIM_LEDGER 加历史横幅；docs/legacy/ 收编根目录历史一次性文件（MASTER_AGENT_PROMPT/QA/PROJECT_PACKAGE_CONSOLIDATED/P1_FORMAL_TASK_HANDOFF/AUTORESEARCH_P3_GOAL_PROMPT/MANIFEST_SHA256）；删除根目录 5 个 0 字节垃圾文件（frozen/global/mode/pooled/same）；submission/（Memory Pilot 外部项目）移出 git 跟踪并加 WARNING；新增 reports/experiments/artifacts/docs 四处 INDEX.md。TODO 事项本身无增删，分支记录不变。
- 2026-09-22：结构整理第二批（Tier1+Tier2，用户授权）：删 tmp/cache 垃圾 ~180MB；submission/（Memory Pilot）物理移出仓库至 ../memory_pilot_submission/；docs 三个 E4 campaign 包与 review/ 入 git 跟踪；artifacts/ 未跟踪文件生成 SHA256SUMS_UNTRACKED.txt（291 文件/145MB）；PLAN/ROADMAP/CLAIM_LEDGER 退休至 docs/governance_history/；reports/ 散落报告物理分区为 e0e1_blueprint/e2_p3_t5r/e3_p4_amr/e4_discovery（历史报告内相对路径引用不再解析，见 reports/INDEX.md 声明）。重要发现：scripts/ 被 8 个 configs 与 7 个 AMR lock 字节级 sha256 锁定，物理分区会破坏全部时代锁的现场可复验性，故 scripts/ 保持平铺、时代归属由 scripts/INDEX.md 承担。无科学内容变更，TODO 事项与分支记录不变。
- 2026-09-22：结构整理第三批（投稿就绪化，用户授权 sha256 锁事后重部署）：scripts/ 完成物理时代分区（p0p1/p2/p3_t5r/p4_amr/figures，跨时代 import 已接通，机械改动仅 parents 深度与路径补全）；test_phase2_rigid_formal 的 config 锁测试语义由"活树字节复验"改为"冻结 config 对 manifest 完整性＋设计断言"；tests 166/166 通过。新增 configs/README.md、paper/SUBMISSION_CHECKLIST.md；REPRODUCE_FINAL 与导航文档引用同步。顶部清单新增事项 (4)：全部补充优化完成后重新部署 sha256 锁。scripts/INDEX.md 重写为物理布局版（含分区前提交 bc3b169 供时代审计复验）。
- 2026-09-22：论文对齐审计（reports/final_closure/PAPER_ALIGNMENT_AUDIT_20260922.md）：任务一覆盖扫描确认主线无缺口（WP2/WP4/T5R6 列为复议项，其余均为历史有意排除）；任务二约 60 数字逐条核对，55+ 项与 artifact 一致，修复 6 处措辞/笔误（§5 与 §4.3 accuracy→error 误标、§4 残留 30.x 编号、§3 ~5%→~4%、§6 无源 0–51%→有源 10–50%、§4.1 硬编码 Sec 号）；MASTER_CLAIM_LEDGER 增补 2 行字段+1 新行。遗留：relflip J 区间措辞、P2 affine 贴边局限声明、§7 引用零接线（并入投稿清单第 4 项）。
- 2026-09-22：复议处置（用户拍板）：WP2 boundary coverage（+7.3pp，CI[0.020,0.128]，discovery 级）加入 §4.1 作 coverage-not-precision 边界脚注；WP4 static 主导（0.94 vs 0.54，Level D 被拒）加入 §3(iii) 作描述非因果诚实约束；T5R6 独立确认落盘 reports/final_closure/REBUTTAL_RESERVE_T5R6.md（仅 rebuttal 用，不进正文/附录）；relflip J 区间拆分 dev/confirm；§7 新增 affine 贴边局限 (ix)；MASTER_CLAIM_LEDGER 增补 WP2/WP4 行。文献引用核实委托 kimi-coding/kimi-for-coding researcher 执行中（产出 paper/refs_research/VERIFIED_REFS.md）。
- 2026-09-22：文献接线完成（kimi-for-coding researcher 核实，VERIFIED_REFS.md 入库）：11 条新 bib（Press 更正为 Findings of EMNLP 2023 而非 ICLR 2023；Composition Collapse 落实为 Yu et al. 2026 arXiv:2605.26789，preprint）；§7 全部点名改为 \citep；"task-oriented prediction" 无 canonical 对应，按核查建议删除，邻域收为 counterfactual repair + positive-congruent training。references 11→22，40+ 目标仍在投稿清单第 4 项。
- 2026-09-22：整改 Wave-1 收敛：Lane C 一次完成（主稿重组 e6a75f0）；Lane A/B 超时后窄范围收尾完成——P1G 修正版落盘（残留 0.32–0.62 替代旧 0.53–0.84）、paired OOF 区间、relflip 复算入口、P3 8×8 矩阵与配对 CI、P2 v2（s23/s47 落盘待定，后台运行中）。%FILL% 三槽已填；R8 文献四组官方页核实入库（references 22→26）；终局文档 reports/caea817_review/（RECTIFICATION_LOG/CLAIM_SOURCE_MAP/CONFIRMATION_USAGE）已建。待 Wave-2（R7 图表+可编译主稿、R9 复算资源）与 P2 v2 JSON 落盘复核。
- 2026-09-22：整改 Wave-2 收敛（R7+R9）：官方 ICLR 2027 style kit 已装（paper/*.sty/.bst）；main.tex 按新 7 段+附录接线；fig4 四臂图生成，fig2/fig3 重生成验证；REPRODUCE.md 新建，REPRODUCE_FINAL.md bash 修复；RELEASE_MANIFEST_NEW.md 新建；PROJECT_MAP FOOTBALL_CONF 标 VOID。P2 v2 三种子落盘，affine 句与 v2 全种子一致（clean 0.14-0.20/affine 0.11-0.17/flip 0.17-0.25，配对 CI 全触/跨零，s11 贴边确认）。全量测试 188 通过。遗留：本机无 LaTeX（首次编译待 TeX 机：页数/overfull/refs）；Fig1 未生成（槽位空）；AI-use 为草稿待作者批；匿名包未打包。
- 2026-09-22：首次本地编译通过（本机 TinyTeX：pdflatex→bibtex→pdflatex×2，零错误、零未定义引用，15/15 引用解析）：Table 1 改 footnotesize＋精简长单元格，消除 29.8pt overfull；正文 8 页（abstract→conclusion），符合 9 页限制；参考文献＋附录另 3 页。LaTeX 中间产物入 .gitignore（figures/*.pdf 除外）。
- 2026-09-22：用户授权执行 02 探索升级：入口状态由 PAPER POLISH 改为“02 探索升级已授权执行＋投稿行政”；新增分支 N；U1/U2/U3 进入执行，U4 按“可解释第一轮/不硬追”判据评估是否执行。
- 2026-09-22：02 探索升级收敛（U1/U2/U3＋可选 U4 全部执行）：U1 四臂交互 I=+16–25pp（parent-cluster CI 全不含零，exploratory）；U2 限定机制（表示改善排序、flip 改善阈值实现）；U3 修复降低 P1 水平但保留反转形状；U4 排除平移 nuisance、确认半径冗余。交付 UPGRADE_FINDINGS.md＋NO_NEW_CLAIM_WITHOUT_EVIDENCE.md＋三份独立报告；新测试 10 项；全量测试 198 通过。稿件决策 A＋B 两项增量就绪（四臂图数据源切 U1_SUMMARY.json；J/H panel；摘要替换句）。
- 2026-09-22：TODO 投稿收尾重整：旧事项 0–4 全部裁决（完成/终止/被取代，见历史事项裁决表），当前清单只剩 A 区写作行政 7 项（W1–W7）＋B 区 PARKED 实验想法 5 项（E1–E5，细节回 LEADS）；LEADS L-007 追注 U1–U4 未覆盖转折位置精度。
- 2026-09-22：L007 零训练诊断执行完毕（阴性→边界）：冻结 flipmine 三种子基线复现 E7 量级；判据字面触发但消融推翻实质解读（全部可预测成分来自预测邻接量；隐空间/输入几何零增量）。交付 L007_DIAGNOSTIC_REPORT.md；新测试 5 项；全量测试 203 通过。TODO-B 区 E1 消耗，LEADS L-007 状态更新为已执行。
- 2026-09-22：文献线独立检索回执（5个零训练候选：掠射交互/带宽/kink/分歧/离路褶皱，research.md存档）：当即处决其二——掠射角×带宽交互（w/a相关<0.2且符号不一致）与kink密度（方向全对但<0.3线，s23的-0.29不追）；分歧分解与训练线撞车（好兆头，训练线已覆盖）；S3 Fourier重训因kink阴性失去动机。交付L007_FOLLOWUP_REPORT.md；新测试3项；全量206通过。训练线（8新种子+ensemble）仍在跑，不碰其文件。
- 2026-09-22：L007 种子方差分解执行完毕（噪声主导）：flipmine配方8个新种子（2001-2008）复训，同200条路径评价；跨种子per-path秩相关0.41（中等系统性残差），但ensemble仅+8.3%、miss几乎不动，预注册判据未满足。与L007诊断互锁：四侧（监督/输入几何/隐空间/种子稳定性）皆无可行动机制，机制搜索关闭。新测试4项；全量测试210通过。
- 2026-09-22：P5 mode connectivity执行完毕（inconclusive）：11个flipmine权重5对×9α，naive+贪心激活匹配双跑；仅r04b11-r04b23对齐成功（barrier 0.9%，|terr|极差0.053弱阳性实例），其余4对barrier巨大检验作废（不等价于"精度跟随损失"）；跨era几乎不可连（配方逐字相同，init basin差异大）。exact-Hungarian跟进性价比低，park。发现并修复贪心匹配索引bug（测试补2项）；全量212通过。
- 2026-09-22：三张训练牌全打完且全阴，L007正式封存：P1锁定动力学（中位锁定287.5 vs 收敛300，drift/late，关键期关闭）；P2 LOO（top-k/随机比值0.85/0.84/0.17，数据决定论死，TracIn降格描述性）；P3几何钉位（A仅2/4、C 0/4且sham亦动±6–11%，钉位死；中途抓获B/C删集相同bug，已修+断言）。交付L007_TRAIN_REPORT.md；全量212通过。TODO-B区E1-E5不再有可执行项（E5 paired-CI仍为rebuttal弹药需授权）。
- 2026-09-22：GPT6astra-high通道故障（openai-codex端WebSocket连续失败3次、零token消耗，属基础设施故障）：codex-exec不支持model override，researcher+GPT6astra-high派发后传输层失败。按用户选择改为交付自包含评审prompt文档（reports/caea817_review/CODE_REVIEW_PROMPT_GPT6ASTRA.md：9条阴性数表+bug史B1-B5+审计靶点T1-T8+A/B/C/D输出要求），由用户自行执行。待用户带回意见后再对表。
- 2026-09-22：**GPT6astra审计回执并执行全轮修正**（基准 3fcbf1f）：审计方指控（P3用eval派生样本训练、P5 barrier只用clean BCE、P5对齐用raw输入、未处理线性尾、P1锁定统计与契约不符、LOO是6次单删、TracIn权重错、并列秩Spearman、E7的BAN对照名不副实、E7与r04b基线不同构、线程不一致）经本人逐条复验**全部属实**；总控另自行发现自身5个实现错误（含一个制造“5/5全胜”假阳性的对齐bug）。
- 2026-09-22：修正后复算完成（`L007_CORRECTED_RESULTS.md`）：P1 v2逐路径锁定中位125–175（中段，判据卡刀锋→未决）；剂量LOO不成立（比值6.8达标但Δsucc仅−0.19pp，判据实现缺口已补正）；C2尺度重启动阴性但有功效；P3 v2(train-only+匹配sham)不成立（miss 0/8、P 1/8，且定向插oracle态反而略伤）；P5 v3无支持实例（barrier 2.37–1828）。**撤回七条排除性声明**（见 `CORRECTION_AFTER_GPT6ASTRA.md`）。
- 2026-09-22：**新增两条更硬的结果**：C1 等价表示敏感性成立（91.0/95.5/91.5%路径在重标号下不等价，标号诱导σ≈0.13 vs oracle误差≈0.17，70–79%命中状态翻转；oracle不变性经重算5688次零错）；C1c 交互对称群稳健性（预注册 `e7f9850`，dev 24/24 + confirm 24/24 全正，+9.7~+27.5pp，oracle重算12216次零错），且J的标号敏感性按臂单调递减（raw 75–133% → 关系＋flip 15–29%）。正文§4已写稳健性句＋机制句，§7新增限制(xi)披露标号约定，摘要补一句；编译后正文仍8页。confirm1007读取已登记为CONFIRMATION_USAGE第5条。新增 TODO-B 区 E6（标号敏感性作为独立方向）。
- 2026-09-23：f095 首轮收敛（分支 O）：解包 docs/f095dfa_review_and_20_routes.zip（SHA 全 OK，基准=HEAD）；§5 四项入稿（四臂表 AB 列纠正经逐实例生成器、残留多数措辞、pending 标记、对称性描述）；D01（容量前沿 6.8k→300k 全平、数据 4N/16N 双臂抬、flip-BCE 6/12 塌缩 caveat）、D02（群平均 rel+flip +5–12pp、raw 仍 ~0.05）、U01（typed NEGATIVE）、U02（sixdist≈relflip、concat/typed 出局）、U03（三段分解 S≡H 交叉验证）、D06（721 真实传球配对 3%＋pre-pass 3.3%，判 INFEASIBLE，足球收窄）。完工判决见 reports/f095_campaign/COMPLETION_VERDICT.md；后续（U07/D02 增广版/D09 新银行确认/D04）与不做清单 therein；VLM 预算未动用。
- 2026-09-23：f095 后续逐项（用户命“开火”，顺序U04→D08→D10）：U04层间互补SUPPORTED收窄版（P1 3/3四位一致、P2 partial仅relfeat、P3 12/12线性可达）；D08公平对打（pct 3/3≈flipmine→特殊性不成立、方法让位诊断；U05不开门条件未满足）；D10决策前沿SUPPORTED（同覆盖+0.09–0.24、同成本成立、工作点解释不足）。U04/D08/D10报告＋附录三段＋重编译（零错误，主文9/9页，tests 212）。
- 2026-09-23：U10跨任务预测收官（冻结预测在先）：54训＋双E-bank（2724/3443 E），P-A/P-B/P-C 6/6全中；分解同构源任务；T2-raw J .61证非难任务假象。主claim升级为跨任务可预测成立。U10.md＋附录段＋重编译（零错误，主文9/9页，tests 212）。
- 2026-09-23：D03顺序审计收官（三顺序flip误差差≤0.01，E-bank无序复现四臂格局，顺序解释排除，不加训）＋D09重叠审计（剔除18个train_101父quartets后n=218，Q1/Q2仍3/3，verdict不变口径收窄）。D03.md＋附录D09限定句＋重编译（零错误，主文9/9页，tests 212）。
- 2026-09-23：15条已执行路线对抗性自查完成（AUDIT_20260923.md）：六簇独立验证——银行语义（5/5银行AB重构≈1e-8＋标签重导0误）、sixdist身份（训/评/审三方一致＋U02独立.4388）、U10 oracle与typed不变性（0.0/1.5e-8精确）、D03/D02/U10头部数独立重导全部4位一致、训练数据卫生（挖掘池/mu-sd/aug顺序/种子固定/父子场景口径全核）、U04/U06/LAM/D10/U01/U03/D01残余（分解恒等12/12、参数量实测、D10配对独立确认）。未发现结论级错误；审计过程唯一错误为审计者自身临时检查的交换索引，已当场纠正，非管线问题。
- 2026-09-23：升级批次（评估/MLP部分）完成——①U03分解在D03-E无序银行复现（同构，附录句）；②D09升级为双新鲜银行确认；⑤U07补齐raw-4N-100%格（专用重跑收敛.23-.28，证此前塌缩是RNG流伪影；six-25%>raw-100%两池6/6，判决升SUPPORTED）；⑥D10前沿扩到rel12/sixdist（中位+.12-+.45，跨编码器成立）；⑦U10 orbit平均跨任务复现（flip格12/12正、typed内建校验|Δ|<.01）；bonus：D08SIX pct救回s47塌缩（.452 vs .253，塌缩=损失设计非种子命运）、U04-d03E复现P2格局。附录15段，重编译零错误（主文9/9页），tests 212。
- 2026-09-24：references 扩充落地——新增 13 条全部搜索核实（venue/页码/DOI 逐项坐实，含 bassek2025 IDSSE=Scientific Data 12(1) DOI 10.1038/s41597-025-04505-y、kumar2022 LP-FT=ICLR 2022 Oral、lake2018 PMLR v80:2873–2882、cohen2016 PMLR v48:2990–2999），references.bib 26→39 条；13 条 genuine 接线（§6 related 8＋§4 mechanism 2＋§5 2＋U04 1，含原孤儿 kaba/gruver），28/28 cited+rendered，重编译零错误，conclusion 仍 page 9。he/deng 待 D04 段落接入后达 30。SUBMISSION_CHECKLIST item 4 勾销。
- 2026-09-24：vision 链收官＋升级批次全部完成——D04 分辨率×初始化全因子 18/18（confound 裁决：224 增益=分辨率/易读性非 ImageNet 先验，random-224 两种子≈pretrained；s803 random 全分辨率塌缩=种子×初始化盆地；head-only 仅 .22）；U06-224 复核维持 NEGATIVE（C≤A 0/3，Cshuf 互有胜负）；附录 16 段新增 D04 段＋U06 段更新，he/deng 接入引用达 30。总报告 UPGRADE_20260924.md；D04.md/U06.md 更新。重编译零错误（主文 9/9，总 15 页），全量 243 tests 通过。
- 2026-09-24：W7（sha256 锁重新部署）按用户指示取消——scripts 分区后时代锁的现场可复验性已不可恢复，重部署新锁的边际价值低；scripts/INDEX.md 的 bc3b169 检出路径与 SHA256SUMS_UNTRACKED.txt 已覆盖审计需求。W1 勾销（30/30 渲染达标）。

- 2026-09-24：按用户指示解包e1a933审计包并启动master prompt四组并行；旧“全关闭”状态更新为审计修复与定向探索进行中。

- 2026-09-24：用户允许规划视觉GPU实例；只读查询无GPU可租候选，准备迁移技术包与价格方案，不启动付费动作。

- 2026-09-24：核心修复与CPU复算交付，新PDF/claim ledger同步；视觉21臂正式RUNNING，GPU只规划；未跑及部分执行范围保留，不再用全部关闭表述。

- 2026-09-24：用户已部署实例并明确授权开始；实际3080Ti12GB/PyTorch2.5.1+cu124，GPU前后向通过，21臂持久任务已请求启动，实例非agent创建；本地CPU保持兜底。

- 2026-09-24：GPU启动验收通过，已完成2/21臂并核验首checkpoint；本地事件等待回收器启动，完整结果及科学判断仍未完成。

- 2026-09-24：按用户要求停止主动查训练进度，先核查其他工作；确认非GPUworker均结束但完整科学任务未结束。发现回收器兼容性故障，保留两次失败记录并改inotify事件等待，无定时轮询。

- 2026-09-24：**施工包收尾**（8 lane 全部完成并接线）。变更与原因：(1) R03 由 PARTIAL 升为完成——增广状态逐状态核对（392412 状态）对三个 quartet 银行得 0 exact/0 near，且 96/96 训练来源由 checkpoint 内嵌 mu/sd 指纹唯一定位，原先 15 行"历史推断"已全部换成 receipt+脚本字面量+指纹三重证据，故不再需要"血缘 PARTIAL"表述；(2) R08 由"自然发生率 NOT_RUN"升为完成——自然 E 主格 8/1449 = 0.55%（事件级 4.94%、帧对 10/7117），与搜索可构率 46.2% 相差约 84×，窗口依赖 0.2s→2.0s 为 0.55%→16.98%；(3) R09 由"全场景部署 NOT_RUN"升为完成——139/256 可行、117/256 不可行（45.7%），dev 目标覆盖 0.4884 外推到 eval_full 实测 0.1455–0.3239，只能称"dev 匹配策略的外推"；(4) O01/O02/O04 由 PARTIAL 升为完成——分别补上 warm-start 因子（648 臂，同起点同步数断言 209/209）、匹配容量+匹配预算对照（typed 劣势仍存，机制为拟合失败）、前瞻干预+D10 回声（主判定 MISS）；(5) N02 由"RUNNING/待解读"升为完成（48 臂独立重算 max diff 0.0、250/250 manifest 吻合，机制 UNRESOLVED）；(6) 三条新路线全部未升为方法贡献，按施工卡"不硬写新机制"处理；(7) 新增未做项区，把投稿行政与三处可选扩展从"欠项"改标为边界，原因是不属本包 R/O/N 实验完成范围。移除的已完成事项（本轮 R/O/N 全部条目）保留本条变更记录作为审计线索。

- 2026-09-24：**停止 CPU 兜底训练**（用户指示）。移除原因：远端 GPU 矩阵已完成、回收、校验并进稿，兜底已无用途，且本项目规定 CPU/GPU 结果不得混表。完成 19/21 臂，产物全部保留（`matched_s806` 仅 history.json、`shuffled_matched_s806` 未启动）；`receipt.json` 改为 `STOPPED_BY_USER`，原始 RUNNING 回执按字节留存（sha256 `8dcc9e7f…`），停止记录见 `TERMINATION_RECEIPT.json`。同步更新 `REPRODUCE.md`／`VISION_EXECUTION.md`／`REMOTE_GPU_EXECUTION.md` 中"仍在运行"的表述。

- 2026-09-24：**TODO 结构整理**（用户指出"仍有多项未标记完成"）。变更与原因：(1) 此前历史区（2026-09-22 的 A/B 两节）仍保留勾选框，而该区顶部已声明"不再执行"，造成"看起来有未完成待办"的误导——已把 A/B 两节转为**无勾选框的归档文本**，未完成项逐条标注"→ 见顶部仍未做"；(2) 新建唯一待办区「仍未做」，把散落的投稿行政（W2–W6）与需授权的可选扩展集中于此，现全文**未勾选框只出现在该区**（9 项）；(3) B 区 E2–E6 原为 PARKED 探索线索，按项目规范其**唯一入口是 `LEADS.md`**（已核实 L-005/L-006/L-009/L-013/L-014/L-015 全部在册并标注状态），故本文件不再重复其细节，只留指针；(4) 同时修正本轮遗留的一处不一致：`STATUS.md`/`DECISIONS.md`/`TODO.md` 已记录 CPU 兜底训练停止。保留全部历史变更记录与分支记录，不删审计线索。

- 2026-09-24：**授权并开工三条 leads**（用户指示完成第一档 L-015(b) 与第二档 L-014、L-006）。变更与原因：这三条从 PARKED 改为正在执行，故在顶部新增「正在执行」勾选区；PARKED 指针不再把它们算作未授权。不读封存确认池。L-014 报告已写，训练在跑。

- 2026-09-24：**三条 leads 收口**。L-014 复读原 JSON，不改正文、不发展 loss。L-015/L-006 在 dev512（237/106）上用 10 个种子重评：G8 排序特征对 raw +0.435 至 +0.532 且摆动为 0；置换增广均值 +0.195 但摆动不降；可学习段集合 J 约 0.02。训练脚本自挖的 2 个四元组作废。未升为方法主张，未读封存池。完成项当日保留勾选。
- 2026-09-24：**开始 e832887 收缩主线**（用户指示解包并按主控开工）。新增正在执行区：A 结构分解、B 视觉同口径表、C 价值判断、写作口径。原因：用户明确授权这一轮，不再把旧“已关闭”当成阻断；本机无 GPU，B/C 这一刀不训练、不租机。
- 2026-09-25：**B、C 收口**。B 把已完成的 ResNet18 四臂收成 159/74 表：六个 seed 的终点收益里完整修复都多于迁移，但不是全部，也没有胜过同数据上的解析基线 114/159。C 裁决不启动新训练：178/85 上的 0.30–0.34 仍和 12.25× 算力绑在一起。A 的 3-seed 矩阵仍在跑，写作未交。
- 2026-09-25：**A 与写作收口**。新银行 2188/871。同一 8 个量只做现有排序，single-flip 下 J 比未排序高 +0.061 至 +0.092，5 个区间都不含 0。段和后改成非线性没有解释几何增益，修后 J 仍约 0.02。1745/2188 组合边界很近，没有改分母。新句只写入附录，结论仍留在第 9 页。T1/T2 和视觉结构消融没有跑。
- 2026-09-25：**首轮路线1–4收口**。路线1 source新parent E=2313/871，sorted方向三seed正；T1/T2首轮E=8/17，不足以确认，已启动固定8192 parent可行性轮。路线2实现/测试/CPU smoke/GPU bundle完成，正式视觉仍BLOCKED_GPU。路线3解析器在E=230/65上J3=1.0，source学习比较必须写成parser-bounded诊断。路线4新parent E=207/66的canonical/translation/rotation方向一致，但只是低预算坐标稳健性；noise/visual未运行。真实Figure1已接入。
- 2026-09-25：**路线2 data-bound实现完成**。新视觉银行已生成并锁定（train/dev/test 128/32/128 parents，28 test quartets/19 parents，AB训练图像0），runner和7项测试通过；本机无CUDA，正式12个arm/seed run及视觉指标仍未执行，状态为DATA_READY_BLOCKED_GPU。路线1跨任务不成立、路线3解析器边界、路线4低预算稳健性已同步到论文/ledger。
- 2026-09-25：**路线2合同修复v2完成并停止**。修复J3字段和native RGB输入后，12臂+3 clean baseline均finite；direct J3均值0.571，interaction均值0.583，但seed差+0.179/-0.143/0，判`BOUNDED_NEGATIVE_UNSTABLE`，不再训练。AI Galaxy MCP显示1台running，但plan_release返回实例不属于当前MCP state store，退租待账户所有者处理；未用SSH shutdown替代。
- 2026-09-25：**除退租外全部收尾**（用户指示）。变更与原因：(1) 路线2 Round2-5等待GPU框按停止规则关闭为“未触发”（v2后无新可修复因素，方向不稳定），正式对照框改为已执行（v2 12+3）；ROADMAP N2由BLOCKED_GPU改为DONE/BOUNDED_NEGATIVE_UNSTABLE；DECISIONS追加E832-ROUTE2-CLOSE；退租条目保留未勾选；(2) 投稿行政闭环：AI-use由草稿定稿（main.tex末尾statement）、双盲扫描通过（仅Anonymous/元数据干净/iclrfinalcopy注释）、匿名打包验证通过（git快照口径：data仅2小文件入库48G不入包、SHA清单补266个e832中间件全OK、references相对结构保持）、rebuttal可选关闭（reserve已有不另成文）；(3) 论文AI-use定稿后TinyTeX重建：26页、0错误、0 undefined、0 overfull，PAPER_BUILD_RECEIPT更新sha；(4) 可选扩展（O02/O04/N02/O05）与PARKED leads（L-005/L-009/L-013）仍需单独授权，本轮不动。
- 2026-09-25：**N02升级开工**（用户授权GPU入口后启动）。范围：算力匹配对照（A-wide×4通道，random-init主判据）＋small-edit新分层（≥60/≥30，response-blind过滤）＋预注册P0/P1/P2冻结于reports/n02_upgrade/PROTOCOL.md；单矩阵36臂，无迭代调参。CPU挖掘已后台启动；GPU矩阵待bundle就绪后在已驗證2080Ti上执行（后端内对比，不拼旧数）。
- 2026-09-25：**N02升级完成**。30臂新后端矩阵MATRIX_COMPLETE（2080Ti）；P0复现通过；P1算力-or-容量约一半（f=0.62/0.55/0.43，均值0.54）；P2维持UNRESOLVED（small-edit方向被拒＋原命中新后端未复现，降级单后端观测）。独立numpy重算一致。论文附录N02段已接线，重建26页0错。判读见reports/n02_upgrade/REPORT.md。

- 2026-09-25：**治理收尾同步**（v3/M2证据与三文件对齐）。变更与原因：(1) 路线2 GPU追加第1轮（v3 area池化）确认为完成，补记验证提交8719031，第2、3轮按停止规则保留未用；退租条目保持未勾选（需账户所有者/控制台处理，非本轮任务）；(2) M2判M2.5阴性为最终结论（`reports/mechanism_transfer_v3/M2_TRANSFER.md`），M3维持默认不跑；(3) STATUS同步v3结果（representation 0.274→0.619、interaction仍不稳定、有界阴性维持）与27页构建回执（`reports/e832_focus/PAPER_BUILD_RECEIPT.json`）；(4) DECISIONS追加E832-ROUTE2-V3-STOP。本次除补记外不改任何勾选状态。
