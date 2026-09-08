# TODO — ICLR 2027 Action-Mode Spectrum

更新时间：2026-09-04

## 当前固定决定

由于 SNGAR 当前无法访问，IDSSE 暂时替代 SNGAR 的“开发数据”角色。这个替代只覆盖当前 T5R 开发链，不改变 P2/P3 历史，也不把 7 场 IDSSE 说成 SNGAR 的 45/9/10 场划分。

同一批 IDSSE 数据不能在本轮同时充当开发数据和独立外部确认数据。SNGAR 恢复、真正的独立 provider 确认和 P4 放行仍是后续条件。

## 最近一次官方在线核验（2026-09-04）

- 官方数据集页确认：7 场 2022/23 德国 Bundesliga（含一、二级联赛）比赛；每场包含 25 fps tracking、同步 event 和 metadata；许可为 CC BY 4.0；页面总大小显示 2.63 GB。
- 官方 `main` 文件树及提交 `a715a38dfbaf5f58e431727c2b78d174101a703c` 列出的实际文件 ID 是 `J03WMX`、`J03WN1`、`J03WOH`、`J03WOY`、`J03WPY`、`J03WQQ`、`J03WR9`，与本地三类 XML 文件的内部 `MatchId` 和文件名一致。
- 官方数据卡正文仍列出 `J03WPF`、`J03WQF`；这是官方页面内部的文档差异。当前以官方文件树作为 raw 文件身份依据，保留差异，不改原始文件。
- 官方 Dataset Viewer 当前对 `default/train` 报 `FeaturesError`/`FileNotFoundError`，所以字段级结构核验仍由本地 XML QC 完成，不能把 Viewer 不可用写成数据缺失。

## 主线待办（按顺序和重要程度）

- [x] 1. 完成 IDSSE 来源登记与原始数据核验
    - [x] 已按官方页面和 `main` 文件树登记 dataset revision、文件清单、比赛 ID、许可和数据说明；数据卡正文的 ID 差异已记录。
    - [x] 已确认官方文件树的 7 个比赛 ID 与本地 raw 文件一致；不按数据卡中的旧 ID 猜测或覆盖原始文件。
    - [x] 为 `data/raw/sports/idsse-data/` 建立来源记录、文件清单和 SHA-256；23/23 原始文件复核通过。

- [x] 2. 把 IDSSE 原始 XML 转成项目统一格式并完成逐场 QC（依赖 1）
    - [x] 保留 raw→canonical 映射、parser 版本、坐标单位、时间轴、缺失率、长间隔和事件对齐记录；7 场 receipt 均完成。
    - [x] 输出每场 conversion receipt、data card 和可复核的 canonical manifest；整体 QC 审计通过。

- [x] 3. 冻结 IDSSE 的 match-level 开发/保留方案（依赖 2；训练前必须完成）
    - [x] 只按比赛划分，不按帧或事件随机切分；train/valid/reserved 已写入 split lock。
    - [x] 明确 reserved 集只是同源未见 match，不等同于 SNGAR hidden test 或独立 provider confirmation。

- [x] 4. 冻结 T5R2 任务、指标和 baseline lock（依赖 3）
    - [x] context 使用 raw positions 的 `z_ctx`，intrinsic 使用 centered positions 的 `z_mode`。
    - [x] 固定 natural pair ranking / retrieval、translation robustness、cross-readout leakage、match-level bootstrap 和 non-inferiority 规则；lock 中明确 model results 尚未使用。

- [x] 5. 运行 fixed dual-channel sanity（依赖 4）
    - [x] 比较 raw single-channel、centered single-channel、relational pooling 和 fixed dual-channel；4×3 neural grid 的 capacity 与 seeds 可审计，另有 Procrustes analytic control。
    - [x] context 在 non-inferiority margin 内，intrinsic 三个 seed 均改善，`z_mode` 平移响应近零，natural-pair geometry relation 未坍缩；T5R3 closure review 通过，允许进入 T5R4 Round 1。

- [x] 6. 按预算运行 bounded autoresearch（两轮已完成并关闭）
    - [x] Round 1：4 个候选、每个候选 seeds 11/23/47，共 12 个模型；正式输出为 `artifacts/phase3/task_semantic_repair_v1/t5r4_round1_v1/`；无全维度 Pareto 支配者，不自动晋级。
    - [x] Round 2（最后一轮）：只检验 encoder update balance，3:1/2:1、总步数 420 持平 reference，共 6 个模型；正式输出为 `artifacts/phase3/task_semantic_repair_v1/t5r4_round2_v1/`；按训练前冻结规则选出 `update_ratio_2to1`，关闭搜索。语义 addendum 与授权报告已冻结。

- [x] 7. candidate lock 后运行一次保留 match（T5R5 PASS_STRONG；外部确认仍待后续）
    - [x] 已锁定候选（`update_ratio_2to1`）、配置、数据 hash、干预协议、指标和失败条件；lock 审计通过后执行一次性读取。
    - [x] `J03WQQ` 读取完成（3575 snapshots/515 pairs，只读一次，无重训练）：任务 gate 全过，干预 full mean 0.63 强通过，路由 A（T5R6 独立确认）。
    - [ ] 不把已用于开发的 IDSSE 结果写成 external confirmation；SNGAR 或 SoccerTrack 仍是独立确认恢复分支。

- [ ] 8. 根据 T5R gate 决定 P4 或诊断论文路线（依赖 7）
    - [ ] gate 未闭合时冻结为诊断/机制版本，不启动 AMR，不提前推进书法确认。

- [x] 9. T5R6 独立外部确认（SoccerTrack-v2，8 场同方向，CONFIRMED）子项串行执行完毕
    - [x] N1 下载登记 8 场 GSR 并写 SHA/receipt（45.16GB，修订 eae51793）
    - [x] N2 GSR 转 canonical 与逐场 QC（44,401 快照）
    - [x] N3 构建任务视图并审计（7,534 对）
    - [x] N4 跑任务确认（2:1 逐场 8/8 占优）
    - [x] N5 跑干预确认（8/8 full 高于基线）
    - [x] N6 写报告回写 ledger 定路由
- [ ] 10. ICLR 2027 投稿冲刺（评审已完成：`reports/PUBLICATION_REVIEW_ICLR2027.md`）子项 10a 阻塞其余全部
    - [ ] 10a 行政合规：OpenReview profile（机构邮箱，审核可长达 2 周）、互审资格、作者名单——9/18 摘要截止前冻结，今天必须启动
    - [ ] 10b 冻结论文形状并开始写作：机制先行 9 页；AMR/书法/篮球进 future work（可并行，依赖 10a 的作者决定）
    - [ ] 10c 修复增益 headline 表＋8 场 bootstrap CI（已有数字，只需整理）
    - [ ] 10d SSL 基线（SimSiam/BYOL 类，仅开发集评估，不新增保留场/外部读取）
    - [ ] 10e 配比扫掠 1:0/1:1 是否补做——需用户拍板（与"无 Round 3"承诺有张力，见评审 3.3）

## 分支记录

| 分支 | 状态 | 说明 |
|---|---|---|
| A：IDSSE 暂代 SNGAR 的开发主线 | 当前执行 | T5R1–T5R5 已闭合（`T5R5_PASS_STRONG`，路由 A）；T5R6 独立确认待执行，7 场 IDSSE 不复用为外部确认。 |
| B：SNGAR 恢复 | 暂缓 | 保留原访问和 test firewall；获得可信访问或带 revision/mapping/SHA-256 的镜像后，可升级独立样本量与确认强度。 |
| C：IDSSE 独立外部确认 | 本轮关闭 | IDSSE 一旦被用于开发，就不能在同一轮再作为独立 external confirmation；需要另一个 provider/source，或明确降级为同源保留 match。 |
| D：动态 support | 可并行、非阻塞 | 继续使用历史 SkillCorner 的 response-blind 规则开发，不能反向选择主线候选。 |
| E：AMR 与书法 | 阻塞 | 只有 T5R 任务—鲁棒性—外部确认门闭合后才重新评估。 |
| F：SoccerTrack-v2 外部确认 | 完成 | T5R6 CONFIRMED；8 场同方向；P4/诊断决策待事项 8。 |
| G：ICLR 2027 投稿冲刺 | 当前执行 | 摘要截止 2026-09-18、全文 2026-09-25（官方核实）；论文形状为机制先行＋修复＋外部确认；AMR/书法/篮球明确移出正文。 |

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
