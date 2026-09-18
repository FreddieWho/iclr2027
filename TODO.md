# TODO — ICLR 2027 Action-Mode Spectrum

更新时间：2026-09-16（B30 重整版；旧 1–13 项已归档，见文末）

## 研究结论一句话（为什么是这份清单）

- B30 头脑风暴 30/30 完成：`reports/research_b30/` 34 文件（W1-M1–M10 / W2-A1–A10 / W3-S1–S10＋ANGLE_MAP/LEDGER/SYNTHESIS_TOPLIST/VERIFY_FINAL），目标 `mu3wl782-yxuns1` 已关闭。
- VERIFY_FINAL 复核 18 项主来源：17 项维持、仅 SimCLR 长 schedule 降级为 B（方向可信、量级弱）；5 项淘汰维持禁令；开火顺序冻结。
- 含义：写作侧 6 项零训练成本先行；训练侧剩 8 轮只打 T2-1→T2-2→T2-4；Tier3 只按 non-claim 口径取用；行政 deadline（摘要 9/18、全文 9/25）阻塞其余。

## 接下来要做（按执行顺序和重要程度排列）

- [ ] 0. 行政与冻结——阻塞其余全部，今天必须启动          子项须串行，a 不定则 b/c 返工
    - [ ] a OpenReview 资料冻结：机构邮箱 profile（审核可长达 2 周）、互审资格、作者名单——9/18 摘要截止前冻结
    - [ ] b 论文形状冻结：机制先行 9 页；AMR/书法/篮球明确进 future work
    - [ ] c 样式与引用基建：官方 ICLR 2027 样式替换占位 preamble；references 11 条→40+ 条

- [ ] 1. Tier1 写作六件套（零训练成本，推荐开工顺序即此序）   子项可并行，但 a/b 先行
    - [ ] a T1-2 前置复核 1 小时：TacticAI PDF 全文 D2 细节复核（B＋升 A 的唯一门）
    - [ ] b T1-3 前置 source-check：split/外部集/基线/种子/算力五处冻结数核对
    - [ ] c T1-1 标题摘要机制先行重构（S3 方案 A，写作半天）
    - [ ] d T1-2 related-work 柔道四段（写作半天，依赖 a）
    - [ ] e T1-4 claim 分级制＋全表 min–max 化（写作半天；禁 mean±std、禁 winner 语言）
    - [ ] f T1-3 rebuttal 十大异议＋现成答案（写作 1 天，依赖 b）
    - [ ] g T1-5 证伪 K1–K3 预注册（K1 permutation null 为 CPU 小任务；K2/K3 先注册再跑）
    - [ ] h T1-6 图表三件套：证伪 ledger 表＋validity-regime 图＋机制 panel（作图 1–2 天；Spec A 轴范围从 epsilon/support 日志锁定）

- [ ] 2. 训练开火（10 轮预算剩 8 轮，独立记账）              子项 T2-1 与 T2-2 可并行（改动正交）；T2-4 待前两项判据
    - [ ] a T2-1 margin–SupCon 混合＋τ sweep（首选开火，1 轮；τ∈{0.07,0.1,0.2}×m∈{0.1,0.2,0.3}，τ=0.1×m=0.2 起步）
    - [ ] b T2-2 attention readout（次选，1 轮；mean＋att＋max＋energy concat，α-entropy 防塌）
    - [ ] c T2-4 结构等变分离路线 D1→D2→D3（占 1–2 轮；novelty 野心最大，需新架构＋审计）
    - [ ] d T2-3 长 schedule 备选（seed-23 单跑 2–3×＋cosine；叙事包装成收敛修正）

- [ ] 3. Tier3 按需取用（non-claim 口径，不占主线）          子项可并行，随手稿进度取用
    - [ ] a W-1 书法 outlook 一段＋定性图（不 eval）
    - [ ] b W-5 B1 probe-discipline audit 进贡献（禁 leaderboard 承诺）
    - [ ] c W-2/W-3/W-4 进 broader impact／related 段落（禁 adversarial/certificate 词；Rank 3 禁入正文）
    - [ ] d W-7 层级探针 E1–E3 零成本重算已有 checkpoint（机制深度证据）
    - [ ] e W-8 耦合标度律 240 点拟合（只许 2 参数饱和形＋hold-out 审计）

- [ ] 4. 手稿组装（骨架已有：`paper/main.tex`＋8 节＋EVIDENCE_MAP） 子项可并行，依赖事项 0 的 b/c
    - [ ] a 摘要初稿复核（`paper/sections/00_abstract.tex`，用户暂缓后重启）
    - [ ] b 正文 prose 填充（按 EVIDENCE_MAP 措辞规则；11d 措辞修正随此执行）
    - [ ] c Figure 1 概念图生成
    - [ ] d §4.5 证伪表述＋淘汰项诚实声明（X1–X5 禁复活；μ 重做禁返工）

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
