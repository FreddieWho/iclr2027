# Decisions

## D-20260901-P3-001：P2 收口后的路线

- 日期：2026-09-01
- 决定：将当前 active phase 设为既有 P3_CAUSAL_MECHANISM；不增加 P2.5/P3a/P3b，不继续调 P2，不直接训练 AMR。
- 原因：P2 exact-spectrum 与 fracture continuity 已证明普通谱功率不足，但异质性为 mixed_or_graph_specific，方向依赖条件。
- 受影响文件：README.md、MASTER_AGENT_PROMPT.md、configs/project.yaml、configs/experiment_matrix.yaml、P3 canonical docs。

## D-20260901-P3-002：中心机制对象

- 日期：2026-09-01
- 决定：保留 Action-Mode Spectrum 作为边缘汇总，新增待检验的 support-conditioned normalized-embedding local geometry：
  q_f(X, delta)=1/2 ||J_f(X) delta||_2^2，G_f=J_f^T J_f。
- 原因：P2 fracture 在保持非零位移 vector multiset 时重分配端点支持，正好可检验 diagonal 与 off-diagonal block。
- 当前解释：工作假设，不是 P3 结果；role 仅为诊断分层，不是通用方法输入。

## D-20260901-P3-003：最小施工路径

- 日期：2026-09-01
- 决定：按 T0 接口/provenance → T1 retrospective prediction → T2 prospective → T3 layer/block → T4 单一因果开关 → T5 任务联系执行；完整 augmentation × pooling × constraint matrix 仅为 gate 后扩展。
- 原因：避免大型后验可学习预测器、结果驱动选 support/layer，以及先训练 AMR 再找解释。
- 受影响文件：docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md、docs/05_AGENT_EXECUTION_MANUAL.md、configs/phase3_support_geometry_v1.yaml。

## D-20260901-P3-004：统计与跨域边界

- 日期：2026-09-01
- 决定：统计单位为 source match；frame、pair、draw、seed 不是独立比赛重复。书法已有正式结果继续标记 exploratory，不用于选择体育端机制；本轮不新增外部数据。
- 原因：保持 P2 provenance 和防止跨域/结果选择泄漏。
- 受影响文件：QA.md、STATUS.md、CLAIM_LEDGER.md、P3 config。

## D-20260901-P3-005：当前不一致项审计

- 日期：2026-09-01
- 结果：canonical source docs、README、MASTER prompt 和机器配置已切换到 P3；PROJECT_PACKAGE_CONSOLIDATED.md 的旧正文保留为历史合并内容，并以文件顶部新增的 P3 synchronization addendum 明确当前状态和优先级。P2 reports/artifacts 未改动。
- 备注：在生成新的正式 P3 结果后，必须再次同步 consolidated package 和 claim ledger；不得用 addendum 改写 P2 历史。

## D-20260901-P3-006：prospective 设计与结果边界
- 决定：T2 只保留一个在查看 response 前锁定的同资产新 draw（seed 20260901）；不因早期运行成本而补选或筛选干预。
- 结果：250 samples、10 matches、5,700 intervention sets 中 4,475 组完整，9 个冻结模型产生 90,432 arm rows 和 40,275 pair rows；response-blind generation、公式、层和指标均未在 response 后改变。
- 影响：T2 支持同资产 prospective 外推，但不等同于新比赛/新数据域验证。

## D-20260901-P3-007：T3 证据选择的唯一开关
- 决定：依据 layer/block 结果只选择 `pooling_accessibility`，限于 Phase-GAT；比较无新增参数的 `team_mean` 与 `relational_pairwise` pooling，3 seeds、matched capacity。
- 原因：DeepSets 的 off-diagonal 主要在 pooling 后出现，GAT/Phase-GAT 的 off-diagonal 在 message passing 中形成并在 pooling 保留；pooled/pooling-pre geometry 的预测力最高。
- 受影响路径：T4/T5；完整 augmentation × pooling × constraint matrix 仍不运行。

## D-20260901-P3-008：P3 gate 路由
- 决定：将 P3 标记为 `T4_T5_COMPLETE_RESPONSE_SHAPING_ONLY_P4_GATE_NOT_MET`，不创建 Commit 5 的 AMR/机制成功版本，不进入 P4。
- 事实：relational pooling 三 seed 均提高 off-diagonal/full 比例和 geometry-response Spearman，但三 seed context response 均上升；heldout accuracy 均值从 0.3542 升至 0.3854，macro-F1 均值从 0.1861 降至 0.1839，未满足客观任务与 robustness 条件。
- 解释边界：结果支持“可按预测塑造表示几何”，不支持“修复表示”或 AMR-Fixed 已有明确任务目标。
- 受影响文件：README.md、MASTER_AGENT_PROMPT.md、docs/01_SCIENTIFIC_BLUEPRINT.md、docs/02_METHOD_SPEC_AMR.md、docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md、docs/05_AGENT_EXECUTION_MANUAL.md、configs/project.yaml、configs/experiment_matrix.yaml、PROJECT_PACKAGE_CONSOLIDATED.md、QA.md、STATUS.md、CLAIM_LEDGER.md、reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md。

## D-20260901-P3-009：T5 support localization 的实际边界
- 决定：补做不读取 response 的 nodewise Jacobian-sensitivity top-k localization；同时保留早期 support-mass sanity 表但不把它作为科学结果。
- 结果：6 个 switch checkpoint、53,700 个 complete arms、60 个 match-group rows；relational pairwise mean top-k recall 0.1546，team mean 0.1487，uniform support-size baseline 0.1458；q-full 重算最大绝对误差 `8.9e-9`。
- 解释边界：定位增益接近 uniform，不能支撑有用 localization task 或 geometry-task repair；P3-G5 仍不支持，P4 继续 blocked。
- 证据：`artifacts/phase3/selected_causal_switch_v1/node_sensitivity_localization_v2_summary.parquet`。

## D-20260901-P3-010：Autoresearch 候选搜索（task-geometry Pareto）

- 日期：2026-09-01
- 决定：执行 `AUTORESEARCH_P3_GOAL_PROMPT.md` 的单轴小改动搜索，复用 train 113/dev 37 分组内验证，heldout 32 仅冻结后审计，不用于排序；候选 `team_mean` / `relational_pairwise` / `centered_team_mean` / `centered_relational` / `team_centered`，seeds 11,23,47，matched capacity 149639，80 epochs Adam 1e-3。
- 结果：lexicographic 排序首位 `centered_team_mean` dev 0.3702>0.3392 且 context~0，但 geom 0.728<0.738 且 heldout 0.139<0.186；单轴 `relational_pairwise` context 上升；`centered_relational` 虽满足三阈值（dev 0.352、context~0、geom 0.746、heldout 0.189）但为 2 轴组合且 dev/heldout 均由单 seed 驱动（seed 11 dev +0.052、seed 47 heldout 0.300），不满足“不能只由单个 seed 驱动”与“一次一轴”门槛。
- 判定：`表示塑形成功，但 task repair 未支持。 / NOT SUPPORTED`，保留 `team_mean` incumbent，不扩大矩阵、不新增数据/模型动物园、不改 P2 冻结资产、不启动 P4 AMR。
- 证据：`artifacts/phase3/candidate_search_v1/`（`ranking.csv`、`task_*.parquet`、`context_*.parquet`、`comparison_vs_team_mean.json`、`manifest.json`、`SHA256SUMS`）、`reports/P3_CANDIDATE_SEARCH_REPORT.md`、`scripts/p3_candidate_search.py` sha256 `102d7afe8`。

## D-20260902-P3-011：旧 heldout 暴露降级

- 日期：2026-09-02
- 决定：旧 candidate search 虽未将 heldout 显式放入排序公式，但在候选循环中计算、打印和保存了 heldout；其状态固定为 `EXPOSED_DURING_CANDIDATE_SEARCH`。
- 允许用途：`exploratory_audit_only`。
- 禁止用途：final confirmation、candidate selection、超参数/layer/epsilon/architecture/support 选择和早停。
- 影响：SNGAR test、IDSSE external-confirmatory 和 SoccerTrack final matches 必须在 candidate lock 后再读取。

## D-20260902-P3-012：拆分 context 与 intrinsic 任务语义

- 日期：2026-09-02
- 决定：context task 允许使用 absolute deployment/field position；intrinsic task 以 centered input 为主，并要求 global translation robustness；两者不再共用单一 global-response gate。
- 原因：旧 T5 同时要求模型记住绝对部署位置且对绝对位置变化不敏感，形成语义冲突。
- 影响：新增 fixed dual-channel sanity；旧 task gate 结论保留为 `NOT_SUPPORTED_UNDER_OLD_CONFLATED_TASK_GATE`。

## D-20260902-P3-013：数据扩展与冻结 split

- 日期：2026-09-02
- 决定：SNGAR train/valid 作为开发数据，SNGAR test 作为 candidate-lock 后一次性同源确认；IDSSE 作为外部 provider/league confirmation；SkillCorner 仅作历史/dynamic-support 规则开发；SoccerTrack 先做 parser smoke。
- 原因：扩大独立比赛并恢复真正未暴露的确认集，不能以同一比赛更多帧代替独立样本。
- 影响：数据访问状态按 `configs/dataset_acquisition_manifest_v2.yaml` 和 `configs/data_manifest.yaml` 记录，不把 gated 数据写成已下载。

## D-20260902-P3-014：P3-T5R 有界优化

- 日期：2026-09-02
- 决定：在既有 `P3_CAUSAL_MECHANISM` 内授权 `P3-T5R0`–`P3-T5R6`；最多两轮 autoresearch，每轮最多六个候选，一次只改一个主要机制轴；P4 继续 blocked。
- 原因：先修正 task semantics、split 和 test firewall，再判断固定双通道是否有可迁移的任务—鲁棒性 Pareto。
- 影响：不新增 Phase，不扩大完整 causal matrix，不先实现 AMR。

## D-20260902-P3-015：动态 support 状态

## D-20260902-P3-016: SNGAR access blocked
- Date: 2026-09-02
- Decision: T5R1 only ran a train/valid dry-run; the local SNGAR directory is absent, normal access failed on DNS, and the elevated probe failed with `Network is unreachable`. Status is `BLOCKED_EXTERNAL_ACCESS`.
- Impact: do not run canonical conversion, task construction, or fixed dual-channel; do not treat the existing SkillCorner historical asset as SNGAR continuous tracking. SNGAR test, IDSSE, and SoccerTrack remain behind the existing firewall.
- Recovery: provide gated Hugging Face access or a raw-data mirror with source revision, license snapshot, raw-to-canonical mapping, and SHA-256.

- 日期：2026-09-02
- 决定：static role universal effect 仍为 `NOT_SUPPORTED`；dynamic semantic support 作为 response-blind、可人工审计的新 operationalization，属于非阻塞子任务。
- 影响：dynamic support 不能反向选择主 task-repair 候选；规则、匹配和人工审计必须先于 response 计算冻结。

## D-20260904-P3-017：IDSSE 暂代 SNGAR 的开发数据角色
- 日期：2026-09-04
- 决定：由于 SNGAR gated access 当前不可用，暂时使用本地 `data/raw/sports/idsse-data/` 的 IDSSE 数据推进 T5R 的开发链；替代范围为 T5R1–T5R4 的数据转换、任务构造、baseline lock、fixed dual-channel sanity 和有界候选开发。
- 数据边界：IDSSE 只有 7 场，不能伪装成 SNGAR 的 45 train / 9 valid / 10 test；必须按 `source_match_id` 划分，训练前冻结开发、验证和保留 match。
- 确认边界：同一批 IDSSE 一旦用于本轮开发，不得再次被称为同一轮的独立 external confirmation。SNGAR 恢复或其他 provider/source 仍是独立确认分支；IDSSE 保留 match 最多只能称为同源未见确认。
- 不变项：P2/P3 冻结资产只读；不新增 Phase；旧 heldout 仍仅 `exploratory_audit_only`；P4 AMR、书法和 candidate-lock 后的确认门继续保留。

## D-20260904-P3-018：IDSSE 官方页面核验与文件身份裁决
- 日期：2026-09-04
- 核验依据：官方数据集页 `https://huggingface.co/datasets/pysport/idsse-data` 及其 `main` 文件树；当前文件提交为 `a715a38dfbaf5f58e431727c2b78d174101a703c`。
- 结果：官方页面确认 7 场 2022/23 Bundesliga 数据、tracking/event/metadata、25 fps、CC BY 4.0、Figshare DOI 和论文 DOI；`main` 文件树列出的 7 个实际比赛 ID 与本地 XML 文件名及内部 `MatchId` 一致。
- 差异处理：数据卡正文列出的 `J03WPF`、`J03WQF` 与 `main` 文件树中的 `J03WQQ`、`J03WR9` 不一致。文件身份以官方 `main` 文件树为准；该差异写入 provenance，禁止修改或重命名 raw 文件来消除差异。
- 限制：官方 Dataset Viewer 当前对 `default/train` 报 `FeaturesError`/`FileNotFoundError`，因此字段级 schema、样本统计和 XML 完整性不能由 Viewer 证明，仍需本地 receipt、SHA-256 和 canonical QC。
- 影响：T5R1 的“官方来源核验”已完成；本地 provenance/manifest/SHA-256 和 canonical conversion 仍未完成，T5R2 及后续训练继续不启动。
- 复查条件：IDSSE 官方数据说明、文件/比赛 ID、来源 revision、许可和 SHA-256 完成登记并通过 canonical QC 后，才可进入 T5R2；若 schema 或覆盖不足，停止并记录失败，不用结果反向放宽协议。

## D-20260904-P3-019：T5R2 IDSSE task 与 baseline/metric lock 闭合
- 日期：2026-09-04
- 数据来源：IDSSE 官方 Hugging Face revision `a715a38dfbaf5f58e431727c2b78d174101a703c`；raw manifest、`RAW_SHA256SUMS`、canonical manifest 和 7 场 conversion receipts 已生成，原始文件哈希复核 23/23 通过。
- 固定 split：train=`J03WMX,J03WOH,J03WPY,J03WR9`；valid=`J03WN1,J03WOY`；reserved holdout=`J03WQQ`。只按 source match 划分，禁止 frame/event 随机切分；reserved match 在 T5R2 未加载，candidate lock 前不得读取。
- 固定任务：raw positions → `z_ctx` 用于 phase/field-zone/absolute-centroid context；20 个出场球员的 globally centered positions → `z_mode` 用于 natural pair ranking。自然配对为同 match、同 half、至少 4 秒分离；正样本 geometry≤0.28 且 centroid≥0.35，hard negative centroid≤0.12 且 geometry≥0.35。
- 规模：train 23,052 snapshots/3,750 pairs；valid 6,127 snapshots/980 pairs。配对阈值只由响应盲的几何协议固定，不使用模型结果；`J03WQQ` 无任务数组输出。
- baseline/metric lock：raw single-channel、centered single-channel、raw relational、fixed dual-channel（仅 T5R3 sanity）和 raw-coordinate/Procrustes；seeds 11/23/47；match-level bootstrap 2,000 次，seed `20260904`；context non-inferiority margin 为 macro-F1 最多下降 0.05、centroid MAE 最多增加 0.05。
- 状态边界：T5R2 已闭合，但没有运行 fixed dual-channel、autoresearch、candidate lock、reserved-match 检查、独立 provider confirmation 或任何模型训练；P4 仍 blocked。IDSSE 不能在本轮同时被写成 independent external confirmation。

## D-20260904-P3-020：T5R3 fixed dual-channel sanity 的方向性门判断
- 日期：2026-09-04
- 运行：`artifacts/phase3/task_semantic_repair_v1/t5r3_sanity_v3/`；仅 train/valid，4 个 neural variants × seeds `11/23/47`，所有 neural variants 均为 141,769 参数；`J03WQQ` 未加载。
- 结果：fixed dual valid natural-pair ranking accuracy 均值 `0.9487`，raw single-channel 为 `0.8030`；dual 相对 raw 的 phase macro-F1 变化 `-0.0126`、field-zone macro-F1 变化 `-0.0087`、centroid MAE 变化 `+0.0095`，均在 `0.05` non-inferiority margin 内；三个 seed 的 dual intrinsic ranking 均高于 raw。
- 机制边界：dual 的 `z_mode` global translation response 约为 `0`，cross-readout 受限，natural-pair geometry→latent-distance proxy 为正且三个 seed 同方向。该 proxy 不是旧 P3 intervention-response Spearman；不能写成旧机制在 IDSSE 上的复现。
- 决定：T5R3 记为 `DIRECTIONAL_PASS_T5R4_REVIEW_REQUIRED`，允许审阅后考虑 T5R4 bounded autoresearch；不得据此创建 candidate lock、读取 reserved/test、放行 P4 或宣称最终方法收益。v1/v2 中止尝试不纳入证据。

## D-20260904-P3-021：T5R3 closure review 与 T5R4 Round 1 启动
- 日期：2026-09-04
- 判断：没有路线级偏移；IDSSE 暂代 SNGAR、train/valid firewall 和 T5R3 的 proxy/等价实现边界均保留。
- 决定：允许 T5R4 Round 1 bounded autoresearch。搜索保持轻量和可探索：最多 2 轮、每轮最多 6 个候选、每候选一个主要机制轴；具体候选值不预先写死，不自动晋级。
- 当前范围：Round 1 仅使用 T5R2 train/valid，候选参考 T5R3 fixed dual；不读取 `J03WQQ`、SNGAR test、旧 exposed heldout 或外部结果。T5R4 结果仍不构成 P4 或独立确认。

## D-20260904-P3-022：T5R4 Round 1 结果边界
- 日期：2026-09-04
- 运行：4 个候选 × seeds `11/23/47`，共 12 个模型；所有模型 141,769 参数，正式输出为 `artifacts/phase3/task_semantic_repair_v1/t5r4_round1_v1/`。
- 结果：没有全维度 Pareto 支配者。`loss_context_up` 与 `head_context_layernorm` 各有局部改善，`routing_mode_head_only` 的 intrinsic ranking 明显下降；Procrustes control 仍显示 ranking 接近几何上限。
- 决定：Round 1 记为 `COMPLETE_REVIEW_REQUIRED`，不自动进入 Round 2，不创建 candidate lock，不读取 `J03WQQ`，不放行 P4。Round 2 是否进行由结果审阅决定，探索轴和具体值保持开放。

## D-20260904-P3-023：T5R4 Round 1 审阅、语义漂移修复与 Round 2 授权（最后一轮）
- 日期：2026-09-04
- 判断：Round 1 无全维度 Pareto winner；`routing_mode_head_only` 构成“intrinsic objective 必须塑造 encoder”的负机制证据；`head_context_layernorm` 实为 post-pooling readout-normalization control；`1.5× loss` 受 Adam 与约 361:59 update imbalance 影响，不能解释为精确机制量。
- 修复：新增 `artifacts/phase3/task_semantic_repair_v1/metric_semantics_addendum.json`（match-half context probe、intrinsic structural accessibility proxy、geometry 非干预 alias、readout-norm control、loss-scale 说明）；冻结 lock 内容/hash 不变，历史报告不重写。repair config 预填的 `round_2_axes: [pooling, constraint_placement]` 被本授权覆盖。
- 决定：授权 Round 2 为最后一轮，只检验 `shared_encoder_task_update_balance`（3:1=315/105、2:1=280/140，总步数 420 持平 reference），判断规则训练前冻结于 `configs/t5r4_round2.yaml` 与本授权报告。`J03WQQ`/SNGAR/旧 heldout/外部结果均不读取，不创建 candidate lock。
- 冲突记录：Round 1 protocol 的 `config_sha256=f27643d5…` 与提交的 `configs/t5r4_round1.yaml`（`ffd9d9cf…`）不一致（运行后提交前被编辑）；冻结产物未动，不重写历史。

## D-20260904-P3-024：T5R4 Round 2 结果与 bounded autoresearch 关闭
- 日期：2026-09-04
- 运行：`artifacts/phase3/task_semantic_repair_v1/t5r4_round2_v1/`；2 候选 × seeds `11/23/47`，6 模型，141,769 参数；`T5R4_ROUND2_AUDIT: PASS`。
- 结果：`update_ratio_2to1` 通过全部资格 gate（pair `0.9511`≥floor、`0.7034` geometry、3/3 seeds 双 floor、`z_mode`~1e-8、context 非劣效），且满足冻结的强推荐条件 A（geometry 3 seeds 同方向 +0.050/+0.010/+0.021，pair/context 保持）与条件 B 支持（centroid 3 seeds 同方向改善 0.0554→0.0376）；`update_ratio_3to1` 通过资格 gate 但 geometry 在 seed23 跌破 floor，主要故事为 ceiling 附近 ranking 增益，不晋级。
- 决定：`T5R4_selected_candidate=update_ratio_2to1`；bounded autoresearch 关闭，不做 Round 3，不扫描 ratio，不微调 2:1。P3-R7 记为 `SUPPORTED_CONDITIONALLY`（边界 = IDSSE 开发集/固定结构/2 场 valid），P3-R3 仍 `NOT_ESTABLISHED`，P4 继续 blocked。下一步 `T5R5_candidate_lock_then_single_reserved_read`（本轮不执行 T5R5）。

## D-20260905-P3-025：T5R5-0 闭包审计（lock 前）
- 日期：2026-09-05
- 范围：只读已可见 train/valid 产物；`J03WQQ` 未打开；无重训练；无协议改动。
- 结果：逐场重算 fixed dual 与入选 2:1（3 seeds × J03WN1/J03WOY）：pooled Spearman 几乎完全由 J03WOY（951/980 pairs）决定；match-macro 方向一致；J03WN1 仅 29 pairs，逐场值处噪声带（标不稳定），不构成反向证据。2:1 在主导场 3 seeds 几何均高于同 seed reference。
- 文档清理：STATUS 残留 Round1 待审阅句、task-geometry 行新旧分隔、phase/deployment 显示别名、P3-G1–G4 标历史假设；核查无 head-placement 误写、无 1.5× 精确影响误写。
- 结论：收紧 claim 边界，不重开 T5R4 选择。证据：`reports/P3_T5R4_TO_T5R5_CLOSURE_AUDIT.md`、`artifacts/phase3/task_semantic_repair_v1/t5r5_prelock_audit_v1/`。

## D-20260905-P3-026：T5R5 candidate lock、干预协议锁与 T5R6 shadow lock
- 日期：2026-09-05
- 锁定：`update_ratio_2to1`（280/140/420）及 9 个对照 checkpoint hash；T5R4 报告/config/protocol hash；数据与 split hash；隐藏任务规则逐字沿用 T5R2（阈值不变）；任务语义 gate（§9.1）与干预机制 gate（§9.2）预冻结；C 路径诊断 fallback（T5R3 fixed dual 仅作 T5R6 诊断参照）已预声明。
- 干预协议：表示映射 f 与 response 同层；fracture 端点重分配（同 multiset、同 support 大小 4、不同端点，response-blind，不用静态 role）；单 epsilon 0.25（归一化半 pitch 单位，与 P2 形式范围最低档精确对应，可见 6 场全 105×68）；最多 250 等距快照；full/diagonal 主预测器＋Rayleigh 简单基线；不训练预测器。
- 状态：`T5R5_LOCK_AUDIT: PASS`；保留 match 只允许读一次；P4 blocked。

## D-20260905-P3-027：T5R5 一次性保留 match 读取结果与路由 A
- 日期：2026-09-05
- 运行：`artifacts/phase3/task_semantic_repair_v1/t5r5_reserved_j03wqq_v1/`；只读一次，无重训练；9 个冻结 checkpoint＋解析对照；250 采样快照→159 complete＋91 incomplete（70 anchor 越界、21 无合法 target，均计失败率）；`T5R5_HIDDEN_AUDIT: PASS`。
- 结果：任务 gate 全过（`z_mode`≈0；zone/centroid 相对隐藏 raw 基线无退化；pair −0.0026；geometry +0.0051 且 2/3 seeds 非负；515 pairs 充分）；干预强通过（2:1 mean full 0.63≥0.5、mean 差 0.66≥0.2、3/3 超基线、方向 0.76、无坍缩）；fixed dual 家族机制同样复现（次要比较，2:1 三 seeds 占优）。
- 决定：verdict `T5R5_PASS_STRONG`，路由 A（`T5R6_EXTERNAL_PROVIDER_CONFIRMATION`）。P3-R2→单 match 条件支持，P3-R7→未见 match 条件支持，新增 P3-R8（单未见 match 干预预测），P3-R3 仍未建立，P4 blocked。不做新搜索。
