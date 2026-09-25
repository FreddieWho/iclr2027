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

## D-20260905-T5R6-001：T5R6 用 SoccerTrack-v2 做独立外部确认
- 日期：2026-09-05
- 决定：路由 A 指定的独立确认用 SoccerTrack-v2（`atomscott/soccertrack-v2`，CC BY 4.0，10 场中排除 smoke 场 117092/117093，用其余 8 场）；坐标系为 105×68 中心原点米制，与 IDSSE 归一化一致，无需物理转换。
- 原因：SNGAR test 仍不可访问；SoccerTrack 是已登记的独立 provider 恢复分支；shadow lock 已预冻结 high-level 规则。
- 复查触发：若 8 场中可用场次不足（如大面积缺帧/队伍不满员），或数据条款变化，则停下报告，不降标准硬做。
- 状态：gated 数据集需用户授权（HF_TOKEN 且账号已接受数据条款）后才能下载数据文件；已建立 PLAN/ROADMAP/LEADS，N1 阻塞中。

## D-20260905-T5R6-002：T5R6 外部确认完成，verdict CONFIRMED
- 日期：2026-09-05
- 事实：用户授权 HF 后下载 8 场 GSR（45.16GB，修订 eae51793，SHA 登记）；转换 44,401 快照/7,534 对；冻结 9 checkpoints 无训练推理；干预与 T5R5 同算子（eps 0.25、support 4、150 场/场）。
- 结果：任务 macro（2:1 zone 0.9653/pair 0.9219/geom 0.5738/cent 0.0365，逐场 8/8 占优）；干预 8/8 full 高于基线、方向 0.66–0.73；审计与 verdict 复算通过。
- 决定：`T5R6_CONFIRMED`；P3-R3 升级为独立 provider 条件支持；P3-R5 门条件满足，是否启动 P4 由事项 8 决定，不自动放行。
- 复查触发：事项 8 若决定启动 P4，需另行冻结 AMR 目标与门槛；若冻结为诊断版本，本分支即封存。

## D-20260905-P4-001：用户裁决 AMR 走完整流程（选项 C）＋配比扫掠授权
- 日期：2026-09-05
- 事实：P3-R5 门已闭合（T5R6 CONFIRMED）；评审给出 A（本期不做）/B（dev-gated PoC）/C（P4 全周期）三选项。
- 用户决定：① AMR 走完整流程（选项 C：M1 实现→锁→训练→对照→候选锁→确认读取）；② 授权配比扫掠（1:0、1:1 补格点）作为 dev-only 事后消融（选择仍冻结为 2:1，不触碰任何已消耗读取）；③ 论文 brief 暂缓评审。
- 配套约束：P4 的保留场/外部二次读取属于新协议事件，到达 P4-N5/N6 时需用户显式授权；在此之前全部工作限 train/valid dev 集。
- 复查触发：M1 未达 spec §13 判据时收缩并不启动 M2；配比扫掠若显示 2:1 非单调异常点，仅作 characterization 报告，不重开选择。

## D-20260905-P4-002：AMR M1 固定路由分配与训练协议冻结（P4-N2）
- 日期：2026-09-05
- 冻结内容（`artifacts/phase4_amr/m1_config_lock.json`，锁哈希 a76ef1b3…，审计 PASS）：
  ① 路由 α=[1,0,0,0,0,0]——band 0（含近全局/DC 模式）对 intrinsic 通道取不变目标
  （依据：整体平移是已确认 context-nuisance，T5R3 构造保证＋T5R5 raw 对照 0.256）；
  bands 1–5 取可恢复目标（依据：P2 诚实负结果 20/36 vs 16/36，禁止把中频预设为修复对象）；
  ② 支持维度经"支持条件白化采样"进入（ξ 先掩码到同队随机子集再滤波），不设学习门；
  ③ 损失权重 task/ctx/route/var/orth = 1.0/1.0/1.0/0.1/0.01；
  ④ 更新预算 210 ctx / 140 pair / 70 route = 420/epoch（与冻结配方等计算量）；seeds 11/23/47。
- 当时理由：M1 的最小可证伪形态；路由分配必须在任何 AMR 结果之前写入配置（spec §8）。
- 复查触发：M1 dev 出现任一系统性反转 → 收缩 claim 并停下（不启动 M2）；
  若 band-0 不变性无收益而某 coupling 频带可恢复性崩坏，回到证据层重估而非改锁。

## D-20260905-P4-003：M1 v1 塌缩失败处置与 v2 锁附录
- 日期：2026-09-05
- 事实：M1 v1（锁 v1）dev 3/3 seeds 模态通道近完全塌缩（pair=0、margin=0、干预退化）；根因=路由损失尺度碾压＋塌缩同时满足未归一化的 L_inv/L_eqv＋防塌缩权重不足＋三元组塌缩不动点。
- 决定：① v1 记为机制负结果（报告 P4_N3_V1_COLLAPSE_REPORT.md），不构成对 H1/H2 的判决；② 锁版本化附录 v2（哈希 65ee8599…）：仅改 L_eqv 为归一化方向恢复＋W_var 0.1→1.0，α/预算/seeds/评估/判据全部不动；③ 审计脚本"无训练记录先于锁"检查精化为"除已关闭的 m1_v1 外无记录"。
- 当时理由：dev 期迭代是锁体系的合法用途；附录在 v2 训练前写入，遵守"先冻结后训练"。
- 复查触发：v2 若再塌缩或出现新系统性反转 → 停下重估 M1 设计本身（不再局部调参）；v2 若通过 dev → 进入 N4 matched-capacity 对照。

## D-20260905-P4-004：M1 v3 设计重估——路由改为全频带可恢复
- 日期：2026-09-05
- 触发：v2 系统性失败（3/3 seeds）：seeds 23/47 模态通道再塌缩；seed 11 总塌缩（共享主干被毁，context 通道陪葬，z_ctx 平移响应 1.0）。
- 重估结论：① band-0 不变性冗余（居中视图已构造保证平移不变，原证据依据误配）；② 共享主干上的不变性压力是破坏性的（v1 毁模态、v2 毁主干）；③ "频带 × 支持"存在图不确定性张力（记为设计风险，v3 不修）。
- 决定：锁附录 v3（哈希 aefb86cb…）：α=[0,0,0,0,0,0] 全频带可恢复（归一化方向读出）；单变量变更隔离"不变性压力"这一破坏因子；其余全部不动。审计精化并通过。
- 停止规则：v3 若仍系统性失败 → 停止 M1 迭代，三轮证据报用户裁决 M1 存废。

## D-20260905-P4-005：v3 若失败则转 research 驱动（用户指令，提前并行启动）
- 日期：2026-09-05
- 用户指令：v3 若仍不理想，不再自行猜测，启动并行 research subagent 探查近似问题，综合出最有希望的 3 条路线后自主尝试。
- 执行决定：不等 v3 判据，立即并行启动三条 researcher 轨道（workflow 9f5cfea；与 v3 训练零干扰；结论对论文亦有用）：
  A. 小图谱带近似质量（Chebyshev R=5 在 20 节点图上是否合理 vs 精确谱分解；离散谱与连续带窗的泄漏；替代构造排名）；
  B. 不变性＋判别联合训练的塌缩机制（SSL 塌缩文献、共享主干梯度冲突证据、与"路由塑造表征"目标相容的修复排名）；
  C. 图不确定性/频带×支持张力（Slepian/局部谱帧文献；mask-then-filter vs filter-then-mask；成功与失败先例）。
- 后续：research 简报＋v3 判据双就绪后，综合 top-3 路线，以锁附录形式预注册后在 dev 上自主尝试（仍受 P4-N5/N6 二次读取授权约束；v3 若通过，research 转入 N4/论文支撑）。

## D-20260905-P4-006：Research 综合 top-3 路线（v3 失败时的尝试顺序）
- 日期：2026-09-05
- 来源：三条 researcher 轨道简报（reports/research_p4/）＋综合 SYNTHESIS_TOP3.md。
- 跨轨道收敛：① n=20 精确 eigh 在各方面占优，Chebyshev 应删除；② 6 等宽频带过 resolving 离散谱，应 2–3 密度均衡带；③ 塌缩=complete collapse 类，有因果验证解（SimSiam/VICReg/GradNorm）；④ Shannon 数 K≈0.3–0.8：中/高频小支持干预本质非局域，无论 v3 成败进论文 Limitations。
- 尝试顺序（v3 失败才启动，每条锁附录预注册、dev-only）：路线 1 精确谱基础（eigh＋密度均衡带＋Slepian 采样＋μ 报告）→ 路线 2 最终嵌入防护（VICReg 最终层＋stop-grad 非对称＋mining 守卫）→ 路线 3 梯度尺度平衡（GradNorm＋warm-start＋PCGrad 备用）。
- v3 预读：v3 仍保留中间层 floor 旁路与对称梯度两个塌缩通道；存活则功劳归可恢复性惩罚塌缩。

## D-20260905-P4-007：v3 失败确认，停止自主迭代，启动 research 驱动重设计
- 日期：2026-09-05
- 事实：v3（α 全零、无任何不变性压力）仍同签名塌缩 3/3（P4_N3_V3_FAILURE_AND_STOP.md）。
- 隔离结论：不变性压力不是必要破坏因子（证伪 v2→v3 假设）；残存驱动锁定 Track B 三机制（对称路由梯度、中间层 floor 旁路、三元组不动点）。
- 决定：① 触发停止规则，盲猜式 M1 迭代终结（三轮总账见报告）；② 按用户既有授权（D-20260905-P4-005"自主尝试"）启动 research 驱动重设计 v4：路线 1（精确谱＋密度均衡带＋Slepian 采样＋μ）与路线 2（最终嵌入 VICReg＋stop-grad 非对称＋mining 守卫）合并实施——路线 1 单独不触塌缩机制，单独尝试预期仍塌，故合并为一次重设计（偏离单变量原则，但属新设计而非调参，如实记录）；路线 3（GradNorm）作备用。
- 复查触发：v4 若仍塌缩 → M1 方法存废交用户裁决，不再开新设计。

## D-20260905-P4-008：v4 重设计冻结并开训（路线 1＋2 合并）
- 日期：2026-09-15（锁时间戳；P4 决策链属 09-05）
- 新设计锁 `m1_v4_config_lock.json`（哈希 37cbb0db…，审计 PASS），相对 v3 的变更全部有测量依据：
  ① 精确 eigh 谱投影替代 Chebyshev（Track A＋C；旧 band5 在 99% snapshot 上为空）；
  ② 3 密度均衡频带 [0,1.0)/[1.0,1.3)/[1.3,2.0]（2000-snapshot 经验谱三等分）；
  ③ Slepian 向量干预采样＋逐 update μ 报告（Track C）；
  ④ VICReg 方差 hinge＋within-mode 协方差移到最终 64 维嵌入（Track B；跨通道惩罚删除）；
  ⑤ stop-grad 非对称路由分支（Track B/SimSiam）；
  ⑥ mean＋energy 并池化（实测 bands1-2 均值池化相消 10×）；
  ⑦ mode_head 去 bias＋LayerNorm(affine=False)（实测 bias 主导 1e-2 信号致 init 死亡）；
  ⑧ W_ROUTE=12.0（init 主干梯度范数 parity 实测 11.98）、VIC_GAMMA=0.02（实测校准）；
  ⑨ active-triplet 0×3 early-stop＋NaN guard（预注册）。
- 冻结保持：seeds/80 epochs/210-140-70 预算/dev-only/冻结 pair 与 context 损失/冻结干预评估＋新增奇异谱与 μ 诊断。
- mini 2-epoch 全链路 smoke 通过（ctx 下降、active-triplet=1.0、eqv 下降）；3 worker 已发射。

## D-20260905-P4-009：v4b Route-3 备用启动（最后一次有界尝试）
- 日期：2026-09-15
- 触发：v4 H1 未通过（P4-A4）：模态 3/3 存活但 12× 路由吃掉 context（2/3 塌缩）。
- 决定：按 D-20260905-P4-007 预注册备用条款开 v4b（锁附录 v4b，哈希 37c52427…，审计 PASS）：
  W_ROUTE 12→2、1–20 epoch route-free warm-start、21–40 ramp、之后全 2.0；
  模型/频带/采样/VICReg/守卫/评估全冻结的新 Runner 调度。
- mini 调度 smoke 通过（warm-start 期 ctx 1.59→0.29、pair→0.003，ramp 期路由扰动三元组后 active 回 0.41）。
- 终局规则：v4b 若未通过 H1 → M1 方法存废交用户裁决，不再开任何尝试。

## D-20260905-P4-010：v4b 终局——M1 存废交用户裁决（terminal）
- 日期：2026-09-15
- 事实：v4b H1 未通过（context 2/3 总塌缩＋1/3 侵蚀，P4_TERMINAL_V4B.md）；
  隔离结论：Slepian 可恢复性目标在共享主干上与 context 保存根本不相容
  （权重 12→2、warm-start 均未能修复）。
- 总账：15 次训练/5 配置/模式完全一致——任何路由压力杀死至少一通道；
  唯一健康阶段是 route-free warm-start。
- agent 建议（仅建议）：选项 A，杀死 M1，负结果进论文，AMR 回 outlook。
  选项 B（解耦主干）可能 work 但让渡核心 claim，需用户授权。
- 状态：P4 冻结，不再开任何尝试，等用户裁决（事项 8 的延续）。

## D-20260905-P4-011：用户新批 5 轮 AMR 预算（覆盖 terminal 停止规则）
- 日期：2026-09-15
- 用户指令：再给 5 轮预算实现 AMR，鼓励多用 research。
- 轮次规划（每轮 = 1 配置 × 3 seeds，H1 通过即提前停止以省预算）：
  R1（v5）：软共享架构（cross-stitch/sluice 式双主干，学共享模式而非硬共享），
  保留 v4 全部有效机器（精确谱、Slepian、VICReg-final、stop-grad）；
  R2（v6）：两阶段 M2-lite（冻结健康的 T5R-2:1 主干，只学路由头；测"路由本身是否有价值"，让渡 shaping claim）；
  R3–R5：视 R1/R2 结果而定（胜者精化/消融），由 research 结论定。
- H1 bar 不变：context 非劣（zone≥0.90 且 centroid 健康）＋ intrinsic 信号 ＋ 干预有限。
- 先行 research（D/E/F 三轨并行）：D 软共享解决破坏性干扰的证据；E GradNorm 之后的 MTL 平衡＋两阶段路由先例；F 联合训练成功的正面先例（与我们有何不同）。

## D-20260905-P4-012：R1（v5）全解耦双塔启动（5 轮预算第 1 轮）
- 日期：2026-09-15
- 来源：D/E/F 综合（SYNTHESIS_5ROUNDS.md）。
- 设计（锁 `m1_v5_config_lock.json`，哈希 a60b3092…，审计 PASS）：
  ctx 塔与 mode 塔零共享参数（双向绝缘单测通过）；mode 侧 v4 机器原样；
  故意无 gate（冻结评估接口要求 train/eval 同构；学习共享延至 R3＋）；
  与 v4b 唯一变量差＝编码器共享→分离；schedule/权重/采样/评估全同 v4b；
  参数 288,757（≈2× 主干，诚实下限）。
- mini 调度 smoke：warm-start ctx 1.65→0.10、pair→0.004（健康双塔起点）。
- R1 预注册判定：H1 通过→AMR（分支塑造版）成立，进 N4，剩 3 轮转对照；
  mode 仍塌缩→路由目标本身 broken→R2 freeze-then-route；
  context 异常（构造不可能）→实现 bug，修 bug 不计轮次。

## D-20260905-P4-013：R1 H1 通过，AMR（分支塑造版）成立，剩 4 轮转 N4
- 日期：2026-09-15
- 事实：v5 双通道 3/3 存活（P4_R1_V5_RESULTS.md）：context 冻结配方水平、
  模态信号＋干预 0.81–0.96 全有限；seed-23 pair 弱（0.52）但机制信号最强，
  记 limitation；seed-11 diag 负/full 正复现 P3-R12。
- 决定：① H1 PASS，AMR 分支塑造版成立（P4-B1）；② 与 v4b 的干净隔离证明
  硬共享是 context 杀手；③ 剩余 4 轮预算转 N4：R2＝CAP-双塔对照（须适配，
  现脚本为共享主干版），centering/canonicalization/relational 复用冻结数。

## D-20260905-P4-014：R2（CAP-双塔对照）启动（5 轮预算第 2 轮）
- 日期：2026-09-15
- 设计（锁 `m1_v5cap_config_lock.json`，哈希 753e5d25…，审计 PASS）：
  v5 双塔原样＋受限线性 CAP 头 64→40；与 v5 唯一变量差＝机制目标
  （Slepian 路由→变换预测 M0，忠实无 stop-grad）；schedule/权重/采样/评估全同；
  mini smoke 通过（warm-start 健康，ramp 期 CAP 扰动三元组为预期瞬态）。
- R2 预注册判定（锁内）：CAP≥v5 且 context 健康 ⇒ 路由相对 CAP 无增量
  （AMR 收缩为分支塑造）；CAP 塌缩而 v5 健康 ⇒ 路由 load-bearing（强化 P4-B1）；
  双健康 ⇒ 比幅度＋N4 行文。

## D-20260905-P4-014b：R2 启动 bug 修复（不计轮次）
- R2 首次发射全部 worker 启动即崩（run_one 读锁中不存在的 alpha 键；训练前崩溃，
  零算力浪费、零 record）。修复：route_alpha 直接置零（锁本就无 alpha，by design）。
- 锁附录＋重哈希（a3f9f6bd…）＋审计 PASS 后重发射。此为实现 bug 修复，
  不消耗轮次预算（5 轮用 2 剩 3 不变）。

## D-20260905-P4-015：R2 判定——路由≈CAP，训练阶段完结，银行剩余预算
- 日期：2026-09-15
- 事实：head-to-head 全打平（pair/几何/干预均值差 ≤0.03，逐种子互有胜负；
  context 逐位相同）。路由相对变换预测无增量。
- 决定：① AMR 收缩为分支塑造版（P4-B2）；② 训练问题已回答完毕，
  建议银行剩余 3 轮不再开训练（R3 CAGrad/momentum-teacher 已无可修的共享冲突，
  预期收益低）；③ 转 N4 行文＋论文写作；用户若坚持用完额度再议。

## D-20260905-P4-016：R3（v6 动量教师缝合包）启动（5 轮预算第 3 轮）
- 日期：2026-09-15
- 来源：G/H/I 综合。用户指令"胆子大一点缝合时髦东西"——v6 把三件 15 炉未试之物
  缝进同一个非对称预测故事（JEPA/BYOL/MAE 一家，非乱炖）：
  ① 动量教师＋per-band predictor（G：stop-grad 非 negotiable，predictor 10×LR，
  τ cosine 0.99→1.0 短 schedule 快教师）；② 双 corruption
  （Slepian 位移／MAE 幽灵球员＋可学习 mask token，S4L 旋转预测同构）；
  ③ KoLeo spread（DINOv2 从属权重 0.1）＋VICReg-final 安全网。
- 设计（锁 `m1_v6_config_lock.json`，哈希 4646783e…，审计 PASS）：
  SHARED 主干回归（救援对象）；teacher 仅镜像表征参数；triplet/context 冻结；
  W_ROUTE=1.0（F#3 0.03–0.10 为 R4 fallback）；v4b schedule 原样；7 单测＋mini 全链路通过。
- R3 预注册判定（锁内）：共享主干 H1 通过⇒联合训练救活（最强 AMR claim），
  剩 2 轮做 ablation（predictor 宽度/F-range 权重）＋N4；
  mode 再塌⇒非对称在该尺度不够⇒R4 JGCL 式单几何损失（Track I P1）；
  context 再塌⇒教师压力亦 destructive⇒R4 freeze-then-route v6 变体（Track E P1）。

## D-20260916-GOAL01：10轮自由预算目标立约（覆盖P4旧预算框架）
- 日期：2026-09-16（目标 mu32v6c0-rywecg）
- 用户授权：全新10轮dev-only训练预算（旧5轮框架关闭；v6不占额度但其verdict仍参与决策）；
  目的=新颖性/科学性/趣味性＋摘要硬数字，两者兼顾；
  本地≤48核并行（保留16核）；GPU仅在CPU不可替代时经aigalaxy租用，总额≤¥50；
  dev-only铁律维持（保留场/外部绝不再碰）。
- H1 bar沿用：context非劣（zone≥0.90＋centroid健康）＋intrinsic信号＋干预有限；
  每轮锁先行＋审计＋3种子＋报告＋ledger行；H1通过即停以省预算。
- 台账文件：`artifacts/phase4_amr/BUDGET_10R.md`（每轮锁哈希＋状态＋花费）。

## D-20260916-GOAL02：v6 verdict（H1 通过）＋第 1 训练轮内容锁定
- 日期：2026-09-16
- 事实：v6 共享主干 H1 通过 3/3（P4_R3_V6_RESULTS.md，P4-C1）：15连败终结；
  seed-47 pair/干预分裂（机制边界）；margin仍薄。
- 决定：动用新预算第 1 轮＝mask 消融（slepian-only 教师，其余冻结）：
  若通过⇒幽灵球员非必要（故事简化）；若退化⇒幽灵球员 load-bearing
  （新颖性 claim＋摘要级"幽灵球员"故事）。predictor加宽/F-range权重降级为
  后备（v6 已健康，diagnostic 动机消失）。

## D-20260916-GOAL03：第 1 训练轮发射（mask 消融，台账 1/10）
- 日期：2026-09-16
- 设计：v6 runner + `AMR_CORRUPTIONS=slepian`（mask 分支零触发，mini 已验证
  mask_trials=0）；其余冻结（锁 `m1_v6nomask_config_lock.json` 462be450…，
  审计 PASS 含输出目录隔离检查）。
- 判定（锁内 R1_decision）：H1 通过⇒幽灵球员非必要；退化⇒幽灵球员 load-bearing。

## D-20260916-GOAL04：R1 判据（H1 通过）＋第 2 轮锁定 JGCL 式单几何损失
- 日期：2026-09-16
- 事实：R1 slepian-only 教师 H1 通过 3/3（GOAL_R1_NOMASK_RESULTS.md，GOAL-R1）；
  一致性反优于 v6 全 bundle；幽灵球员非必要。
- 决定：第 2 轮＝JGCL 式单几何 InfoNCE（新损失本体，消灭 tug-of-war）：
  context 损失保留＋InfoNCE（自然对＋Slepian 视图正例，batch 内负例，τ=0.1），
  warm-start 20＋ramp 20沿用；teacher/predictor/VICReg/KoLeo/triplet 全拆
  （v6 类复用，死头注明）。F-range 权重、CAGrad、freeze-route 降级为后备。

## D-20260916-GOAL05：第 2 训练轮发射（JGCL 单几何，台账 2/10）
- 日期：2026-09-16
- 设计（锁 `m1_jgcl_config_lock.json` feeb99e4…，审计 PASS）：
  v6 架构逐位复用（哈希与 v6 锁相等已验证）；triplet/teacher/predictor/VICReg/
  KoLeo/mask 全拆；单 InfoNCE（自然对＋Slepian 视图正例，batch 内 189 负例，
  τ=0.1）；210 ctx＋210 InfoNCE＝420 等算力；warm-start 20＋ramp 20；
  emb-std 塌缩守卫＋NaN 守卫。
- mini smoke：warm-start ctx 1.82→0.85，ramp 期 InfoNCE 2.86（chance 4.56，有信号）。
- 判定（锁内 R2_decision）：margin 锐于 triplet 系＋H1 保持⇒大数字希望；
  H1 失败⇒单几何不够⇒R3 freeze-then-route；塌缩⇒负例需规模⇒R3 教师＋InfoNCE 混合。

## D-20260916-GOAL06：R2 判据＋建议银行剩余 8 轮转论文
- 日期：2026-09-16
- 事实：JGCL 单几何 H1 通过但全面弱于教师-triplet（GOAL_R2_JGCL_RESULTS.md，
  GOAL-R2）：pair −0.2、margin −0.11、几何代理 −0.3；大数字分支未触发。
- 判断：连续两轮回答"什么不重要"（R1 mask 非必要；R2 单几何更弱），
  训练搜索出现收益递减拐点；剩余候选（CAGrad/momentum 变体/长 schedule）全为
  低预期收益（无可修的共享冲突、无未收敛项、无新机制）。
- 决定（待用户确认）：银行剩余 8 轮，目标转 task-final（N4 行文＋摘要数字＋
  可复现性包）；task-r35/r610 若用户同意则关闭或转为写作任务。

## D-20260916-GOAL06R：R2 判据落盘（H1 通过但弱）＋重申银行建议
- 日期：2026-09-16
- 事实：JGCL 单几何 H1 通过但全面弱于教师-triplet（GOAL_R2_JGCL_RESULTS.md，
  GOAL-R2）：pair −0.2、margin −0.11、几何代理 −0.3；大数字分支未触发。
- 判断：连续两轮回答"什么不重要"（R1 mask 非必要；R2 单几何更弱），
  训练搜索边际回报归零；剩余候选（CAGrad/momentum 变体/长 schedule）预期收益全低。
- 决定（待用户确认，与 D-20260916-GOAL06 合并）：银行剩余 8 轮，转 task-final
  （N4 行文＋摘要数字＋可复现性包）；task-r35/r610 若用户同意则关闭或转为写作任务。

## D-20260916-GOAL07：B30头脑风暴治理冻结（目标 mu3wl782-yxuns1）
- 日期：2026-09-16
- 轮单位：1轮=1份含检索复核的subagent简报；总额30，分三波（W1机制/W2架构/W3故事wild各10）
- 只读边界：禁新训练、禁改代码、禁碰保留场/外部；允许零成本重算已有artifact（注明输入哈希）
- 训练侧并行：BUDGET_10R独立记账（剩8轮）；明显方向不等list可先开（需点名或预注册判据）
- 台账：reports/research_b30/LEDGER.md；角度地图ANGLE_MAP.md（旧13份简报禁区已标出）

## D-20260918-AUDIT01：第三方审计闭合——U02符号bug/U03b nuisance/X03措辞
- 日期：2026-09-18
- 事实：审计在 main@c1837e0 发现 u02_decision.py 校准拟合用 sigmoid(+lg/T) 拟合 feas(yy==0)，
  而 P0=sigmoid(−lg)，符号反；u03b_style.py 同qi三状态复用同一RNG致背景nuisance不一致；
  X03 实现为无约束低秩映射 BAᵀ 而非数学projection。
- 决定：U02 修符号重跑 u02_fixed，“confidence anti-information”解释作废（score/lexico/u02b/argmax结论不受影响）；
  U03b 改同seed新鲜RNG重跑 u03b_fixed，正文暂禁“nuisance完全锁死”口径；
  X03 全库改“rank-4 low-rank patch map”，结论收窄为“该层该补丁装不上”（禁fundamentally cannot compose）。
- 复查触发：u02_fixed/u03b_fixed 落地后更新 FINAL_EVIDENCE_TABLE pending行。

## D-20260918-AUDIT02：科学探索冻结＋证据表＋论文迁移
- 日期：2026-09-18
- 事实：审计结论——科学探索可停；禁开路线：AMR/Jacobian扩展/E7新loss/quartet v3/relation v3/
  X03换rank换layer/temperature sweep/action-rule续调/更多scene family。
- 决定：TODO顶部 SCIENTIFIC_EXPLORATION:FROZEN；reports/FINAL_EVIDENCE_TABLE.md 为正文唯一数字来源；
  主headline切 fresh 151/176=85.8%；84.4%限定随机编辑分母（fresh瞄准44%并列，不平均）；
  paper标题→Knowing the Parts.../摘要引言已迁新线，02–08节重构待办。

## D-20260918-ENV01：后续 HF 下载走镜像
- 日期：2026-09-18
- 事实：本机直连 huggingface.co 经 7890 代理缓慢且出现过下载僵死；`huggingface_hub` 认
  `HF_ENDPOINT`（已验证覆盖默认 `https://huggingface.co`）。
- 决定：`~/.bashrc`、`~/.profile`、`~/.zshrc` 均追加
  `if [ -z "${HF_ENDPOINT:-}" ]; then export HF_ENDPOINT="https://hf-mirror.com"; fi`
  （显式导出优先，不覆盖）；正在跑的 SigLIP2 下载不动，沿用官方源直至完成。
- 复查触发：首次经镜像下载权重后核对 revision/sha 与 protocol lock 一致。


## 2026-09-24 — E1A933-START：审计修复与定向探索
用户指定解包并执行master prompt。HEAD恰等于审计基准 e1a933e6b32dd07c6895c7925cbccd9004da7f54；包内31个文件SHA256一致。R01–R10分四组并行，先局部纠错，随后推进对应优化和新路线；不以旧关闭/PARKED文案否定本轮授权。旧产物只读，修正分析不冒称首次确认；不新增付费资源。当前范围由包内00–04定义，根目录原无PLAN/ROADMAP，以本次合同补建路径文档，不改写历史科学判据。所有工程通过与科学支持分别记录。


## 2026-09-24 — E1A933-GPU-PLAN
用户允许视觉训练耗时过长时规划GPU实例。CPU实测整轮约73秒，21臂估9–12h，继续运行。GPU只读catalog/evaluate均无库存候选；4090/3090价目仅GPU组件，不是整机报价。准备独立CUDA入口及最小传输包，保留FP32/batch32/数据选择合同；不租机、不停CPU、不将CPU/GPU混为同一矩阵。详见reports/e1a933_review/GPU_TRAINING_PLAN.md。


## 2026-09-24 — E1A933-REMOTE-START
用户提供已部署的SSH实例并明确开始工作，覆盖之前仅GPU规划的权限边界。实测设备是RTX3080Ti12GB，不是预设V100；保持既有PyTorch2.5.1+cu124/torchvision0.20.1+cu124。最小离线包SHA256一致，真实CUDA前后向与有限梯度检查通过；batch32/FP32/21臂全部保持。独立远端任务使用监督进程持久运行，完成后验证21个checkpoint哈希并打包；不改CPU冻结源，不混CPU/GPU结果，不安装新环境或租机。新GPU结果因backend和NumPy版本与本地不同，必须独立分析，不承诺跨backend逐位一致。SSH首次按已有accept-new配置记录公钥，其后固定指纹严格校验；密码不存入文件。监督器不支持训练中间态恢复，失败如实保留并另目录重启受影响臂/矩阵。

## E1A933-GPU-FINISH（2026-09-24）
用户要求关机前本轮GPU任务一起做完。已有RTX3080Ti实例继续使用，无新增租机；增加N03六臂、O05十二臂、N02四十八臂和O06固定原生VLM272请求。N03启动由N01辅助学习/任务收益两者共同决定；相对强基线没有稳定收益，不触发进一步瓶颈架构搜索或向足球迁移。O01/O02/R03/O04余项可CPU完成。完成事件驱动回收，全部模型/逐样本产物校验后才生成GPU释放回执，不自动关机。

## E1A933-LANES（2026-09-24）— 8 条并行 lane 收尾与三处口径收窄
施工包剩余缺口改为 8 条并行 lane（n02/r03/r09/o01/o02/o04/r08/n01p1），共享工作区 + 严格文件归属（不用 worktree：artifacts 未被 git 跟踪，worktree 内看不到待分析产物）。全部完成后由总控统一接线进稿、同步状态文档并重生成构建回执。

主动收窄三处、并修复三处缺陷：
1. **N02 机制由"可归因"降为 UNRESOLVED**：B 相对 A 的 ΔJ 很大（+0.340/+0.300）但 B 的 conv/linear MACs 是 A 的 12.25×，算力未匹配；冻结方向预测仅 pretrained 命中、random 未命中（seed803 反向 +0.110），故只写"信息不变条件下的网格/算力配置效应"。
2. **O04 前瞻预测由"可预测"降为未支持**：预先封印的主要预测 P1 只 7/8 命中，按事前判据算 MISS，不放宽为"多数"；次级 P2 8/8 成立。
3. **R09 只能称"外推"**：dev 目标覆盖 0.4884 在无筛选的 213 场景上实测 0.1455–0.3239（目标的 30–66%），且 b0.5 下 net-benefit 差 0/9 对排除 0，故不写"可部署决策增益"。
4. **修复 R05/R06 合同检查**：`normalization_commutes_<p>` 四项不可能失败（点置换只行置换轨道，沿群轴排序后任何逐维仿射归一化都相同），已替换为可失败检查并附 mutation 证据。
5. **披露线程数口径混杂**：附录 U10 表 typed 列是 2 线程评估、I 列来自 4 线程可复现的 raw/six；同一 typed flip 臂最多相差 9.0pp，已在表格 caption 与正文写明不可跨口径比较。
6. **修复 claim ledger 结构**：原表两行缺 denominator 单元格导致其后各列左移，已补占位；新增行中未转义 `|` 改用 `\lvert`/`\rvert`。
复查触发：若后续要升级 N02 机制 claim，须先做算力匹配对照并以新挖的小编辑分层（≥60 quartet）复核；若要在 O04 上"验证"分解，须换与 D10 共享场景/parent 的对象，否则不成立。

## E1A933-CPU-FALLBACK-STOP（2026-09-24）— 停止冗余的 CPU 兜底训练
用户指示"停掉兜底训练"，已执行。

- **对象**：host PID 1886333（`vision_run.py --out artifacts/e1a933_review/vision_canonical224_v2 --epochs 20`），已运行 7 小时 27 分、占 8 线程。
- **理由**：它只是"远端 GPU 矩阵若失败"的保险；GPU 矩阵已完成、回收、逐文件校验并已进稿，而本项目规定 CPU/GPU 结果不得混表，故即使跑完也不进任何报告数字。
- **执行**：SIGTERM，2 秒内退出，核实无残留 `vision_run` 进程。
- **产物保全**：完成 19/21 臂；`matched_s806` 仅有 `history.json`（无 `model.pt`）、`shuffled_matched_s806` 未启动；**未删除任何文件**。`receipt.json` 状态改为 `STOPPED_BY_USER`；原始 RUNNING 回执按字节保留为 `receipt.RUNNING_asof_stop.json`（sha256 `8dcc9e7f0e1edb409d138fe07348bcde9166997870f7be7f215e0a91774497c1`）；停止记录 `TERMINATION_RECEIPT.json`。
- **对结论的影响**：无。报告中的视觉矩阵是远端 GPU 那一套。
- 复查触发：若日后需要跨 backend 数值对照，应把该 19 臂产物**单独归档为新报告**，明确标注为 CPU/GPU 一致性对照，不得与 GPU 数字拼进同一张表。

## E1A933-LEADS-L014-L015-L006（2026-09-24）— 三条线索收口，不升方法主张
用户指示完成第一档 L-015(b) 与第二档 L-014、L-006。

- **L-014**：只复读 `artifacts/event_updater/sided_s{11,23,47}.json`，不重跑、不发展 loss。正文页预算已满，观察句留在 `reports/e1a933_review/L014_RADIUS_REPORT.md`，不写入讨论段。
- **L-015/L-006**：新训练，评价只用反复用过的 dev512（237 四元组 / 106 父场景），不读封存确认池。10 个种子，固定 300 轮，不用评价 J 挑模型，4 线程。
- **作废分母**：训练脚本自挖评价池只有 2 个四元组，`SUMMARY.json` 的 J 不进结论。
- **不把可学习池化的失败写成普遍否定**：宽 32 与一次参数量匹配的宽 48 都没学会，但这是这条配方的阴性，不是“等变架构不可能”。
- **不改写四臂主结果**：这里的 raw 用共享 xy 归一化，绝对 J 不能和论文四臂表相减。
- 复查触发：若要把 G8 特征的增益写成确认级主张，必须换一个未用过的银行，并且不能把构造出来的零摆动说成经验发现。

## E832-FOCUS-START（2026-09-24）— 按用户包收缩为两条主线
用户指示解包 `docs/e832887_focus_upgrade_package.zip` 并按 `01_MASTER_PROMPT.md` 开工。

- **不用旧“已关闭”阻断本轮**。旧 PLAN 仍不改。本轮科学问题以用户包为准：哪些表示与关系计算条件，让单次变化监督改善完整组合正确性，而不只是搬移错误。
- **A0 已用仓库真实类核对**，不是只跑包内镜像。`TypedPairMLP` 与 `G8SetMLP` 在 4 例棋盘上的混合差分残差最大约 1e-16；线性 rho 短拟合阈值准确率 0.5；只把 rho 改成带非线性的包装后，同 4 例可拟合（准确率 1.0）。这是表达能力，不是银行泛化。`TypedTriMLP`/`TypedDiskMLP` 先拼接再非线性融合，不连坐。记录：`artifacts/e832_focus/structure/EXPRESSIVITY_REPO_AUDIT.json`。
- **这一刀不租 GPU**。本机 `torch.cuda.is_available()` 为假，旧实例状态是 SAFE_TO_SHUTDOWN。B 先复用已有视觉结果；C 不训练。新增付费租机单独报告。
- **旧脚本不改字节**。新代码只进 `experiments/e832_focus/`，新结果只进 `artifacts/e832_focus/` 与 `reports/e832_focus/`。
- 复查触发：若修后交互网络在新父场景上仍不能区分竞争解释，不把“加了一层”写成方法主张；若要启动 C 的长训练，必须先有 A/B 的解释缺口或更强独立主张。

## E832-FOCUS-A-RESULT（2026-09-25）— 新银行分开了排序，没有分开“加一层”
结构线在新生成的父场景上完成预先指定的比较。总控核对了 `metrics_continued.json` 与报告中的差值，没有重训。

- **分母锁定**：2188 个无序 E、871 个 parent，`bank_sha256` `15f5bf181b37dc9b4309582490f7fc11dfca8b03491251c3724fcb2e1bdd804c`。到训练名单的最小 L-inf 是 0.119。不读封存池，不读 dev512。
- **被分开的解释**：同一 8 个连续量，只做现有 G8 排序。single-flip 下相对未排序的 J 差为 +0.061 至 +0.092，5 个 parent 区间都不含 0。这是一次具体不变处理，不是信息完全相同的证明，也不是学出来的集合网络。
- **没有被分开的解释**：中点和绝对正弦本身相对 6 个有序距离，区间大多含 0，没有加 seed，没有改分母。
- **加一层不是几何增益的解释**：段和后的线性读出改成非线性后，训练损失下降，但 single-flip 的 J 仍约 0.017–0.046。相对加性负控，flip 差最多约 +0.015，5 个区间里 3 个含 0。旧加性限制解释棋盘反例，不解释这份银行的几何优势。
- **边界**：1745/2188 的组合 margin 低于 0.02。这是报告过的切片，不是替换后的分母。
- **不升方法，不进结论页**。附录写了范围句。T1/T2 未跑。C 维持不启动。视觉结构消融未跑。
- 复查触发：若要把排序优势写成确认级主张，需要在边界不那么近的新父场景上同向复现，并且不能把构造出来的零摆动说成经验发现。

## E832-FIVE-ROUTE-OPTIMIZATION（2026-09-25）— 五条路线并行收束，路线1/2各五轮
用户授权完成五条胜率优化方案；路线1、2除基本实现外各给五轮按实际证据优化和挖掘。

- **共同协议**：`reports/e832_focus/OPTIMIZATION_PROTOCOL.md`。五条路线只服务一个主问题，不按轮数制造阳性。
- **五轮是上限与决策门，不是必须跑满**：Round 0 实现；后续先3 seeds；只对最强1–2个对比扩到10 seeds。阴性、反例和标准基线解决任务都算完成。
- **路线1**：source/T1/T2跨任务，分开输入优势与交互符号。
- **路线2**：同一ResNet18 encoder的可观测段表示＋interaction/additive消融，新parent。当前本机无CUDA、旧授权实例端口拒绝连接；先做实现、CPU冒烟和GPU bundle，正式结果不得用CPU小样本冒充。
- **路线3**：解析、标准集合/关系、容量匹配和有限群轨道强基线。
- **路线4**：新parent、新renderer、预设观测变化，失败时缩窄claim。
- **路线5**：只接最强1–2项结果；真实Figure 1、统一主表、ledger和复现入口。
- **资源边界**：不产生新增租机支出，不改旧冻结脚本字节，不读封存池。
- 复查触发：任何路线若只剩“工程跑完”而没有区分竞争解释，不进入正文；GPU不可用时路线2标记BLOCKED_GPU，不用弱结论填充。

## E832-ROUTE1-4-FIRST-PASS（2026-09-25）— 先收窄主张，再继续优化
父核验路线1–4的首轮实现和结果。

- **路线1**：source新parent E=2313/871；rich_sorted相对rich_unsorted在single-flip三seed为+0.037/+0.024/+0.057，方向支持source范围句。T1/T2仅E=8/17，全部低预算且不可解释；不作跨任务确认，追加固定8192测试parent的可行性轮。
- **路线2**：4项接口测试和16-parent/4-step CPU smoke通过；正式结果`BLOCKED_GPU`，不将CPU smoke写成视觉结论。旧授权实例端口拒绝连接，未产生新租机。
- **路线3**：source E=230/65 的解析segment-intersection J3=1.0；学习式capacity/relational J3约0–0.052。故source学习比较明确改写为“解析器之外的学习诊断”，不再暗示任务需要复杂关系模型。
- **路线4**：新parent E=207/66；canonical/translation/rotation是严格坐标不变量，sorted方向三条件同号但训练只有60 epochs、效应低；noise和visual renderer均未运行，不升级为renderer稳健性。
- **论文影响**：在路线1跨任务和路线2视觉正式结果到来前，主文中心不写“cross-object interaction已解决”；当前最强可防守贡献是repair-flow测量、手工表示的范围性差异和视觉full-repair边界。真实Figure 1已按固定dev512规则接入，路线5审阅为PARTIAL。
- 复查触发：若解析器在扩展任务上同样解决问题，论文必须接受“廉价解析基线足够”的结论，不能继续把学习式排序写成方法优势。

## E832-ROUTE1-ROUND2（2026-09-25）— 跨任务确认不成立，主张收窄
路线1按预先写入的 Round 2 manifest 完成唯一一次新parent扩展，未改 E 规则、指标或候选族。

- source：E=1414、389 parents；rich_sorted−rich_unsorted 在3个seed×clean/flip全部为正，parent区间均排除0。
- T1：E=417、170 parents；所有排序差区间跨0，判 `UNINFORMATIVE`。
- T2：E=912、361 parents；方向随seed/监督变化，仅一个clean seed区间排除0且为负，判 `NEGATIVE_UNSTABLE`。
- 首轮 source 的可审计 parent 数是513而非早期 prose 的871；新结果使用 Round 2 manifest 的1414/389，旧结果保留并在报告中标明。
- 允许：source-specific sorted representation effect；T1/T2边界/不稳定性。禁止：跨任务确认、普遍interaction、修复普遍有效。
- 复查触发：只有新的任务协议或真正不同的关系结构干预，才能重新提出跨任务机制问题；不继续抽取更多parent直到出现阳性。

## E832-ROUTE2-DATA-READY（2026-09-25）— 正式视觉结果仍不写
路线2审计发现原`gpu_run.py`只是状态stub。已补齐fresh data generator、hash/split锁定、data-bound runner和7项测试。

- 新视觉银行：train/dev/test parents=128/32/128；singleton images=323/81/319；test-only quartets=28/112 images/19 parents；AB训练图像=0；parent/image split两两不交。
- 四臂合同固定：同一ResNet18与输入，direct global linear、additive shared red/blue linear、representation concat linear、interaction concat nonlinear。推理只接受图像。
- CPU只做PILOT_ONLY；本机无CUDA，12个arm/seed run、J3/J4/full repair/migration/111 regression全部未运行。状态为`DATA_READY_BLOCKED_GPU`，不是视觉阴性。
- 复查触发：只有GPU结果和独立review完成后，才允许更新路线2 claim；没有GPU时正文不写视觉interaction方向。

## E832-ROUTE2-WAITING-LOGIN（2026-09-25）— 远程执行暂停等待授权入口
用户要求等待其提供登录方式。路线2本地代码、fresh data、manifest和GPU runner均已准备并通过7项测试；不主动连接远程主机、不保存凭据、不创建新租机。恢复入口：`reports/e832_focus/route2/REMOTE_RUNBOOK.md`。在用户提供入口前，正式视觉状态保持 `DATA_READY_BLOCKED_GPU`，正文不写视觉方向。

## E832-ROUTE2-ITERATION-BUDGET（2026-09-25）— 可持续至04:55
用户将原“最多3轮优化”改为可继续迭代至04:55。仍保留提前停止条件：结果满意、确认失败、没有新的可修复因素或数据/实现合同失效时立即停止；不为耗尽时间制造新训练。发现J3字段和native RGB mask输入合同缺陷时，先修合同再重跑，不把旧结果解释成科学结论。

## V3-M1-D1（2026-09-25）— 接受segment_moment块共享统计
`segment_moment`两block独立统计在合法segment swap下破坏不变性（swing~12），改为跨block共享统计后swing为0。旧run保留在`superseded/blockwise_stats/`并排除出结果；bank/seed/supervision/selection未动。接受该修复。

## V3-M2-VERDICT（2026-09-25）— 冻结模块迁移阴性，归因精确
16格GPU矩阵（2080 Ti，692/351 bank，冻结M1 orbit head seed11）：direct_full J3≈0.92为最优；真几何+冻结head 0.796，随机head≈0，head competent且特异；纯几何loss前端（冻结/微调backbone）J3≈0.20，dev几何MSE≈0.51不变。失败在规范化orbit接口不可微回归，不在探针容量。按M2.5阴性分支停止加head；M1源表示结论不受影响。用户提供的GPU入口全程可用，凭据未落盘。

## V3-M1-VERDICT（2026-09-25）— 源端有界结论，T1/T2暂缓
M1.2源矩阵确认：whole-orbit参照去掉高风险独立排序且不付精度代价，但flip侧相对sorted的增益方向不稳定；segment_moment不足；repaired head复现阴性；source未找到精确碰撞。不携带任何跨任务结论。T1/T2移植暂缓，优先M2接口工作。

## E832-ROUTE2-FINAL-V2（2026-09-25）— 合同修复后有界阴性，退租外部阻塞
Round 2修复了J3定义和native RGB输入合同；v2 12臂+3 clean baseline均完成。direct J3均值0.571，interaction均值0.583，但interaction-direct seed差+0.179/-0.143/0，不能声称稳定视觉交互优势；状态为`BOUNDED_NEGATIVE_UNSTABLE`，停止训练。AI Galaxy MCP账户显示1台running，但`plan_release`返回该实例不属于当前MCP state store；退租需账户所有者通过控制台或提供正确MCP instance name，未用SSH shutdown代替。

## E832-ROUTE2-CLOSE（2026-09-25）— 五轮优化预算关闭，退租除外
用户指示除退租外全部收尾。Round2-5优化轮按停止规则关闭：v2之后无新的可修复合同/实现因素，方向不稳定，无继续训练理由，故四轮均记“未触发”，不是“等待GPU”。原DATA_READY_BLOCKED_GPU状态由v2正式结果取代（12+3已执行）。退租条目保留未勾选，待账户所有者处理，本轮不收。

## N02-UPGRADE-START（2026-09-25）— 算力匹配＋小编辑分层
用户提供已授权GPU入口并指示开工。连通性已验证（RTX 2080 Ti 11GB，torch 2.5.1+cu121，双IP同机同host key，指纹已固定 pin，后续 StrictHostKeyChecking=yes）。凭据只用内存、不存文件。协议冻结先于训练（P0复现/P1算力比例/P2小编辑机制升级判据）。旧48臂字节不动；新产物只写 experiments/artifacts/reports/n02_upgrade/。新后端与旧3080Ti/cu124数字不拼因果差值。

## N02-UPGRADE-VERDICT（2026-09-25）— 算力一半量化，机制仍未决
P0通过（新后端B−A两init皆正）。P1：A-wide（×4通道，2.25G≥B 1.81G）关掉约一半差距，
判COMPUTE_OR_CAPACITY_MAJORITY_BY_MEAN（s806的0.43异质性如实保留）。P2：small-edit
564/146上预测方向被拒（random正向两seed显著），原−0.133命中在新后端未复现，
机制维持UNRESOLVED且aliasing路更窄。工程事故两起（smaledit整批插值OOM→分块修复；
残留cache目录撞exist_ok→清理重跑），判据未动。模型.pt留远端（W臂约700MB/个），
hash已在result.json登记。
