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
