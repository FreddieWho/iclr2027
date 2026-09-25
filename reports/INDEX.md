# reports/ INDEX — 报告索引（已按时代物理分区）

> 全局叙事映射见根目录 `PROJECT_MAP.md`；本目录 2026-09-22 起按时代分区。
> ⚠️ **历史报告正文内的相对路径引用**（如 `reports/P3_XXX.md`）对应分区前布局，
> 不再解析；文件本身内容未改。现行位置由下表给出。
> 时代：E0–E1 蓝图/P2 / E2 P3-T5R / E3 P4-AMR / E4 转折发现 / E5 外部确认 / E6 终局。

## 当前写作与审查入口（2026-09-25）

- `submission_audit_20260925/` — 本轮结论与代码审查、必要 GPU 补算、敏感性分析、统一 CLAIMS、允许措辞、复算和剩余缺口。写作优先读此目录。
- `e832_focus/`、`mechanism_transfer_v3/`、`n02_upgrade/` — 后续实验的原始报告；受本轮更正的字段与解释以审查目录为准。

## 历史终局（E6）

- `final_closure/` — **历史终局包**：FINAL_PROJECT_VERDICT（15 问）、
  MASTER_CLAIM_LEDGER（历史数字来源）、FINAL_EVIDENCE_TABLE（9-22 版）、
  REPRODUCE_FINAL（一键复算）、P1/P2/P3_CLOSURE_REPORT、EVIDENCE_CORRECTIONS_FINAL、
  PAPER_MIGRATION_AUDIT、P2_DECISION、RELFEAT_DECISION、PAPER_ALIGNMENT_AUDIT_20260922、
  REBUTTAL_RESERVE_T5R6（T5R6 独立确认储备，仅 rebuttal 用）
- `next_novelty/` — P1 线终局（FINAL_NOVELTY_DECISION、EVIDENCE_CORRECTIONS_V2）
- `p123_upgrade/` — P123 三件套（PAPER_INSERTS、ledger、复算包）
- `confident_blindspots/` — P1/P2/P3 三件套草稿与足球复现
- `exploration_log.yaml` — 探索日志（根目录保留）

## E4 转折发现（最终叙事诞生）

- `last15h/` — N01–N10 卡、SELECTION/M_SELECTION、E1_PREREG/E1_VERDICT、
  E2_*（足球）、E7_*（方法网全灭）、ABSTRACT_DRAFT、HEADLINE_FREEZE、FIGURE_SOURCE
- `discovery_campaign/` — R01–R06 各轮卡、TRACK6、MAIN_FIGURE_SOURCE、AUTH_LOG_R6
- `next6_ef0f7a3/` — SIX_ROUTE_RESULTS/SELECTED_STORY/NUMBERS，U01–U03/X01–X03 卡
- `e4_discovery/FINAL_EVIDENCE_TABLE_20260918.md` — ⚠️ 9-18 审计闭合版，已被
  `final_closure/FINAL_EVIDENCE_TABLE.md`（9-22）取代

## E5 外部确认（全阴性，进论文 §6）

- `bridge/B01_PILOT.md` — DINOv2-B/CLIP pilot（NOGO）
- `bridge_r/` — BRIDGE_R_FINAL_REPORT 为总；SIZE_SWEEP/SPATIAL_READOUT/SEMANTIC_DELTA/
  BALANCED_TRANSITION/COMPETENCE_FIRST 各决策与报告

## E3 P4-AMR（方法线，已放弃；论文 §7 背景）

- `e3_p4_amr/` — P4_MU_THRESHOLD、P4_N3_V{1,2,3} 失败报告、P4_N4_V4、P4_R{1,2,3}、
  P4_TERMINAL_V4B、GOAL_R{1,2}、N4_HEADTOHEAD_DRAFT、PUBLICATION_REVIEW_ICLR2027（9-05 胜算评审）
- `research_p4/`、`research_p4_round2/`、`research_p4_round3/` — researcher 简报
- `research_b30/` — B30 头脑风暴 34 文件（W1–W3＋VERIFY_FINAL）
- `research_briefs_20260905/` — 投稿胜算评审 4 简报

## E2 P3-T5R（几何可预测+任务语义修复；CONFIRMED 但仅表示塑形）

- `e2_p3_t5r/` — 主报告 P3_SUPPORT_CONDITIONED_GEOMETRY；T5R 链
  （TASK_SEMANTIC_REPAIR_PLAN→T5R3_SANITY→T5R4_ROUND{1,2}→T5R5_HIDDEN_CONFIRMATION→
  T5R6_EXTERNAL_CONFIRMATION）；补强（HEADLINE_EFFECT/EPSILON_SWEEP/SUPPORT_SCALING/
  PREDICTOR_BASELINES/SSL_FAMILY_CONTROL/RATIO_SWEEP/CANDIDATE_SEARCH）；治理
  （T5R5_CANDIDATE_LOCK、CLOSURE_AUDIT、DOCUMENT_MIGRATION、EXECUTION_EVIDENCE 等 21 份）

## E0–E1 蓝图与 P2（Action-Mode Spectrum 旧线）

- `e0e1_blueprint/` — PHASE0_REPORT、p0-overview.html（P0 样本可视化）、
  P2_CLAIM_LEDGER、P2_HETEROGENEITY_DIAGNOSIS、P2_MATCHING_SMOKE、
  EXPLORATION_CHECKPOINT{1,1_POINT_MAINLINE,2,2_FRACTURE,2_FRACTURE_PRE_RESIDUAL_FIX}、
  CALLIGRAPHY_*（书法线 EXPLORATORY_FROZEN）、RELEASE_CHANGELOG_20260901

## E6 已关闭旁路（结论均为阴性/工程级，勿当主线读）

- `update_geometry/`（composition-as-update 归因）、`event_updater/`（SIDE_DIAGNOSTIC_ONLY）、
  `l002_ball/`（运动混杂）、`radius_loss/`（反向失败）、`novelty_round3/`（四包收敛）
