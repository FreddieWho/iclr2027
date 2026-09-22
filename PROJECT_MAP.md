# PROJECT_MAP — 叙事主线 → 文件系统全映射

> 用途：回答"这个目录/文件是干什么的、属于哪条叙事线、现在还算不算数"。
> 配合 `README.md`（前门）使用。2026-09-22 结构整理建立。
> 时代划分见 README「路线调整简史」：E0 蓝图 / E1 P2 / E2 P3-T5R / E3 P4-AMR /
> E4 转折发现 / E5 外部确认 / E6 终局收官。

## 1. 最终叙事主线（paper 每一节的证据在哪）

| paper 节 | 内容 | 关键代码 | 关键产物 | 关键报告 |
|---|---|---|---|---|
| §2 受控设定 | 坐标 quartet/transition 仪器（oracle 标签+已知效应编辑） | `experiments/discovery_campaign/core/`、`experiments/last15h/shared/` | `artifacts/discovery_campaign/scenes/` | `reports/discovery_campaign/ROUND1_SUMMARY.md` |
| §3 三失败模式 | N01 转折漏检 / N03 事件条件漏检 / N04 组合(emergent)漏检；E1 fresh holdout 确认（151/176=85.8%） | `experiments/last15h/`（N01–N10 worker） | `artifacts/last15h/`（含 `E1/holdout_909/result.json`） | `reports/last15h/N01.md N03.md N04.md`、`reports/last15h/E1_VERDICT.md`+`E1_PREREG.md` |
| §4 flip 诊断修复 | flipmine（oracle 挖掘边界穿越样本训练）；P1 置信反转；P3 错误迁移 | `experiments/discovery_campaign/`（R02/R04）、`experiments/next_novelty/`（p1_unified/p1b/p1g）、`experiments/ccm_audit/`（p3_final/p3d_fair/relflip_train） | `artifacts/next_novelty/p1_unified{,_confirm}`、`p1b_*`、`p1g_*`、`p3/`、`relflip*/`、`relfeat/` | `reports/discovery_campaign/R3_flipmine_card.md`、`reports/final_closure/P1_CLOSURE_REPORT.md`、`P3_CLOSURE_REPORT.md` |
| §4b 足球真实数据 | IDSSE 三场×三种子复现三失败+修复（坐标→真数据） | `experiments/last15h/`（E2）、`experiments/confident_blindspots{,_v2}/` | `artifacts/phase3/task_semantic_repair_v1/`（T5R3 冻结模型）、`artifacts/confident_blindspots/` | `reports/last15h/E2_N01.md E2_N03.md E2_N06.md`、`reports/confident_blindspots/P1_REPORT.md` |
| §5 像素确认 | 同 quartet 渲染为像素+CNN；五风格全同向 | `experiments/last15h/`（N08）、`experiments/next6_ef0f7a3/`（U03 pixflip） | `artifacts/last15h/`、`artifacts/next6_ef0f7a3/` | `reports/last15h/N08.md`、`reports/next6_ef0f7a3/U03.md` |
| §6 外部确认（阴性） | DINOv2-B/CLIP pilot→DINOv3 S/B/L sweep→读出/语义Δ/均衡修复/competence，全阴性 | `experiments/bridge/`、`experiments/bridge_r/` | `artifacts/bridge/`、`artifacts/bridge_r/` | `reports/bridge/B01_PILOT.md`、`reports/bridge_r/*.md`（BRIDGE_R_FINAL_REPORT 为总） |
| §7 相关工作+局限 | 旧 Action-Mode/Jacobian 线退为半段背景；局限清单 | （历史线，见 §2 下表） | — | — |
| §8 可复现性 | 五层锁/单次读取/预注册/审计链 | `scripts/audit_*.py` | 各 artifact 的 lock/manifest/SHA256SUMS | `reports/final_closure/REPRODUCE_FINAL.md` |

正文每个数字的分母/边界/禁用措辞：`reports/final_closure/MASTER_CLAIM_LEDGER.md`。

## 2. experiments/（16 个子目录，按时代）

| 目录 | 时代 | 内容 | 状态 |
|---|---|---|---|
| `discovery_campaign/` | E4 | 转折现象发现引擎（quartet 渲染/编辑/R01–R06 路由） | 主线证据，冻结 |
| `last15h/` | E4 | N01–N10 + E1 holdout + E2 足球 + E7 方法网 | 主线证据，冻结 |
| `next6_ef0f7a3/` | E4 | U01–U03/X01–X03 六路（像素修复 U03 为主） | U03 进正文，其余 park |
| `bridge/` | E5 | foundation model B01 pilot | 阴性，进 §6 |
| `bridge_r/` | E5 | DINOv3 sweep/读出/语义Δ/均衡修复/competence | 阴性，进 §6 |
| `ccm_audit/` | E6 | P3 迁移终局（p3_final/p3d_fair/relflip/relfeat/paired_compare） | **主线证据，当前** |
| `next_novelty/` | E6 | P1 置信反转终局（p1_unified/p1b/p1g＋确认池） | **主线证据，当前** |
| `p123_upgrade/` | E6 | dev512 样本银行构建 | **主线输入，当前** |
| `confident_blindspots/`、`confident_blindspots_v2/` | E6 | P1 足球侧复现 | 主线辅助 |
| `repair_decomposition/` | E6 | P2 覆盖/精度分解 | 主线辅助 |
| `novelty_round3/` | E6 | Round3 四包（干预双头/transfer/分区/运动混杂，全阴性或工程级） | 关闭 |
| `update_geometry/` | E6 | WP1–WP5（composition-as-update 归因 111/111） | 关闭（进 §3 机制句） |
| `event_updater/` | E6 | EventUpdater（阴性，SIDE_DIAGNOSTIC_ONLY） | 关闭 |
| `l002_ball/` | E6 | ball 轨迹试点（弱 null，S2 混杂） | 关闭 |
| `radius_loss/` | E6 | 半径 loss 试点（反向失败，埋） | 关闭 |

## 3. artifacts/（23 个子目录，按时代；大文件本地，sha 见各自 manifest）

| 目录 | 时代 | 内容 | 状态 |
|---|---|---|---|
| `phase0/` `phase1/` `phase2/` | E0–E1 | 250 canonical samples、9 冻结模型、P2 fracture 权威输出 | 历史冻结（§7 背景），只读 |
| `phase3/` | E2 | support geometry/prospective/T5R3 模型/SSL 对照/ε-sweep 等 | 历史冻结；T5R3 模型被 §4b 复用 |
| `data_v2/` | E2 | IDSSE + SoccerTrack-v2 canonical 数据（provenance 链） | 数据冻结，只读 |
| `phase4_amr/` | E3 | AMR v1–v6 全部训练产物 | 历史（方法位放弃，§7 背景） |
| `discovery_campaign/` | E4 | 转折现象原始证据（含 confirm_1007 **封存**） | 主线，冻结 |
| `last15h/` | E4 | N01–N10/E1(holdout_909 **封存**)/E2 产物 | 主线，冻结 |
| `next6_ef0f7a3/` | E4 | 六路产物 | 主线（U03），冻结 |
| `bridge/` `bridge_r/` | E5 | foundation model 特征与评价（holdout_895 **封存**） | 阴性证据，冻结 |
| `p123_upgrade/` | E6 | dev512 银行（sha 21786139…）+ P1/P2/P3 中间产物 | **主线输入**，冻结 |
| `next_novelty/` | E6 | P1 unified/confirm、p1b/p1g、p3、relflip/relfeat、p3d_fair | **主线证据**，冻结 |
| `confident_blindspots/` | E6 | 足球 P1（FOOTBALL_CONF.json） | 主线辅助 |
| `update_geometry/` `event_updater/` `l002_ball/` `radius_loss/` `novelty_round3/` | E6 | 已关闭旁路产物 | 关闭，仅审计 |
| `calligraphy_pilot/` | E0 | 书法试点（EXPLORATORY_FROZEN，不进本期论文） | 历史冻结 |
| `remote/` | E0 | 远端书法/检索运行收据 | 历史 |

## 4. reports/（散落报告按时代归类；子目录详见 `reports/INDEX.md`）

- **E6 终局权威**：`final_closure/`（全部）、`next_novelty/`、`p123_upgrade/`、
  `confident_blindspots/`、`repair_decomposition/`（如有）、`exploration_log.yaml`
- **E4 转折**：`last15h/`、`discovery_campaign/`、`next6_ef0f7a3/`、
  `FINAL_EVIDENCE_TABLE.md`（9-18 版，已被 final_closure 版取代）
- **E5 外部确认**：`bridge/`、`bridge_r/`
- **E3 P4-AMR**：`P4_*.md`、`GOAL_R*.md`、`N4_HEADTOHEAD_DRAFT.md`、`research_p4*/`、
  `research_b30/`、`research_briefs_20260905/`
- **E2 P3-T5R**：`P3_*.md`（SUPPORT_CONDITIONED_GEOMETRY 为该期主报告）
- **E0–E1 蓝图/P2**：`PHASE0_REPORT.md`、`P2_*.md`、`EXPLORATION_CHECKPOINT*.md`、
  `PUBLICATION_REVIEW_ICLR2027.md`、`p0-overview.html`
- **书法（EXPLORATORY_FROZEN）**：`CALLIGRAPHY_*.md`
- **E6 关闭旁路**：`update_geometry/`、`event_updater/`、`l002_ball/`、`radius_loss/`、
  `novelty_round3/`、`next_novelty/FINAL_NOVELTY_DECISION.md`
- 历史发布记录：`RELEASE_CHANGELOG_20260901.md`

## 5. 文档对照：权威 vs 历史（避免双"唯一来源"混淆）

| 主题 | ✅ 当前权威 | ⛔ 历史版本（勿再引用数字） |
|---|---|---|
| 正文数字 | `reports/final_closure/MASTER_CLAIM_LEDGER.md` | `CLAIM_LEDGER.md`（P2/P3 时代）、`paper/EVIDENCE_MAP.md`（旧论文骨架） |
| 证据总表 | `reports/final_closure/FINAL_EVIDENCE_TABLE.md`（9-22 重建） | `reports/FINAL_EVIDENCE_TABLE.md`（9-18，该文件顶部已标 superseded） |
| 项目状态 | `STATUS.md`（POLISH） | `configs/project.yaml`（P3 时代字段）、`PLAN.md`/`ROADMAP.md`（P4 时代） |
| 行动史 | `TODO.md` 变更记录（最新在文末） | TODO 中部 B30 清单（已标历史区） |
| 复算入口 | `reports/final_closure/REPRODUCE_FINAL.md` | 各时代报告内的运行命令（仅溯源用） |
| 决策史 | `DECISIONS.md`（追加至 9-18）+ TODO 变更记录（9-18 后） | — |

## 6. 已消耗并封存的确认资源（任何二次读取需用户授权）

| 资源 | 用途 | 状态 |
|---|---|---|
| `artifacts/last15h/E1/holdout_909/` | E1 fresh holdout battery | 已单次读取（E1 通过），封存 |
| `artifacts/discovery_campaign/scenes/confirm_1007/` | P1 确认池 | 已单次读取（P1 CONFIRMED），封存 |
| bridge holdout_895 | foundation model 确认 | 未拆封（所有 dev 门未过），封存 |
| `J03WQQ`（IDSSE reserved） | T5R5 单次读取 | 已消耗（PASS_STRONG，E2 时代） |
| SoccerTrack-v2 | T5R6 外部确认 | 已消耗（CONFIRMED，E2 时代） |
