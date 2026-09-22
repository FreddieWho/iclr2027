# experiments/ INDEX — 实验代码按时代归类

> 全局叙事映射见根目录 `PROJECT_MAP.md`。复算入口：`reports/final_closure/REPRODUCE_FINAL.md`。
> 时代：E0 蓝图 / E2 P3-T5R / E3 P4-AMR / E4 转折发现 / E5 外部确认 / E6 终局。
> （scripts/ 根目录脚本为 E0–E3 各期管线脚本，未在本目录。）

## ⭐ 主线证据代码（终局叙事，REPRODUCE_FINAL 链内）

| 目录 | 时代 | 内容 |
|---|---|---|
| `p123_upgrade/` | E6 | dev512 样本银行构建（build_bank.py） |
| `next_novelty/` | E6 | P1 置信反转：p1_unified.py（dev/confirm 银行）、p1b_analysis.py（分位/CV）、p1g_repair.py（修复残留） |
| `ccm_audit/` | E6 | P3 迁移：p3_final.py、p3_migration.py、p3d_fair.py（keepbal 公平重跑）、p3d_eval.py、relflip_train.py、relfeat_eval.py、paired_compare.py |
| `confident_blindspots/`、`confident_blindspots_v2/` | E6 | P1 足球侧复现（football_conf.py、p1_football.py） |
| `repair_decomposition/` | E6 | P2 覆盖/精度分解（p2_decompose/p2_operating/p2_affine_atoms/p2_table） |
| `discovery_campaign/` | E4 | 转折现象发现引擎：core/（quartet 渲染与编辑）、routes/（R01–R06）、flipmine 训练与评价 |
| `last15h/` | E4 | N01–N10 worker、E1 fresh holdout battery、E2 足球三件套、E7 方法网（shared/paths.py 为共享路径模块） |
| `next6_ef0f7a3/` | E4 | U01–U03/X01–X03 六路（u03b_style.py 像素五风格修复进正文 §5） |

## 外部确认（E5，全阴性 → 论文 §6）

| 目录 | 内容 |
|---|---|
| `bridge/` | b01_pilot.py：DINOv2-B/CLIP ViT-B frozen + linear probe |
| `bridge_r/` | DINOv3 S/B/L 特征提取与读出/语义Δ/均衡修复/competence-first 各任务 |

## 已关闭旁路（E6，阴性/工程级，勿重开）

| 目录 | 结局 |
|---|---|
| `update_geometry/` | WP1–WP5；composition-as-update 111/111 归因成立，incidence law 死亡 |
| `event_updater/` | EVENT_UPDATER_NEGATIVE；SIDE_DIAGNOSTIC_ONLY |
| `l002_ball/` | 主线弱 null；S2 被运动混杂解释（WP-D） |
| `radius_loss/` | 三种子反向失败，掩埋 |
| `novelty_round3/` | 四包全收敛（A/B 阴性、C 工程级、D 混杂） |
