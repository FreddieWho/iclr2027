# artifacts/ INDEX — 产物目录按时代归类

> 大文件（.pt/.npy/.npz/.parquet）按根 `.gitignore` 留本地（全目录约 36G）；
> 各子目录的 receipt/manifest/SHA256SUMS 为 provenance 凭证。全局叙事映射见
> 根目录 `PROJECT_MAP.md`。时代：E0–E1 蓝图/P2 / E2 P3-T5R / E3 P4-AMR /
> E4 转折发现 / E5 外部确认 / E6 终局。

## ⭐ 终局叙事证据（E4–E6）

| 目录 | 时代 | 内容与地位 |
|---|---|---|
| `p123_upgrade/` | E6 | **dev512 样本银行**（sha 21786139…，P1/P2/P3 共用输入）＋P1/P2/P3 中间产物 |
| `next_novelty/` | E6 | P1：p1_unified{,_confirm}、p1b_res/p1b_confirm、p1g_res/p1g_confirm、p1_freeze.json；P3：p3/、p3d_fair*/、relflip*/、relfeat/；P2：p2/ |
| `discovery_campaign/` | E4 | 转折现象原始证据；含 `scenes/confirm_1007/`（**P1 确认池，已封存**） |
| `last15h/` | E4 | N01–N10 产物；`E1/holdout_909/`（**fresh holdout，已单次读取并封存**）；E2 足球产物 |
| `next6_ef0f7a3/` | E4 | 六路产物（U03 像素五风格） |
| `confident_blindspots/` | E6 | FOOTBALL_CONF.json（足球 P1） |
| `bridge/`、`bridge_r/` | E5 | foundation model 特征/评价（holdout_895 **未拆封，封存**） |

## 已关闭旁路产物（E6，仅审计价值）

`update_geometry/`、`event_updater/`、`l002_ball/`、`radius_loss/`、`novelty_round3/`

## 历史冻结（E0–E3；被引用的背景证据仍有效，但不构成主线）

| 目录 | 时代 | 内容 |
|---|---|---|
| `phase0/`、`phase1/` | E0–E1 | 250 canonical samples（phase1/canonical_samples.npz）、9 冻结模型 |
| `phase2/` | E1 | P2 fracture continuity 唯一权威输出＋异质性诊断（只读） |
| `phase3/` | E2 | support_geometry_v1、prospective、T5R3 冻结模型（被 §4b 复用）、SSL/ε/support 扫掠、causal switch |
| `data_v2/` | E2 | IDSSE＋SoccerTrack-v2 canonical 数据与 provenance（只读） |
| `phase4_amr/` | E3 | AMR v1–v6 训练产物（方法线已放弃） |
| `calligraphy_pilot/`、`remote/` | E0 | 书法试点与远端运行收据（EXPLORATORY_FROZEN） |
