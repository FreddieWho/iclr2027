# ICLR 2027 — Compositional and Transitional Blind Spots

> **当前状态（2026-09-22 起）：EXPLORATORY PHASE CLOSED，PAPER POLISH 阶段。**
> 终局裁决见 `reports/final_closure/FINAL_PROJECT_VERDICT.md`。
> 全文截止 2026-09-25 AOE。新方向需用户单独授权。

## 这个项目最终是什么

最终论文：**"Knowing the Parts Does Not Mean Knowing the Change:
Compositional and Transitional Blind Spots in Learned Representations"**
（`paper/main.tex`，双盲匿名）。

一句话故事：**模型此刻可以是对的，却在意含变化时不会更新，并且看起来被修复了
却没有恢复一致的能力。** 静态置信、端点正确、完全组合一致性和下游成功，
是"变化下的可靠性"的四个不同概念。

三大贡献（终局裁决 #10）：
1. **P1 置信反转**——高置信样本在真变化下错误率反而最高（坐标域，确认池 CONFIRMED）；
2. **P3 错误迁移主导**——修复把错误从组合失效迁移到别处，relflip 转向 full consistency；
3. **P2 能力/操作点分离**——覆盖而非精度主导，规则依赖。

Headline：原子编辑都判对时，联合组合变化在 fresh holdout 上 151/176（85.8%）漏检；
单语义边界穿越漏检 43.8%→修复后 31.3%；置信最高组更新错误率 0.98–1.00。

## 论文与数字的唯一权威来源

| 用途 | 唯一权威文件 | 说明 |
|---|---|---|
| 论文正文 | `paper/main.tex` + `paper/sections/` | 已迁移到最终叙事（02 受控设定→08 可复现性） |
| 正文每个数字 | `reports/final_closure/MASTER_CLAIM_LEDGER.md` | the ONLY source for paper numbers |
| 证据总表 | `reports/final_closure/FINAL_EVIDENCE_TABLE.md` | 2026-09-22 重建版，取代根下旧表 |
| 终局裁决 | `reports/final_closure/FINAL_PROJECT_VERDICT.md` | 15 问 15 答，含已删除的旧贡献清单 |
| 一键复算 | `reports/final_closure/REPRODUCE_FINAL.md` | 每个正文数字的重算命令 |
| P1/P2/P3 收官 | `reports/final_closure/P{1,2,3}_CLOSURE_REPORT.md` | 各线终局报告 |
| 纠错记录 | `reports/final_closure/EVIDENCE_CORRECTIONS_FINAL.md` | 8 项已确认纠错（旧数作废在此登记） |

⚠️ 根目录 `CLAIM_LEDGER.md`、`reports/FINAL_EVIDENCE_TABLE.md`（9-18 版）、
`paper/EVIDENCE_MAP.md` 均为**历史版本**，措辞与数字以 final_closure 为准。

## 推荐阅读顺序

**Reviewer / 快速了解（30 分钟）：**
1. 本文件 → 2. `reports/final_closure/FINAL_PROJECT_VERDICT.md`（终局 15 问）
→ 3. `paper/sections/00_abstract.tex` + `01_introduction.tex`
→ 4. `reports/final_closure/MASTER_CLAIM_LEDGER.md`（每个数字的分母与边界）。

**新 agent 接手（半天）：**
1. 本文件 → 2. `STATUS.md`（当前状态）→ 3. `PROJECT_MAP.md`（叙事→目录全映射）
→ 4. `reports/final_closure/REPRODUCE_FINAL.md`（复算链）
→ 5. `TODO.md` 顶部状态块与文末变更记录（完整行动史）。

## 目录速览

| 目录 | 角色 | 状态 |
|---|---|---|
| `paper/` | 最终论文（LaTeX） | **当前** |
| `reports/final_closure/` | 终局权威包（ verdict/ledger/证据表/复算/纠错 ） | **当前** |
| `experiments/` | 全部实验代码（16 个子目录，按时代） | 当前+历史混合，见 `experiments/INDEX.md` |
| `artifacts/` | 全部实验产物（23 个子目录，36G 本地） | 当前+历史混合，见 `artifacts/INDEX.md` |
| `reports/` | 全部报告（按时代），索引见 `reports/INDEX.md` | 当前+历史混合 |
| `docs/` | 各时代方案包与蓝图文档，索引见 `docs/INDEX.md` | 历史为主 |
| `docs/legacy/` | 根目录移出的历史一次性文件 | 历史 |
| `configs/` | 各期 config 锁与 data manifest | 历史冻结 |
| `scripts/`、`tests/` | 各期脚本与单测（P0→终局全跨度） | 历史冻结+当前 |
| `data/` | 原始数据（48G，gitignore） | 本地 |
| `submission/` | ⚠️ **另一个项目（Memory Pilot）误入，与本论文无关，已移出 git 跟踪** | 忽略 |
| `templates/` | 报告模板 | 工具 |

治理文档（项目根）：`PLAN.md` / `ROADMAP.md` / `CLAIM_LEDGER.md` 保留为
**历史档案**（顶部有横幅标注其对应时代）；当前状态以 `STATUS.md`、
`TODO.md`、`reports/final_closure/` 为准。

## 为什么会有这么多"旧线"目录（路线调整简史）

本项目是探索性研究，经历了六次路线调整，最终叙事在 2026-09-18 才确立。
每个时代的产物全部保留（provenance 纪律），按时间排列：

| 时代 | 时间 | 主线 | 结局 |
|---|---|---|---|
| 0 蓝图 | 08-18→08-31 | Action-Mode Spectrum 立项（体育阵型+书法+AMR 方法） | P0/P1 完成 |
| 1 P2 | 08-31→09-01 | fracture continuity 测量 | mixed_or_graph_specific |
| 2 P3-T5R | 09-01→09-05 | support-conditioned 几何可预测+任务语义修复+SoccerTrack 外部确认 | CONFIRMED，但仅是表示塑形 |
| 3 P4-AMR | 09-05→09-16 | AMR 方法线 v1–v6 | 15 连败后 H1 过但≈CAP，方法位放弃 |
| 4 转折 | 09-17→09-18 | discovery campaign→last15h→next6：**发现更新失败现象**，85.8% headline，flipmine 修复 | **最终叙事诞生** |
| 5 外部确认 | 09-18→09-20 | foundation model（DINOv2/v3/CLIP/SigLIP2）桥接 | 全阴性，进论文 §6 |
| 6 终局 | 09-21→09-23 | P1/P2/P3 三线收官+确认池 | **EXPLORATORY CLOSED** |

各时代目录归属详见 `PROJECT_MAP.md`。旧 Action-Mode/Jacobian/AMR 线在论文中
退为 §7 半段背景，相关证据仍有效（被引用处见 ledger），但不构成主线。

## 不要先读什么（常见的坑）

- ❌ `docs/01–10` 蓝图文档：立项时的 AMR/书法叙事，**已被路线调整取代**（历史价值在 `docs/INDEX.md` 说明）。
- ❌ `docs/legacy/PROJECT_PACKAGE_CONSOLIDATED.md`：旧方案包合并版。
- ❌ `STATUS.md` 以外的任何"P4 AMR 进行中"描述：P4 已终结（v6≈CAP，方法位放弃）。
- ❌ `submission/`：另一个项目。
- ❌ 任何单一目录名猜测归属：`radius_loss`、`l002_ball`、`event_updater` 等均为
  终局期的已关闭旁路，先看 `PROJECT_MAP.md` 再读。

## 硬纪律（任何新工作必须遵守）

1. 正文数字只能来自 `reports/final_closure/MASTER_CLAIM_LEDGER.md`；
   discovery 数与 fresh/confirm 数不得混用分母。
2. `holdout_909`（坐标）、`confirm_1007`（P1 池）、`holdout_895`（bridge）均已
   **单次消耗并封存**；任何二次读取是新的协议事件，需用户显式授权。
3. 禁开方向清单见 `TODO.md` 顶部（AMR/Jacobian 扩展/E7 新 loss/第 61 个小 MLP 等）。
4. 大文件（.pt/.npy/.npz/.parquet）按 `.gitignore` 留本地；sha 校验见各 artifact manifest。
