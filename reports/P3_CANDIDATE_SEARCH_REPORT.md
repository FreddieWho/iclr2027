# P3 Candidate Search Report — Task–Geometry Alignment

状态：`NOT_SUPPORTED`（保留 `team_mean` incumbent，不启动 P4）

## 1 搜索协议

- 依据 `AUTORESEARCH_P3_GOAL_PROMPT.md`，heldout 标签禁止用于候选选择、超参数、layer/epsilon/architecture/support 选择及早停。
- 复用 `artifacts/phase1/canonical_samples.jsonl` 的 train 113 / dev 37 / heldout 32 划分（非 heldout 搜索，heldout 仅冻结后审计）。
- seeds 11、23、47；每个候选单独训练 Phase-GAT 80 epochs、Adam 1e-3、CPU；matched capacity 149639 参数（见 `artifacts/phase3/candidate_search_v1/task_*.parquet` 的 `parameter_count`）。
- 一次只改一个机制轴：constraint placement（全局平移不变性，输入中心化 `positions - mean(positions, dim=1)`）或 pooling（`team_mean` vs `relational_pairwise`）；`centered_relational` 为两轴组合，仅作参考，不计入单轴门槛。
- 排序：lexicographic `dev/grouped macro-F1 ↑`、`context response ↓`、`geometry Spearman ↑`，权重事后不调整。
- 产物：`artifacts/phase3/candidate_search_v1/`（`ranking.csv`、`task_*.parquet`、`context_*.parquet`、`comparison_vs_team_mean.json`、`manifest.json`、`SHA256SUMS`），代码 `scripts/p3_candidate_search.py` sha256 `102d7afe8`，config `configs/phase3_support_geometry_v1.yaml` sha256 `e2842cea`。

## 2 搜索期（非 heldout）汇总

| candidate | dev macro-F1 mean | dev 95% CI (seed bootstrap, n=3) | context mean | geom Spearman mean | 单轴 vs incumbent |
|---|---:|---|---|---:|---|
| `team_mean` (incumbent) | 0.339243 | [0.3044, 0.3633] | 0.013235 | 0.738346 | — |
| `centered_team_mean` | 0.370199 | [0.3439, 0.4176] | ~0 (1.16e-13) | 0.728018 | constraint 1 轴 |
| `centered_relational` | 0.352027 | [0.3409, 0.3582] | ~0 (1.21e-13) | 0.746236 | 2 轴组合 |
| `relational_pairwise` | 0.355579 | [0.3355, 0.3721] | 0.015705 | 0.845307 | pooling 1 轴 |
| `team_centered` | 0.069971 | — | ~0 | 0.015269 | 破坏性对照 |

Lexicographic 排序：`centered_team_mean` > `relational_pairwise` > `centered_relational` > `team_mean` > `team_centered`。

Pareto 三条件（dev↑ 且 context↓ 且 geom≥0.738）：

- `centered_team_mean`：dev↑、context↓，但 geom 0.728 < 0.738（-0.010），且 per-seed geom 三 seed 均为 0.74/0.74/0.69 低于 incumbent 的 0.77/0.75/0.68 均值，不满足“保持或提高”。
- `relational_pairwise`：dev↑、geom↑，但 context 0.0157 > 0.0132，不满足。
- `centered_relational`：三条件均满足（dev +0.0128、context ~0、geom +0.0079），但 per-seed dev 提升仅 seed 11 (+0.052)，seed 23 (-0.009)、seed 47 (-0.005) 均略降，改善由单 seed 驱动；且为 2 轴组合，不符合“一次一轴”冻结门槛。
- `team_centered`：dev 大幅下降、geom 崩溃。

因此**没有单轴候选同时满足 dev 稳定提升 + context 降低 + geom 保持**。

## 3 冻结后 heldout 审计（仅用于报告，不参与选择）

| candidate | heldout macro-F1 mean | per-seed heldout | vs incumbent 0.186120 |
|---|---:|---|---|
| `centered_relational` | 0.188965 | 0.1401 / 0.1264 / 0.3004 | +0.0028（由 seed 47 单点驱动，2/3 seed 低于 incumbent） |
| `centered_team_mean` | 0.139307 | 0.1281 / 0.1952 / 0.0945 | -0.0468 |
| `relational_pairwise` | 0.183934 | 0.1714 / 0.1879 / 0.1925 | -0.0022 |

若以 heldout 审计的 incumbent 门槛（heldout >0.18612、context ≤0.013235、geom ≥0.738）为准，`centered_relational` 的 heldout 均值虽略高于门槛，但 2/3 seed 低于门槛且依赖单 outlier，不满足“改善不能只由单个 seed 或少数样本驱动”。`centered_team_mean` 明显低于门槛。

## 4 失败条件与诊断

- `relational_pairwise` 证实 P3 已有结论：几何预测力提升（0.738→0.845）可与任务/鲁棒性背离，属 `RESPONSE_SHAPING_ONLY`。
- 输入中心化候选通过硬编码零化全局平移响应（context ~0），但 dev 增益不稳、geom 未显著提升或略降，heldout 无一致改善。
- `team_centered`（对 team 内 h 减均值后再池化）使分类坍缩，确认 pooling 信息丢失需团队级统计而非零均值残差。

**判定**：

> 表示塑形成功，但 task repair 未支持。 / NOT SUPPORTED

按指令：保留 `team_mean` incumbent，不扩大搜索矩阵，不新增数据/模型动物园，不修改 P2 冻结资产，不启动 P4 AMR。

## 5 所需输出清单（已生成）

- 配置与代码版本：`configs/phase3_support_geometry_v1.yaml`、`scripts/p3_candidate_search.py`（sha 见 §1，`candidate_search_v1/manifest.json`）。
- 每个 seed 的 train/dev/heldout macro-F1：`task_*.parquet`。
- 冻结候选的 heldout（本轮无冻结，仅审计）：`comparison_vs_team_mean.json` 的 `heldout_means`。
- 每个 seed 的 context mean/median：`context_*.parquet`（含 median）。
- geometry Spearman：`ranking.csv` 的 `geom_mean`（per-seed 详见搜索日志：`team_mean` 0.7762/0.7559/0.6829、`centered_relational` 0.7729/0.7625/0.7033 等）。
- match-level bootstrap：dev 均值的 seed bootstrap 95% CI 见上表；geometry 的 match-level bootstrap 需 prospective pair 产物，本轮 search 未重跑 pair geometry，已在 `selected_causal_switch_v1` 中报告。
- 参数量与训练成本：149639 参数，80 epochs × 3 seeds × 5 candidates，CPU。
- manifest / provenance / SHA-256：`manifest.json`、`SHA256SUMS`。
- 与 `team_mean` 差值：`comparison_vs_team_mean.json` 的 `dev_delta_vs_team_mean` / `context_delta_vs_team_mean` / `geom_delta_vs_team_mean`。
- Pareto 判定：`pareto_all_three` 仅 `centered_relational` 为 true（但为 2 轴组合且 seed 不一致，不予冻结）。

## 6 限制

- Prospective geometry 的 match-level 统计单位 (n=10 source matches) 较小，CIs 宽；已在 `reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md` 保留。
- Context 的“~0”由输入中心化构造性保证，非学习所得鲁棒性。
- Heldout 仅 32 样本，单 seed outlier 可翻转均值，故以 seed 一致性约束为准。
