# P2-H1 有界异质性诊断

- checkpoint: `P2_H1_HETEROGENEITY_DIAGNOSIS_COMPLETE`
- scientific route retained: `mixed_or_graph_specific`
- source: `artifacts/phase2/p2_fracture_continuity_v1/statistics`
- status: descriptive reanalysis only; no model inference was run

## 目的与边界

本诊断复用已经冻结的 P2 fracture continuity 统计产物，在比赛层级检查架构、图、角色和能量的异质性，并分别计算频率/拓扑匹配残差与效应的相关性。它不使用固定数值阈值或 p-value 作为升级到 P3 的门槛，也不把比赛、模型 seed 或 intervention draw 当作独立重复。

## 总览

- 条件数：36；residual strata：108。
- 条件方向：positive=20，negative=16，zero=0。
- LOMO 同号比例：median=1，min=0.8；其中精确为 1 的条件占 0.833333。
- 模型 seed 方向一致的条件占 0.833333；seed range 的 median/max 为 3.91014e-05/0.000423002。
- |frequency residual correlation|：median=0.0315643，max=0.1315。
- |topology residual correlation|：median=0.0478934，max=0.348405。
- 简单 residual adjustment 在完整秩 strata 中的 normalized SCG 中位绝对偏移为 4.58907e-06，最大为 0.000101452；该调整仅作敏感性检查。
- 最大 topology residual 相关性出现在 `gat_ae_seed47/knn4/midfielder/eps=0.5`，相关系数=0.348405；因此不能把整体中位数较小表述为所有条件均不受残差影响。

## 因素范围（描述性）

以下范围是在其他因素固定后，跨该因素水平的 normalized SCG 范围。不同因素的水平数不同，因此只作异质性定位，不宣称因素之间存在严格的方差贡献排序。

| factor | backgrounds | median range (normalized SCG) | max range | pairwise contrasts |
|---|---:|---:|---:|---:|
| architecture | 12 | 9.27087e-05 | 0.00026244 | 36 |
| epsilon | 18 | 1.96355e-05 | 9.99026e-05 | 18 |
| graph_id | 18 | 1.37406e-05 | 8.79857e-05 | 18 |
| role_group | 12 | 0.000128984 | 0.000252742 | 36 |

## 极端条件示例

最大正向条件：

- `Phase-GAT/knn4/forward/eps=0.5`: normalized SCG=0.000207743，match positive fraction=0.7。
- `GAT-small-AE/delaunay/midfielder/eps=0.5`: normalized SCG=0.000183557，match positive fraction=1。
- `GAT-small-AE/knn4/midfielder/eps=0.5`: normalized SCG=0.000127042，match positive fraction=1。

最大负向条件：

- `Phase-GAT/delaunay/defender/eps=0.5`: normalized SCG=-9.64761e-05，match positive fraction=0.5。
- `Phase-GAT/delaunay/defender/eps=0.25`: normalized SCG=-6.55644e-05，match positive fraction=0.4。
- `GAT-small-AE/delaunay/forward/eps=0.5`: normalized SCG=-6.40438e-05，match positive fraction=0.2。

## 路由解释

当前结果继续保留 `mixed_or_graph_specific`。本报告只把异质性结构显式化，不把某一条件的正负方向提升为普遍机制，也不因为 LOMO 同号就消除架构/图/角色依赖。是否进入 P3，应在阅读这些分层结果后作开放式研究决策。

## 产物

- `condition_profile.csv`：36 个架构×图×角色×能量条件。
- `factor_contrasts.csv`、`factor_ranges.csv`、`factor_summary.csv`：因素对比与描述性范围。
- `residual_sensitivity.csv`：108 个 model×图×角色×能量 residual strata，同时包含频率和拓扑相关性。
- `residual_adjustment.csv`：每个 residual stratum 的简单线性敏感性调整，仅用于诊断。
- `figures/heterogeneity_heatmap.png`、`figures/factor_ranges.png`：快速查看图。
- `diagnosis_summary.json`、`source_provenance.json`、`MANIFEST_SHA256.txt`：状态与 provenance。
