# P2-H1 有界异质性诊断

- 所属阶段：`P2_NOVELTY_DISCRIMINATOR`
- checkpoint：`P2_H1_HETEROGENEITY_DIAGNOSIS_COMPLETE`
- 来源：`artifacts/phase2/p2_fracture_continuity_v1/statistics/`
- 诊断性质：冻结 P2 fracture 结果的描述性再分析，不重新推理模型
- 当前科学路由：`mixed_or_graph_specific`

## 目的和边界

P2 fracture continuity 已经完成，但总体结果不是统一方向。H1 只使用已完成的 match-level、model-match、LOMO 和 pair-effect 表，检查异质性来自哪些条件，并检查效应是否随 response-blind matching residual 改变。

本诊断不使用固定阈值或 p-value 作为升级到 P3 的门槛；比赛、模型 seed 和 intervention draw 也没有被当作独立重复。residual adjustment 只是线性敏感性检查，不替代 paired estimand。

## 主要结果

- 共分析 36 个 `architecture × graph × role × epsilon` 条件：20 个总体正向，16 个总体负向，没有零方向条件。
- leave-one-match-out 后的同号比例中位数为 `1.0`，最低为 `0.8`；30/36 个条件的 LOMO 结果与全量方向完全同号。方向具有一定跨比赛稳定性，但这不消除架构、图和角色依赖。
- 模型 seed 方向完全一致的条件占 `83.3%`；normalized SCG 的 seed range 中位数为 `3.91e-05`，最大为 `4.23e-04`。因此不能把 architecture-averaged 结果当作所有 seed 都一致。
- 描述性因素范围的中位数为：role `1.29e-04`、architecture `9.27e-05`、epsilon `1.96e-05`、graph `1.37e-05`。因素水平数不同，这些数值只用于定位异质性，不构成严格的方差贡献排序。

| 条件 | normalized SCG | match positive fraction |
|---|---:|---:|
| 最大正向：Phase-GAT / kNN-4 / forward / ε=0.5 | `2.07743e-04` | `0.7` |
| 最大正向：GAT-small-AE / Delaunay / midfielder / ε=0.5 | `1.83557e-04` | `1.0` |
| 最大负向：Phase-GAT / Delaunay / defender / ε=0.5 | `-9.64761e-05` | `0.5` |
| 最大负向：GAT-small-AE / Delaunay / forward / ε=0.5 | `-6.40438e-05` | `0.2` |

## residual 敏感性

108 个 `model × graph × role × epsilon` strata 的 raw SCG 与 residual 的相关性为：

- frequency residual：绝对相关系数中位数 `0.0316`，最大 `0.1315`；
- topology residual：绝对相关系数中位数 `0.0479`，最大 `0.3484`。

最大 topology residual 相关性出现在 `gat_ae_seed47 / kNN-4 / midfielder / ε=0.5`。因此“整体相关性较小”不能被写成“所有条件都不受 residual 影响”。

对两个 residual 做简单线性敏感性调整时，108/108 个 strata 的设计矩阵为满秩；normalized SCG 在中位 residual 处的绝对调整量中位数为 `4.59e-06`，最大为 `1.01e-04`。该结果支持保留 residual 作为明确限制，而不是把调整后的数值当作新的主结论。

## 科学路由

H1 不改变 `mixed_or_graph_specific` 路由。现有证据更适合表述为：冻结点集编码器对同向量集端点重分配的响应在部分角色和架构中具有跨比赛稳定性，但其幅度和方向明显依赖模型架构、图结构和角色；它还不能支持统一的 endpoint-organization 因果机制。

因此当前不直接启动 P3 causal-mechanism matrix、AMR 训练或中国书法复现。若后续仍要进入 P3，应先明确只检验一个由上述异质性结果支持的最小机制，而不是扩大模型或数据范围。

## 产物

- 实现：[scripts/p2_heterogeneity_diagnosis.py](/home/huyudi/012_conference/iclr2027/scripts/p2_heterogeneity_diagnosis.py)
- 定向测试：[tests/test_p2_heterogeneity_diagnosis.py](/home/huyudi/012_conference/iclr2027/tests/test_p2_heterogeneity_diagnosis.py)
- 汇总报告：[artifacts/phase2/p2_heterogeneity_diagnosis_v1/report.md](/home/huyudi/012_conference/iclr2027/artifacts/phase2/p2_heterogeneity_diagnosis_v1/report.md)
- 条件剖面：`artifacts/phase2/p2_heterogeneity_diagnosis_v1/condition_profile.csv`
- 因素对比：`artifacts/phase2/p2_heterogeneity_diagnosis_v1/factor_contrasts.csv`
- residual 敏感性：`artifacts/phase2/p2_heterogeneity_diagnosis_v1/residual_sensitivity.csv` 和 `residual_adjustment.csv`
- 可视化：`artifacts/phase2/p2_heterogeneity_diagnosis_v1/figures/heterogeneity_heatmap.png`
- provenance：`artifacts/phase2/p2_heterogeneity_diagnosis_v1/source_provenance.json` 和 `MANIFEST_SHA256.txt`

本任务只读取足球点集实验的既有产物；未下载新数据，未运行视觉模型，未读取或更新任何生信索引。
