# P2 匹配优化、接线验证与资源门结果

日期：2026-08-31  
范围：P2 matching v2、P2-C DeepSets wiring smoke、P2-D 运行前资源门  
研究模式：开放式探索研究；无预注册或预设停止规则

> **状态更新（2026-08-31）：** 本文主体记录正式运行前的 matching/P2-C/资源门历史状态。provenance-locked 的 250 样本 × 9 模型 rigid formal v2 已关闭 `P2_RIGID_C2_COMPLETE`；fracture continuity 随后以独立 operator 完成，关闭为 `P2_FRACTURE_C2_COMPLETE`，探索性解释为 `mixed_or_graph_specific`。权威 fracture 结果见 `reports/EXPLORATION_CHECKPOINT_2_FRACTURE.md`；本文其余 fracture 未运行描述均为历史记录。

## 1. 当前结论

本轮实现已完成，但没有产生 P2 科学结论：

- matcher 已产生可解释的探索版本：`MATCHER_VERSION_SELECTED`；
- P2-C 已通过工程接线：`WIRING_ONLY_NOT_SCIENTIFIC`；
- P2-D 判断本地 CPU 足够，GPU 当前没有必要；
- 250 样本 × 9 模型正式响应运行仍为 `NOT_RUN`；
- `epsilon=2.0` 压力测试仍保留，其大量边界无效没有被删除或改写；
- 未读取、更新或重建任何生信索引。

## 2. Matching v2 做了什么

实现包括：

- 10 场比赛每场确定性抽取 2 帧，共 20 个样本；
- 候选几何和谱特征按 `sample × graph × coalition × candidate` 只计算一次；
- 能量和方向层只重算边界可行性；
- 新增稳定 `candidate_id`、边界余量、候选复用、失败分类和覆盖—平衡曲线；
- exact-spectrum 最多进行 32 次确定性符号重采样，每次保持逐模态功率；
- 诊断 schema 拒绝 model、embedding、response、SCG 等字段，防止响应泄漏到 matcher。

20 样本 response-blind 诊断显示，在 Rayleigh 差仍限制为 `0.05`、语义重叠仍不超过 `0.5` 时，将六频段功率比例 L1 从 `0.20` 调整到 `0.30`，低能量共同支持从 617/1824 增至 1260/1824。这个选择没有读取任何模型输出。

按方案规定先以 frequency ratio、再以 topology 距离排序后，低能量 selected control 的平均 Rayleigh 差由 `0.012074` 变为 `0.012359`；平均 band L1 由 `0.152016` 增至 `0.200934`；平均 topology match score 由 `1.955936` 变为 `2.035500`。因此这里的收益明确是共同支持增加，代价是频率和拓扑残差均小幅变宽；完整残差均保存在候选与 selected-control 表中。

选定配置的 3648 个 matched set 中：

| 图 | epsilon | 完整集 | 总集 | 完整比例 |
|---|---:|---:|---:|---:|
| Delaunay | 0.25 | 324 | 456 | 71.1% |
| Delaunay | 0.50 | 301 | 456 | 66.0% |
| Delaunay | 1.00 | 219 | 456 | 48.0% |
| Delaunay | 2.00 | 48 | 456 | 10.5% |
| kNN-4 | 0.25 | 329 | 456 | 72.1% |
| kNN-4 | 0.50 | 306 | 456 | 67.1% |
| kNN-4 | 1.00 | 221 | 456 | 48.5% |
| kNN-4 | 2.00 | 42 | 456 | 9.2% |

总完整集为 1790/3648；低能量完整集为 1260/1824。最大能量误差为 `1.33e-14`，exact-spectrum 最大逐模态功率误差为 `0`，没有无效 probe graph。

## 3. P2-C 接线结果

P2-C 只使用 `deepsets_ae_seed11`：

- 模型由 P1 的 `build_torch_models()` 构造，没有复制或重写架构；
- checkpoint 通过 manifest SHA256 校验，并使用 `weights_only=True`、CPU、`eval()` 和 inference mode；
- 对 20 个选中样本重算 baseline，保存 embedding 与重算结果通过 float32 数值等价；最终运行逐元素完全相等，最大绝对差为 `0`，容差和 exact-equal 状态均写入 receipt；
- 只消费 `epsilon=0.25/0.5` 的 1260 个完整四臂集；
- 共生成 5040 条有限 response，四臂配对完整且唯一；
- 端到端用时 `13.80 s`，峰值 RSS `979 MiB`。

该结果只证明 checkpoint、baseline、干预、embedding 和 response 表可以正确连接。没有比较模型、没有统计效应、没有 SCG/C2 结论。

## 4. P2-D 资源门

有界 benchmark 使用每类一个 seed11 模型：DeepSets-AE、GAT-small-AE、Phase-GAT。GAT 两类始终读取 P1 canonical baseline weighted kNN-4 adjacency，未按扰动后坐标重建图。

按 20 样本中每样本低能量完整集密度外推到 250 样本：

- 预计 15750 个 matched set；
- 每模型约 63000 条四臂 response；
- 9 模型约 567000 条 response；
- 当前保守 CPU 时间外推约 2.3 分钟，response Parquet 约 12.3 MiB；共享机器上的短 benchmark 有波动，应把它理解为“数分钟量级”而非承诺时长；
- 顺序加载模型时峰值内存以实测约 1 GiB 为量级，不能把单模型峰值机械乘 9。

风险：20 样本的完整集密度未必能代表 250 样本；三架构只各测一个 seed；完整 250 样本 matching 成本未包含在 response 推理外推中；共享机器负载会影响微基准。即使按这些风险保守判断，本地 CPU 仍足够，当前不应租 GPU。

## 5. 下一步决策点

正式 250×9 运行前需要做的是探索分析范围选择，而不是购买算力。建议先确定：

1. 首个正式分析是否只从 `epsilon=0.25/0.5` 开始；
2. 主要描述是否先按 match 聚合，避免把同一比赛内大量 set 当成独立重复；
3. 首轮是否先运行 9 个点模型，图像模型继续保持旁证；
4. 如何并列报告完整集、缺失边界和 matcher 残差，而不把 `epsilon=2.0` 的缺失静默丢弃。

这些是开放式探索选择，不是预注册约束。作出选择后可直接本地推进；在此之前正式运行保持 `NOT_RUN`。

## 6. 证据路径

- 诊断配置：`configs/phase2_matching_v2.yaml`
- 选定配置：`configs/phase2_matching_v2_selected.yaml`
- response-blind 诊断：`artifacts/phase2/matching_v2_stratified_smoke/`
- 选定 matcher：`artifacts/phase2/matching_v2_selected_smoke/`
- P2-C：`artifacts/phase2/p2c_deepsets_seed11_smoke/`
- P2-D：`artifacts/phase2/p2d_resource_gate/`
- 实现：`scripts/p2_matched_controls.py`、`scripts/p2_matching_diagnostics.py`、`scripts/run_phase2_pipeline.py`、`scripts/p2_point_model_adapter.py`、`scripts/run_phase2_response_smoke.py`、`scripts/run_phase2_resource_gate.py`
- 测试：`tests/test_phase2_pipeline.py`、`tests/test_phase2_matching_optimization.py`、`tests/test_phase2_response_smoke.py`
