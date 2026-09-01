# Current Status

更新时间：2026-09-01

## 当前路线

- active_phase: P3_CAUSAL_MECHANISM
- current_checkpoint: P3_T4_T5_COMPLETE_RESPONSE_SHAPING_ONLY
- current_route: mixed_or_graph_specific
- current_phase_status: P3_T4_T5_COMPLETE_RESPONSE_SHAPING_ONLY_P4_GATE_NOT_MET
- next_action: 冻结 P3 claim，停止当前机制复杂度，不启动 P4 AMR

P2 已冻结：rigid formal v2、独立 fracture continuity 和 P2-H1 异质性诊断均已完成。P2 结论是 representation response，不是下游任务性能；20/36 条件正向、16/36 条件负向，方向依赖 architecture、graph、role、energy。该结果不被重算、覆盖或改写。

## P3 当前证据等级

| 对象 | 状态 | 说明 |
|---|---|---|
| support-conditioned local geometry 机制 | CONDITIONALLY_SUPPORTED | retrospective/prospective 均有预测力；因果开关只达表示塑形 |
| normalized embedding Jacobian/JVP 接口 | SUPPORTED_T0 | parity、JVP/有限差分和二阶 smoke 完成 |
| retrospective geometry prediction | SUPPORTED_T1 | full geometry pooled mean Spearman 0.8466，baseline 0.1222；match bootstrap 仍按 n=10 谨慎解释 |
| prospective intervention | SUPPORTED_T2 | 同资产、同模型、同 operator 的 response-blind 新 reallocation；full geometry mean Spearman 0.8526，baseline 0.1294 |
| causal switch | RESPONSE_SHAPING_ONLY | Phase-GAT pooling accessibility 改变 off-diagonal 比例和 response，但未闭合任务/鲁棒性门 |
| task-geometry alignment | NOT_SUPPORTED_AS_METHOD_REPAIR | nodewise support localization 仅接近 uniform；heldout macro-F1 均值略降，context response 三 seed 均上升 |
| 书法路线 | EXPLORATORY_FROZEN | 不用于选择体育端机制；本轮不调参 |
| P4 AMR | BLOCKED_BY_P3_GATE | 不直接训练 AMR |

## 冻结资产与权威路径

- 250 个 canonical samples：artifacts/phase1/canonical_samples.npz
- 9 个冻结模型与 checkpoint：artifacts/phase1/models/
- P2 唯一权威输出：artifacts/phase2/p2_fracture_continuity_v1/
- P2-H1 diagnosis：artifacts/phase2/p2_heterogeneity_diagnosis_v1/
- P3 design lock：configs/phase3_support_geometry_v1.yaml
- P3 主报告：reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md
- P3 候选搜索（autoresearch）：artifacts/phase3/candidate_search_v1/、reports/P3_CANDIDATE_SEARCH_REPORT.md（结论 NOT_SUPPORTED，保留 team_mean）

关键 provenance、版本和 hash 见 P3 config、各 phase3 artifact 的 receipt/SHA256SUMS 与 `reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md`。P2 报告、manifest、checksum 和历史 claim 只读。

## 范围边界

本轮不新增数据，不访问、修改或重建 infra/bioinf-data-index/，不扩大到篮球、视觉模型或新的书法数据。正式计算若遇到容量问题，先记录 CPU profiling；未获得证据前不申请 GPU。

## 升级/停止条件

- 只有 retrospective + prospective 预测、layer/block 定位、一个 matched-capacity 因果开关和客观任务联系同时闭合，才进入 P4 AMR-Fixed。
- 只有 diagonal geometry 有效时，claim 收缩为节点/支持敏感度各向异性。
- retrospective 有效但 prospective 失败、或 response 可控但任务无收益时，不进入 AMR。
- geometry 与允许的 integrated metric 均不能预测时，停止当前机制复杂度，保留 P2 的 mixed_or_graph_specific。
- 当前实际路由：geometry 预测和 anatomy 获支持，但 causal switch 未带来稳定任务—鲁棒性 Pareto，故保留 P2 route、停止机制堆叠，P4 blocked。
