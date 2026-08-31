# Current Status

更新时间：2026-09-01

## 当前路线

- active_phase: P3_CAUSAL_MECHANISM
- current_checkpoint: P2_H1_HETEROGENEITY_DIAGNOSIS_COMPLETE
- current_route: mixed_or_graph_specific
- current_phase_status: P3_DOCUMENT_MIGRATION_COMPLETE_T0_NOT_RUN
- next_action: P3-T0 接口、provenance 与数值等价 CPU smoke

P2 已冻结：rigid formal v2、独立 fracture continuity 和 P2-H1 异质性诊断均已完成。P2 结论是 representation response，不是下游任务性能；20/36 条件正向、16/36 条件负向，方向依赖 architecture、graph、role、energy。该结果不被重算、覆盖或改写。

## P3 当前证据等级

| 对象 | 状态 | 说明 |
|---|---|---|
| support-conditioned local geometry 机制 | NOT_TESTED | P3-G1–G5 是 post-P2 工作假设 |
| normalized embedding Jacobian/JVP 接口 | DESIGN_LOCKED_NOT_RUN | 代码和 T0 receipt 尚未产生 |
| retrospective geometry prediction | NOT_RUN | 不得从 P2 response 倒推结论 |
| prospective intervention | NOT_RUN | 需在 T1 公式、层和指标冻结后 response-blind 生成 |
| causal switch | NOT_SELECTED | 只有 T3 gate 后选择一个 |
| task-geometry alignment | NOT_RUN | 任务定义须先冻结 |
| 书法路线 | EXPLORATORY_FROZEN | 不用于选择体育端机制；本轮不调参 |
| P4 AMR | BLOCKED_BY_P3_GATE | 不直接训练 AMR |

## 冻结资产与权威路径

- 250 个 canonical samples：artifacts/phase1/canonical_samples.npz
- 9 个冻结模型与 checkpoint：artifacts/phase1/models/
- P2 唯一权威输出：artifacts/phase2/p2_fracture_continuity_v1/
- P2-H1 diagnosis：artifacts/phase2/p2_heterogeneity_diagnosis_v1/
- P3 design lock：configs/phase3_support_geometry_v1.yaml
- P3 主报告：reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md

关键 provenance、版本和 hash 见 P3 config 与 reports/P3_DOCUMENT_MIGRATION.md。P2 报告、manifest、checksum 和历史 claim 只读。

## 范围边界

本轮不新增数据，不访问、修改或重建 infra/bioinf-data-index/，不扩大到篮球、视觉模型或新的书法数据。正式计算若遇到容量问题，先记录 CPU profiling；未获得证据前不申请 GPU。

## 升级/停止条件

- 只有 retrospective + prospective 预测、layer/block 定位、一个 matched-capacity 因果开关和客观任务联系同时闭合，才进入 P4 AMR-Fixed。
- 只有 diagonal geometry 有效时，claim 收缩为节点/支持敏感度各向异性。
- retrospective 有效但 prospective 失败、或 response 可控但任务无收益时，不进入 AMR。
- geometry 与允许的 integrated metric 均不能预测时，停止当前机制复杂度，保留 P2 的 mixed_or_graph_specific。
