# Current Status

更新时间：2026-09-02

## 当前路线

- active_phase: P3_CAUSAL_MECHANISM
- current_checkpoint: P3_T5R0_DOCUMENT_MIGRATION_COMPLETE
- current_route: support_conditioned_geometry_with_task_semantic_repair
- current_phase_status: P3_T5R0_COMPLETE_T5R1_DATA_ACCESS_PENDING_P4_BLOCKED
- p4_status: blocked_pending_p3_t5r_gate
- p5_status: blocked_pending_sports_prediction_lock
- next_action: verify_sngar_train_valid_access_and_canonical_conversion

P2 已冻结：rigid formal v2、独立 fracture continuity 和 P2-H1 异质性诊断均已完成。P2 结论是 representation response，不是下游任务性能；20/36 条件正向、16/36 条件负向，方向依赖 architecture、graph、role、energy。该结果不被重算、覆盖或改写。

旧 candidate search 的 heldout 状态为 `EXPOSED_DURING_CANDIDATE_SEARCH`，允许用途仅为 `exploratory_audit_only`，禁止作为 final confirmation 或 candidate selection。旧 T5 阴性结果保留，但明确限定为 `NOT_SUPPORTED_UNDER_OLD_CONFLATED_TASK_GATE`。

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

P3-T5R 允许在既有阶段内引入经批准的独立体育数据，但不访问、修改或重建 infra/bioinf-data-index/，不扩大到篮球、视觉模型或新的书法数据。SNGAR 只按 train/valid 开发，test 需 candidate lock；IDSSE 只作外部确认。正式计算若遇到容量问题，先记录 CPU profiling；未获得证据前不申请 GPU。

## 升级/停止条件

- 只有 retrospective + prospective 预测、layer/block 定位、一个 matched-capacity 因果开关和客观任务联系同时闭合，才进入 P4 AMR-Fixed。
- 只有 diagonal geometry 有效时，claim 收缩为节点/支持敏感度各向异性。
- retrospective 有效但 prospective 失败、或 response 可控但任务无收益时，不进入 AMR。
- geometry 与允许的 integrated metric 均不能预测时，停止当前机制复杂度，保留 P2 的 mixed_or_graph_specific。
- 当前实际路由：旧 causal switch 仍为 response shaping only；T5R0 文档/协议迁移已完成，下一步仅验证 SNGAR train/valid 访问与 canonical conversion，P4 继续 blocked。

## P3-T5R 当前状态（2026-09-02）

- T5R0：协议/文档迁移，已完成；
- T5R1：SNGAR train/valid 获取与 canonical conversion，未运行；
- T5R2：context/intrinsic task 与 baseline lock，未运行；
- T5R3：fixed dual-channel sanity，未运行；
- T5R4：bounded autoresearch，未授权启动；
- T5R5/T5R6：candidate-lock test 与 IDSSE external confirmation，未运行。

本轮目标是判断任务语义修复是否能把 geometry/response shaping 转化为客观 task–robustness Pareto；任何 T5R claim 在实验前均为 `WORKING_HYPOTHESIS` 或 `NOT_ESTABLISHED`。
