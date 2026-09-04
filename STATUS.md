# Current Status

更新时间：2026-09-04

## 当前路线

- active_phase: P3_CAUSAL_MECHANISM
- current_checkpoint: P3_T5R4_ROUND2_COMPLETE
- current_route: support_conditioned_geometry_with_task_semantic_repair
- current_phase_status: P3_T5R4_BOUNDED_AUTORESEARCH_CLOSED_READY_FOR_T5R5_LOCK
- p4_status: blocked_pending_p3_t5r_gate
- p5_status: blocked_pending_sports_prediction_lock
- next_action: t5r5_candidate_lock_review

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

P3-T5R 允许在既有阶段内引入经批准的独立体育数据，但不访问、修改或重建 infra/bioinf-data-index/，不扩大到篮球、视觉模型或新的书法数据。当前固定使用 IDSSE 暂代 SNGAR 的开发数据角色；7 场 IDSSE 不伪装成 SNGAR 的 45/9/10 场划分，同一批数据不能同时充当开发集和独立外部确认集。正式计算若遇到容量问题，先记录 CPU profiling；未获得证据前不申请 GPU。

## 升级/停止条件

- 只有 retrospective + prospective 预测、layer/block 定位、一个 matched-capacity 因果开关和客观任务联系同时闭合，才进入 P4 AMR-Fixed。
- 只有 diagonal geometry 有效时，claim 收缩为节点/支持敏感度各向异性。
- retrospective 有效但 prospective 失败、或 response 可控但任务无收益时，不进入 AMR。
- geometry 与允许的 integrated metric 均不能预测时，停止当前机制复杂度，保留 P2 的 mixed_or_graph_specific。
- 当前实际路由：旧 causal switch 仍为 response shaping only；T5R0–T5R3 已完成，T5R4 Round 1 已完成待审阅，IDSSE 暂代 SNGAR 的开发分支已固定，P4 继续 blocked。

## P3-T5R 当前状态（2026-09-04）

- T5R0：协议/文档迁移，已完成；
- T5R1：IDSSE 暂代 SNGAR 的开发数据，官方页面、raw manifest、SHA-256、canonical conversion 和逐场 QC 已完成；
- T5R2：context/intrinsic task、match-level split、自然配对规则和 baseline/metric lock 已闭合；任务审计通过；
- T5R3：fixed dual-channel sanity 已完成；dual 相对 raw 满足方向性 sanity 条件，但 geometry 指标是 natural-pair proxy，不能写成旧 P3 intervention-response 复现；
- T5R4：bounded autoresearch 两轮已完成并关闭；Round 2（encoder update balance，3:1/2:1，总步数 420 持平）按训练前冻结规则选出 `update_ratio_2to1`（geometry 3 seeds 同方向改善、pair/context 保持），T5R3 不再是 incumbent；candidate lock、reserved match 和外部确认仍未运行；
- T5R5/T5R6：candidate lock、IDSSE 同源保留 match 检查与独立 provider confirmation，未运行。

本轮目标是判断任务语义修复是否能把 geometry/response shaping 转化为客观 task–robustness Pareto；T5R3 closure review 已允许 T5R4 Round 1，Round 1 已完成但无全维度 Pareto 支配者，尚未形成 P4 或外部确认 claim。

## IDSSE 临时替代说明

IDSSE 当前只承担 T5R 的开发输入。官方 Hugging Face `main` 文件树（提交 `a715a38dfbaf5f58e431727c2b78d174101a703c`）列出的 7 场文件与本地 raw 文件 ID 一致；数据卡正文仍列出 `J03WPF`、`J03WQF`，这一文档差异已保留，不能据此改写 raw。由于 IDSSE 只有 7 场且没有等同于 SNGAR 的官方隐藏测试划分，必须在训练前锁定比赛级开发、验证和保留方案；保留比赛只能作为同源未见检查，不能写成独立 provider confirmation。SNGAR 恢复和真正独立确认仍记录在 [TODO.md](TODO.md) 的分支记录中。

T5R2 已锁定：train=`J03WMX,J03WOH,J03WPY,J03WR9`，valid=`J03WN1,J03WOY`，reserved holdout=`J03WQQ`。训练/验证任务视图分别为 23,052/6,127 个 snapshot，自然配对分别为 3,750/980；reserved match 未加载，candidate lock 仍是测试前置条件。原始哈希复核为 23/23 通过，IDSSE T5R2 结构审计为 `PASS`。权威 artifacts 为 `artifacts/data_v2/idsse/` 和 `artifacts/phase3/task_semantic_repair_v1/`；T5R3 v3 已完成并通过独立审计，结果仅为方向性 sanity pass。

T5R3 已完成：4 个 neural variants × 3 个 seeds，所有 neural models 为 141,769 参数；dual valid natural-pair ranking accuracy 均值 0.9487，raw baseline 为 0.8030；dual valid context 的 phase/field-zone macro-F1 为 0.5382/0.9619，centroid MAE 为 0.0554；`z_mode` translation response 约为 0。结果只说明可进入 T5R4 审阅，不说明 P4 已放行。

T5R4 Round 2 已完成（`update_ratio_2to1` 入选，autoresearch 关闭），结果见 `reports/P3_T5R4_ROUND2_REPORT.md`。
