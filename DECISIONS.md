# Decisions

## D-20260901-P3-001：P2 收口后的路线

- 日期：2026-09-01
- 决定：将当前 active phase 设为既有 P3_CAUSAL_MECHANISM；不增加 P2.5/P3a/P3b，不继续调 P2，不直接训练 AMR。
- 原因：P2 exact-spectrum 与 fracture continuity 已证明普通谱功率不足，但异质性为 mixed_or_graph_specific，方向依赖条件。
- 受影响文件：README.md、MASTER_AGENT_PROMPT.md、configs/project.yaml、configs/experiment_matrix.yaml、P3 canonical docs。

## D-20260901-P3-002：中心机制对象

- 日期：2026-09-01
- 决定：保留 Action-Mode Spectrum 作为边缘汇总，新增待检验的 support-conditioned normalized-embedding local geometry：
  q_f(X, delta)=1/2 ||J_f(X) delta||_2^2，G_f=J_f^T J_f。
- 原因：P2 fracture 在保持非零位移 vector multiset 时重分配端点支持，正好可检验 diagonal 与 off-diagonal block。
- 当前解释：工作假设，不是 P3 结果；role 仅为诊断分层，不是通用方法输入。

## D-20260901-P3-003：最小施工路径

- 日期：2026-09-01
- 决定：按 T0 接口/provenance → T1 retrospective prediction → T2 prospective → T3 layer/block → T4 单一因果开关 → T5 任务联系执行；完整 augmentation × pooling × constraint matrix 仅为 gate 后扩展。
- 原因：避免大型后验可学习预测器、结果驱动选 support/layer，以及先训练 AMR 再找解释。
- 受影响文件：docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md、docs/05_AGENT_EXECUTION_MANUAL.md、configs/phase3_support_geometry_v1.yaml。

## D-20260901-P3-004：统计与跨域边界

- 日期：2026-09-01
- 决定：统计单位为 source match；frame、pair、draw、seed 不是独立比赛重复。书法已有正式结果继续标记 exploratory，不用于选择体育端机制；本轮不新增外部数据。
- 原因：保持 P2 provenance 和防止跨域/结果选择泄漏。
- 受影响文件：QA.md、STATUS.md、CLAIM_LEDGER.md、P3 config。

## D-20260901-P3-005：当前不一致项审计

- 日期：2026-09-01
- 结果：canonical source docs、README、MASTER prompt 和机器配置已切换到 P3；PROJECT_PACKAGE_CONSOLIDATED.md 的旧正文保留为历史合并内容，并以文件顶部新增的 P3 synchronization addendum 明确当前状态和优先级。P2 reports/artifacts 未改动。
- 备注：在生成新的正式 P3 结果后，必须再次同步 consolidated package 和 claim ledger；不得用 addendum 改写 P2 历史。
