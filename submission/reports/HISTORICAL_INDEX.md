# 历史收据索引（只标注范围与 successor，不篡改原文）

- `artifacts/model_manifest.json`（2026-09-12，`BLOCKED_C1_LONG_CONTEXT…`）：continuation 前的技术阻塞快照。successor：`reports/V4_FAST_PILOT_REPORT.md`（P2 已完成 72 memories）。
- `artifacts/dataset_manifest.json`：同上阶段的数据快照；当前 P2 数据见 `reports/CURRENT_STATUS.json` 的 current_data。
- `artifacts/cost_summary.json`（全零/`NOT_RUN`）：旧预检占位，不是当前费用。当前费用：项目实付＋不确定 `$8.29376795`（已含 `$0.1529898` 预留），见 `reports/V4_ROUTE_DECISION.json` 的 resource_controls。
- `reports/decision.json`（`NO_GO/TECHNICAL`）：continuation 前的技术停机决策。successor：`reports/V4_ROUTE_DECISION.json`（`INCONCLUSIVE_NO_GO_SIGNAL`，2026-09-14）。
- `artifacts/v4_confirmation_rows.jsonl`、`artifacts/v4_exploration_rows.jsonl`：continuation 前 `NOT_RUN` 快照；当前行见 `artifacts/v4_confirmation_rows_compact_v2_20260914.jsonl` 与 `artifacts/v4_exploration_rows_owner_continuation_p1_20260914.jsonl`。
- C1/R1 runner `55eabd3a…`：原版源码快照 `UNAVAILABLE_ORIGINAL_RUNNER_SNAPSHOT`（待从 Git/备份只读查找）；当前源码对应 R2 `8f7afcbc…`，不得把当前版称为原 C1 精确版。
