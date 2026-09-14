# V3 状态：路线决策用途已被 V4 取代

V3 `visible_budget_preflight_v3` 自 2026-09-13 起冻结为 `INCOMPLETE_PROTOCOL_DEVIATION / SUPERSEDED_FOR_ROUTE_SELECTION`。V4 只在路线选择上取代 V3，不覆盖其历史证据。

- V3 的 40 个响应格中 4 个完成、1 个请求在等待响应时被本地中断、35 个未尝试；本地提前停止偏离当时冻结的停止规则。
- 中断请求仍保留 `$0.088638` 不确定费用预留；V3 已用与不确定费用合计 `$1.37211145`。
- 不续跑剩余 35 格，不重试中断的 1024 格，也不把 V3 输出并入 V4 统计。
- 原始 V3 文件、请求缓存、账本、配置与哈希均保持不变。V4 采用 thinking-disabled 的新压缩配置和实际长度敏感性分析。
- 旧 `reports/decision.json` 仍是历史 `NO_GO/TECHNICAL` 快照，不是 Memory 科学假设的阴性结果，也不作为 V4 终局。

权威依据：`docs/ICLR_MemoryPilot_V4_FastDecision_20260913/docs/05_MIGRATION_FROM_V3.md`、V3 部分结果及其 `run_03` manifest/账本。
