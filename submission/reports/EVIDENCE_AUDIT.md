# EVIDENCE_AUDIT

> **范围说明：** 完整证据审计为 `NOT_RUN`（pilot 之后的工作），不等于工程缺陷。当前结论见 `reports/CURRENT_STATUS.json`。

状态：`NOT_RUN_FORMAL_EVIDENCE_AUDIT`

正式要求是至少 24 个可追到 gold+memory 的成功/失败 pair，并区分完整语义证据、reader 使用失败、缺失/扭曲、猜测和 unknown；还需按适用机制完成 highlight、restore 或等长 placebo 控制。本轮没有冻结任何 live memory，也没有生成 live prediction，因此：

- 可审计 formal pairs：`0/24`，不是零效应。
- retention/accessibility 分类：`NOT_RUN`。
- semantic evidence audit：`NOT_RUN`。
- highlight/restore/placebo：`NOT_RUN`。
- 失败类别反例与分母：`NOT_RUN`。

P0 synthetic gold 已在 query 文件中，LongMemEval gold 已在 imported query 文件中；它们只是审计输入准备，未反流到 compressor。`work/smoke_run` 中的所有 predictions/scores/steps 都带 `mock=true`，在本审计之外。

结论：证据审计门未关闭，不能把技术准备或 mock 结果升级成 retention、不可逆遗忘或 reader accessibility 结论。
