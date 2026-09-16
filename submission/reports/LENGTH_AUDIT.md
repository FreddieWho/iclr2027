# LENGTH_AUDIT

> **范围说明：** 严格等长 endpoint 实验为 `NOT_RUN`；P2 长度分析见主报告“压缩器与实际长度”与 rewrite/费用边界附录。

状态：`NOT_RUN_FORMAL_LENGTH_AUDIT`

2026-09-12：DeepSeek-V4.1-Flash、V4-Flash和 GLM C1 fallback 均使用固定 native tokenizer、`reasoning_effort=low` 和同一完整原始 history；V4.1 在 12000/16000 下、GLM 在 12000 下均 `visible memory=0/finish=length`，V4 在 16000 下经直连等待 600s 后无响应；V4.1 32000 首次返回 HTTP 500，重试后仍 `visible memory=0/finish=length`。没有任何 memory row 进入评分。见 `reports/REASONING_BUDGET_V2.md`。

历史单次授权诊断：原始32K输入、原始提示、provider output=24576/max，实测reasoning=24553、visible memory=0、finish_reason=length。该结果与 V2 结果均不进入评分，完整长度门仍未通过。见 `reports/P0_C1_24576_PROBE.md`。

2026-09-11技术修复：固定模型/max reasoning/10240输出的32K与16K探针均无visible memory、finish_reason=length；R1独立短答为4 native answer tokens/18 JSON tokens、finish_reason=stop。这些是校准而非正式长度门通过；详见 `reports/P0_REPAIR_20260911.md`。

## 已核验的输入长度

| 数据 | n history | tokenizer | canonical tokens | 结果 |
|---|---:|---|---:|---|
| synthetic P0 | 8 | `tiktoken:cl100k_base` | 32,722–32,753 | 可用于 calibration，不计推断 |
| LongMemEval cleaned 抽样 | 32 | `tiktoken:cl100k_base` | 112,845–118,608 | 完整保留；超过默认 32K 施工配置 |

`configs/pilot.json` 的目标是 `L=32,768`、`B∈{1,024,2,048}`；中间最大目标为 `4B`。这只证明计划可构建，不证明目标模型能看见自然 history。自然数据需要已验证的 128K 级窗口或协议允许的、与 H/Q 无关的完整短历史子集；本轮没有自行截断、删 session 或改用 oracle。

## 未执行项目

- C1 的 native tokenizer 已核验；API/native usage 仅有 technical probes：V5 在同一首个 32K synthetic history 上，2048 target 有一次 natural-stop 且正文 1801 tokens 的 pass；1024 target 正文 2104 tokens 超限。该单次 pass 不构成正式 length gate 或 90% compliance 证据。
- 正式每个压缩 step 的 actual visible memory、clip fraction 和 reader 长度：`NOT_RUN`；隐藏 reasoning 只作为阻塞证据，不作为 memory。
- `>5%` 输出被裁剪的技术门槛：`NOT_TESTABLE`。
- cap-matched 与共同实际长度 `t=min(actual lengths,B)` 重读：`NOT_RUN`。
- reader context 截断/服务商隐藏 compaction 证明：`NOT_RUN`。
- formal path 的 length-sensitive CI：`NOT_RUN`。

mock smoke 的 `work/smoke_contrast.json` 仅显示 `MOCK_ONLY`，不可用作长度或效应证据。

## 非正式 reasoning 长度诊断

2026-09-11 UTC 完成 10 次 `deepseek-v4.1-flash/low` full-history 诊断（8 条独立 history、2 条 replicate）。所有请求均自然 `stop`；provider usage 的 reasoning tokens 为 6,189–29,913，均值 14,153.6，中位数 13,907。该请求不施加正式 memory 输出上限，故可见正文未作为合格 memory 使用；没有 R1、score 或 formal length gate 通过结论。逐条元数据见 `artifacts/reasoning_length_diagnostic_v1.json`。

结论：长度门槛尚未关闭；不能宣称同一实际 memory budget 下的 direct/staged 比较成立。

## VISIBLE_BUDGET_PROTOCOL_V3（已配置，预检待运行）

候选最终 budgets 为 `[1024, 1536, 2048, 3072, 4096]` native tokens，保留 `deepseek-v4.1-flash/low` 和 thinking enabled，provider `max_tokens=65536`。只有满足自然合规率≥0.90 的候选才可冻结；首选 `[2048,4096]` 仍是待检验预期，不是已选结果。

预算预检样本的来源/样本数还未预先冻结，因此实际合规率、选定预算、formal length distribution 和 actual compression rate 均为 `NOT_RUN`。配置禁止预检 histories 与 formal P0 histories 重叠，且正式 runner 在预算冻结、artifact 落盘、P0 plan 与预算一致前会阻止执行。

V3 reporting 将 `actual_compression_rate` 定义为 `1 - visible_native_tokens / original_history_native_tokens`，以每个 compressor step 的原始 history 为分母；reasoning tokens 优先使用 provider usage，缺失时估算为 `max(0, total_completion_tokens - visible_native_tokens)` 并标注估算。
