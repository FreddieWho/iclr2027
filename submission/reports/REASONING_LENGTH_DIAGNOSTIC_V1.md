# Low-effort reasoning length diagnostic

## 结果

2026-09-11 21:55 UTC 在 OpenCode Go 上完成 10 次串行请求。模型固定为 `deepseek-v4.1-flash`，`reasoning_effort=low`，8 条独立 history 加前两条各 1 次 replicate。provider 返回的 reasoning usage 均可测且均以 `finish_reason=stop` 结束。

| 汇总 | 值 |
|---|---:|
| reasoning tokens 总计 | 141,536 |
| reasoning tokens 范围 | 6,189–29,913 |
| reasoning tokens 均值 | 14,153.6 |
| reasoning tokens 中位数 | 13,907 |
| 8 条独立 history 均值 | 16,101.0 |
| provider completion tokens 总计 | 164,773 |
| 长度截断 | 0/10 |

逐条值：`18,100, 7,450, 13,216, 29,913, 23,117, 15,209, 14,598, 7,205`（8 条独立 history），replicate 为 `6,539, 6,189`。

## 配置与边界

- 模型、returned model、`reasoning_effort`、provider output budget、HF tokenizer revision、`/v1/models` SHA、完整请求 tokens、provider usage、cache read 和 finish reason 均写入逐条 JSONL。
- 本次没有启用原先的 12k/24k/32k 普通输出上限；由于兼容 API 的 wire 请求需要 `max_tokens` 字段，使用 131,072 作为技术承接窗口。这不是正式 P0 的 memory budget，也没有进行事后截断。
- 可见正文没有被压到 1,024 native tokens，因此这些响应不是合格 formal memory；未调用 R1、R2，也未计算 score。
- 本轮共享账本增量为保守估计 `$0.29638140`；累计 `$0.79275835`，低于项目 `$18` 和 DeepSeek `$3` 硬上限。该价格是账本估算，不是 provider billing receipt。

## 结论

`low` 下 reasoning 长度不是固定值，当前 full-history 诊断观察到约 6.2k–29.9k tokens；32k 附近的个别请求在 131,072 技术窗口下自然停止。该结果只回答长度诊断问题，不解除 C1 formal memory gate，P0 的 `NO_GO_TECHNICAL` 状态保持不变。

原始记录：`work/p0_reasoning_length_diag_v1_20260912/calls.jsonl`；汇总 artifact：`artifacts/reasoning_length_diagnostic_v1.json`。
