# REASONING_BUDGET_AMENDMENT_V2

## 后续长度诊断（非正式 P0）

随后按用户要求对 `deepseek-v4.1-flash/low` 做了 10 次 reasoning-length-only 请求：8 条独立 history，加前两条各 1 次 replicate。普通 12k/24k/32k 输出限制未启用；兼容 API 的 wire 请求使用 131,072 token 技术承接窗口。10/10 为 `finish_reason=stop`，provider-reported reasoning 为 6,189–29,913 tokens，均值 14,153.6，中位数 13,907；没有调用 R1、没有生成 formal memory 或 score。该诊断不改变本文件的 `NO_GO_TECHNICAL` 边界。

本诊断完成后共享账本保守累计为 `$0.79275835`（本轮增量 `$0.29638140`，含原有 5 个 uncertain reservations）；文中原有的 `$0.49637695` 是诊断前 V2/fallback 快照。完整逐条记录见 `artifacts/reasoning_length_diagnostic_v1.json` 和 `work/p0_reasoning_length_diag_v1_20260912/calls.jsonl`。

## 当前状态

基础 V2 已写入 `configs/models_opencode_go_p0_reasoning_v2.json`；其真实首个 C1 请求已记录为技术失败。按该失败触发的预设规则，升级配置已冻结在 `configs/models_opencode_go_p0_reasoning_v2_escalated.json`，随后又记录了用户授权的 32K extension、DeepSeek V4 fallback 和 GLM C1 fallback；每次都使用新的隔离输出目录。

用户随后授权增加额度直至任务完成；当前共享 P0 cap 为 `$6.00`，仍低于项目 `$18.00` 与模型 caps，自动充值保持关闭。transport retry 固定为 1 次，避免不确定重试把预算边界变成隐藏变量；两次代理失败和一次直连超时的保留额均留在共享账本中。

实际执行结果：DeepSeek V4.1 C1 `low/12000` 与 `low/16000` 均在同一首个 32K history 上以 `finish_reason=length` 结束，visible memory 均为 0；一次 `low/32000` 返回 HTTP 500，用户要求重试后第二次 `low/32000` 仍为 `finish_reason=length`、visible memory=0。DeepSeek V4 `low/16000` 先遇到代理失败，重试一次仍失败，直连后在 600 秒 read timeout 失败。回到已授权 GLM C1 的 `low/12000` 后仍以 `finish_reason=length`、visible memory=0 结束。所有尝试均未进入 R1，不能产生 score。共享账本累计保守占用 `$0.49637695`，其中 5 个 uncertain reservations；没有一次是 cap 中断。

V2 首选使用此前用户要求尝试的 `deepseek-v4.1-flash` 作为 C1 compressor，随后按用户授权尝试 `deepseek-v4-flash` 和原始 `glm-5.3-flash` C1 fallback；R1 始终固定为 `glm-5.3-flash`、low、provider 12,000、可见答案≤512。旧配置
`configs/models_opencode_go_p0_r1_2048.json` 保留为历史合同，不覆盖、不重写其结果含义。

## 已冻结的执行规则

| 角色/阶段 | 模型 | reasoning effort | provider output budget | 可见输出/记忆限制 |
|---|---|---:|---:|---:|
| C1 final memory（基础/升级） | `deepseek-v4.1-flash` | `low` | 12,000 / 16,000 | target 1,024 或 2,048 native tokens |
| C1 intermediate memory（基础/升级） | `deepseek-v4.1-flash` | `low` | 24,000 / 32,000 | ≤8,192 native tokens |
| C1 fallback gate | `deepseek-v4-flash` 或 `glm-5.3-flash` | `low` | 16,000 / 12,000 | 仅首个 long history；成功才扩展 |
| R1 initial/formal | `glm-5.3-flash` | `low` | 12,000 | final answer ≤512 native tokens |
| R2 | `deepseek-v4.1-flash` | match final R1 | 12,000 | 仅 primary signal 后执行 |

R1 校准状态为 `PENDING`。只有 oracle accuracy <0.95，或在 relational multihop / exception_or_retraction 上出现系统性失败，才把 R1 固定提升到 `high`，并将该设置应用到全部正式 R1 runs。提升需要生成新的冻结配置；本文件不自动提升。

`max` reasoning 不进入 primary pilot，仅允许 GO 后的 robustness。C1/R1 的 effort 在同一角色的所有 path 固定，不作为科学 treatment。

## 协议保护

- runner 按 final/intermediate 阶段读取独立的 `max_completion_tokens`，不从目标 memory 大小推导 provider 预算。
- R1 先通过 prompt/schema 约束短答；超过 512 不截断，记为 protocol violation，最多重试一次。
- 只接受配置允许的 natural stop；`length` 记为 `TECHNICAL_INVALID`。当截断率超过 1% 时生成新的冻结配置启用 final 16,000 / intermediate 32,000；本次升级、两次 32K extension、DeepSeek V4 fallback 和 GLM fallback 均未得到合格 memory。所有 network/HTTP failure 保留为技术失败，不作为 `finish_reason` 或科学结果。
- provider 的 `reasoning_content` 不写入 artifact；显式 `<think>` block 从最终 body 移除，未闭合 block 不会进入 memory/answer。memory 只使用 final memory body。
- 每个阶段都是 fresh request，下一阶段只收到上一阶段的最终 memory body，不传 reasoning。
- C1 与 R1 使用各自冻结的 native tokenizer；artifact 记录 `provider_model_id`、returned model、GET `/v1/models` SHA256、HF tokenizer revision、native memory tokens、完整请求 tokens、provider usage、cache read、finish reason、reasoning effort、provider output budget 和最终可见 token 数。

## 验证与科学边界

`python3 -m unittest discover -s tests`：56 tests，全部通过。另完成一次本地 tokenizer mock 联调，仅验证执行路径，不产生网络费用，也不进入统计。

当前科学裁决仍为 `NO_GO_TECHNICAL`：所有 C1 fallback gate 均没有合格 formal memory/score rows，R1 未执行。V2 技术校准未解锁长上下文 C1；在实际 C1 natural-stop memory、R1 controls 和完整 P0 完成前，不得声称 Memory 假设得到验证或被证伪。完整失败清单和每次 provider/model/budget/token 元数据见 `artifacts/reasoning_v2_live_result.json`。
