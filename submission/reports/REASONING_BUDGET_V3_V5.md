# REASONING_BUDGET_V3_V5

## 裁决

`NO_GO_TECHNICAL`，`formal_p0=false`。本轮只做 C1 compressor 的协议探测，没有调用 R1，没有生成 score，也没有进入科学统计。

当前组合固定记录为：C1=`deepseek-v4.1-flash`，R1=`glm-5.3-flash`，provider=`OpenCode Go`。实时 `/v1/models` 快照中没有 `glm-3-flash`，因此不能把用户提出的 `deepseek-v4.1-flash + glm-3-flash` 作为可执行组合；R1 保留 GLM-5.3-Flash。

## 探测结果

所有请求使用同一首个 synthetic history `syn-20260910-00000`，因此结果是技术可行性证据，不是跨 history 的科学结果。

| 配置 | C1 设置 | target | provider output | reasoning | final memory | finish | 结果 |
|---|---|---:|---:|---:|---:|---|---|
| V3 | low，thinking enabled，65536 | 1024 | 47309 | 42902 | 4406 | stop | `TECHNICAL_INVALID_MEMORY_OVER_BUDGET` |
| V3 | low，thinking enabled，65536 | 2048 | 26441 | 21931 | 4509 | stop | `TECHNICAL_INVALID_MEMORY_OVER_BUDGET` |
| V4 | low，thinking enabled，65536，单行 + newline stop | 1024 | 8234 | 7175 | 1058 | stop | `TECHNICAL_INVALID_MEMORY_OVER_BUDGET` |
| V4 | low，thinking enabled，65536，单行 + newline stop | 2048 | 17781 | 14319 | 3461 | stop | `TECHNICAL_INVALID_MEMORY_OVER_BUDGET` |
| V5 | low，thinking enabled，65536，单行 + 更紧 body 上限 | 1024 | 17246 | 15141 | 2104 | stop | `TECHNICAL_INVALID_MEMORY_OVER_BUDGET` |
| V5 | low，thinking enabled，65536，单行 + 更紧 body 上限 | 2048 | 15799 | 13997 | 1801 | stop | `TECHNICAL_PASS` |
| isolated no-thinking | low，thinking disabled，1024 | 1024 | 1024 | null | 1024 | length | `TECHNICAL_INVALID_FINISH_REASON` |

`TECHNICAL_PASS` 只表示该次 2048 目标满足长度和 natural-stop 门；它不能替代 1024 目标，也不能证明内容覆盖、R1 可读性或科学效应。

## 解释与边界

V3–V5 的 prompt、单行格式和 stop sequence 能提高 natural-stop 概率，但不能给最终 memory body 提供独立的 provider 硬上限。关闭 thinking 后把 wire `max_tokens` 设成 1024 又得到 `finish_reason=length`，说明简单降低 provider budget 也不能满足“自然停止且不截断”的合同。

因此当前不能安全地声称“正式 P0 已可顺利完成”。若强行采用单一 stop 分隔符、关闭 thinking 或只保留 2048 cell，都需要新的协议/科学选择，可能改变 memory 覆盖或 treatment 定义，不能自动升级为正式配置。

## 成本与可复跑证据

截至最后一次探测，共 33 条账本记录，保守占用 `$1.02277825`，累计 provider input `990561`、output `555113` tokens，其中 5 条 uncertain reservations。没有一次因 cap 中断；正式 P0 仍未启动。

逐条 metadata、请求 hash 和缓存正文见 [reasoning_budget_v3_v5_probe_result.json](../artifacts/reasoning_budget_v3_v5_probe_result.json) 及各自 `work/` output root。配置和 prompt 保持隔离：

- `configs/models_opencode_go_p0_reasoning_v3_naturalstop.json`
- `configs/models_opencode_go_p0_reasoning_v4_singleline.json`
- `configs/models_opencode_go_p0_reasoning_v5_boundedbody.json`
- `configs/probe_deepseek_v41_nothinking_1024.json`

结论仍是技术阻塞，不是科学阴性；在获得可独立约束 final memory body 的 provider/API 能力，或批准新的不等价协议前，P0、P1、P2、P3 均保持 `NOT_RUN`/`BLOCKED`。
