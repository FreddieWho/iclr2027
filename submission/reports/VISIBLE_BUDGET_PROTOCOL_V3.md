# VISIBLE_BUDGET_PROTOCOL_V3

## 状态

新协议已配置，状态为 `PREFLIGHT_REQUIRED_NOT_FORMAL`。没有发送新的 provider 请求；正式预算没有冻结，正式路径测试不可启动。当前科学裁决仍为 `NO_GO_TECHNICAL`，不代表科学假设被证伪。

## 配置

- C1：OpenCode Go `deepseek-v4.1-flash`，thinking enabled，`reasoning_effort=low`，provider `max_tokens=65536`。该预算覆盖 reasoning 与 visible content；65536 高于已观测的最大 completion 47309，留有输出余量。
- 候选最终 memory budgets：1024、1536、2048、3072、4096 native tokens。
- 候选预算只在预检自然合规率至少 0.90 时有资格进入正式配置。合规定义为：单次预先安排的请求返回非空正文、`finish_reason=stop`，且正文 native token 数不超过候选预算；provider/API 错误计入不合规分母。
- 优先考察 `[2048, 4096]`，但只有两者都达标后才可冻结为正式预算对。其他组合需要显式审阅。
- R1 保持 OpenCode Go `glm-5.3-flash`、low、provider 12000、最终可见答案≤512。R1 尚未运行。

## 尚未冻结的预检设计

用户协议没有指定预检样本数和历史来源。新配置将 `sample_size_per_candidate`、`preflight_histories_path` 保持为空，并禁止预检历史与正式路径历史重叠。预算合规率、正式预算对和 preflight artifact 因此均待测；配置中的 `final_memory_budgets_native_tokens` 与 C1 的 formal target list 当前为空。正式 runner 会拒绝运行，直到预检结果、合格预算、预算冻结状态和 artifact 均已登记，并且 P0 plan 的预算与冻结值一致。

V5 的旧结果只作为历史技术证据，不用于估计 V3 预检合规率：在同一条 history 上，1024 target 曾生成 2104 tokens，2048 target 曾生成 1801 tokens；每个候选只有单次结果，提示版本也不同，不能据此宣称 90% 自然合规。

## 正式运行处理

正式调用若自然停止但正文超出目标，会记录为 `BUDGET_NONCOMPLIANT`，对应 score 为空、不会调用 R1，也不会记成 memory 内容质量失败；其他矩阵单元可以继续。`finish_reason` 不为 `stop` 或正文为空仍是技术无效。默认不做拒绝采样或重试；只有在调用前预注册拒绝采样规则后才允许重试。正文不截断，不流式切断。

每个 compressor step 记录 provider 报告的 reasoning tokens；缺失时记录 `max(0, total_completion_tokens - visible_native_tokens)` 并标注为估算。还记录 visible native tokens、total completion tokens、finish reason、budget compliance 和实际压缩率。压缩率定义为 `1 - visible_native_tokens / original_history_native_tokens`，即原始历史长度减少的比例。

允许的结论范围是“相同最终预算上限下的压缩路径依赖”。若要声称 endpoint-equivalent final rate，必须对实际长度差异做调整。

## 验证

本地验证覆盖：继承配置会合并到 DeepSeek C1 / GLM R1；预算未预检冻结时 fail-closed；超预算输出记 `BUDGET_NONCOMPLIANT`、score 为空且不调用 R1；reporting 字段按配置写入。59 项离线测试通过。没有执行 live preflight 或正式 P0。
