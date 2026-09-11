# PREFLIGHT

日期：2026-09-11

状态：`BLOCKED_C1_LONG_CONTEXT_MAX_REASONING_OUTPUT_EXHAUSTION`

## 结论

离线 runner、目标 native tokenizer、合成 P0 数据和公开 LongMemEval 数据准备完成。`.env` 已在单次进程内成功加载，provider probe 已到达 OpenCode Go；R1 新合同已经冻结为 provider `2048`、可见答案 `512`、`reasoning_effort=max`。正式 P0 在首个 C1 长上下文请求停止：即使 C1 使用授权上限 `10240`，仍返回 `content=null`、`finish_reason=length`，只有隐藏 `reasoning_content`，因此没有产生科学效应估计。

## 工作区与依赖

- 工作目录：`/home/huyudi/012_conference/iclr2027/submission`。
- 当前上层 Git root：`/home/huyudi/012_conference/iclr2027`；本次只在 submission 目录内工作，未修改上层未跟踪项或 `sources/` 原文。
- branch/HEAD：`main` / `5cdd585`（上层仓库当前状态；submission 内容在该仓库中仍未跟踪）。
- Python：`/opt/anaconda3/bin/python3`；`numpy 1.26.4`、`requests 2.34.2`、`tiktoken 0.12.0`、`transformers 5.6.2`、`tokenizers 0.22.2`、`huggingface_hub 1.8.0` 可用。
- 离线测试：`python3 -m unittest discover -s tests -v`，44/44 passed。
- `squeez doctor`：hooks、registration、interpreter、compression、freshness、blob store 均报告 ok。

## 模型、endpoint、授权

`configs/models.example.json` 仍是不可运行示例。新增 `configs/models_opencode_go_p0.json`，引用环境变量 `PILOT_OPENCODE_GO_API_KEY`，不记录或展示 key 值；`.env` 已被 submission 目录级 `.gitignore` 忽略并收紧为 `600`。用户授权中的 provider/model/预算已转为非敏感配置字段。

本机只读检查发现已安装 Ollama 权重：`qwen2.5:1.5b`、`qwen2.5:0.5b`、`bge-m3:latest`。`ollama ps` 为空，检查时无运行模型；这些本地权重尚未完成本 pilot 要求的 C1/R1 能力、长度和模型适用性核验，`bge-m3` 也不是 reader/compressor。未启动服务、未下载权重、未把它们算作正式 C1/R1。

正式模型能力状态：

| 项目 | 状态 |
|---|---|
| C1/R1 精确 model ID/version | `glm-5.3-flash`；`/v1/models` 返回；P0 配置冻结 |
| 可访问 endpoint 与协议 | OpenCode Go chat-completions endpoint 返回 200；session header 后 C1 probe 成功 |
| native tokenizer/version | GLM tokenizer revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`，本地文件已加载 |
| 32K+ context 与 8K 可见写入输出 | 同一 32K+ history 在 C1 `2048` 和授权上限 `10240` 下均未得到 natural-stop visible memory；gateway 未公开逐模型 context 元数据 |
| reasoning/hidden output 计费规则 | 短请求 C1 可 natural stop；长请求在 `10240 + reasoning=max` 下耗尽输出并返回空 visible content；旧 R1 `512` 合同曾被拒绝，新 R1 未因 C1 阻塞而 live 执行；详见 `artifacts/provider_probe_manifest.json` |
| 输入/输出价格与核验日期 | GLM `$0.15/$0.50` per 1M，cache read `$0.03`；官方页核验于 2026-09-11 |
| 明确 live 授权与正美元 cap | 授权已收到并从 `.env` 加载；P0 cap `$0.50` |
| 已发送 live 请求 | 账本 attempts `13`，另有 1 个短控制请求；formal P0 score rows `0` |

历史 canonical tokenizer 仍为 `tiktoken:cl100k_base`；运行时 C1/R1 已另行冻结并验证 GLM native tokenizer。两者不混用。

## 工程预算与已运行的本地流程

`configs/pilot.json` 的工程上限是 15,000 calls、150,000,000 input tokens、15,000,000 output tokens；这不是消费授权。默认探索计划在 32 history、两预算下为 4,224 次保守调用上限；calibration 计划为 264 次。独立 probe roots 和短控制请求合计约 `$0.05332325` actual/uncertain；未把 probe 当 formal score。

已运行的本地准备：

1. `work/p0_calibration_data_20260911`：8 条合成校准 history、64 题，历史 canonical tokenizer，32,722–32,753 tokens/history；仅数据合同校验，不计入推断。
2. `work/p0_calibration_data_glm_20260911`：8 条合成校准 history、64 题，GLM native tokenizer，32,720–32,751 tokens/history；仅 P0 calibration 数据准备，不计入推断。
3. `work/smoke_data` + `work/smoke_run`：4 条 demo-tokenizer、mock provider 的流水线 smoke；计划 200 次调用上限，`cost_ledger` 实际 attempts 为 0，明确 `mock=True`，不进入 `reports/decision.json`。
4. `data/raw/longmemeval_20260911`：文档指定的公开 cleaned/oracle 文件已下载并记录 SHA256；`data/longmemeval_20260911` 已抽取 32 条完整 history。没有启动自然数据 reader/compressor。

## 继续所需的最小输入

下一步只需用户明确：是否在保持 OpenCode Go / `glm-5.3-flash` / `reasoning_effort=max` 下，将 C1 provider 总输出预算提高到 `>10240`，并记录新的 cap。R1 新合同保持 provider `2048`、visible `512`；在 C1 未解锁前不重跑 P0，也不进入 P1/P2。若不授权扩展，当前 `NO_GO_TECHNICAL` 保持不变。

## 关键 hash

- `MASTER_AGENT_PROMPT.md`：`50e772c4bc7c876b7be4e2e11465b64927e4b6abb038ab7ce30af562e17f3005`
- `configs/pilot.json`：`25fe92396146a7218aebdf200bd60393e07b5079a30afa33f45c3f4c463f0569`
- `configs/calibration.json`：`e8b77dc86ed7604f78f9b2b658e3507f3b5ff52fe787cf5c13ddb700a7b73a46`
- `configs/calibration_opencode_go.json`、`configs/models_opencode_go_p0.json`：以当前文件 SHA256 为准。
- `artifacts/provider_probe_manifest.json`：provider/session/参数错误与独立 ledger 汇总。
- OpenCode `/v1/models` response：`e859652c6d13b5233309d8f53d0525325225ba4300ecabe7bb8cc0d31967cffd`。
- HF tokenizer revisions：GLM `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`；DeepSeek `60d8d70770c6776ff598c94bb586a859a38244f1`。
- P0 histories/query/manifest：见 `artifacts/dataset_manifest.json`。
- smoke code/prompt/run hash：见 `work/smoke_run/run_manifest.json`。
