# Phase Report — ICLR Memory Pilot preflight and credential-bound handoff

日期：2026-09-11

## Scope and status

本阶段负责把 bounded pilot 从施工包推进到可运行状态，并冻结当前可安全交给后续执行者的输入/输出面。离线代码、C1/R1 native tokenizer、P0 synthetic data、LongMemEval 数据准备和 mock smoke 已完成；R1 新合同已冻结为 provider `2048`、visible answer `512`、`reasoning=max`。正式 P0 首个 C1 长上下文请求在授权 `10240` 上限内耗尽 reasoning，返回 `content=null`/`finish_reason=length`，真实 C1/R1 pilot 为 `BLOCKED_C1_LONG_CONTEXT_MAX_REASONING_OUTPUT_EXHAUSTION`，没有科学效应结果。

## Canonical inputs

| 输入 | 位置 | 用途 |
|---|---|---|
| 主启动约束 | `MASTER_AGENT_PROMPT.md` | 阶段顺序、GO/NO_GO 和资源边界 |
| 研究/统计合同 | `docs/01–06`, `docs/09` | 实验矩阵、测量、决策、产物合同 |
| Action-Mode 备忘 | `docs/08_ACTION_MODE_AUDIT_MEMO.md` | 只读后续备忘，本阶段不施工 |
| pilot/calibration 配置 | `configs/pilot.json`, `configs/calibration.json` | H/B/路径/工程上限 |
| 模型示例配置 | `configs/models.example.json` | 当前为 REPLACE 示例，不能 live |
| P0 运行配置 | `configs/models_opencode_go_p0_r1_2048.json`, `configs/calibration_opencode_go.json` | OpenCode Go/GLM-5.3-Flash；R1 provider 2048/visible 512；key 只通过 env |
| prompts | `prompts/*.txt` | compressor/reader/judge/audit 原始提示 |

## Canonical outputs

- `reports/PILOT_REPORT.md`：唯一阶段结论与非主张边界。
- `reports/decision.json`：由 `scripts/decide.py` 生成的二元结果。
- `reports/decision_input.json`：本次真实 decision input，mock 明确排除。
- `reports/PREFLIGHT.md`、`NOVELTY_AUDIT.md`、`LENGTH_AUDIT.md`、`EVIDENCE_AUDIT.md`、`ROBUSTNESS.md`：门状态和证据边界。
- `artifacts/dataset_manifest.json`、`model_manifest.json`、`confirmation_selection.json`：机器可读 provenance/status。
- `artifacts/steps.jsonl`、`predictions.jsonl`、`scores.jsonl`：canonical score surface 保留 `NOT_RUN` 标记；独立失败 P0 output roots 不并入其中。
- `artifacts/cost_ledger.sqlite`、`cost_summary.json`：canonical score surface 未产生有效 formal row；失败/probe ledger 另列于 `artifacts/provider_probe_manifest.json`。
- `artifacts/provider_probe_manifest.json`：独立 provider probes、响应字段形状、错误摘要、session 与 reservation 汇总。

## Directory surface

- `data/raw/longmemeval_20260911/`：本地原始下载；不进入公共发布包。
- `data/longmemeval_20260911/`：32 条完整 cleaned history/query 的本地导入；query/gold 只供未来冻结 memory 后读取。
- `work/p0_calibration_data_20260911/`：8 history/64 question 的真实-tokenizer calibration data，不计推断。
- `work/p0_calibration_data_glm_20260911/`：8 history/64 question 的冻结 GLM native-tokenizer P0 数据，不计推断。
- `work/calibration_plan.json`、`work/exploration_plan.json`：无网络的调用计划。
- `work/smoke_data/`、`work/smoke_run/`、`work/smoke_contrast.json`：独立 mock-only smoke surface，不得与 formal artifacts 合并。

## Upstream/downstream contract

压缩器上游只能是 `histories.jsonl` 的 public text；query、answer、gold evidence、question type 和 dataset labels 不得进入写入请求。所有 memory 完成后，reader 才能读取 `queries.jsonl`；自然数据必须使用 semantic grader。后续正式 run 必须以新的、独立的 output root 写入，并保留每步 input/output/memory hash、usage、finish reason、length 和 cost。

当前下游可安全使用：代码和配置审计、P0 数据合同、LongMemEval 原始文件 hash、计划调用上限和阻塞原因。当前下游不可使用：任何 pilot effect、模型排序、GO route、自然数据分数或等价性结论。

## Processing flow and scripts

1. `python3 -m unittest discover -s tests -v`：44/44 offline tests passed。
2. `scripts/make_synthetic.py`：生成 P0 synthetic histories/queries/manifest。
3. `scripts/fetch_longmemeval.py`：从文档指定公开入口下载并记录 checksum；raw 数据不发布。
4. `scripts/import_longmemeval.py`：保留完整 session，剥离 compressor 不可见标签，输出 `semantic_required`。
5. `scripts/build_plan.py`：生成 calibration/exploration call upper bounds，无网络。
6. `scripts/run_pilot.py --mock`：仅验证解析、缓存、路径和落盘；live adapter 已验证至 provider 响应，但 C1 长上下文输出能力仍阻塞。
7. `scripts/analyze.py`、`scripts/decide.py`：统计/机械决策；没有 formal scores 时不产生科学结论。
8. `scripts/core.py`/`run_pilot.py`：记录 run start、provider/model/usage/cache、reasoning effort、provider output budget、visible token limit/count、HF revision、native/request token counts、finish reason、models endpoint SHA 和 User-Agent；provider 参数不自动切换，隐藏 reasoning 不作为 memory。

## Key parameters

- history target：32,768 canonical tokens；B：1,024/2,048。
- 核心路径：raw、direct、staged2_wide、staged2_tight、staged3、rewrite3、wait3。
- P0：6–8 histories；计划实际生成 8 条 synthetic 校准 history。
- P1：32 histories；P2：至少 64、建议 96 个全新 histories；确认总上限 192。
- 工程上限：15,000 calls / 150M input / 15M output；非消费授权。
- 统计单位：history-clustered paired utility；确认 family size 按锁定对比记录。

## Reading order for the next executor

先读 `reports/PILOT_REPORT.md` 和 `reports/PREFLIGHT.md`，再读三个 machine-readable manifests；若获得 C1 `>10240` provider output 的明确授权，重新核验当前 git/data hashes，并以新的 output root 重做 8 条 P0 live calibration。不得复用 probe/mock output 或把本阶段 `NO_GO` 改成科学否定。
