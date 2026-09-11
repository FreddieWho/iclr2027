# PILOT_REPORT

## 一页裁决

- Decision：`NO_GO`
- Reason code：`NO_GO_TECHNICAL`（C1 长上下文在授权输出上限内耗尽 max reasoning，P0/正式长度与评分门尚未关闭；不是科学阴性）
- Route：无；没有任何确认效应可评估
- Claim：截至 2026-09-11，本包完成了可复跑的本地流水线、运行时配置和 native tokenizer 准备；R1 新合同已冻结为 provider `2048`、可见答案 `512`、`reasoning=max`，但正式 P0 的首个 C1 长上下文请求在授权 `10240` 上限内返回 `content=null`/`finish_reason=length`，尚未完成真实 bounded memory pilot，无法判断 direct/staged memory 的科学效应。
- 最强确认对比：`NOT_RUN`；estimate/CI/history/question/replicate 均无
- capability/probe attempts：账本 `13`，另有 1 个短控制请求；formal P0 score rows：`0`；全部 probe actual/uncertain：约 `$0.05332325`（不含科学 score）
- mock：存在，但只在 `work/smoke_run`，`contains_mock=false` 的决策输入明确排除
- 下一步：用户需明确是否允许 C1 在保持 `glm-5.3-flash`、OpenCode Go、`reasoning=max` 下把 provider 总输出预算提高到 `>10240`；在此之前不运行 P0/P1/P2。

## 1. 实际矩阵

| 阶段 | 计划/数据 | 状态 | 是否进入科学统计 |
|---|---|---|---|
| T00 preflight | 依赖、tokenizer、endpoint、授权 | 部分完成；live capability blocked | 否 |
| T01 novelty | 三篇指定直接近邻的主要方法/实验核验 | `INCOMPLETE`；附录/code 未核完 | 否 |
| T02 synthetic P0 | 8 histories/64 questions，真实 canonical tokenizer | 数据合同完成 | 否 |
| T02 natural prep | LongMemEval cleaned 32 complete histories | 数据准备完成；semantic eval not run | 否 |
| P0 live calibration | 8 native-tokenizer histories，raw/no-memory/oracle/direct | `BLOCKED_C1_LONG_CONTEXT_MAX_REASONING_OUTPUT_EXHAUSTION`；首个 C1 请求失败 | 否 |
| P1 exploration | C1/R1，32 histories，all core paths，2 budgets | `NOT_RUN` | 否 |
| P2 confirmation | ≥64 new histories，≤2 locked contrasts | `NOT_RUN` | 否 |
| P3/generalization | R2/C2/natural + evidence/length audits | `NOT_RUN` | 否 |
| local smoke | 4 demo histories，mock provider | 完成；仅 pipeline smoke | 明确排除 |

完整计划文件：`work/calibration_plan.json`（264 calls upper bound）、`work/exploration_plan.json`（4,224 calls upper bound）和 `work/p0_calibration_plan_glm_20260911.json`（264 calls upper bound）。

## 2. 结果与统计

没有有效 formal `scores.jsonl`，所以没有 paired utility、history-cluster bootstrap CI、type interaction、rewrite/same-K 对照或 cost-saving estimate。正式 P0 在首个 C1 压缩请求停止；`artifacts/scores.jsonl` 的单行是 `NOT_RUN` 状态标记，不是 score row。

smoke 的 `direct - staged3 = 0.0000`、CI `[0.0000, 0.0000]`、n=4 是 mock-only 输出，不能支持零效应或等价结论。

## 3. 长度、证据和泛化门

- 长度：P0 native-GLM 数据准备为 32,720–32,751 tokens；同一首个 32K history 在 C1 provider `2048` 和授权上限 `10240` 探针均未产生 natural-stop visible memory；正式 compressor/reader actual memory、共同 `t`、reader 长度和隐藏截断仍未运行。详见 `reports/LENGTH_AUDIT.md`。
- 证据：formal 可审计 pair 为 `0/24`；restore/highlight/placebo 与语义 evidence audit `NOT_RUN`。详见 `reports/EVIDENCE_AUDIT.md`。
- 泛化：R2/C2/natural reader 均 `NOT_RUN`；自然数据仅完成下载、完整导入和标签隔离。详见 `reports/ROBUSTNESS.md`。
- 查重：三篇指定近邻主要方法/实验已核验，但附录/code/actual-length 全覆盖未完成，故 novelty gate `INCOMPLETE`。

## 4. 成本和资源

工程上限来自 `configs/pilot.json`，不是消费授权：15,000 calls、150M input tokens、15M output tokens。正式 P0 score rows 为 0；provider probe roots 的账本/控制请求 actual-or-uncertain 汇总约 `$0.05332325`，详见 `artifacts/provider_probe_manifest.json`。公开 LongMemEval raw 文件仅作本地数据准备，原始数据不放入公开交付包。

## 5. 证据边界

已排除：代码层面的 query allowlist、真实 GLM tokenizer 可用性、计划预算合法性、session header 缺失这一适配问题、短请求的 C1 schema/natural stop、旧 R1 `512 + reasoning=max` 合同冲突。

尚未排除：C1 `>10240` provider 总输出需求、32K+ C1 visible memory、R1 新 `2048/512` 合同的 live 行为、服务商隐藏 compaction、长度差异、生成波动、reader 使用失败、存储丢失、semantic scoring 偏差、cross-family 泛化及新颖性重叠。

因此不能写“效应为零”“staged 等价 direct”“Memory 假设被证伪”，也不能把技术阻塞伪装成 scientific NO_GO_SMALL_EFFECT。

## 6. 复跑命令

```bash
python3 -m unittest discover -s tests -v
python3 scripts/make_synthetic.py --out work/smoke_data --n 4 --tokens 3000 --tokenizer demo
python3 scripts/build_plan.py --profile configs/smoke.json --out work/smoke_plan.json
python3 scripts/run_pilot.py --histories work/smoke_data/histories.jsonl \
  --queries work/smoke_data/queries.jsonl --plan work/smoke_plan.json \
  --models configs/models.example.json --out work/smoke_run --mock
python3 scripts/analyze.py --scores work/smoke_run/scores.jsonl \
  --contrast direct,staged3 --budget 256 --out work/smoke_contrast.json
```

正式 P0 命令已按 R1 新合同启动，但因 C1 长上下文 max reasoning 输出耗尽暂停；失败/探针 output roots 和成本见 `artifacts/provider_probe_manifest.json`。自然数据 semantic grader 仍未配置。不得把 mock 命令改称 formal pilot。

## 7. 产物索引

- `reports/PREFLIGHT.md`
- `reports/NOVELTY_AUDIT.md`
- `reports/LENGTH_AUDIT.md`
- `reports/EVIDENCE_AUDIT.md`
- `reports/ROBUSTNESS.md`
- `reports/PHASE_REPORT_20260911.md`、`reports/RELEASE_CHANGELOG_20260911.md`
- `reports/decision_input.json` 与由 `scripts/decide.py` 生成的 `reports/decision.json`
- `artifacts/dataset_manifest.json`
- `artifacts/model_manifest.json`
- `artifacts/confirmation_selection.json`
- `artifacts/provider_probe_manifest.json`
- `artifacts/steps.jsonl`、`predictions.jsonl`、`scores.jsonl`、`cost_summary.json`、`cost_ledger.sqlite`
- mock-only 参考运行：`work/smoke_run/run_manifest.json`、`steps.jsonl`、`predictions.jsonl`、`scores.jsonl`、`work/smoke_contrast.json`

## 8. Action-Mode

只引用 `docs/08_ACTION_MODE_AUDIT_MEMO.md` 作为后续备忘；本轮未执行其中 A 的施工建议、未修改 `sources/`、未改变 AMR 方法目标。
