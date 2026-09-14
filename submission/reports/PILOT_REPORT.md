# PILOT_REPORT

- 最新 P0 状态（2026-09-13）：V3 可见预算预检完成 4/40 格；观察到的 2,048/4,096 失败格已排除首选预算对通过的可能。但冻结计划规定除全局授权、配置或预算阻塞外继续执行，本轮未遇到这些阻塞仍提前停止，构成协议偏离；故整份预检不完整、正式预算未冻结，R1 与 formal scoring 均 `NOT_RUN`。项目级 `reports/decision.json` 当前仍为 `NO_GO/TECHNICAL`（多个审计门未过），不是科学阴性；该决策尚未从这份部分预检重新生成。详情见 `reports/P0_VISIBLE_BUDGET_V3_RESULT_20260913.md`、`artifacts/visible_budget_preflight_v3_partial_20260913.json`。

- 最新非正式诊断：`deepseek-v4.1-flash/low` 对 8 条独立 full history 加 2 条 replicate 共 10 次请求，reasoning tokens 为 6,189–29,913（均值 14,153.6，中位数 13,907），10/10 `finish_reason=stop`。普通 formal memory 输出上限未启用，因此未生成 formal memory、未调用 R1、无新 score；共享账本累计保守占用 `$0.79275835`。详见 `reports/REASONING_LENGTH_DIAGNOSTIC_V1.md`。
- 最新修复探测：C1 改为 `deepseek-v4.1-flash/low` 后，V3/V4/V5 均在同一首个 32K history 上自然停止；V5 的 2048 target 通过，但 1024 target 仍超预算。关闭 thinking 并把 wire budget 设为 1024 又得到 `finish_reason=length`。因此仍未解锁 formal C1，R1 未调用；当前账本为 33 条记录、保守占用 `$1.02277825`。详见 `reports/REASONING_BUDGET_V3_V5.md` 与 `artifacts/reasoning_budget_v3_v5_probe_result.json`。
- `VISIBLE_BUDGET_PROTOCOL_V3` 启动前配置快照：C1 为 `deepseek-v4.1-flash/low`、thinking enabled、provider `max_tokens=65536`；候选 memory budgets 为 1024/1536/2048/3072/4096。该快照当时尚未冻结预检样本；后续真实预检及其未完成状态见本报告首条与 `reports/P0_VISIBLE_BUDGET_V3_RESULT_20260913.md`。

## 一页裁决

- 2026-09-12真实执行 V2：C1 `deepseek-v4.1-flash/low` 在 final provider budget `12000`、随后按 >1% 截断规则升级至 `16000` 时，均返回 `finish_reason=length`、visible memory=0；首次 `32000` 返回 HTTP 500，用户要求重试后仍为 `finish_reason=length`、visible=0。已授权的 `deepseek-v4-flash/low` fallback 经代理重试与直连仍分别失败/超时；原始 `glm-5.3-flash/low` C1 fallback 也在 `12000` 返回 `length`、visible=0。R1 未执行、无 score。该 V2/fallback 快照的共享账本累计保守占用 `$0.49637695`，含 5 个 uncertain reservations；没有一次因 cap 中断。详见 `reports/REASONING_BUDGET_V2.md` 与 `artifacts/reasoning_v2_live_result.json`；不改变科学结论。
- 2026-09-12此前诊断：优先DeepSeek-V4.1-Flash、随后V4-Flash各一次替换探针均失败；max reasoning、24576总输出、1024 memory目标下可见memory均为0，finish=length。见 `reports/P0_DEEPSEEK_SWITCH_20260912.md`。
- 诊断前共享账本累计保守占用 `$0.49637695`，其中保留 5 个 uncertain reservations；V2 及 fallback 共新增 8 次真实 C1 尝试，56 项离线测试通过，无新科学评分。该数值保留为正式 V2/fallback 快照。

- 最新已授权探针：原始32K历史/原始提示、C1 `24576/max` 仍返回 `finish_reason=length`、visible memory=0，provider reasoning=24553；只调用一次，无重试。完整P0仍阻塞。详见 `reports/P0_C1_24576_PROBE.md` 与 `artifacts/c1_24576_probe_result.json`。
- 正式 V2/fallback 快照成本：共享账本累计保守占用 `$0.49637695`，其中 5 个 uncertain reservations；最新诊断后累计为 `$0.79275835`。旧 `$0.05332325` 仅为历史 probe 汇总，不能替代当前账本。
- 工程验证：56 项离线测试通过；R1 短答控制已成功（答案4 tokens），完整 reader 校准仍未完成。
- Decision：`NO_GO`
- Reason code：`NO_GO_TECHNICAL`（满足自然停止与 memory token 上限的正式预算组合尚未通过预检并冻结；P0/正式评分门未关闭；不是科学阴性）
- Route：无；没有任何确认效应可评估
- Claim：截至 2026-09-12，本包完成了可复跑的本地流水线、运行时配置和 native tokenizer 准备；R1 合同为 provider `12000`、可见答案 `512`、`reasoning_effort=low`。DeepSeek V4.1 在 V5 的单次 2048-token 探测中产生过 natural-stop 且长度合格的 memory，但预算组合尚未通过预检或冻结；尚未完成真实 bounded memory pilot，无法判断 direct/staged memory 的科学效应。
- 最强确认对比：`NOT_RUN`；estimate/CI/history/question/replicate 均无
- capability/probe attempts：正式 V2/fallback 快照的共享账本共 16 行，其中 1 行是 accounting carry-forward；实际 provider attempts 为 15，10 次响应已结算，5 次保留为 uncertain。随后新增 10 次 reasoning-length-only 请求，账本共 26 行、25 次 provider attempts，全部新增请求已结算。另有 1 个短控制请求未进入该账本。formal P0 score rows：0。
- mock：存在，但只在 `work/smoke_run`，`contains_mock=false` 的决策输入明确排除
- 下一步：先冻结预算预检样本设计，再对候选 budgets 计算自然合规率；仅在合格预算对和预检 artifact 冻结后重建 P0 plan。P0/P1/P2保持未完成；预算不合规单元记为缺失协议结果，不作为 memory 质量失败。

## 1. 实际矩阵

| 阶段 | 计划/数据 | 状态 | 是否进入科学统计 |
|---|---|---|---|
| T00 preflight | 依赖、tokenizer、endpoint、授权 | 部分完成；live capability blocked | 否 |
| T01 novelty | 三篇指定直接近邻的主要方法/实验核验 | `INCOMPLETE`；附录/code 未核完 | 否 |
| T02 synthetic P0 | 8 histories/64 questions，真实 canonical tokenizer | 数据合同完成 | 否 |
| T02 natural prep | LongMemEval cleaned 32 complete histories | 数据准备完成；semantic eval not run | 否 |
| P0 live calibration | 8 native-tokenizer histories，raw/no-memory/oracle/direct | `NO_GO_TECHNICAL_GATE`；V3 预检 4/40 提前停止，预算未冻结；正式 raw/no-memory/oracle/direct 与 R1 校准 `NOT_RUN` | 否 |
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

工程上限来自 `configs/pilot.json`，不是消费授权：15,000 calls、150M input tokens、15M output tokens。正式P0 score rows为0；截至最新修复探测共享账本累计保守占用 `$1.02277825`，详见 `artifacts/reasoning_budget_v3_v5_probe_result.json`；包含 uncertain reservation，不是 provider 账单。此前 `$0.79275835` 是 reasoning-length diagnostic 快照，`artifacts/provider_probe_manifest.json` 中旧约 `$0.05332325` 是更早历史记录，均不能替代当前共享账本。公开LongMemEval raw文件仅本地保留，不放入公开交付包。

## 5. 证据边界

已排除：代码层面的 query allowlist、真实 GLM tokenizer 可用性、计划预算合法性、session header 缺失这一适配问题、短请求的 C1 schema/natural stop、旧 R1 `512 + reasoning=max` 合同冲突。

最新正式技术诊断：DeepSeek V4.1 low 的 12K/16K/32K、DeepSeek V4 low 的 16K（含直连 600s）及 GLM low 的 12K 均未取得 32K 输入的可见 memory；另有 10 条高技术窗口的 low reasoning-length-only 请求自然 stop，但可见正文未按 formal memory target 约束。仍未解决：C1合格可见写入、R1新合同的完整raw/oracle/no-memory校准（短答接口已通过）、服务商隐藏 compaction、长度差异、生成波动、reader使用失败、存储丢失、semantic scoring偏差、cross-family泛化及新颖性重叠。

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
# low reasoning length diagnostic (requires the authorized .env credential)
set -a; . ./.env; set +a
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  PYTHONPATH=scripts python scripts/probe_reasoning_length.py \
  --output work/p0_reasoning_length_diag_v1_20260912 --limit 10
```

较早的 formal P0 尝试因 C1 输出耗尽而暂停，历史探针见 `artifacts/provider_probe_manifest.json`。最新 V3 预算门状态见 `reports/P0_VISIBLE_BUDGET_V3_RESULT_20260913.md`；formal P0 与 reader calibration 仍未运行。自然数据 semantic grader 仍未配置。不得把 mock 命令改称 formal pilot。

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
- `artifacts/reasoning_budget_v3_v5_probe_result.json`、`reports/REASONING_BUDGET_V3_V5.md`
- `configs/models_opencode_go_p0_visible_budget_v3.json`、`prompts/compress_neutral_formal_v6.txt`、`reports/VISIBLE_BUDGET_PROTOCOL_V3.md`
- `reports/P0_VISIBLE_BUDGET_V3_RESULT_20260913.md`、`artifacts/visible_budget_preflight_v3_partial_20260913.json`
- `reports/PROJECT_PROGRESS_SUMMARY_20260913.md`
- `artifacts/steps.jsonl`、`predictions.jsonl`、`scores.jsonl`、`cost_summary.json`、`cost_ledger.sqlite`
- mock-only 参考运行：`work/smoke_run/run_manifest.json`、`steps.jsonl`、`predictions.jsonl`、`scores.jsonl`、`work/smoke_contrast.json`

## 8. Action-Mode

只引用 `docs/08_ACTION_MODE_AUDIT_MEMO.md` 作为后续备忘；本轮未执行其中 A 的施工建议、未修改 `sources/`、未改变 AMR 方法目标。

## 9. 2026-09-13 P0 预算门更新

- 最新共享账本 actual-plus-uncertain 为 `$1.37211145`，其中含历史结转、本轮 4 个已结算请求及 1 个被中断请求的 `$0.088638` 保守预留；详见 P0 部分结果 artifact。
- 预检仅有部分技术证据，未生成完整 canonical 预检 artifact；本轮提前停止违反冻结计划的继续执行条件。不能据此声称自然合规率已完整估计，也未据此运行 formal P0。
