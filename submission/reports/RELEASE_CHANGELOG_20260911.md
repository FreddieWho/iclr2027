# Release Changelog — 2026-09-11

## Release scope

本次仅发布 ICLR Memory Pilot 的 preflight、数据准备、查重输入、阻塞审计、OpenCode Go P0 运行配置和可复跑 smoke surface；未发布真实 pilot 结果。

## Included updates

- 离线测试通过：44/44。
- 冻结 OpenCode Go P0 配置：GLM-5.3-Flash、User-Agent、价格/预算、models endpoint SHA、GLM/DeepSeek tokenizer revisions；凭据值未写入工作区。
- 下载并本地验证 GLM native tokenizer；生成 8 history/64 question 的 native-tokenizer P0 calibration data。
- 完成 OpenCode Go session/capability probes：短 C1 `max_tokens=2048` 返回 natural stop；R1 旧 `max_tokens=512 + reasoning=max` 被 provider 拒绝；首个 32K C1 请求在授权 `max_tokens=10240 + reasoning=max` 下仍返回 `content=null`/`finish_reason=length`；R1 新 `2048 provider / 512 visible` 合同已写入配置但尚未 live 执行。独立 reservation/actual ledger 汇总写入 `artifacts/provider_probe_manifest.json`。
- 生成 8 条真实 `tiktoken:cl100k_base` synthetic P0 history、64 题并保存 manifest/hash。
- 生成 calibration/exploration 计划：264 / 4,224 次保守调用上限，均未联网。
- 下载并校验 LongMemEval cleaned/oracle；导入 32 条完整 cleaned histories，public history 无禁止字段，评分标为 `semantic_required`。
- 完成三篇指定直接近邻的主要方法/实验核验；附录、release artifact、代码和 exact actual-length 覆盖仍标 `INCOMPLETE`。
- 生成 formal `NOT_RUN` artifact markers、C1 长上下文阻塞模型 manifest、probe ledger 和最终 `NO_GO/TECHNICAL` decision；probe 不进入科学统计。

## Audit outcomes

| Gate | 状态 | 原因 |
|---|---|---|
| technical validity | `FAIL/BLOCKED` | C1 32K 长上下文在授权 10240 output 内耗尽 max reasoning，正式 P0 未完成 |
| query blind | `NOT_RUN` | 仅静态数据合同通过，无 formal live request |
| budget/length | `NOT_RUN` | 无 actual memory、clip、common-length 重读或 hidden truncation 证据 |
| rewrite/same-K | `NOT_RUN` | 没有 formal scores |
| independent confirmation | `NOT_RUN` | 未完成 exploration/selection |
| generalization | `NOT_RUN` | R2/C2/natural reader 未运行 |
| novelty | `INCOMPLETE` | 尚未检查全部 appendix/code/released artifacts |
| task relevance/evidence | `NOT_RUN` | 没有可审计 live memory/prediction pairs |

## Verification assets

- `reports/decision.json` 由 `scripts/decide.py` 机械生成：`decision=NO_GO`、`reason=TECHNICAL`。
- `artifacts/cost_ledger.sqlite`：canonical formal score surface 仍为 `NOT_RUN`；已启动的失败 output roots 和 probe ledgers 见 `artifacts/provider_probe_manifest.json`。
- smoke-only run 的 `run_manifest.json` 保存 code/prompt hash；其 score contrast 标为 `MOCK_ONLY`，未进入 decision input。
- `git diff --check` 无输出；必需报告/JSON/JSONL/SQLite 文件检查通过。

## Canonical surface changes

新增当前日期的 phase report、release changelog、五类审计报告、decision files、formal status artifacts 和 OpenCode Go P0 配置/tokenizer preparation。未修改 `sources/` 原文、已有文档、Action-Mode 施工内容或任何生信索引；LongMemEval 是对话 benchmark，不触发 `infra/bioinf-data-index/` 更新。

## Final release state

最终状态：`NO_GO`，机械 reason `TECHNICAL`；对应人类可读理由是 `NO_GO_TECHNICAL`。该状态表示 C1 长上下文输出能力和 P0 门未满足，不表示 direct/staged 假设已被证伪，也不表示效应为零。

## Known post-release notes

- raw LongMemEval 文件约 280M，仅保留本地并以 hash/provenance 引用；公开交付不应包含原始正文。
- 继续执行需要用户明确是否允许 C1 在保持 OpenCode Go / `glm-5.3-flash` / `reasoning=max` 下将 provider output 提高到 `>10240`；收到后重新核验并从新的 P0 live calibration output root 开始，不直接跳到 P1/P2。
- 本 changelog 不授权启动 Action-Mode 实验、重写历史证据或扩大模型/预算搜索。
