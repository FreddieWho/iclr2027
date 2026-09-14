# 05｜从 V3 继续，而不是重启项目

## 1. 保留什么

保留并继续使用：
- 已实现 OpenCode Go adapter；
- DeepSeek native tokenizer revision lock；
- model list snapshot/hash 机制；
- cost ledger；
- query-blind 数据合同；
- synthetic histories/questions；
- LongMemEval 已准备数据；
- row-level artifact / prompt hash / usage 记录；
- 现有 mock/smoke tests。

## 2. 冻结什么

以下只读归档：
- `work/p0_visible_budget_preflight_v3_20260913/run_03/*`
- `artifacts/visible_budget_preflight_v3_partial_20260913.json`
- V3 plan hash / prompt hash /旧 `decision.json`

新增一个状态说明，不修改旧 artifact：

`V3_STATUS = INCOMPLETE_PROTOCOL_DEVIATION / SUPERSEDED_FOR_ROUTE_SELECTION`

原因：V3 的 hard-budget preflight 不再作为 V4 science gate。早停偏离继续披露；不通过“补跑剩余格”来擦除历史。

## 3. 明确不要做

- 不续发 V3 的 35 个未尝试格；
- 不重试被中断的 1024 格；
- 不把 V3 的 4 个 visible-budget 输出拿进 V4 统计；
- 不让旧 `configs/pilot.json [1024,2048]` 静默覆盖 V4；
- 不用旧 `reports/decision.json` 作为 V4 终局。

## 4. V4 新文件建议

在仓库内建立：

```text
configs/v4_fast_decision.json
configs/v4_frozen_runtime.json
work/v4_p0/
work/v4_p1_exploration/
work/v4_p2_confirmation/
artifacts/v4_exploration_rows.*
artifacts/v4_confirmation_rows.*
reports/V4_PREFLIGHT.md
reports/V4_FAST_PILOT_REPORT.md
reports/V4_ROUTE_DECISION.json
```

## 5. 最小代码改动

agent 优先复用现有 runner。只需要支持：
1. C1 thinking disabled 的显式请求；
2. 同一 prompt 传 requested target B，但不做 hard compliance rejection；
3. 对 natural-stop content 用冻结 tokenizer 记录 visible tokens；
4. path DAG：direct / staged2 / rewrite；
5. batch reader scoring；
6. history-level paired statistics；
7. length-balanced 与 regression sensitivity；
8. R2 读取冻结 memories。

不要在这轮重构完整框架。
