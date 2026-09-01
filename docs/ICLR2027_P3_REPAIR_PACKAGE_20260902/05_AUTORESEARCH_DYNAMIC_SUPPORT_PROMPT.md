# Autoresearch Prompt：动态语义 Support 构造与检验

你是一个独立子问题 agent。你的目标不是优化主模型，也不是证明某个方向，而是判断：当前 static role support 是否过于粗糙；能否用 response-blind 的动态事件与运动信息构造更可信的协同小组，并在冻结后检验其相对 matched random support 的表示效应。

该任务在 `P3_CAUSAL_MECHANISM` 内，非 P4 阻塞条件。阴性结果必须保留。

## 1. 数据

优先：

- SkillCorner `dynamic_events.csv` + tracking + phases；
- SNGAR train/valid event + whole-match tracking。

不得读取 SNGAR test。不得用模型 response 定义 support。

## 2. 研究问题

区分：

- static role group：同为 defender/midfielder/forward；
- dynamic semantic support：同一时刻共同逼抢、同时跑位、共同传接或阵线协同的球员；
- matched random support：同队、同 support size、相近局部图/频率/位移分布，但不共享动态事件。

主问题：

> 动态 support 是否比静态 role 更稳定地解释 semantic-vs-random response 差异？

这不是预设为正。

## 3. 允许的 support families

最多开发四类：

1. pressing chain；
2. simultaneous defensive engagement；
3. off-ball-run / passing-option unit；
4. event-window movement coherence。

每类最多两个规则版本。总规则数不超过八个。

## 4. Response-blind 规则

每个规则运行前注册：

- source columns；
- temporal window；
- actor/teammate inclusion；
- minimum/maximum support size；
- velocity/distance/connectivity thresholds；
- exclusion rules；
- confidence levels；
- matched-random construction；
- expected failure modes。

注册后生成 support manifest 和 hash，提交 commit，然后才允许计算模型 response。

## 5. 人工审计

分层抽样 200 条 support：

- 两名评阅者独立判定合理/不合理/不确定；
- 不显示模型 response；
- 报 agreement；
- 规则不得根据 response 调整；
- 若只根据人工审计调整，必须新建规则版本，并将旧版本结果完整保留。

人工审计缺失时，只能称 `algorithmic dynamic support`。

## 6. Matching

matched random 至少控制：

- same match/frame/team；
- support size；
- node validity；
- intervention energy；
- vector multiset 或方向分布；
- graph density/cut；
- frequency residual；
- boundary legality。

匹配失败必须显式记录，不能插补为 0。

## 7. 评价

- primary unit = match；
- 按 architecture、support family 和 energy 报告；
- dynamic-vs-random effect；
- static-role-vs-random effect；
- dynamic minus static incremental effect；
- LOMO；
- match bootstrap；
- residual sensitivity；
- support audit confidence strata；
- 不以所有条件同号为机械目标。

## 8. 结果分类

- `DYNAMIC_SUPPORT_SUPPORTED_CONDITIONALLY`：至少一个预注册 family 在多 match/seed 稳定，且不由 residual/audit failure 解释；
- `STATIC_PROXY_WAS_INSUFFICIENT`：dynamic 明显优于 static；
- `ORGANIZATION_EFFECT_REMAINS_MIXED`：dynamic 仍强异质；
- `NOT_ESTABLISHED`：支持量/匹配/审计不足；
- `NOT_SUPPORTED`：高质量 dynamic supports 下仍无增量。

不得将单一 family 阳性写成普遍战术理解。

## 9. 输出

```text
artifacts/phase3/dynamic_support_v1/
  rule_registry.yaml
  support_manifest.parquet
  matching_manifest.json
  audit_sample.csv
  audit_summary.json
  response_results.parquet
  match_level_effects.parquet
  residual_sensitivity.parquet
  receipt.json
  SHA256SUMS
```

报告：`reports/P3_DYNAMIC_SUPPORT_REPORT.md`

## 10. 禁止事项

- 用 static role 直接命名 tactical ground truth；
- 看 response 后改 support；
- 搜几十个 threshold；
- 选择阳性比赛/role/energy；
- 把 frame 当独立重复；
- 用这个子任务反向选择主 task-repair 候选；
- 因结果阴性而扩大到新领域。
