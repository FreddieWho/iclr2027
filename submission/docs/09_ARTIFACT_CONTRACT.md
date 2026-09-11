# 09｜最终产物合同

`reports/PILOT_REPORT.md`：开头一页必须有decision、route/reason、最强效应/CI/history数、实际完成矩阵、总成本、claim边界。随后完整结果，不只胜者；exploration与confirmation分开。

`reports/decision.json`：按模板与脚本输出，二元决策；contains_mock=false；每个audited gate的证据路径存在。自动化不替代最终科学审核。

`reports/NOVELTY_AUDIT.md`：三篇直接近邻方法/表格/代码定位，8维对比及剩余新知识；未访问项明确写未验证。

`reports/PREFLIGHT.md`：授权依据不含secret、精确模型版本、context/output/tokenizer、price/date、工程cap。

`reports/LENGTH_AUDIT.md`：target与actual分布、截断/不合规率、相同实际长度敏感性、reader截断证明。

`reports/EVIDENCE_AUDIT.md`：至少24个可追到gold+memory的pair；完整/缺失/扭曲/unknown，oracle/highlight/restore/placebo至少按主要机制适配一项，不把restoration当首创。

`reports/ROBUSTNESS.md`：R2/C2/自然数据至少一项的独立证据；claim范围。

`artifacts/steps.jsonl`、`predictions.jsonl`、`scores.jsonl`、`dataset_manifest.json`、`model_manifest.json`、`cost_ledger.sqlite`、`confirmation_selection.json`与全部prompt/code hashes：每个数可重建。

数据集原始文本按原许可保留，不在公共包重新分发。分享结果时保留可复现的公开URL/revision/hash。没有真实实验的目录必须在名称或manifest标记SMOKE，不与PILOT结果合并。

## 必需图表（不以图好看决定去留）

1. 同history同B的paired差异与CI，按探索/确认分面或独立图。
2. budget×path×information类型表（含n与实际长度），不要求相变。
3. D/S3/R3/W3关键对照与成本表。
4. 证据状态×读出对错的计数/分母及真实反例。

可以只做清晰表格完成pilot；不要求美术制作、正文成稿或宣传图。
