# L-014 半径诊断（复算，不施工）

日期：2026-09-24。状态：完成观察，不发展 loss。

## 问题

转折没命中的样本，是任务本身更难，还是模型自己的决策边界更钝？

## 来源

- 脚本：`experiments/event_updater/sided_radius.py`
- 冻结模型：`artifacts/discovery_campaign/r04b_s11/clean`（三种子挖掘，同一冻结 clean 预测器）
- 数字文件：`artifacts/event_updater/sided_s{11,23,47}.json`
- 原报告：`reports/event_updater/EVENT_UPDATER_REPORT.md` Side D
- 本文件只复读上述 JSON，未重跑探测，未读封存池

定义：在 24 个方向上二分，`r_s` 是 oracle 翻转的最短步长，`r_m` 是模型 logit 变号的最短步长，`R = r_m / r_s`。分组是 transition-miss 对 hit。

## 复算

| seed | n_hit | n_miss | R_hit q25/med/q75 | R_miss q25/med/q75 | r_s med hit/miss | r_m med hit/miss |
|---|---:|---:|---|---|---|---|
| 11 | 32 | 55 | 0.388 / 0.638 / 1.441 | 1.031 / 1.896 / 4.058 | 0.182 / 0.204 | 0.212 / 0.406 |
| 23 | 38 | 46 | 0.279 / 0.808 / 1.713 | 1.207 / 1.870 / 3.758 | 0.178 / 0.213 | 0.165 / 0.459 |
| 47 | 28 | 49 | 0.225 / 0.935 / 1.735 | 1.029 / 1.871 / 6.469 | 0.182 / 0.196 | 0.156 / 0.411 |

三种子同向：miss 组 R 中位 1.87–1.90，hit 组 0.64–0.94。oracle 半径中位几乎重合（0.18 vs 0.20），分离来自模型半径（hit 0.16–0.21，miss 0.41–0.46）。原报告的四舍五入表与这些 JSON 一致，没有改数。

## 边界

- 探测翻转与 `r_m` 共用方向，有部分循环；可解释性靠任务难度匹配和 R 跨过 1。
- 这是一个冻结 clean 模型上的诊断，不是三种子独立重训。
- 不据此做 margin-matching loss。L-014b 已测过，三种子反向，保持放弃。

## 可引用的一句

Transition misses sit at a larger model-to-oracle radius ratio than hits (median R 1.87–1.90 vs 0.64–0.94, three mining seeds, same frozen predictor); the oracle radius is matched, so the gap is the model's own decision radius, not task difficulty. This is a diagnostic, not a training result.

正文页预算紧（结论在第 9 页）。这句话先放在本报告；若作者确认还有一行空位，再写入 `paper/sections/discussion.tex`。本轮不改正文，避免为观察性诊断挤掉主文。
