# U1 确认附录 — 确认池完整 2×2（存档摘要后续分析，零新读数）

> 状态更正：`UPGRADE_FINDINGS.md` 与禁令第 9 条曾写“确认池缺 raw+flip 整行”。
> 复核发现 `artifacts/next_novelty/relflip_v2/RELFLIP_CONFIRM.json` 的四个臂
> （raw_clean / raw_flipmine / relfeat / relflip）在 confirm1007（n=509）上
> 三种子全有 J 行——缺的是“把它们拼成交互”的后续分析，不是数据。
> 本附录只做该后续分析，不跑任何新前向、不读新封存输入。

## 确认池 2×2（J 点估计，同 bank 配对评价）

| seed | raw | raw+flip | relfeat | relflip | I_confirm (pp) |
|---|---|---|---|---|---|
| 11 | 0.0589 | 0.1493 | 0.1631 | 0.4735 | +22.0 |
| 23 | 0.0472 | 0.1257 | 0.2220 | 0.4479 | +14.7 |
| 47 | 0.0413 | 0.1434 | 0.2259 | 0.4361 | +10.8 |

## 证据等级更新

- U1 交互从 **exploratory** 升级为 **confirm-direction**：
  dev 配对交互（I=+16–25pp，parent-cluster CI 全不含零）＋确认池同方向
  点估计（I=+11–22pp，n=509）。
- 仍不是 paired-CI 级确认：确认池只有汇总 J，没有逐 quartet CI；
  要 CI 需在 confirm1007 上重跑逐状态前向——那是新协议事件，需用户授权。
  在那之前，正文措辞一律用“确认池同方向”，不用“确认交互显著”。
- §4 表格与正文中的“确认 flipmine 格为空（gap, by design）”已过时，
  随本附录同步改写为实测值（0.149/0.126/0.143）；图注同步。
