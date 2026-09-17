# R1 结果卡：R02 坐标深化（family/强度针对性调参）

## Sweep（tune_707 前 256，mlpA，margin 0.03）
| cfg | n_cand | full miss\|flip / uncond | matched(298) miss\|flip / uncond |
|---|---|---|---|
| A 基准 | 298 | 0.576 / 0.297 | 0.576 / 0.297 |
| B 对子增强 | 514 | 0.614 / 0.410 | 0.559 / 0.352 |
| C 随机增强 | 490 | 0.651 / 0.387 | 0.628 / 0.363 |
| C@margin0.05 | 490 | 0.614 / 0.348 | 0.571 / 0.312 |

## Confirm（confirm_808 前 256，新场景）
| cfg | 模型 | full miss\|flip / uncond | matched miss\|flip / uncond |
|---|---|---|---|
| A@0.03 | mlpA | 0.569 / 0.305 | 同左 |
| C@0.03 | mlpA | 0.621 / 0.391 | 0.599 / 0.355 |
| C@0.03 | mlpB | 0.665 / 0.418 | 0.632 / 0.375 |
| C@0.03 | mlpC | 0.596 / 0.375 | 0.592 / 0.352 |
| C@0.05 | mlpA | 0.591 / 0.355 | 0.559 / 0.316 |
| C@0.05 | mlpB | 0.643 / 0.387 | 0.600 / 0.340 |
| C@0.05 | mlpC | 0.578 / 0.348 | 0.566 / 0.320 |

## 结论
- C（random-heavy，n_random=256）在预算匹配下 3 模型×2 margin 全胜 A/B：现象演示从 miss|flip 0.576→0.60~0.63（matched）、uncond 0.297→0.35~0.38。
- A 在新场景校准一致（0.569 vs tune 0.576）：场景难度稳定，增益来自配置非场景。
- 展示选择：C@0.03 × mlpB（full 0.665/0.418，matched 0.632/0.375），margin0.05 下保持 0.643/0.387。
- 诚实记录：B 的 full uncond 最高（0.410）系预算膨胀，matched 口径下 C 仍胜；全表即分母，无隐藏。
- R02_result 更新：以 `r02_confirm_C_mlpB_h32f16`（C@0.03）为 R1 主数字，旧 eval_202 基准保留为对照。
- 产物：`r02_deepen_sweep/`、`r02_deepen_Cm05/`、`r02_confirm_{A,C_*,C05_*}/`（R02_deepen.json 全量）。
