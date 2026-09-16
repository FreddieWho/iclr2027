# Memory V4/P2 独立复算报告

**日期：2026-09-14｜状态：离线复算完成；不构成完整重复实验。**

## 范围与结论

本报告只对 P2 compact-v2 已发布的 history-level utility 汇总做独立统计复算：24 个 history、R1/R2 各 24 行，共 48 个 reader-history 记录；路径为 `direct`、`staged2`、`rewrite`。没有新增历史、补答案、重算问题级评分、重新压缩记忆或调用模型/API。

主要结论仍为 **`INCONCLUSIVE_NO_GO_SIGNAL`**。完整三路径样本上的差值与主报告一致。只要求某个对比的两条路径有效时，可恢复部分 history，点估计更接近零，但区间仍宽且全部跨零。此结果既不支持将 Memory 升为主线，也不能证明路径效应为零或两方案等价。

## 输入、来源与统计单位

- 锁定源：`artifacts/v4_confirmation_rows_compact_v2_20260914.jsonl`，仓库提交 `5646f1d69154d2294817a74e1dc5f148265a72f2`。源文件 SHA-256：`9a5502b2e00f4f9fcaf2daffeaa820f28e0eeba25cca4eef7d03fc4c3e18497e`。
- 输入：[`history_inputs.csv`](history_inputs.csv)，48 个唯一 reader-history 键。复算脚本逐项校验源文件和 CSV 的 288 个 utility/长度字段；null utility 保留为空白。
- `U_*` 是 history-level utility，范围 `[0,1]`；表中效应乘 100，以百分点（pp）报告。`L_*` 是对应 memory 的 native token 长度。源中的长度 0 表示无可评分 memory，不解释为成功生成了零 token 的 memory。
- 不能从这份汇总表复核 memory 正文、question-level 答案/评分或逐题证据；所以这是对已提交派生统计输入的复算，不是第三方完整重复实验，也不独立验证原始评分正确性。

## 估计对象与方法

差值定义：`SD = staged2 − direct`，`RD = rewrite − direct`，`SR = staged2 − rewrite`。统计单位始终为 history，而不是问题数。

- `three_arm_complete`：仅保留三条路径都有效的 history。该结果可与主报告的 complete-case 比较直接核对。
- `pair_specific`：每项对比只要求该对比涉及的两条路径有效，不要求第三条路径有效。它增加可用 history，但改变了纳入规则和对应样本，因此与三路径完整估计不是完全相同的目标。
- 每位 reader、每种纳入规则、每项对比独立做 20,000 次 history bootstrap；报告 percentile 90% CI，seed `2026091303`。没有做多重比较校正，也不把 CI 跨零解释为等价或无效应。

## 复算结果

效应和区间单位均为 pp；方括号内为 90% CI。

| Reader | 对比 | 三路径完整 | Pair-specific |
|---|---|---|---|
| R1 | SD | n=11；−4.545 [−15.909, +4.545] | n=13；−1.923 [−13.462, +9.615] |
| R1 | RD | n=11；−6.818 [−15.909, +2.273] | n=15；−1.667 [−10.000, +6.667] |
| R1 | SR | n=11；+2.273 [−9.091, +13.636] | n=12；+4.167 [−6.250, +14.583] |
| R2 | SD | n=12；−3.125 [−16.667, +9.375] | n=14；−0.893 [−12.500, +9.821] |
| R2 | RD | n=12；−2.083 [−10.417, +6.250] | n=16；−3.125 [−10.938, +4.688] |
| R2 | SR | n=12；−1.042 [−9.375, +7.292] | n=12；−1.042 [−9.375, +7.292] |

三路径完整的 SD 与既有 V4 主报告一致：R1 −4.545 pp（n=11），R2 −3.125 pp（n=12）。Pair-specific SD 分别纳入 13 和 14 个 history，估计为 −1.923 与 −0.893 pp；两者 90% CI 都宽且跨零。纳入更多可用 history 后点估计向零移动，说明结果对失效样本的纳入规则敏感；这不能消除按路径成功条件化带来的选择偏差。

## 解释边界与路线建议

1. 当前证据不足以支持 `GO_MEMORY`，也没有满足强 `NO_GO_MEMORY` 条件；保留 `INCONCLUSIVE_NO_GO_SIGNAL`，不据此切换 Memory 主线。
2. 区间宽且跨零，不构成“无效应”或“等价”的证据。差异性技术缺失仍在：只要目标路径可用，pair-specific 仍条件化于该路径成功。
3. P1 使用旧 prompt，不能与 P2 合并成同 prompt 的独立复制。R2 是对冻结 memory 的敏感性读取；其执行是在 owner continuation 下覆盖先前 trigger，不是 trigger-compliant 的独立确认。
4. P2 的 8192 native-token 值是软目标，52 个自然停止 memory 均超过目标；本复算不能支持严格等长 endpoint 主张。
5. 当前结果不触发 P3。若未来另行开展补测，应作为新方案冻结：先解决真实长度操纵和技术缺失，再以新 histories 检验预设类型交互；不要把这次 pair-specific 复算包装为新确认实验。

## 文件与下游接口

| 文件 | 用途 / 格式 | 下游可用范围 |
|---|---|---|
| `history_inputs.csv` | 48 行、每位 reader-history 一行；utility 与 native 长度 | 独立重算本报告的 history-level contrasts |
| `recompute.py` | Python 标准库脚本；先核对本地源 SHA 与 288 个转录字段，再生成结果 | 从本目录运行 `python recompute.py`；不访问网络或 provider |
| `paired_reanalysis.csv` | 12 行；reader × 纳入规则 × 对比，含 n、均值、90% CI、seed 和重复次数 | 审计/敏感性分析，不作为新的 reader/question-level 原始证据 |
| `RELEASE_CHANGELOG_20260914.md` | 本次审计、重算和交付记录 | 查核本次产物的来源与验证 |

上游科学结果和路线判定仍以 [`../../reports/V4_FAST_PILOT_REPORT.md`](../../reports/V4_FAST_PILOT_REPORT.md) 与 [`../../reports/V4_ROUTE_DECISION.json`](../../reports/V4_ROUTE_DECISION.json) 为准；本复算不修改它们，也不覆盖原 P1/P2 receipts。阅读顺序：本报告 → changelog → `README.md` → 输入 CSV 与脚本。
