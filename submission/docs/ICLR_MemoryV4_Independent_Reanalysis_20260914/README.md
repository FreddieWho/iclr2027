# Memory V4/P2 独立复算

本离线复算已完成。完整结论、方法、接口与可用边界见
[`PHASE_REPORT_20260914.md`](PHASE_REPORT_20260914.md)；本次核验和产物记录见
[`RELEASE_CHANGELOG_20260914.md`](RELEASE_CHANGELOG_20260914.md)。

审计对象：FreddieWho/iclr2027，commit `5646f1d69154d2294817a74e1dc5f148265a72f2`。
源文件：https://github.com/FreddieWho/iclr2027/blob/5646f1d69154d2294817a74e1dc5f148265a72f2/submission/artifacts/v4_confirmation_rows_compact_v2_20260914.jsonl

## 数据来源与边界

`history_inputs.csv` 的 48 个 reader-history 行由上述已提交文件中的 U 与 L_native 数组逐项转录，
没有新增样本、补齐答案、合并 P1 或重跑模型。原文件中 null 保留为空白。
原文件中长度为零是无可评分记忆的状态值，不是实际零长度压缩。

本次能复算 history-level 差值，不能核验未提交到仓库的 memory 正文、question-level 答案/评分或
信息类型级逐 history 数据。仓库 .gitignore 明确排除了 work/。
因此复算不是第三方完整重复实验，也不证明原始评分无误。

## 计算

运行 `python recompute.py`，只读本地 CSV，无网络调用和 API 费用。
统计单位为 history；每位 reader 分开分析。20,000 次 history bootstrap，90% percentile CI。
所有对比使用 seed=2026091303；这是统一种子的独立敏感性复算，不应冒充原 runner 每个对比的精确随机抽样序列。
SD=staged2−direct；RD=rewrite−direct；SR=staged2−rewrite。输出数值单位为百分点。

- `three_arm_complete`：与主报告相同的三路径共同完整样本限制。
- `pair_specific`：某个对比只要求涉及的两条路径有效。未涉及路径的失效不再排除该历史。

两种估计对象并不完全相同；三路径共同完整可作为对照一致性的敏感性分析。
pair-specific 回收已有可用数据，但仍然条件化于对应路径的成功，不能消除非随机失效的选择偏差。
当前区间宽，不能依据不显著宣布等价；本轮重分析属于审计/探索而非新确认。

## 主要复算结果

SD：R1 n=13，−1.923 pp，90% CI [−13.462,+9.615]。
SD：R2 n=14，−0.893 pp，90% CI [−12.500,+9.821]。

## 路线意见

不据此切换 Memory 主线，也不把当前结果写为科学无效。
Action-Mode 按当前已完成外部确认的状态推进；Memory 最多保留一轮针对性补测：
修复真实预算操纵、共享 direct 起点的 rewrite 对照、技术缺失，并在新历史上检验关系多跳与
撤回/例外的类型交互，控制证据相对位置与模板。不要原样扩大当前宽矩阵。
