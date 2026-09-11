# ROBUSTNESS

状态：`NOT_RUN`

本轮没有正式 C1/R1 memory 可供 R2、C2 或自然数据 reader 复现。LongMemEval cleaned 已下载并抽样 32 条完整 history，但每题标记为 `semantic_required`，尚未运行 semantic grader；自然 history 也明显超过默认 32K 配置，不能在未验证 context window 时强行评估。

| 泛化方向 | 状态 | 非主张 |
|---|---|---|
| R2 reader 读取冻结 C1 memory | `NOT_RUN` | 不声称 reader-family 泛化 |
| C2 compressor 复现关键路径 | `NOT_RUN` | 不声称 compressor-family 泛化 |
| LongMemEval-S 32 history | 数据已准备，评估 `NOT_RUN` | 不声称自然数据方向或 benchmark 分数 |

没有第二设置的独立效应、CI 或 semantic scoring 结果；因此所有 GO route 的泛化底线均未满足。
