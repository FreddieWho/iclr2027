# PLAN — P4 AMR-Fixed（M1）全周期

科学问题：P3/T5R 已用冻结证据建立"位移如何在构件联盟上分配是表征的一等变量，
且局部几何可预测、修复可被外部确认"。AMR-Fixed（M1）问的是下一步：**把 P3
机制证据转化为固定的"频段 × 支持/关系"路由约束后，一个显式方法能否在
robustness–structure Pareto 上超过既有 baseline**（CAP、centering、
canonicalization、relational pooling），而不只是把响应压大。

假设：

- H1（路由有效性）：M1 的固定机制路由在 dev 上改善 robustness–structure Pareto
  （相对 CAP/centering/canonicalization/relational pooling 的 matched-capacity
  对照），不要求所有频段响应变正。
- H2（机制一致性）：M1 学到的模式指纹与 P3 证据相容——不变性收益集中在
  context-nuisance 频段，coupling 频段保持可恢复性；不要求逐频段方向预设。

判据（描述性）：以 dev 任务指标（zone/pair/geometry/centroid）＋干预指标
（full/diag/基线）联合判定；任一系统性反转即收缩 claim 并停下报告。
M1 不过即不启动 M2（spec §13）。

为什么值得做：P3-R5 门已闭合（T5R6 CONFIRMED）；用户 2026-09-05 裁决选项 C
（D-20260905-P4-001）。AMR 是项目立项时的核心方法贡献，M1 是它的最小可证伪形态。

读取纪律：J03WQQ 与 SoccerTrack-v2 在 T5R 锁下各已消耗一次读取。P4 的任何
保留场/外部确认读取都是**新的协议事件，届时需用户显式授权**；在那之前全部
工作限 train/valid dev 集。
