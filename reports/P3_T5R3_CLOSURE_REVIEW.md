# P3-T5R3 Closure Review

日期：2026-09-04  
结论：`ACCEPTED_FOR_T5R4_ROUND1_EXPLORATION`

T5R3 没有发生路线级偏移。IDSSE 暂代 SNGAR 的开发角色是已固定的项目决定；train/valid 防火墙有效，`J03WQQ`、SNGAR test、旧 exposed heldout 和外部结果均未读取。T5R3 的 dual 结果只记为方向性 sanity pass。

保留三项证据边界：

1. 固定 kNN-4 message-passing 是锁定 Phase-GAT 的等价实现，后续比较必须沿用同一实现口径；
2. geometry 数值是 natural-pair geometry–latent-distance proxy，不是 intervention-response；
3. valid 只有 2 场，T5R4 仍是探索，不产生 P4 或最终方法收益 claim。

因此允许启动 T5R4 Round 1。搜索只需遵守已有硬边界：train/valid only、每个候选一个主要机制轴、seeds `11/23/47`、最多 2 轮且每轮最多 6 个候选；候选的具体强度和组合保留探索空间，不自动创建 candidate lock 或读取 reserved match。

Round 1 先测试 loss balance、gradient routing 和 head placement 的少量代表点。候选只做结果汇总，不自动晋级；是否进入 Round 2 由 Round 1 结果审阅决定。该 review 不要求重跑 T5R3 或补建大规模统计矩阵。
