# PLAN — T5R6 独立外部确认（SoccerTrack-v2）

科学问题：T5R5 在单场同源保留比赛上复现的“任务语义分离＋局部几何预测受控响应”机制，
换到独立 provider（不同联赛级别、不同采集系统）与多场比赛时是否保持方向。

假设：

- H1（任务确认）：`update_ratio_2to1` 在 SoccerTrack-v2 多比赛上的 task–geometry 表现为
  与 T5R5 同方向（context 非劣效、intrinsic pair 保持、geometry 不坍缩、translation 近零）。
- H2（机制确认）：归一化局部几何（full/diagonal）预测受控 support-reallocation 响应的能力
  在独立 provider 上复现，多比赛逐场 full 相对简单谱基线保持优势方向。

判据（描述性、非预注册门控）：逐场 Spearman 为主、match-macro 为主汇总、pooled 为辅；
多比赛同方向即算支持，不逐场设硬门槛；任一方向系统性反转即记为机制边界并停下报告。

为什么值得做：T5R5 只是单场同源确认；P3-R3（跨比赛泛化）与 P4（AMR） gate 要求独立确认。
这是路由 A 指定的唯一正路，不开新搜索、不调参。
