> ⚠️ **历史档案（P4-AMR 时代，2026-09-05）**：本 ROADMAP 已终结（P4-N1–N4 完成，
> N5/N6 未授权即随路线调整关闭）。最终叙事与收官节点见
> `reports/final_closure/FINAL_PROJECT_VERDICT.md`；当前状态见 `STATUS.md`。
> 以下原文保留作历史。

# ROADMAP — P4 AMR-Fixed（M1）全周期

| 节点 | 内容 | 检验的假设 | 状态 |
|---|---|---|---|
| P4-N1 [infra] | AMRModel 实现：双通道＋Chebyshev 谱带滤波（B=6）＋谱白化采样＋路由损失；JVP 与有限差分校验；单元测试 | —（[infra] 前提） | 待执行 |
| P4-N2 | M1 config lock：固定"频段 × 支持/关系"路由分配（由 P3 证据导出）、损失权重、白化采样协议、seeds/splits；锁审计。**任何 AMR 结果产生前必须冻结** | —（协议前提） | 待执行 |
| P4-N3 | M1 训练（3 seeds）＋dev 评估（任务＋干预＋几何剖面） | H1、H2 | 待执行 |
| P4-N4 | matched-capacity 对照（CAP、centering、canonicalization、relational pooling 子集，dev） | H1 | 待执行 |
| P4-N5 | candidate lock＋保留场确认读取（**需用户授权 J03WQQ 二次读取**） | H1/H2 确认 | 待授权 |
| P4-N6 | 外部确认读取（**需用户授权 SoccerTrack-v2 二次读取**）＋P4 报告＋ledger 回写 | H1/H2 外部确认 | 待授权 |

说明：N1–N4 全部限 train/valid dev 集，无需任何授权即可推进；N5/N6 是新的
协议事件（T5R 的单次读取已消耗），到达该节点时停下等用户授权。
M1 未达到 Pareto 改善判据时，按 spec §13 收缩，不启动 M2。
