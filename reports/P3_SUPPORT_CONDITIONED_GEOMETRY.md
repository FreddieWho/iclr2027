# P3 Support-Conditioned Local Geometry

状态：DESIGN_LOCKED_NOT_RUN  
阶段：P3_CAUSAL_MECHANISM  
性质：post-P2 mechanistic follow-up，exploratory

## 当前结论边界

P2 的唯一冻结 route 为 mixed_or_graph_specific。本报告不重算、不覆盖 P2，也不把 P2 representation response 写成任务改善。P3 的待检验对象是归一化 embedding 的局部几何：

\[
q_f(X,\delta)=\frac12\|J_f(X)\delta\|_2^2,\qquad
G_f(X)=J_f(X)^\top J_f(X).
\]

Action-Mode Spectrum 继续作为该条件化对象的边缘汇总。当前没有 P3 结果；所有 P3-G1–G5 均为工作假设。

## 冻结施工路线

1. T0：复现 250 samples × 9 frozen models 的 P2 baseline embedding，暴露节点/interaction/pooling 层并验证 JVP、有限差分和二阶近似。
2. T1：在 P2 fracture response 上，以 match 为统计单位比较频率/residual/support 基线、diagonal geometry、full geometry 和 off-diagonal geometry。
3. T2：冻结 T1 公式、层、指标和统计单位后，生成 response-blind 的同资产 prospective reallocations。
4. T3：分解 diagonal/off-diagonal、team/support/edge block，并定位信息形成或丢失的层。
5. T4：只选择一条有证据支持的 matched-capacity causal switch。
6. T5：在方法结果前冻结任务定义，完成 perturbation localization 和一个非 intervention-label 的客观任务。

完整 causal matrix 只是 gate 后 optional extension；P4 AMR 在 gate 前不运行。

## 结果记录模板

正式运行时，本报告需回写：

- retrospective 与 prospective 分开；
- match-level Spearman、MAE、R²、方向准确率、calibration、效应量和 bootstrap 区间；
- architecture、seed、graph、role、epsilon 的失败条件；
- geometry 相对于最佳 frequency/residual baseline 的增量；
- layer/block 定位、因果开关的预测—操作—几何—任务链；
- manifest、receipt、SHA-256、figure source tables 和 known limitations。

若 T1/T2 不支持预测，或 T4 只有 response 无任务收益，必须在本报告保留失败结果并降低 claim；不得补选干预、扩大模型或直接训练 AMR。
