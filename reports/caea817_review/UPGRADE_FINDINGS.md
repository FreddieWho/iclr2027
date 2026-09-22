# UPGRADE_FINDINGS — 02 探索升级总览（dev512 为主；exploratory 除非另注）

## 已知事实（本轮复用，未改动）

- 四臂 J（三种子）：raw 0.051–0.055；raw+flip 0.059–0.076；
  relfeat 0.131–0.211；relflip 0.384–0.422（U1 重算逐点一致）。
- 同一 flip 监督：`mine_flips(seed=5)` 在 train_101 上重算与 r04b 三种子
  `mined.npz` 逐条一致（1534 条，sha `abf78112…`）。
- 无泄漏：训练场景 / mined 终点 / dev 起点 / dev AB 终点四集合坐标哈希
  两两交集为 0。
- 既有确认：relflip J 在另一 parent 池仍高；确认池完整 2×2（n=509）同方向
  I=+11–22pp 点估计，见 U1_CONFIRM_ADDENDUM（存档摘要后续分析，零新读数）。
- 代数事实：四半径由完整点对距离决定（U4 实证对应：sixdist≈tenfeat）。

## 重新计算（本轮新推导）

- U1：I = +24.1/+24.5/+16.5pp，parent-cluster CI 三种子全不含零；
  配对差 relflip−relfeat +0.19–0.25（CI 全不含零），raw+flip−raw 跨零；
  H 集恒等式 R_endpoint = R_full + M 三种子成立。
- U2：H：raw 0.54–0.57 → raw+flip 0.67–0.73 → relfeat 0.81–0.87 →
  relflip 0.87–0.92；relfeat→relflip 的 J 翻倍主要来自阈值实现而非新可分性；
  parent 对半阈值转移保持臂排序（relflip 评价半 J 0.31–0.47 vs H 0.88–0.90）。
- U3：固定高风险集上 relflip 终点正确率 0.74–0.84（基线 0.36–0.56），
  完整修复 0.26–0.32（基线 0.00）；Q2 反转形状保留但绝对水平下移；
  preserve 三项无代价（static 0.04、single-flip 0.17–0.19、FA ~0.01）。
- U4：centered≈raw（排除平移 nuisance 解释）；sixdist≈tenfeat（半径冗余）。

## 推断（带范围限定）

1. J 尺度经验正交互成立（dev 配对 CI＋确认池同方向；限所测表示/任务）。
2. 表示改善状态排序，flip 监督（在关系输入上）主要改善固定阈值实现。
3. 有效修复减轻 P1 的水平、保留其形状——修复边界，不是 P1 被修好。
4. 关系收益不是去姿态 nuisance 处理；四半径无实质贡献。

## 未支持项（见 NO_NEW_CLAIM_WITHOUT_EVIDENCE.md）

完整 2×2 交互的 paired-CI 级确认（点估计已有，见 U1_CONFIRM_ADDENDUM）、encoder-vs-head 归因、因果中介链、H 作部署准确率、
跨模型 logit 尺度比较、向 foundation models 推广。

## 稿件决策（02 §3：A＋B 两项都拿到）

- A（U1+U2）：主文四臂图（数据源切到 U1_SUMMARY.json）＋ J/H 对照一 panel（主文或附录）。
- B（U3）：主图一 panel 或附录（固定集 0.74–0.84；反转形状保留边界句）。
- 摘要/贡献替换句（按结果选择）：

  > On 237 controlled quartets, combining label-free relational inputs with
  > flip supervision raises full joint correctness from 0.05 to 0.38–0.42
  > (exploratory interaction, +17–25pp over the additive expectation):
  > the representation improves state separability while flip supervision
  > mainly realizes it at a fixed threshold, and the effective repair lowers
  > but does not remove the confidence inversion on later flips.
