# FINAL_PROJECT_VERDICT (2026-09-22, 通俗中文)

1. P1 closure 后支持多强 claim？P1-STRONG（坐标域）：6/6 模型 dual 反转，
   确认池冻结复现，分组 CV 显示难度变量几乎无 held-out 信号，配对残留 CI
   全不含 0。限定：坐标受控变化，不做 universal 定理。
2. confidence inversion 是否还有未解决 confound？有：oracle margin/distance
   之外的未测量难度（已声明）。已解决：起点 margin（真值已审计）、位移、
   编辑族、类别、parent 聚类、阈值距离（AUC 0.99 仍留 confidence 残留）。
3. P3 的 endpoint repair 中多少是 full repair？基线 110 子集上 3-10%
   （dev，parent-cluster CI），确认池 relflip 45-50%。
4. migration 经 parent-cluster 后稳定吗？稳定：dev M=0.36-0.40，CI 全不含 0；
   keepbal 公平重跑后结论不变。
5. fair keepbal 是否改变判断？否：J/R_full/M 三种子方向不一致，迁移未被
   可靠降低。方法分支维持关闭。
6. relfeat 是否多 seed 复现？是：J 3/3 提升（dev 0.13-0.21，confirm
   0.16-0.23），静态同步改善，无灾难性代价。
7. relation representation 是否成为新的正向结论？是（P3-METHOD）：
   relflip J 0.38-0.47、R_full 超过 M（确认池），明确按 §16 边界表述
   （手工 relation-aware 输入，不 general architecture claim）。
8. P2 的 affine/code mismatch 如何解决？已解决：报告重写为与冻结宽网格
   代码逐字对应；s11 贴边如实列为局限，未再放宽。
9. equal-coverage 后 P2 是 Main 还是 Discussion？P2-MAIN_SCOPED：等覆盖质量
   占优多数组稳定＋affine 失败核心行为＋代码可复现；但全在 dev 场景/银行，
   故 repair 小节短 subsection＋Discussion 回声，不 standalone。
10. 当前最强的 3 个 contribution：(a) P1 置信反转＋筛选后果（确认池）；
    (b) P3 error migration 主导＋relflip 转向 full consistency；
    (c) P2 能力/operating-point 分离＋规则依赖。
11. 哪些旧 contribution 已被删除：①"首次条件化"；②"compositional C1"；
    ③ incidence/切向决定论；④"修复不提高精度"全称；⑤足球 P1 旧数；
    ⑥"修复治愈更新失败"；⑦"缺陷在更新不在看"口号。
12. 摘要讲什么、不讲什么：讲 85.8%、双分母 transition 数、P1 反转（坐标）、
    组件修复数、P3 迁移主导、足球/像素定位、"正确性不是一维"收束。不讲：
    C1 组合口径、切向故事、治愈叙事、跨域 P1、首创 conditional。
13. Figure 主线：Fig1 现象链（headline）→ Fig2 置信反转 → Fig3 迁移 →
    Fig4 下游/外部确认。relflip 可作 Fig3 panel 或附录。
14. 是否还存在必须完成的新实验？否。本轮授权的新训练（keepbal 公平重跑、
    relfeat×2、relflip×3）全部完成；确认池 P1 部分已用。剩余缺口（pixel P1、
    更多 rel 变体）已明确列为局限/未来工作，不阻塞投稿。
15. 项目是否正式进入 PAPER POLISH / SUBMISSION CLOSURE？是。
    EXPLORATORY PHASE CLOSED（2026-09-22）。后续只做写作打磨与审稿响应；
    新方向需单独授权。

最终故事（数据支持才用）：Models can be right now, fail to update when
meaning changes, and appear repaired without recovering consistent
competence. Static confidence, endpoint correctness, full compositional
consistency, and downstream success are distinct notions of reliability
under change.
