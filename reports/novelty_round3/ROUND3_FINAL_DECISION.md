# ROUND3 Final Decision (2026-09-22)

Verdicts: A `INTERVENTION_SUFFICIENCY_NEGATIVE` / B
`TRANSFERABLE_REPAIR_SET_NEGATIVE` / C `PATH_REPAIR_ENGINEERING_ONLY` /
D `MOTION_CONFOUND_EXPLAINS_S2`.

## 十问
1. Intervention训练比同数据static augmentation更能改善composition吗？不能。
   trans 0.454 vs 0.450，comp双双饱和，A6对A0仅1/3。
2. 同标签不同后果的状态在新表示中分得更开吗？没有（A2>A0仅1/3；倒是有趣：
   A1端点增广把区分压得最低）。
3. 一模型的repair examples能跨seed/arch修别模型吗？不能稳定。src11有
   9.4pp/2方向性hint，不及bar且一次强反转——记观察，不记claim。
4. 比flipmine更省样本吗？没跑出效率差（挖掘59–82k queries才留150）。
5. Path repair优于point并泛化吗？否。seen上point 0.04 vs path 0.24完胜；
   unseen无样本。分区工具 verified 留用。
6. 争夺信号匹配运动后还在吗？基本不在：运动解释70–90%，残差+2~7pp小且
   不稳定。S2降级为混杂观察。
7. Cover提高真实事件selectivity了吗？问题已随S2死亡而moot，未测。
8. 哪条进正文？没有。四包全军覆没是结论本身：update缺陷的边界又收窄一圈。
9. 哪些进appendix？A1-collapse curiosity一行；path分区verified一行；
   src11-hint诚实注记；S2混杂分析。
10. 哪些永久关闭？Intervention双头、transfer mining、path repair、matched
    selectivity（L-002彻底消耗）。半径诊断（L-014）与confirm额度（L-013）
    保持不动。

## Paper fate
正文零修改。Round3的唯一论文价值是failed-alternative护城河：
intervention-loss inert、counterexample不transfer、path-repair输给point、
真事件信号是运动混杂——四个"此路不通"钉死，Discussion收尸段备料。
No sixth direction.
