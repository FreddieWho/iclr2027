# 来源与最近邻定位

## 仓库（固定2c3e2d9）

- `reports/e832_focus/STRUCTURE_DECISION.md`：有效source对照2188/871、BCE及五seed排序结果。
- `reports/e832_focus/route1/REPORT.md`：新source与T1/T2结果、样本/训练合同变化。
- `experiments/e832_focus/route1_cross_task/common_runner.py`：T1/T2 rich、独立排序、source四点求和、first12候选、生成去重与不保留权重。
- `experiments/e832_focus/structure/a_source_compare.py`：另一套真正保留segment psi的修正结构，不与上条混为一谈。
- `experiments/f095_campaign/u10_oracles.py`：oracle与合法角色。
- `reports/e832_focus/route2/REPORT.md`：v2 28 quartet/19parent与seed方向。
- `experiments/e832_focus/route2_visual/{visual_mechanism,gpu_run,data_generator}.py`：mask、whole-image context、J3/J4、数据来源。
- `reports/e832_focus/route3/REPORT.md`、`route4/REPORT.md`：解析基线和构造不变性范围。

URL基底：`https://github.com/FreddieWho/iclr2027/tree/2c3e2d9640607c32aacf86c8758cd1e2f8ef317b`

## 研究文献（本轮检索核验，引用主文/作者原文，不用二手摘要当理论）

1. Santoro et al., A simple neural network module for relational reasoning, NeurIPS 2017.
   `https://proceedings.neurips.cc/paper/2017/hash/e6acf4b0f69f6f6e60e9a815938aa1ff-Abstract.html`
   对象关系模块不是本项目首创。
2. Wagstaff et al., Universal Approximation of Functions on Sets, JMLR 23(151), 2022.
   `https://jmlr.org/beta/papers/v23/21-0730.html`
   集合网络的表达性条件已有系统理论，不能把单一小池化失败推成集合学习失败。
3. Dym, Lawrence & Siegel, Equivariant Frames and the Impossibility of Continuous Canonicalization, ICML 2024.
   `https://proceedings.mlr.press/v235/dym24a.html`
   完整字典序规范化可能引入不连续；本项目固定有限群全平均保持连续，不能误套“不可能”结论。
4. Lin et al., Equivariance via Minimal Frame Averaging for More Symmetries and Efficiency, ICML 2024.
   `https://proceedings.mlr.press/v235/lin24i.html`
   群/框架平均作为强基线，不作为新方法。
5. Koh et al., Concept Bottleneck Models, ICML 2020.
   `https://proceedings.mlr.press/v119/koh20a`
   分开感知与概念决策、干预概念是已有基础。
6. Espinosa Zarlenga et al., Avoiding Leakage Poisoning: Concept Interventions Under Distribution Shifts, ICML 2025.
   `https://proceedings.mlr.press/v267/espinosa-zarlenga25a.html`
   不能把概念头看起来可读当成可靠可干预接口；分布外替换可能出问题。
7. Veličković & Blundell, Neural Algorithmic Reasoning, Patterns 2021.
   `https://arxiv.org/abs/2105.02761`
   已有“将算法计算迁至原先不可直接访问的观测”的研究方向；本项目须靠具体可控干预和泛化增量区别。
8. Bevilacqua et al., Neural Algorithmic Reasoning with Causal Regularisation, ICML 2023.
   `https://proceedings.mlr.press/v202/bevilacqua23a.html`
   依据中间计算构造不变性与外推并非空白，模块迁移需清晰比较。
9. Park et al., Relational Knowledge Distillation, CVPR 2019.
   `https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html`
   “迁移关系”不是新概念；其对象是样本间关系，和本项目场景内几何应区分。

检索边界：以上足以界定关键近邻，不是穷尽2026全部投稿的优先权证明。不宣称无人做过任务充分表示、关系计算迁移或匹配损失。
