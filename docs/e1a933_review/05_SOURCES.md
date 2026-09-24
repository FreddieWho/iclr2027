# 溯源与外部文献

所有仓库路径固定于 `e1a933e6b32dd07c6895c7925cbccd9004da7f54`。以下是已实际读取的主要来源；列表不是声称全仓库逐行审计。

## 仓库来源

| 对象 | 来源 |
|---|---|
| 完成计数、补测状态 | reports/f095_campaign/COMPLETION_VERDICT.md；UPGRADE_20260924.md；AUDIT_20260923.md |
| D01 | reports/f095_campaign/D01.md；experiments/f095_campaign/d01_capacity.py；d01_make_data.py |
| D02/U01/U02 | reports/f095_campaign/D02.md；U01.md；U02.md；对应symmetry/u01模型接口 |
| D03 | reports/f095_campaign/D03.md；experiments/f095_campaign/d03_build_bank.py；d03_eval.py |
| D04 | reports/f095_campaign/D04.md；experiments/f095_campaign/d04_vision_train.py；experiments/last15h/n08_visual.py |
| D06 | reports/f095_campaign/D06.md；experiments/f095_campaign/d06_probe.py；d06_roles.py |
| D08 | reports/f095_campaign/D08.md；experiments/f095_campaign/d08_fairfight.py |
| D09 | reports/f095_campaign/D09.md（含n236→218去重历史） |
| D10 | reports/f095_campaign/D10.md；experiments/f095_campaign/d10_frontier.py |
| U03/U04 | reports/f095_campaign/U03.md；U04.md；experiments/f095_campaign/u04_refit_heads.py |
| U06 | reports/f095_campaign/U06.md；experiments/f095_campaign/u06_relation_distill.py |
| U07 | reports/f095_campaign/U07.md；d01_capacity.py |
| U10 | reports/f095_campaign/U10.md；U10_SOURCE_PREDICTION.md；experiments/f095_campaign/u10_targets.py；u10_models.py；u10_orbit_eval.py |
| U10独立算术 | artifacts/f095_campaign/U10/U10_T1_EVAL.json；U10_T2_EVAL.json |
| paper | paper/sections/00_abstract.tex；04_mechanism.tex；STATUS.md |

构造某个来源的固定链接：
`https://github.com/FreddieWho/iclr2027/blob/e1a933e6b32dd07c6895c7925cbccd9004da7f54/<path>`。

## 实际运行与未运行

实际运行：本包 `checks/reproduce_contract_failures.py` 九项本地生成样例；`checks/recompute_archived_contrasts.py` 对已归档汇总数字的转录算术；JSON与ZIP完整性检查。
未运行：原始数据全量评估、项目模型重训、全部项目tests、LaTeX全文编译。未改远程仓库。镜像函数只是为了最小合同反例；并不代替现场从production模块导入后的测试。

## 外部研究（原始论文/官方文档）

1. Lopez-Paz, Bottou, Schölkopf, Vapnik. *Unifying distillation and privileged information* (2015 preprint/ICLR 2016 lineage).
   https://arxiv.org/abs/1511.03643
   用途：特权信息与蒸馏是既有框架；不是本项目的新概念。本次没有声称已穷尽可观测性匹配邻域。

2. Carion et al. *End-to-End Object Detection with Transformers*, ECCV 2020.
   https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/832_ECCV_2020_paper.php
   用途：集合预测和匹配loss已有成熟先例。引用不意味必须使用Transformer或大规模DETR。

3. Zhang. *Making Convolutional Networks Shift-Invariant Again*, ICML 2019, PMLR 97:7324–7334.
   https://proceedings.mlr.press/v97/zhang19a.html
   用途：降采样、抗混叠与网络敏感性的已有研究；本项目必须通过固定观测信息的实验验证，不能直接把该论文当作D04原因。

4. scikit-learn官方 IsotonicRegression 文档。
   https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html
   用途：替换手写保序回归并测试ties/训练点预测；现场应锁定实际安装版本。

5. PyTorch官方 BatchNorm2d 文档。
   https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
   用途：requires_grad冻结参数不等于冻结running statistics；应测试eval模式与buffer哈希。

6. Park et al. *Relational Knowledge Distillation*, CVPR 2019.
   https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html
   用途：关系型蒸馏的近邻。官方页面本次直接打开遇到403，先前检索可见摘要；正式bib请现场核实。不要误说该方法本身专门解决同一场景中的不可见端点身份。

上述来源只支持相应背景，不替本项目证明新机制。N01条件方差结论、N02固定插值不增加观测信息、N03瓶颈干预逻辑在本包中明确作为数学推理/实验假设；最终科学效应必须由新数据支持。
