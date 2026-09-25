# 来源与证据边界

仓库固定commit：e832887c938948b23e8783719d493f2b9b8c3a17。
仓库路径的标准定位为：`https://github.com/FreddieWho/iclr2027/blob/e832887c938948b23e8783719d493f2b9b8c3a17/<path>`。

## 已读取的主要项目来源

|ID|路径|用途|
|---|---|---|
|R01|reports/e1a933_review/FINAL_COMPLETION_MATRIX.csv|本轮实际完成与范围|
|R02|reports/e1a933_review/L015_L006_REPORT.md|10seed、G8特征与池化阴性|
|R03|experiments/e1a933_review/leads_l014_l015_l006.py|8维特征和线性rho实现|
|R04|experiments/f095_campaign/u01_models.py|源TypedPairMLP加性结构|
|R05|reports/e1a933_review/O01_DATA_OPTIMIZATION_REPORT.md|数据/优化/续训因子|
|R06|reports/e1a933_review/VISION_GPU_RESULTS.md|21臂视觉与迁移/辅助学习|
|R07|reports/e1a933_review/N02_SAMPLING_ACCESS_REPORT.md|48臂采样/MAC/新池|
|R08|reports/e1a933_review/N03_DECISION_BOTTLENECK_REPORT.md|瓶颈与经典基线|
|R09|reports/e1a933_review/O05_LEARNED_PARTIAL_TRACKING_GPU.md|足球边界|
|R10|reports/e1a933_review/O06_NATIVE_VLM_GPU.md|原生VLM已执行的窄阴性|
|R11|reports/e1a933_review/R03_LINEAGE_REPORT.md|新血缘纠正、dev512实际未见parent|
|R12|paper/main.tex|结构、Figure1与AI-use现状|
|R13|paper/sections/00_abstract.tex|摘要|
|R14|paper/sections/01_introduction.tex|贡献与P1中心|
|R15|paper/sections/04_mechanism.tex|主文结构比较、跨任务与优化|
|R16|paper/sections/04b_boundary.tex|视觉/足球/P2位置|
|R17|paper/sections/discussion.tex|现有限制与残留口径|

本包中的数值未独立重训；若现场生成文件与报告冲突，以重新核验的逐样本源结果为准，并保留修正记录。尤其O01长报告有历史段和新增段，不要把108与648计数不加区分合并为独立样本数。

## 已检索的原始文献/官方来源

1. ICLR 2027 Reviewer Guidelines — https://iclr.cc/Conferences/2027/ReviewerGuidelines
   问题、证据、意义和社区新知识是核心，不要求SOTA。不能据此预测某篇论文的精确胜率。
2. Santoro et al., A simple neural network module for relational reasoning, NeurIPS 2017.
   https://papers.nips.cc/paper/2017/hash/e6acf4b0f69f6f6e60e9a815938aa1ff-Abstract.html
   关系交互模块不是新思想；本项目应提出更具体的监督到联合正确性的科学比较。
3. Lee et al., Set Transformer, ICML 2019.
   https://proceedings.mlr.press/v97/lee19d.html
   集合交互的直接强基线；不变与交互是不同设计需求。
4. Wagstaff et al., On the Limitations of Representing Functions on Sets, ICML 2019.
   https://proceedings.mlr.press/v97/wagstaff19a.html
   集合函数表达能力有条件；本包加性反例是独立初等推导，不冒称论文原定理。
5. Kaba et al., Equivariance with Learned Canonicalization Functions, ICML 2023.
   https://proceedings.mlr.press/v202/kaba23a.html
6. Dym et al., Equivariant Frames and the Impossibility of Continuous Canonicalization, ICML 2024.
   https://proceedings.mlr.press/v235/dym24a.html
   不把任意规范化都当平滑；也不能将该论文关于特定frame的结论错套到完整有限群平均。
7. Kamath et al., The Hard Positive Truth about Vision-Language Compositionality, ECCV 2024.
   https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2149_ECCV_2024_paper.php
   正负例权衡不是新发现；本项目区别应在同quartet错误迁移、完整修复与干预。
8. Yan et al., Positive-Congruent Training, CVPR 2021.
   https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html
9. Press et al., Measuring and Narrowing the Compositionality Gap in Language Models, Findings EMNLP 2023.
   https://aclanthology.org/2023.findings-emnlp.378/
10. Jones et al., Selective Classification Can Magnify Disparities Across Groups, ICLR 2021.
    https://arxiv.org/abs/2010.14134
11. Koh et al., Concept Bottleneck Models, ICML 2020.
    https://proceedings.mlr.press/v119/koh20a.html

这里是定向最近邻定位，不是穷尽文献综述；未找到同名论文不能证明新颖性。所有假设与包内路线是建议，不属于已经完成的项目结果。
