# 来源与已有工作边界

审阅基准为 `f095dfaa1341a057efee52c56fe2aea9c39ded6a`；公开资料核查日期2026-09-23。
repo来源通过已连接GitHub读取。外部资料只用于评审标准、邻近研究与路线扩展，不替代仓库事实。下面的引用定位不是对novelty的穷尽证明；执行前对所选路线补针对性查重，不追求机械引用数量。

## 仓库来源

- **[R01] 当前证据与稿件。** `paper/sections/00_abstract.tex`、`02_instrument.tex`、`03_phenomenon.tex`、`04_mechanism.tex`；`reports/caea817_review/UPGRADE_FINDINGS.md`；`STATUS.md`。基准URL格式：`https://github.com/FreddieWho/iclr2027/blob/f095dfaa1341a057efee52c56fe2aea9c39ded6a/<path>`。
- **[R02] 任务与训练。** `experiments/discovery_campaign/{coord_mlp,r04b_methods,r02_search,common}.py`；`docs/iclr2027_discovery_campaign_20260917/core/relations.py`。原始MLP 6,849参数为按代码的算术结果，不是读取权重文件统计。
- **[R03] 重标号及审计。** `reports/caea817_review/{C1C_CONFIRM_RESULT,C1C_RELABELING_ROBUSTNESS,L007_CORRECTED_RESULTS,CORRECTION_AFTER_GPT6ASTRA,CONFIRMATION_USAGE}.md`。8标号是同对象相关变换，confirm1007被重复使用。
- **[R04] Foundation分支。** `reports/bridge_r/COMPETENCE_FIRST_DECISION.md`及相关最终patch/readout结果；`BRIDGE_R_FINAL_REPORT.md`是较早状态，不能用它的访问阻塞覆盖后续已运行的DINOv3。
- **[R05] 足球。** `experiments/last15h/e2_n01_t5r3.py`；`reports/last15h/E2_N01.md`；`paper/sections/04b_boundary.tex`。核心controlled测试固定WOY，oracle是home mean-x zone；natural P1未复现。
- **[R06] 排序、交互和确认归档。** `reports/caea817_review/{U1_FACTORIAL_REPORT,U1_CONFIRM_ADDENDUM,U2_ORDERING_REPORT,U3_CONFIDENCE_AFTER_REPAIR,U4_CONTRAST_REPORT}.md`；`artifacts/next_novelty/{relfeat/RELFEAT.json,relflip_eval/RELFLIP.json,p3/P3_FINAL.json,relflip/RELFLIP_CONFIRM.json}`。U03是基于已有S/J诊断的新分析建议，不是仓库既有结论。

## 最接近的方法与理论（primary sources）

- **[R07] Frame Averaging for Invariant and Equivariant Network Design.** Puny等；`https://arxiv.org/abs/2110.03336`。有限群平均保证不变/等变不是新方法思想；D02将其作为强反证baseline。
- **[R08] The Hard Positive Truth about Vision-Language Compositionality.** ECCV 2024；`https://arxiv.org/abs/2409.17958`。hard-negative训练可伤害hard-positive，需正面比较，不能声称首次观察repair损坏别的正确预测。
- **[R09] Positive-Congruent Training: Towards Regression-Free Model Updates.** Yan等，CVPR 2021；`https://arxiv.org/abs/2011.09161`。negative flips与Focal Distillation直接相关。
- **[R10] Torchvision ResNet18官方文档。** `https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html`。只是可复现标准视觉baseline示例，非当前最强模型推荐。
- **[R11] Qwen2.5-VL-3B-Instruct官方模型卡。** `https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct`。示例reference，不声称最新/最好；许可证、确切revision和资源用量由执行时官方文件核验。
- **[R12] SoccerTrack-v2官方项目与论文。** `https://atomscott.github.io/SoccerTrack-v2/`；`https://arxiv.org/abs/2508.01802`。数据已有本项目使用历史，不能因独立provider就称当前新holdout。
- **[R13] Task-based End-to-end Model Learning in Stochastic Optimization.** Donti等，NeurIPS 2017；`https://papers.neurips.cc/paper_files/paper/2017/hash/3fc2c60b5782f641f76bcefc39fb2392-Abstract.html`。预测误差与决策效用的差别已有成熟基础。
- **[R14] Deep Sets.** Zaheer等，NeurIPS 2017；`https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html`。合法不变性取决于任务结构；四点无类型集合会丢掉配对。
- **[R15] E(n) Equivariant Graph Neural Networks.** Satorras等，ICML 2021；`https://proceedings.mlr.press/v139/satorras21a.html`。现成等变架构可作baseline，使用它本身不构成新颖性。
- **[R16] Actions Speak Louder than Goals: Valuing Player Actions in Soccer.** Decroos等，KDD 2019；`https://doi.org/10.1145/3292500.3330758`。足球行为的实际价值已有成熟问题，几何代理不可等同进球/传球反事实收益。
- **[R17] Scalars are universal: Equivariant machine learning, structured like classical physics.** NeurIPS 2021；`https://proceedings.neurips.cc/paper/2021/hash/f1b0775946bc0329b35b823b86eeb5f5-Abstract.html`。距离/内积的不变表示不是本项目首创。
- **[R18] ICLR 2027官方指南。** `https://iclr.cc/Conferences/2027/ReviewerGuidelines`；`https://iclr.cc/Conferences/2027/AuthorGuidelines`。强调新知识、意义和证据，不要求一定SOTA；全文2026-09-25 23:59 AoE、主文≤9页、官方样式、匿名、AI-use声明。涉及截止日期以AuthorGuidelines为准，Reviewer FAQ中存在跨年份模板残留。
- **[R19] Excessive Invariance Causes Adversarial Vulnerability.** Jacobsen等，ICLR 2019；`https://arxiv.org/abs/1811.00401`。明确研究任务相关变化被忽略；不能错误区分成“它只研究固定标签鲁棒性”。
- **[R20] Selective Classification Can Magnify Disparities Across Groups.** ICLR 2021；`https://arxiv.org/abs/2010.14134`。置信筛选改善总体却伤害子群已知，P1新意须落到受控变化条件与具体预测问题。
- **[R21] Measuring and Narrowing the Compositionality Gap in Language Models.** Press等，Findings of EMNLP 2023；`https://aclanthology.org/2023.findings-emnlp.378/`。原子正确、组合错误的条件测量有直接先例；不是ICLR正式论文。
- **[R22] CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning.** Johnson等，CVPR 2017；`https://cs.stanford.edu/people/jcjohns/clevr/`；`https://arxiv.org/abs/1612.06890`。可复用其带场景图的诊断思想和开源生成基础；不能把新四状态评估称原CLEVR标准分数，具体下载许可须核对。

## 新结论与已有工具的区分

本包的20项都是**计划**。精确群平均、嵌套集合网络、关系蒸馏、约束保护、阈值区间恒等式均有已知方法或基础数学支撑。可发表的新内容应来自：此前未分开的失败来源、能被反证的干预效果、跨任务预测或实用模型的有效迁移，而不是给基础工具换名字。
