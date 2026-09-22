# 核验来源

阅读日期2026-09-18。下面文献用于辨别最近邻，不宣称本轮完成了系统综述或所有公式的全文核验。

## 仓库原始依据

- `reports/discovery_campaign/PAPER_NUMBERS_FILLABLE.md`
  `https://github.com/FreddieWho/iclr2027/blob/3ab758d65d68aca9204736e04dcb7c60e5a8f6ad/reports/discovery_campaign/PAPER_NUMBERS_FILLABLE.md`
- `reports/discovery_campaign/MAIN_FIGURE_SOURCE.json`
  `https://github.com/FreddieWho/iclr2027/blob/3ab758d65d68aca9204736e04dcb7c60e5a8f6ad/reports/discovery_campaign/MAIN_FIGURE_SOURCE.json`
- `reports/discovery_campaign/TRACK6_card.md`
  `https://github.com/FreddieWho/iclr2027/blob/3ab758d65d68aca9204736e04dcb7c60e5a8f6ad/reports/discovery_campaign/TRACK6_card.md`
- `reports/discovery_campaign/METHOD_ROUND1_screen.md`
  `https://github.com/FreddieWho/iclr2027/blob/3ab758d65d68aca9204736e04dcb7c60e5a8f6ad/reports/discovery_campaign/METHOD_ROUND1_screen.md`
- `reports/discovery_campaign/R06_card_round2.md`
  `https://github.com/FreddieWho/iclr2027/blob/3ab758d65d68aca9204736e04dcb7c60e5a8f6ad/reports/discovery_campaign/R06_card_round2.md`
- `experiments/discovery_campaign/CAMPAIGN_STATE.md`
  `https://github.com/FreddieWho/iclr2027/blob/3ab758d65d68aca9204736e04dcb7c60e5a8f6ad/experiments/discovery_campaign/CAMPAIGN_STATE.md`
- `experiments/discovery_campaign/r02_t5r3.py`
  `https://github.com/FreddieWho/iclr2027/blob/3ab758d65d68aca9204736e04dcb7c60e5a8f6ad/experiments/discovery_campaign/r02_t5r3.py`
- `experiments/discovery_campaign/r04b_methods.py`
  `https://github.com/FreddieWho/iclr2027/blob/3ab758d65d68aca9204736e04dcb7c60e5a8f6ad/experiments/discovery_campaign/r04b_methods.py`

- `reports/discovery_campaign/R04_confirm_card.md`
  `https://github.com/FreddieWho/iclr2027/blob/3ab758d65d68aca9204736e04dcb7c60e5a8f6ad/reports/discovery_campaign/R04_confirm_card.md`

## 检索到的主要近邻

### S1
Jacobsen et al. Excessive Invariance Causes Adversarial Vulnerability. ICLR 2019.

`https://arxiv.org/abs/1811.00401`

对本轮的影响：过度不变性/任务相关变化被遗漏已有直接先例。

### S2
Tramer et al. Fundamental Tradeoffs between Invariance and Sensitivity to Adversarial Perturbations. ICML 2020.

`https://proceedings.mlr.press/v119/tramer20a.html`

对本轮的影响：不变性与敏感性的矛盾不是本项目首创。

### S3
Bender et al. Mitigating Clever Hans Strategies in Image Classifiers through Generating Counterexamples. arXiv, 2025.

`https://arxiv.org/abs/2510.17524`

对本轮的影响：CFKD生成反事实、教师重标注、重训练；是flipmine重要近邻。

### S4
Hamman et al. Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations. NeurIPS 2025.

`https://papers.nips.cc/paper_files/paper/2025/hash/dbb5b038bb6f396ede1da85307a13890-Abstract-Conference.html`

对本轮的影响：CoD以反事实帮助少样本学习教师边界；边界反例/少样本本身不足以宣称首次。

### S5
Czarnecki et al. Sobolev Training for Neural Networks. 2017.

`https://arxiv.org/abs/1706.04859`

对本轮的影响：函数值与导数共同监督是既有方法框架。

### S6
Harel-Canada et al. Sibylvariant Transformations for Robust Text Classification. 2022.

`https://arxiv.org/abs/2205.05137`

对本轮的影响：有意改变标签的增广已有体系；不能称本项目首次突破label-preserving augmentation。

### S7
Santoro et al. A simple neural network module for relational reasoning. 2017.

`https://arxiv.org/abs/1706.01427`

对本轮的影响：RN视觉关系任务已有强先例；多一个关系网络不是本文创新。

## 新颖性说明

十条都是拟探索的设计，尚未获得结果；不能将任何路线直接称为首次提出。研究框架已有先例，拟争取的是具体语义转折规律、对行动/事件的后果，以及超过现有flipmine的机制性收益。
