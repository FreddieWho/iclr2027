# U4 — 小表示对照（两种预注册对照，第一轮即停）

对照合同：只做两种新增表示，各自 clean/flip 两种监督，复用三种子与同一 flip 样本
（mine_seed=5，flip 列表 sha `abf78112…`），配方与 r04b/relflip 完全一致
（300ep、Adam 1e-2、lam 1.0、同种子初始化）。
训练脚本 `experiments/ccm_audit/u4_contrast_train.py`；
评价 `experiments/ccm_audit/u4_contrast_eval.py`；
输出 `artifacts/next_novelty/u4_contrast/`（12 模型＋manifest＋U4_SUMMARY.json）。
实际变更：无（配方零偏离；评价时 featurize 在 eval 侧显式处理，
checkpoint 存预计算特征空间的 in_dim/mu/sd）。

## 结果（dev512，三种子 J [95% parent CI] / H）

| arm | s11 | s23 | s47 |
|---|---|---|---|
| centered_clean | 0.072 / 0.658 | 0.089 / 0.629 | 0.046 / 0.625 |
| raw_clean（U1） | 0.055 / 0.565 | 0.051 / 0.544 | 0.055 / 0.544 |
| centered_flip | 0.114 / 0.793 | 0.110 / 0.768 | 0.110 / 0.764 |
| raw+flip（U1） | 0.068 / 0.730 | 0.059 / 0.671 | 0.076 / 0.692 |
| sixdist_clean | 0.122 / 0.848 | 0.173 / 0.835 | 0.194 / 0.873 |
| relfeat（U1，10 维） | 0.131 / 0.873 | 0.169 / 0.810 | 0.211 / 0.861 |
| sixdist_flip | 0.388 / 0.907 | 0.367 / 0.907 | 0.380 / 0.907 |
| relflip（U1，10 维） | 0.384 / 0.873 | 0.422 / 0.920 | 0.397 / 0.882 |

## 解释（按 02 §U4 预注册分支）

- 居中 raw 未追平关系表示（J 的 CI 与 raw 重叠，H 远低于关系臂 0.81–0.92）：
  收益不是已知平移 nuisance 处理，不宣称新的关系机制，但排除最简单的混杂解释。
- 六距离 ≈ 十特征（J/H 的 CI 大面积重叠；flip 臂 H 三种子同为 0.907）：
  四个半径无实质贡献，与代数预期一致（半径由完整点对距离决定）；
  解释止于“显式关系计算更便于联合学习”，不说信息更多。
- 无后续调参：第一轮结果可解释，停止，不硬追。

## 论文落点

附录小消融一行（sixdist≈tenfeat；centered≈raw）；不进主线贡献。
