# D09 冻结预测（2026-09-23，新银行 d09fresh665 未读数前）

银行：`bank_d09fresh665.npz`（fresh 7680 父场景中 seed665 抽 512，未参与任何训练/选择；manifest sha d184fc57…）。
冻结模型（N 训练，dev512 上已定型）：sixdist-N clean/flip（U02）、raw-N clean/flipmine（r04b 锚）。
预测（dev512 模式：six-flip .43–.44 vs raw-flip .06–.08；six-clean .12–.19 vs raw-clean .05–.06）：
- Q1：新银行上 sixdist-flip J − raw-flip J ≥ +0.15，三种子全中。
- Q2：新银行上 sixdist-clean J − raw-clean J ≥ +0.05，三种子全中。
命中→"几何组织×flip 互补"首个未参与选择的确认；任一种子反向→限域（效应止于 dev512 构造）。
