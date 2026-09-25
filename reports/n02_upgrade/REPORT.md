# N02 升级报告（30 臂新后端矩阵 + small-edit 新分层）

执行：2080 Ti 11GB，torch 2.5.1+cu121（旧矩阵为 3080Ti + cu124；一切对比只在新后端内部）。
分层：small-edit 新挖 564 quartets / 146 parents（`smaledit_bank.npz`，sha `ecd8f48c…`，
与 N02 冻结 splits 查重 0 匹配）。宽网：ResNet18 通道 ×4，178M 参数，MACs 2.25G ≥ B 的 1.81G。
独立重算：本地 numpy 重算与远端 `upgrade_analysis.json` 逐数一致（P1 f 三种子、
P2 s803 交互均复核通过）。

## P0（复现 sanity）：PASS

新后端 B−A standard flip：pretrained +0.330/+0.409/+0.251，random +0.283/+0.330/+0.296，
6/6 CI 排除 0。实现有效，主判据可读。

## P1（算力-or-容量）：约一半可解释，判 `COMPUTE_OR_CAPACITY_MAJORITY_BY_MEAN`（附异质性声明）

random-init（J 均值）：A = 0.500/0.522/0.478；W = 0.702/0.708/0.629；B = 0.809/0.837/0.815。
f = (W−A)/(B−A)：s803 +0.619、s805 +0.552、s806 +0.434，均值 +0.535。
3/3 种子为正且幅度可观；但 s806 未达 0.5 线，冻结规则未写 seed 聚合方式，
故标签取 cross-seed 均值口径并如实报告异质性。
论文口径：把 A 加宽到 ≥B 量级 MACs 能关掉约一半的 B−A 差距；剩下约一半与网格相关，
但仍未干净识别（宽网同时放大了参数，算力与容量不可分）。

## P2（小编辑机制）：维持 `UNRESOLVED`，且有两条反向证据

small-edit 564/146 上 lowstride 交互（预测：两 init 同负）：pretrained +0.183/−0.119/−0.014
（符号混杂），random +0.136/+0.114/+0.050（方向与预测相反，s803/s805 CI 在正侧排除 0）。
"lowstride 主要消掉小编辑优势"不成立；random 下反而是 lowstride 拉大了 B−A。
附加：旧 178 bank 上原 pretrained −0.133 命中在新后端未复现（新后端 old-bank 交互 6 格 CI 全含 0），
原命中降级为单后端观测，不再作为机制证据引用。

## 对论文的影响（PLAN H2：采样/网格机制）

- MAC 混杂从"完全未处理"变为"约一半已量化"：正文附录 N02 段补一句话（新后端、独立编号、不与旧数拼合）。
- 机制结论不变（仍 UNRESOLVED），但未决的含义变强了：small-edit 大分母下预测方向被拒，
  且原命中被证伪为后端特异——aliasing 归因的路更窄了，如实写。
- 不升级任何主 claim；不碰 P1/P2/P3 正文数字。
