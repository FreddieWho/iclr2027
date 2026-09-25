# N02 升级协议（冻结，训练前）

冻结时间：2026-09-25。状态：`FROZEN_PRE_TRAIN`。任何训练启动后不得修改判据；
数值幅度一律事后报告，只冻结符号/比例/规则。

## 回答的问题（服务 PLAN 假设 H2：采样/网格机制）

1. B−A 的 +0.30–0.34 有多少是算力（12.25× MACs），多少是网格？
2. lowstride 交互在小编辑分层上是否成立（原 5 行/3 parent 无辨识力）？

## 矩阵（flip-only，有界 36 臂）

- obs：A（native64）、A-wide（native64，ResNet18 通道 ×4，实测 MACs 2.25G/图 ≥ B 的 1.81G，参数 178M；匹配的是算力/容量联合体，判据措辞为算力-or-容量）、B（bilinear→224）
- stem：standard / lowstride；init：pretrained / random；seed：803/805/806
- 全部 flip 监督；20 epoch、Adam 3e-4、batch32（2080Ti 11GB 若 OOM 则全矩阵统一改 batch16，声明后执行）、FP32、TF32/AMP 关闭
- 选型：只按 singleton-dev BCE，无 AB、无 J 选型，不删 seed
- 训练数据：复用冻结 N02 `data.npz` 的 train/dev（同分布，无新泄漏）；test = 旧 178 + 新 small-edit 分层（fresh parents，见下）
- 后端纪律：全部对比只在新后端（2080Ti + cu121）内部成立；不与 3080Ti/cu124 旧数拼因果差值

## A-wide 定义（random-init 为主）

- 通道数 ×4 的 ResNet18（conv/bn/fc 逐层放宽，参数量/MACs 实测写入 manifest；目标 MACs 与 B 同量级 ≥1.8G/图）
- pretrained 权重对宽网不存在，故**算力分离主判据只在 random-init 下执行**（A/A-wide/B 全 random，干净三方比较）；pretrained 只复现 B−A（P0）与 lowstride 交互，不做算力归因
- Net2WiderNet 函数保持初始化列为可选探索，不进入主判据

## 新 small-edit 分层（CPU 挖掘，与训练并行）

- 同一 miner（`mine_quartets`，seed 链新值）、同一渲染合同、同一 margin/资格规则，只 sweep 更多 test parents 后按 `quartet_small_edit` 掩码过滤
- 目标：≥60 quartet / ≥30 parent，cap 4/parent；与 train/dev/test + 19 个暴露 bank 做 1e-6 canonical 查重，0 匹配才可用
- 选择是输入几何过滤（response-blind），如实声明为 filtered stratum

## 修正案 A1（2026-09-25，训练前）：小半径提案挖掘
直接 sweep 原 miner 产率过低（4096 parents→17 small，实测），改用小半径提案：
`atomic_edits(x, rng, radius∈{0.03,0.05,0.07})` × 4 trials/parent（96 候选/parent，
原 miner 8 候选/parent）。oracle/margin≥0.005/标签资格/配对逻辑与原 miner 逐字一致，
试点 32 parents→1014 small 候选。选择仍是输入几何过滤（response-blind），
记为 filtered stratum；cap 4/parent 与查重规则不变。

## 修正案 A2（2026-09-25，训练前）：矩阵 36→30 臂
W（宽网）不存在 pretrained 权重，故 W×pretrained 6 臂取消（设计使然，非执行缺口）。
实际矩阵：A/B × standard/lowstride × pre/rand（24 臂）+ W × standard/lowstride × rand（6 臂）= 30 臂。
P1 只在 random-init 下执行（协议正文已写）；pretrained 只做 P0 复现与 old-bank 交互对照。

## 预注册判据

- P0 复现（sanity）：新后端 (B−A) standard flip，pretrained 与 random 均为 ΔJ>0 且 parent-cluster CI 排除 0。不通过则停下查实现，不动主判据。
- P1 算力（random-init）：f = (Awide−A)/(B−A)（J 尺度，parent 等权）。
  f≥0.5 → `COMPUTE_OR_CAPACITY_MAJORITY`；f≤0.25 → `GRID_STANDS`；之间 → `MIXED`。点估计+CI 照报。
- P2 小编辑机制：在 small-edit 分层上，lowstride 交互 (B−A)_low − (B−A)_std 为负且 CI 排除 0，**两个 init 同时成立** → 由 `UNRESOLVED` 升级为 `SUPPORTED_SCOPED`；任一不成立 → 维持 `UNRESOLVED`。
- P3：全面报告 A-wide 自身 J/atomic/CCM 与 static 无关性（不重测 static，沿用已确立结论）。
- 统计：parent 等权 + 4000 次 parent 聚类 bootstrap；3 seed 全保留；无阈值搜索。

## 预算与停止

- 单矩阵 36 臂，无迭代调参轮；OOM 只允许统一 batch16 一次让步
- small-edit 分层若 sweep 2048 parents 仍不足 60/30，则降级报告实际分母，不放宽 cap/查重规则
- 产物：`artifacts/n02_upgrade/`；代码：`experiments/n02_upgrade/`（复用旧脚本只读拷贝，宽网改动独立文件）；旧字节不动
