# P3-T5R4 Round 2 Report

日期：2026-09-04  
状态：`T5R4_ROUND2_PASS_SELECTED_2TO1_AUTORESEARCH_CLOSED_P4_BLOCKED`  
所属阶段：既有 `P3_CAUSAL_MECHANISM`，任务线 `P3-T5R4`（最后一轮）

显示名称遵循 `artifacts/phase3/task_semantic_repair_v1/metric_semantics_addendum.json`：
match-half context F1（辅助 period-context probe，非战术理解）；
natural-pair ranking = intrinsic structural accessibility proxy（Procrustes control `0.9372` 为 ceiling warning）；
geometry 数值 = natural geometry–latent distance Spearman（非干预 proxy）。

## A. 一页式结论

- Round 2 结论：**PASS**（按训练前冻结规则，有明确 winner，非后验挑选）。
- Selected candidate：**`update_ratio_2to1`**（context:intrinsic = 280:140 updates/epoch，总步数 420 与 reference 持平）。
- Incumbent 替换：T5R3 fixed dual 被替换为 `update_ratio_2to1`，作为 T5R5 candidate lock 的输入候选。
- Bounded autoresearch：**关闭**，不再有 Round 3，不扫描 ratio，不微调 2:1。
- P4：**仍 blocked**（`blocked_pending_p3_t5r_gate`）；reserved `J03WQQ` 未读，外部确认未运行。
- Next action：`T5R5_candidate_lock_then_single_reserved_read`。

## B. 实验设计

Round 2 不是普通 loss sweep。唯一轴是 `shared_encoder_task_update_balance`：
每个 epoch 的 context vs intrinsic optimizer update 数量分配。loss weights 全固定 `1.0`
（Adam 下纯梯度尺度不可解释为影响比例，见授权报告 §4），总步数/epoch 固定 `420`
（reference 为 361+59），候选 3:1 = 315/105、2:1 = 280/140。
其余全部冻结：IDSSE T5R2 数据/split、kNN-4 图、encoder 结构与 `team_mean` pooling、
heads、无 LayerNorm、shared routing、Adam lr `1e-3`、80 epochs、batch 64、seeds `11/23/47`、
参数量 `141,769`、train/valid only。
调度保留 Round 1 的 blockwise context→intrinsic 顺序，每类 step 取自独立
seed-deterministic shuffle/cycle 流（context 配额不足整轮时从连续循环流中取数，
intrinsic 超配额时跨 cycle 取数、每 cycle 确定性新 permutation），每 step 仍只对应一个
task loss，无 curriculum/early stopping/动态调整。protocol 见
`artifacts/phase3/task_semantic_repair_v1/t5r4_round2_v1/protocol.json`。

## C. 指标语义

- match-half context F1（`firstHalf/secondHalf`）是辅助 period-context probe，不是战术阶段理解；不作为晋级理由。
- natural-pair ranking 是 response-blind 几何构造的 intrinsic structural accessibility proxy；
  Procrustes analytic control（`0.9372`）说明任务本身强几何可解，ceiling 附近的微小 ranking 提升不单独作为晋级理由。
- geometry–latent Spearman 是 natural-pair internal-geometry distance ↔ latent distance 的
  非干预 proxy，不是旧 P3 intervention-response 机制在 IDSSE 上的复现。

## D. 结果表

Valid 均值（2 matches × 3 seeds；match-half F1 仅为辅助 probe）：

| Candidate | match-half F1 | zone F1 | centroid MAE | pair acc | MRR@2 | geometry Spearman | `z_mode` translation |
|---|---|---:|---:|---:|---:|---:|---:|
| T5R3 fixed dual (ref) | 0.5382 | 0.9619 | 0.0554 | 0.9487 | 0.9743 | 0.6761 | ~1e-8 |
| R1 `loss_intrinsic_up`（探索对照） | 0.5944 | 0.9697 | 0.0571 | 0.9460 | 0.9730 | 0.6811 | ~1e-8 |
| R2 `update_ratio_3to1` | 0.5607 | 0.9568 | 0.0450 | 0.9669 | 0.9835 | 0.6959 | ~1e-8 |
| R2 `update_ratio_2to1`（selected） | 0.6022 | 0.9627 | 0.0376 | 0.9511 | 0.9756 | 0.7034 | ~1e-8 |

资格 gate（训练前冻结，从 reference artifact 读取 floor：pair ≥ `0.9387`、
geometry ≥ `0.6661`、`z_mode` < `1e-6`）：

| Candidate | context 非劣效 (vs raw) | `z_mode` | pair floor | geometry floor | ≥2/3 seeds 双 floor |
|---|---|---|---|---|---|
| 3:1 | 通过（zone −0.014，centroid 改善） | 通过（~9e-9） | 通过（0.9669） | 均值通过（0.6959） | 2/3（seed23 geometry `0.6640` 略低于 floor） |
| 2:1 | 通过（zone −0.008，centroid 改善） | 通过（~1e-8） | 通过（0.9511） | 通过（0.7034） | 3/3 |

Same-seed deltas（candidate − T5R3 reference，同 seed 相减）：

`update_ratio_2to1`：

| seed | pair Δ | geometry Δ | match-half Δ | zone Δ | centroid Δ |
|---|---|---:|---:|---:|---:|
| 11 | −0.0156 | +0.0502 | +0.0378 | +0.0498 | −0.0010 |
| 23 | +0.0006 | +0.0101 | +0.0929 | −0.0227 | −0.0312 |
| 47 | +0.0225 | +0.0213 | +0.0612 | −0.0246 | −0.0215 |

geometry 增益 3/3 seeds 同方向（去掉任一种子后剩余均值仍为正）；
centroid 3/3 seeds 同方向改善；pair 保持（3 seeds 均高于 floor，大 match 上同-seed deltas 全 ≥0）；
zone 变化在 margin 内。match-half F1 在 3 seeds 上同步抬升到 ~0.60 且跨种子方差收窄，
仅记为辅助观察，不作为晋级理由。

`update_ratio_3to1`（未晋级）：pair 增益 3/3 seeds 为正（+0.0027/+0.0173/+0.0350），
但主要故事是 Procrustes ceiling 附近的 ranking 增益；geometry 同-seed deltas 为
+0.0405/−0.0083/+0.0271（seed23 为负且绝对值 `0.6640` 跌破 floor），不满足“稳定”
要求。按冻结规则，ranking 提升不能在 geometry 不稳定时单独支撑晋级。

Per-match deltas（valid 仅 2 场：`J03WN1` 425 snapshots/29 pairs，`J03WOY` 5702/951）：

- `update_ratio_2to1` 在占 97% pairs 的 `J03WOY` 上 3 seeds 的 pair/phase/centroid deltas
  方向一致为正或改善；`J03WN1` 仅 29 pairs（单个 pair 翻转 = 3.4pp），deltas 在噪声带内摆动，
  不提供信息也不构成反向证据。均值方向由大 match 与 3 seeds 共同支撑，不由单 seed/单场大效应支配。
- 统计边界：valid 只有 2 场且 pair 分布极不均衡（match-grouped 均值赋予 29-pair match 同等权重），
  本轮结论仍是开发集内的方向性证据，不是独立确认。

分离/泄漏审计（valid）：2:1 的 context-head-on-`z_mode` zone F1 为 `0.260–0.262`、
mode-head-on-`z_ctx` pair 为 `0.29–0.33`、paired latent correlation 为 `0.13–0.15`，
与 reference（`0.253–0.258` / `0.238–0.327` / `0.124–0.157`）处于同一水平，
无新增泄漏。`z_ctx` accessibility response（`0.276`）与 reference（`0.286`）相当，
context 访问能力保持。

## E. 科学解释

按训练前冻结的强推荐条件 A（geometry–latent Spearman 稳定提高，pair/context 无实质退化），
`update_ratio_2to1` 达标，同时 centroid 的绝对-context 改善为条件 B 提供了支持。
允许的解释（数据支持范围内）：

> shared encoder 的 task-specific structure preservation 对 optimizer update allocation 敏感；
> intrinsic objective 必须以足够频率塑造 backbone，才能维持 task-selective intrinsic geometry，
> 同时保留 absolute context access。

不允许的解释：

> 找到了通用最优 2:1 ratio（本轮只检验了 3:1/2:1 两个点；2:1 的数值本身不是可迁移的方法结论，
> 其状态为 `SUPPORTED_CONDITIONALLY`，边界 = IDSSE 开发集 / 当前固定结构 / 2 场 valid）。

对 PLAN 假设的影响：支持“encoder update balance 是 Round 1 trade-off 的机制轴之一”，
不改变 P3 中心链条；T5R5/T5R6 与 P4 gate 条件不变。

权威产物：`artifacts/phase3/task_semantic_repair_v1/t5r4_round2_v1/`
（manifest/protocol/model_results/summary/comparison_vs_t5r3/semantic_alias_snapshot/
checkpoints/SHA256SUMS）。结构审计 `T5R4_ROUND2_AUDIT: PASS`。
