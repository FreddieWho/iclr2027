# P3-T5R5 Hidden Confirmation Report (J03WQQ, single read)

日期：2026-09-05
状态：`T5R5_PASS_STRONG -> T5R6_EXTERNAL_PROVIDER_CONFIRMATION`
阶段：既有 `P3_CAUSAL_MECHANISM`，任务线 `P3-T5R5`

本报告是 lock 后一次性读取的结果。候选、对照、任务规则、干预协议、判决门均在
`candidate_lock.json` / `t5r5_intervention_lock.json` / `configs/t5r5_hidden_confirmation.yaml`
中预冻结；看到结果后未改动任何规则。显示名遵循 semantic addendum。

权威产物：`artifacts/phase3/task_semantic_repair_v1/t5r5_reserved_j03wqq_v1/`。
`T5R5_HIDDEN_AUDIT: PASS`（9 记录、无训练痕迹、阈值与 T5R2 一致、verdict 复算一致、
仅一个 reserved 输出目录）。

## 1. 隐藏任务确认（J03WQQ：3575 snapshots，515 pairs）

| 模型（3 seeds 均值） | match-half F1（辅助） | zone F1 | centroid MAE | pair acc | MRR@2 | geometry–latent Spearman | `z_mode` translation |
|---|---|---:|---:|---:|---:|---:|---:|
| `update_ratio_2to1` | 0.7600 | 0.9769 | 0.0363 | 0.9605 | 0.9803 | 0.7322 | ~0 |
| T5R3 fixed dual | 0.6553 | 0.9658 | 0.0539 | 0.9631 | 0.9816 | 0.7271 | ~0 |
| raw single-channel | 0.6875 | 0.9701 | 0.0451 | 0.8932 | 0.9466 | −0.4537 | 0.2225 |
| Procrustes analytic | — | — | — | 0.9922 | — | — | — |

任务语义 gate（§9.1）逐项：`z_mode`≈0（<1e-6）✓；zone 相对隐藏 raw 基线
（0.9769 vs 0.9701，无下降）✓；centroid（0.0363 vs 0.0451，无增加）✓；
pair 相对 fixed dual（0.9605 vs 0.9631，Δ=−0.0026 ≥ −0.02）✓；
geometry 均值 Δ=+0.0051 ≥ 0 ✓，同 seed 非负 2/3
（+0.0209/+0.0007/−0.0062）✓；自然对 515 组，充分（非 UNDERPOWERED）✓。

注意：隐藏 match 上 Procrustes ceiling 高达 0.9922，pair 排名本身几乎完全几何可解；
2:1 与 fixed dual 的 pair 差异（−0.0026）在此 ceiling 下不作方法区分，只记“保持”。
分离审计：2:1 的 context-head-on-`z_mode` zone F1 为 0.2145、mode-head-on-`z_ctx`
pair 为 0.39–0.41、paired latent correlation 为 0.13，无新增泄漏；
`z_ctx` accessibility（0.241）与开发集水平相当。

## 2. 实际干预确认（250 采样快照 → 159 complete＋91 incomplete）

| 模型 | seed | pairs | full geometry | diagonal | spectral 基线 | 方向准确率 |
|---|---|---:|---:|---:|---:|---:|
| 2:1 | 11 | 159 | 0.5966 | 0.3146 | −0.0130 | 0.7547 |
| 2:1 | 23 | 159 | 0.6272 | 0.3316 | 0.0645 | 0.7862 |
| 2:1 | 47 | 159 | 0.6664 | 0.4312 | −0.1279 | 0.7484 |
| fixed dual | 11 | 159 | 0.5762 | 0.2627 | −0.0680 | 0.7107 |
| fixed dual | 23 | 159 | 0.4220 | 0.1997 | −0.1355 | 0.6792 |
| fixed dual | 47 | 159 | 0.6428 | 0.2573 | −0.1508 | 0.7107 |

干预机制 gate（§9.2，selected 家族）：mean full 0.630 ≥ 0.50 ✓；
mean(full−baseline) 0.656 ≥ 0.20 ✓；3/3 seeds full > baseline ✓；
方向准确率均值 0.763（高于 chance）✓；无 full<diagonal 坍缩
（full 0.63 vs diag 0.36）✓。STRONG 通过。

次要比较：核心局部几何机制在两个家族均复现（fixed dual mean full 0.547，
同样强通过）；2:1 在 3 个 seeds 上 full 均高于同 seed fixed dual，
开发集方向在未见 match 上保持，但该比较按 lock 声明为次要。

有效性边界：91/250 未完成中 70 为 anchor 越界（严格 bound=1.0 下 epsilon-0.25
位移推出归一化边界，response-blind 构造的固有损耗），21 为无边界合法不交
target；均计入失败率，不纳入 Spearman。complete n=159，response 方差非退化。

## 3. 实际确认了什么

允许的表述（数据支持范围内）：

> 任务语义修复把 absolute context 与 intrinsic structure 分开，得到一个表示，
> 其局部几何在未见源比赛的受控响应上保持可预测。

> 提高 intrinsic objective 直接更新 shared encoder 的频率的做法，改善了开发集
> Pareto，并在未见 match 上保持了同方向（任务保持＋干预 full 3 seeds 占优）。

## 4. 仍未确认什么

- P3-R3（跨比赛泛化）仍未完全建立：这只是一个同源未见 match，不是多比赛总体，
  更不是独立 provider 确认；需 T5R6。
- 2:1 不是通用最优：只检验过两个开发集比例点；不扩写。
- 自然对排名不证明战术/语义理解（Procrustes 已达 0.9922）；match-half 不证明
  战术阶段理解；本轮不验证 AMR（T5R5 更不验证 AMR）。
- 干预 conclusions 限于固定家族、冻结算子、单 epsilon、单 match。

## 5. 防火墙/溯源

候选在隐藏读取前锁定（pre-lock HEAD `c72d66c`，`T5R5_LOCK_AUDIT: PASS`）；
只读一次（仅 `t5r5_reserved_j03wqq_v1/`）；无重训练（9 记录均无 training 痕迹）；
阈值与 T5R2 逐字一致（审计复核）；SNGAR test、旧 heldout、外部结果均未用于选择；
P4 blocked。

## 6. 路由

```text
T5R5_PASS_STRONG -> T5R6_EXTERNAL_PROVIDER_CONFIRMATION
```

P4 仍 blocked；不做新 autoresearch。
