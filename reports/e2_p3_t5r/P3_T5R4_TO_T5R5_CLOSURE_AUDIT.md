# P3-T5R4 → T5R5 Closure Audit

日期：2026-09-05
状态：`T5R5_PRELOCK_AUDIT_COMPLETE`
阶段：既有 `P3_CAUSAL_MECHANISM`，任务线 `P3-T5R5-0`

> This audit refines the claim boundary and does not reopen T5R4 candidate selection.

本审计只读已可见的 train/valid 产物，未打开 `J03WQQ` 及其任务内容，未做任何重训练，
未调整任何协议。计算脚本：`scripts/compute_t5r5_prelock_audit.py`；
产物：`artifacts/phase3/task_semantic_repair_v1/t5r5_prelock_audit_v1/`。

显示名遵循 semantic addendum：match-half context F1 为辅助 probe；
natural-pair ranking 为 intrinsic structural accessibility proxy；
geometry 值为非干预 proxy。

## 1. Valid 逐场几何（已可见证据的重算）

对象：T5R3 fixed dual 与 T5R4 入选 `update_ratio_2to1`，seeds 11/23/47，
逐场（`J03WN1` 425 snapshots/29 pairs；`J03WOY` 5702 snapshots/951 pairs）：

| 模型 | seed | 场次 | pairs | geometry–latent Spearman | pair acc | zone F1 | centroid MAE |
|---|---|---|---|---:|---:|---:|---:|
| fixed dual | 11 | WN1 | 29 | 0.7334 | 0.9655 | 0.9185 | 0.0372 |
| fixed dual | 11 | WOY | 951 | 0.6675 | 0.9537 | 0.9278 | 0.0431 |
| fixed dual | 23 | WN1 | 29 | 0.7264 | 0.9655 | 0.9917 | 0.0681 |
| fixed dual | 23 | WOY | 951 | 0.6729 | 0.9506 | 0.9790 | 0.0690 |
| fixed dual | 47 | WN1 | 29 | 0.7237 | 0.8966 | 0.9837 | 0.0562 |
| fixed dual | 47 | WOY | 951 | 0.6918 | 0.9600 | 0.9706 | 0.0588 |
| 2:1 | 11 | WN1 | 29 | 0.7470 | 0.9310 | 0.9845 | 0.0394 |
| 2:1 | 11 | WOY | 951 | 0.7196 | 0.9569 | 0.9614 | 0.0391 |
| 2:1 | 23 | WN1 | 29 | 0.6853 | 0.9655 | 0.9533 | 0.0375 |
| 2:1 | 23 | WOY | 951 | 0.6860 | 0.9516 | 0.9722 | 0.0374 |
| 2:1 | 47 | WN1 | 29 | 0.7497 | 0.9310 | 0.9369 | 0.0376 |
| 2:1 | 47 | WOY | 951 | 0.7148 | 0.9706 | 0.9681 | 0.0345 |

pooled-pair Spearman（980 pairs 混合）vs match-macro Spearman（逐场均值）：

| 模型 | seed | pooled | macro |
|---|---|---:|---:|
| fixed dual | 11 | 0.6676 | 0.7004 |
| fixed dual | 23 | 0.6723 | 0.6997 |
| fixed dual | 47 | 0.6886 | 0.7078 |
| 2:1 | 11 | 0.7179 | 0.7333 |
| 2:1 | 23 | 0.6825 | 0.6857 |
| 2:1 | 47 | 0.7099 | 0.7322 |

## 2. 边界收紧（§2 重申）

1. pooled Spearman 几乎完全由 `J03WOY`（951/980 pairs）决定，不是跨场泛化证据；
   match-macro 赋予 29-pair 场次同等权重，方向一致但 WN1 单场值不稳定（已标 `spearman_stable=false`）。
2. `J03WN1` 的 29 pairs 使逐场 Spearman/accuracy 处于噪声带（单个 pair 翻转 ≈3.4pp）：
   2:1 在 WN1 上 2 升 1 降（seed23 0.6853 vs 0.7264）不构成反向证据，也不提供支持。
3. 2:1 在占主导的 WOY 场上 3 seeds 的 geometry 全部高于 fixed dual 同 seed 值；
   context（zone/centroid）在两场上均无实质退化。
4. 以上均不改变 T5R4 已冻结结论，只收紧其 claim 边界：开发集内的方向性证据，
   不是独立确认。

## 3. 本次文档清理（§4.1）

1. STATUS 当前路由句的 “T5R4 Round 1 已完成待审阅” 已改为两轮完成并关闭；本轮目标句已改为 T5R5 lock-and-read。
2. STATUS 证据表 `task-geometry alignment` 行已标注“旧 T5 结果，非 T5R”，与 P3-R2/R6/R7 分隔。
3. `docs/01/02/03` 与 T5R plan 中的 `phase/deployment` 显示已加注 match-half/period context probe alias；
   带日期的历史 QA/DECISIONS 条目不动，由 addendum 覆盖。
4. `CLAIM_LEDGER.md` 中 P3-G1–G4 已标为 `HISTORICAL_PRE_RESULT_HYPOTHESIS`，指向完成证据 P3-C1/C3/C4/C5/C6。
5. 核查确认：当前文档无 `head_context_layernorm` 的 head-placement 误写（仅 DECISIONS 历史条目，
   且已正确描述为 readout-normalization control）；无 Round-1 `1.5× loss` 的精确影响误写。
6. 冻结历史产物未改写；`baseline_and_metric_lock.json` 内容与 hash 不变。

## 4. 防火墙状态（lock 前）

- `J03WQQ` 任务数组/结果未读出；`data_views/` 无 reserved 文件；
  `t5r5_reserved_j03wqq_v1/` 不存在（prelock 脚本已做存在性断言）；
- SNGAR test、旧 exposed heldout、外部结果均未用于选择；
- candidate lock 尚未创建；P4 仍 blocked。
