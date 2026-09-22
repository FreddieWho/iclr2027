# P3-T5R4 Round 1 Review and Round 2 Authorization

日期：2026-09-04（Round 2 执行前冻结）  
状态：`T5R4_ROUND1_REVIEWED_ROUND2_AUTHORIZED_FINAL_ROUND`  
所属阶段：既有 `P3_CAUSAL_MECHANISM`，任务线 `P3-T5R4`

本报告是 Round 1 的正式审阅结论和 Round 2 的唯一授权依据。Round 2 的判断规则已在本报告与 `configs/t5r4_round2.yaml` 中预先冻结，禁止在看到 Round 2 结果后重写。

显示名称遵循 `artifacts/phase3/task_semantic_repair_v1/metric_semantics_addendum.json`：
`phase_match_grouped_macro_f1` 显示为 match-half context F1（辅助 period-context probe）；
natural-pair ranking 显示为 intrinsic structural accessibility proxy；
`geometry_response_spearman` 显示为 natural geometry–latent distance Spearman（非干预 proxy）。

## 1. Round 1 没有全维度 Pareto winner

Round 1 运行 4 个候选 × seeds `11/23/47`（`artifacts/phase3/task_semantic_repair_v1/t5r4_round1_v1/`）。
相对 T5R3 fixed dual reference（pair accuracy `0.9487`、geometry–latent Spearman `0.6761`、
`z_mode` translation response `~1e-8`）：

| Candidate | match-half F1 | zone F1 | centroid MAE | pair acc | geometry Spearman |
|---|---|---:|---:|---:|---:|
| T5R3 fixed dual | 0.5382 | 0.9619 | 0.0554 | 0.9487 | 0.6761 |
| `loss_context_up` | 0.5717 | 0.9624 | 0.0462 | 0.9492 | 0.6614 |
| `loss_intrinsic_up` | 0.5944 | 0.9697 | 0.0571 | 0.9460 | 0.6811 |
| `routing_mode_head_only` | 0.5875 | 0.9794 | 0.0317 | 0.9001 | 0.5725 |
| `head_context_layernorm` | 0.5890 | 0.9524 | 0.0543 | 0.9500 | 0.5547 |

没有候选在 context（match-half/zone/centroid）、intrinsic（pair/MRR/geometry）与
robustness（`z_mode` translation）上同时支配 reference。Procrustes analytic control
（`0.9372`）表明 ranking 接近几何可解上限，`0.949–0.950` 量级的 ranking 差异不能单独作为晋级依据。

## 2. `routing_mode_head_only` 是负机制证据

`routing_mode_head_only`（intrinsic loss 不再塑造 encoder，只训练末端 mode head）
context 指标较好（zone `0.9794`、centroid `0.0317`），但 pair accuracy 跌至 `0.9001`、
geometry Spearman 跌至 `0.5725`。结论冻结为：

> intrinsic objective 必须以梯度直接塑造 shared encoder，只训练末端 mode head 不足以保留 task-selective intrinsic geometry。

该候选保留为负机制证据（对应 claim `P3-R6`），gradient-routing 轴就此关闭，不再搜索。

## 3. `head_context_layernorm` 是 readout-normalization control

该候选的实际实现只是 pooled `z` → 无参数 LayerNorm → 既有 context heads，
不是 layer-wise head placement。当前与未来文档统一改称为：

> context readout normalization control（post-pooling context LayerNorm control）

历史 candidate ID 不变。它证明 pair accuracy 可维持 `0.9500` 的同时 geometry proxy
明显下降（`0.5547`），即任务 accuracy 不足以判断 representation geometry 是否正确。
该轴就此关闭，真正的 layer/message-passing head placement 本轮不开放。

## 4. Round 1 loss scaling 不能解释为精确机制量

`1.5× loss` 作用在分开的 optimizer steps 上，且 Adam 对纯梯度尺度自适应归一。
当前每 epoch 约 `361` 个 context batches 对 `59` 个 intrinsic batches（`≈6.12:1`），
因此 Round 1 只支持“training balance 值得检验”，不能把 `loss ×1.5` 当成已识别的
精确机制量（`intrinsic influence +50%` 的读法被禁止）。

## 5. Round 2 唯一允许的轴：encoder update allocation

约 `361:59` 的 optimizer-update imbalance 是唯一允许的 Round 2 检验轴，
命名为 `shared_encoder_task_update_balance`（`encoder_update_ratio`）。
Round 2 只改变每 epoch 的 context vs intrinsic optimizer update 数量分配，
其余全部冻结（§5 清单见 `configs/t5r4_round2.yaml`）。
这不是普通 loss sweep：loss weights 全部固定 `1.0`，总 optimizer steps/epoch
固定 `420`（与 reference 一致），避免“训练 step 更多而变好”的混杂。

偏离记录：`configs/phase3_task_semantic_repair_v1.yaml` 曾预填
`round_2_axes: [pooling, constraint_placement]`。根据 Round 1 证据与本轮冻结的科学边界，
该预填被本授权覆盖，Round 2 只运行 update-ratio 轴；repair config 随本轮同步更新。

## 6. Round 2 只运行 3:1 与 2:1

| Candidate | context updates | intrinsic updates | total |
|---|---|---:|---:|
| `update_ratio_3to1` | 315 | 105 | 420 |
| `update_ratio_2to1` | 280 | 140 | 420 |

调度规则（blockwise context→intrinsic 顺序保留，每类 step 取自独立 seed-deterministic
shuffle/cycle 流，80 epochs 下全样本反复覆盖，无 curriculum/early stopping/动态调整）：
见 `configs/t5r4_round2.yaml` 与 `artifacts/.../t5r4_round2_v1/protocol.json`。

## 7. Round 2 是最后一轮

无论结果如何，Round 2 结束后关闭 bounded autoresearch：不做 Round 3，
不扫描 ratio，不再搜索 loss/head/routing，不微调获胜 ratio。
若无明确 winner，记录 `T5R4_ROUND2_NO_CLEAR_PARETO` 并保留 T5R3 incumbent；
若出现符合预冻结 Pareto 规则的 winner，记录 `T5R4_selected_candidate` 后同样关闭搜索。

## 8. reserved/external/firewall 不变

`J03WQQ` 未读取，SNGAR test 未下载/读取，旧 exposed heldout 未用于选择，
外部结果未使用，candidate lock 未创建（`candidate_lock_created=false`），`P4_release=false`。
Round 2 只使用 T5R2 train/valid。

## 9. 本轮没有路线级漂移，但已修复 proxy/label 语义漂移

- 科学主线仍为 §2 链条（support-conditioned geometry → task-selective accessibility → shared encoder 保留机制）；Round 2 只回答 update-balance 是否为 Round 1 trade-off 的机制轴。
- 已修复并冻结于 semantic addendum：phase label → match-half context probe；
  natural pair → intrinsic structural accessibility proxy（含 Procrustes ceiling warning）；
  geometry proxy → 非干预 alias（legacy key 保留兼容）；
  LayerNorm → readout-normalization control；loss scaling → 不再当作精确 task influence。
- 历史报告不重写，冻结 lock 内容与 hash 不变（`baseline_and_metric_lock.json` 仍为 `d5fe2c85…`）。

## 附：训练前 provenance 冲突记录（§19）

1. Round 1 `protocol.json` 记录的 `config_sha256=f27643d5…` 与当前仓库
   `configs/t5r4_round1.yaml`（`ffd9d9cf…`，与 HEAD 一致）不一致，说明该 config
   在运行后、提交前被编辑过。Round 1 冻结产物（checkpoints/results/summary）未动，
   不重写历史；Round 2 独立记录自己的 config/protocol hash 对齐（见 audit 项 15）。
2. 本授权发布时 HEAD 为 `95ed8a3`；T5R2 lock、T5R3 reference summary、Round 1
   summary/protocol 的 hash 见 `configs/t5r4_round2.yaml` 的 `provenance` 节。
