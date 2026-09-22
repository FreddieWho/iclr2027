# P3-T5R4 Round 1 Report

日期：2026-09-04  
状态：`T5R4_ROUND1_COMPLETE_REVIEW_REQUIRED`

## 运行范围

Round 1 运行 4 个 fixed-dual 周围的代表候选，每个候选使用 seeds `11/23/47`，共 12 个模型。只读取 T5R2 train/valid views；`J03WQQ`、SNGAR test、旧 exposed heldout 和外部结果均未读取，也没有自动 candidate promotion。

| Candidate | Axis | Phase F1 | Zone F1 | Centroid MAE | Pair accuracy | Geometry proxy | `z_mode` response |
|---|---|---:|---:|---:|---:|---:|---:|
| T5R3 fixed dual reference | — | 0.5382 | 0.9619 | 0.0554 | 0.9487 | 0.6761 | ~0 |
| `loss_context_up` | loss balance | 0.5717 | 0.9624 | 0.0462 | 0.9492 | 0.6614 | ~0 |
| `loss_intrinsic_up` | loss balance | 0.5944 | 0.9697 | 0.0571 | 0.9460 | 0.6811 | ~0 |
| `routing_mode_head_only` | gradient routing | 0.5875 | 0.9794 | 0.0317 | 0.9001 | 0.5725 | ~0 |
| `head_context_layernorm` | head placement | 0.5890 | 0.9524 | 0.0543 | 0.9500 | 0.5547 | ~0 |

## 判断

没有候选在所有主要维度上同时支配 T5R3 reference。`routing_mode_head_only` 的 context 指标较好，但 intrinsic ranking 明显下降；`loss_context_up` 和 `head_context_layernorm` 分别展示了 context/centroid 与 phase/pair 的局部改善，仍需人工决定是否继续探索。Procrustes analytic control 已为 `0.9372`，因此 intrinsic ranking 接近几何上限，不能单独作为方法晋级依据。

本轮结果支持“继续审阅”，不支持自动进入 Round 2，也不支持 candidate lock、reserved-match 读取、独立确认或 P4。若继续，下一轮仍应以 task–robustness 的整体平衡为观察对象，具体候选值保持探索性。

权威产物：`artifacts/phase3/task_semantic_repair_v1/t5r4_round1_v1/`。结构审计为 `T5R4_ROUND1_AUDIT: PASS`；参数量均为 `141,769`。
