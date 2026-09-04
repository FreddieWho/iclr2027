# P3-T5R3 Fixed Dual-Channel Sanity Report

日期：2026-09-04  
状态：`T5R3_DIRECTIONAL_PASS_T5R4_REVIEW_REQUIRED`  
所属阶段：既有 `P3_CAUSAL_MECHANISM`

## 1. 运行范围

本次运行只读取 T5R2 已冻结的 train/valid task views：train 4 场、valid 2 场；`J03WQQ` reserved match 未加载，未写出其 task array。没有读取 SNGAR test、旧暴露 heldout、外部结果，也没有执行 candidate selection。

固定训练参数为 seeds `11/23/47`、Adam、learning rate `0.001`、weight decay `0`、80 epochs、batch size `64`、hidden/latent dimension `128`、2 个 message-passing layers、无 augmentation、无 early stopping、无 class weighting。四个 neural variant 均为 `141,769` 参数。

实现使用固定对称加权 kNN-4 图和两层 fixed-weighted message-passing equivalent；没有 learned graph、frequency gate 或 AMR。fixed dual-channel 只让 raw view 进入 `z_ctx`，globally centered view 进入 `z_mode`，两者共享 encoder 和 task heads。

## 2. Valid 结果

下表是两个 valid match 上按 match 平均、再按三个 seed 平均的结果。`geometry relation` 是 natural-pair internal-geometry distance 与 latent distance 的 Spearman proxy，不是旧 P3 的 intervention-response Spearman；IDSSE 本身没有该 intervention response，因此不据此扩写机制 claim。

| Variant | Phase macro-F1 | Field-zone macro-F1 | Centroid MAE | Natural-pair ranking accuracy | Geometry relation proxy | `z_mode` translation response |
|---|---:|---:|---:|---:|---:|---:|
| raw single-channel | 0.5508 | 0.9706 | 0.0459 | 0.8030 | -0.4660 | 0.2404 |
| centered single-channel | 0.5914 | 0.7459 | 0.1908 | 0.9302 | 0.5954 | ~0 |
| raw relational pooling | 0.5831 | 0.9667 | 0.0759 | 0.8332 | -0.3234 | 0.2966 |
| fixed dual-channel | 0.5382 | 0.9619 | 0.0554 | 0.9487 | 0.6761 | ~0 |

相对于 raw single-channel，fixed dual-channel 的 context 变化为：phase macro-F1 `-0.0126`、field-zone macro-F1 `-0.0087`、centroid MAE `+0.0095`，均在 T5R2 锁定的 `0.05` non-inferiority margin 内。natural-pair ranking accuracy 提升 `+0.1456`；三个 seed 均优于 raw baseline，分别为 `0.9596/0.9580/0.9283`。三个 seed 的 geometry proxy 均为正（`0.6676/0.6723/0.6886`）。

Procrustes analytic control 的 valid ranking accuracy 为 `0.9372`，说明该自然配对任务本身已有较强的几何可解性；dual 相对该 control 的增益不能被写成方法优势，只能说明 dual 在当前神经 baseline 中保留并访问了内部结构。

## 3. Cross-readout 与统计边界

fixed dual-channel 的三个 seed 中，context head 读取 `z_mode` 时 field-zone macro-F1 仅 `0.2529–0.2577`、centroid MAE 为 `0.2859–0.2868`；mode head 读取 `z_ctx` 时 ranking accuracy 为 `0.2376–0.3270`。paired latent correlation 为 `0.1240–0.1567`。这支持任务所需信息在两个通道间具有可审计的访问分离，但不证明语义理解。

所有 context/intrinsic 主指标都保留了 2,000 次 match-level bootstrap（seed `20260904`）。valid 只有 2 场比赛，区间和方向稳定性只能作小样本 sanity evidence，不能当作独立确认。

## 4. T5R3 门判断

本次结果满足进入 T5R4 前的方向性条件：

1. context task 相对 raw baseline 未超过 non-inferiority margin；
2. intrinsic natural-pair ranking 在三个 seed 上均改善；
3. `z_mode` global translation response 近零；
4. geometry relation proxy 未坍缩，且三个 seed 同方向；
5. cross-readout 显示 context/intrinsic 访问不是同一条绝对位置捷径。

因此 T5R3 记录为 `DIRECTIONAL_PASS`，下一步可审阅后进入锁定预算内的 T5R4 bounded autoresearch。该判断不等于 P4 放行，也不等于旧 P3 geometry-response 机制已在 IDSSE 上复现。

## 5. 证据与失败尝试

权威运行目录：`artifacts/phase3/task_semantic_repair_v1/t5r3_sanity_v3/`。

- 运行清单：`manifest.json`；12 个 neural checkpoints，4 variants × 3 seeds；
- 协议：`protocol.json`；记录 T5R2 lock hash、输入视图、固定图、loss、translation 和 firewall；
- 结果：`model_results.json`、`summary.json`、`analytic_control.json`；
- 审计：`scripts/audit_t5r3_sanity.py`，当前 `T5R3_AUDIT: PASS`；
- runner：`scripts/run_t5r3_sanity.py`；定向回归测试：`tests/test_t5r3_sanity.py`。

此前的 v1 原始 attention 实现在 raw seed11 的 30/80 epoch 中止，v2 fixed message-passing 实现因会话输出问题在 raw seed11 的 60/80 epoch 中止；两者均没有正式 summary/manifest，均不纳入上述结论。v3 是唯一完整运行和审计的 T5R3 证据。

T5R4、candidate lock、reserved-match 检查、独立 provider confirmation、AMR 和 P4 仍未运行。
