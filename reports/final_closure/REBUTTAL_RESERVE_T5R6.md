# REBUTTAL RESERVE — T5R6 独立 provider 确认（SoccerTrack-v2）

> 用途：**仅用于 rebuttal 回应"外部效度/只在合成数据成立"类质疑**。不进正文、
> 不进附录；属路线调整前 P3 线（support-conditioned geometry + task-semantic
> repair）的外部确认证据，与当前论文主线（update-failure）是不同 claim。
> 建立：2026-09-22（用户授权落盘）。来源报告：
> `reports/e2_p3_t5r/P3_T5R6_EXTERNAL_CONFIRMATION_REPORT.md`、
> `reports/e2_p3_t5r/P3_T5R5_HIDDEN_CONFIRMATION_REPORT.md`。

## 一句话

本项目的表示干预协议曾在**独立 provider 的真实足球数据**（SoccerTrack-v2，
8 场未见比赛，大学级别联赛）上完成过带锁外部确认：任务与干预指标 8/8 场同向
通过（verdict `T5R6_CONFIRMED`），全程零训练、零阈值改动、锁后单次读取。

## 可引用事实（均有 artifact/锁锚点）

| 事实 | 数字 | 锚点 |
|---|---|---|
| 数据 | SoccerTrack-v2（CC BY 4.0，revision `eae51793`），8/10 场（排除 2 个 smoke 场）；44,401 快照 / 7,534 自然对 | `artifacts/data_v2/soccertrack/`（manifest+SHA+收据） |
| 协议 | 五层锁：candidate lock＋干预协议锁＋shadow lock＋审计（`T5R6_AUDIT: PASS`，72/72 记录复算一致）＋单次读取纪律 | `configs/t5r5_hidden_confirmation.yaml`、`scripts/p3_t5r/audit_t5r*.py` |
| 模型 | 冻结 9 checkpoints（2:1 selected / fixed-dual / raw × 3 seeds）＋Procrustes 解析对照 | `artifacts/phase3/` |
| 任务确认（macro） | 2:1：zone F1 0.9653、pair acc 0.9219、geometry 0.5738、centroid MAE 0.0365、z_mode≈0；2:1 在 8 场 geometry/pair/zone 全部高于 fixed-dual 同场值 | T5R6 报告表 |
| 干预确认 | 8/8 场 full 高于基线、方向全部高于 chance（full 区间 0.421–0.602；最弱场 132877 full=0.421 仍过，如实保留） | T5R6 报告逐场表 |
| 前序单次保留场 | IDSSE `J03WQQ`（3575 snapshots/515 pairs）任务 gate 全过、干预 full mean 0.63 强通过，`T5R5_PASS_STRONG` | T5R5 报告 |

## 使用边界（措辞纪律）

- ✅ 可以说："本项目的测量与修复协议曾通过独立 provider、带锁、单次读取的
  外部确认流程检验（SoccerTrack-v2，8/8 场同向）"——证明的是**协议纪律与
  旧线机制证据的可迁移性**。
- ❌ 不能说：当前论文的 update-failure 主结论在 SoccerTrack-v2 上成立
  （T5R6 测的是 P3 线的 geometry→response 与任务语义修复，不是本文的
  transition/compositional blindness）。
- ❌ 不能说：跨域泛化（单一独立 provider、大学级别联赛）。
- 该线的阴性边界（旧 causal switch 仅 response shaping）见
  `docs/governance_history/CLAIM_LEDGER_P2P3_20260916.md` P3-C5/C6。

## 若 reviewer 要求"在新数据上复现本文主结论"

可用回应路径：本文坐标/像素/IDSSE 三线证据已在正文；SoccerTrack-v2 的 GSR
转换管线与冻结模型仍在本仓库（`scripts/p3_t5r/prepare_soccertrack_t5r6.py`、
`scripts/p3_t5r/run_t5r6_external_confirmation.py`），具备在独立 provider 上
复测 turn/event/action 三任务的现成基础设施（E2 范式，`experiments/last15h/`）。
注意：holdout 纪律要求任何新读取是新的协议事件。
