# P3-T5R5 Candidate Lock

日期：2026-09-05
状态：`T5R5_CANDIDATE_LOCKED_SINGLE_READ_AUTHORIZED`
阶段：既有 `P3_CAUSAL_MECHANISM`，任务线 `P3-T5R5`

本 lock 在任何 `J03WQQ` 任务/结果读取之前创建（pre-lock HEAD `c72d66c`）。
此后只允许**一次**保留 match 读取，不做重训练，不改协议，不调超参，不启动 P4。

## 1. 锁定对象

- selected candidate `update_ratio_2to1`（280/140/420，loss 1.0/1.0，shared routing，
  无 readout norm，`team_mean` pooling，未改动的 T5R3 shared encoder，seeds 11/23/47，
  141,769 参数）；3 个 checkpoint 路径与 SHA-256 见
  `artifacts/phase3/task_semantic_repair_v1/candidate_lock.json`（`model.checkpoints`）。
- T5R4 报告/config/protocol/summary/hash 见 lock `identity` 节。
- 冻结对照集：`update_ratio_2to1` ＋ T5R3 fixed dual ＋ T5R3 raw single-channel
  （各 3 seeds，checkpoint hash 全锁定）＋ Procrustes analytic control。不做重训练。

## 2. 数据与任务

- IDSSE 官方 revision `a715a38…`，canonical manifest 与 split lock hash 见 lock `data` 节；
  train/valid/reserved ID 与 T5R2 一致，reserved = `J03WQQ`。
- lock 创建时 reserved 任务数组/结果均未读出（`reserved_task_arrays_written_before_lock=false`）。
- 隐藏任务生成逐字沿用 T5R2（`build_snapshots`＋`build_natural_pairs`，阈值见 lock
  `hidden_task_generation`）；若保留 match 自然对过少/为零，按 UNDERPOWERED/UNDEFINED
  报告，不改阈值。
- 任务语义见 semantic addendum（match-half 为辅助 probe；field-zone/centroid 为
  absolute-context；natural-pair 为几何构造 proxy；translation 为内禀 nuisance 检验）。

## 3. 干预协议（与 T5R 原 P3 机制重连）

- 完整冻结见 `artifacts/phase3/task_semantic_repair_v1/t5r5_intervention_lock.json`：
  表示映射 f（raw→centering→frozen 图/encoder→pool→mode head→L2），Jacobian 与
  observed response 取同一表示层；fracture 端点重分配算子（同 multiset、同 support
  大小、不同端点分配，response-blind，不用静态 role）；单 epsilon `0.25`
  （归一化半 pitch 单位，与 P2 形式范围最低档精确对应，可见 6 场全为 105×68）；
  最多 250 等距快照；full/diagonal 主预测器＋Rayleigh 简单基线；不训练任何预测器。

## 4. 预冻结判决规则

- 任务语义 gate（§9.1）与实际干预机制 gate（§9.2）已写入 lock
  `task_semantic_gate`/`intervention_gate`，hidden config 同步。
  2:1 无需在干预预测上击败 fixed dual 即可复现核心局部几何机制（该比较为次要）。
- 四类结局与路由（A 强通过／B 混合／C 候选失败但机制存活／D 机制失败）见 lock
  `outcome_routes`；C 路径下 T5R3 fixed dual 仅允许作 T5R6 诊断参照（已预声明），
  不得作为新调方法。
- T5R6 shadow rules 已冻结（`t5r6_shadow_confirmation_lock.json`），T5R5 期间不用外部结果。

## 5. 防火墙

Round 3、hidden 前后重训练、阈值/目标改动、旧 heldout/外部结果选择、SNGAR test、
书法确认均禁止；P4 blocked；保留 match 只读一次。
机器审计：`scripts/audit_t5r5_candidate_lock.py`；隐藏运行器：
`scripts/run_t5r5_hidden_confirmation.py`（lock 缺失/非法、checkpoint hash 不符、
输出目录已存在、阈值被改、要求重训练时一律拒绝）。
