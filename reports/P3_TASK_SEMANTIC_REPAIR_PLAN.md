# P3-T5R Task-Semantic Repair Plan

日期：2026-09-04
状态：`T5R4_ROUND2_COMPLETE_AUTORESEARCH_CLOSED`
所属阶段：既有 `P3_CAUSAL_MECHANISM`

## 1. 目的与边界

本报告是 P3-T5R 的执行计划，不是第二套科学蓝图。它承接已经冻结的 P2/P3 结果：support-conditioned normalized-embedding local geometry 能预测并塑造表示 response，但旧 task gate 没有证明 task repair。旧 T5 将需要保留绝对部署信息的 context task 与要求全局平移稳健的 intrinsic task 放入同一 gate，因此其阴性结论限定为：

```text
NOT_SUPPORTED_UNDER_OLD_CONFLATED_TASK_GATE
```

P2 artifacts、P2 reports、P2 manifests、P2 checkpoints 和既有 P3 geometry artifacts 只读。P3-T5R 不新增 Phase，不启动 AMR，不重算 P2/T1/T2。

## 2. 当前路线

```text
P3-T5R0 protocol/document repair
  -> P3-T5R1 IDSSE substitute acquisition and canonical conversion
  -> P3-T5R2 task construction and baseline lock
  -> P3-T5R3 fixed dual-channel sanity
  -> P3-T5R4 bounded autoresearch
  -> P3-T5R5 candidate lock and single hidden test
  -> P3-T5R6 external confirmation and P4 decision
```

所有编号均属于 `P3_CAUSAL_MECHANISM`，不创建 P3.5、P3b 或 P4-pre。

## 3. 数据和 firewall

| 数据 | 当前用途 | 搜索期间状态 |
|---|---|---|
| IDSSE（7 场） | 暂代开发、任务构造、T5R3/T5R4 Round 1 | 来源、SHA-256、canonical QC 和 T5R2 lock 已完成；当前只读 train/valid |
| SNGAR train/valid（45/9 场） | 原计划开发数据 | 暂缓，访问阻塞；不影响当前 IDSSE 替代分支 |
| SNGAR test（10 场） | 同源未见确认 | candidate lock 前不得下载/读取 |
| SkillCorner（10 场） | 历史探索、dynamic support 规则开发 | 不作为最终确认 |
| IDSSE 保留 match | 同源未见检查 | candidate lock 后一次性读取；不称独立 external confirmation |
| SoccerTrack v2 | parser smoke 与 acquisition shift | smoke 不进入最终统计 |

旧 heldout 状态固定为 `EXPOSED_DURING_CANDIDATE_SEARCH`，只允许 `exploratory_audit_only`，不得作为候选选择或最终确认。所有数据源的 raw→canonical mapping、版本、许可和 checksum 必须保留。

## 4. 任务定义

### Context channel

输入 raw positions，主表示为 `z_ctx`，至少包含：

- phase/deployment classification；
- absolute team centroid 或 field-zone prediction。

该通道允许使用 absolute deployment information；global translation response 不能被简单当作越低越好。

### Intrinsic channel

输入 globally centered positions，主表示为 `z_mode`，至少包含一个自然、非 intervention-label 任务：

- natural pair ranking；或
- intrinsic formation retrieval。

该通道要求 global translation robustness，但仍需保留内部构型和 support/relationship information。

### Dynamic support

基于 response-blind dynamic events 构造 pressing chain、simultaneous engagement、off-ball run 或 passing-option support。该子任务非 P4 阻塞条件，不能反向选择主候选。

## 5. 最小固定双通道

```text
shared encoder E
raw view:       X           -> Pool -> z_ctx  -> context head
centered view:  X - mean(X) -> Pool -> z_mode -> intrinsic head
```

两视图先共享 encoder，context head 只读取 `z_ctx`，intrinsic head 只读取 `z_mode`。cross-readout 仅用于 leakage audit，不作为主模型。role 不作为通用运行时必要输入。

冻结 baseline：raw single-channel、centered single-channel、raw relational pooling、fixed dual-channel、raw-coordinate/Procrustes。具体 optimizer、epoch、capacity tolerance、seeds、metrics 和 non-inferiority margin 已在 `artifacts/phase3/task_semantic_repair_v1/baseline_and_metric_lock.json` 冻结；该 lock 只冻结协议，不包含模型结果。

## 6. 搜索和确认 gate

固定双通道 sanity 只有出现 context noninferiority、intrinsic 正向信号、`z_mode` translation robustness 改善且 geometry relation 不坍缩时，才调度 autoresearch。autoresearch 最多两轮，每轮最多六个新候选，一次只改一个主要机制轴，至少使用 seeds 11/23/47。

候选必须在读取 IDSSE 保留 match 前写入 `candidate_lock.json`，并声明保留 match 和独立 external results 在搜索期间均不可见。保留 match 只读取一次；随后不再调参，再等待 SNGAR 或其他 provider/source 的独立确认。

进入 P4 AMR-Fixed 必须同时满足：未见 match geometry prediction、context noninferiority、intrinsic task improvement、`z_mode` nuisance robustness、多 seed/match 一致、保留 match 复现、独立 provider/source 方向保持、无 leakage/role oracle/capacity shortcut，并得到固定的非后验 routing rule。任一核心条件失败，冻结为 diagnostic/mechanistic version，不启动 AMR。

## 7. 当前证据状态

已完成：T5R0 的文档/config 迁移；已固定 IDSSE 暂代 SNGAR 的开发数据角色；T5R1 的官方来源、raw manifest、23/23 SHA-256、7 场 canonical conversion receipts 和逐场 QC；T5R2 的 match-level split、context/intrinsic task views、natural pair protocol 与 baseline/metric lock。
T5R2 任务视图：train=`J03WMX,J03WOH,J03WPY,J03WR9`，valid=`J03WN1,J03WOY`，reserved=`J03WQQ`；train/valid 分别为 23,052/6,127 snapshots 和 3,750/980 natural pairs。`J03WQQ` 未加载，未写出 reserved task arrays。
已运行：fixed dual-channel sanity（仅 train/valid，v3 正式证据）、T5R4 Round 1（4 候选、12 个模型，无全维度 Pareto 支配者）
和 T5R4 Round 2（最后一轮，仅 encoder update balance 轴，3:1/2:1，总步数 420 持平，6 个模型；按训练前冻结规则选出
`update_ratio_2to1`，bounded autoresearch 已关闭）；尚未运行：candidate lock、IDSSE 保留 match 检查、独立 provider confirmation。v1/v2 中止尝试不纳入证据。
当前 P4：`blocked_pending_p3_t5r_gate`。T5R3 closure review 已接受 T5R4 Round 1 探索，具体候选仍不自动晋级。

T5R3 已完成并记录在 `reports/P3_T5R3_SANITY_REPORT.md`。其方向性门为 `PASS`，但 valid 只有 2 场比赛，且 geometry 指标是 natural-pair proxy，不是 intervention-response。T5R4 Round 1 已完成并记录在 `reports/P3_T5R4_ROUND1_REPORT.md`，无全维度 Pareto 支配者。T5R4 Round 2 已完成并记录在 `reports/P3_T5R4_ROUND2_REPORT.md`（入选 `update_ratio_2to1`，搜索关闭）；语义显示名以 `artifacts/phase3/task_semantic_repair_v1/metric_semantics_addendum.json` 为准（match-half context probe、intrinsic structural accessibility proxy）。

机器配置：`configs/phase3_task_semantic_repair_v1.yaml`。  
权威 artifacts：`artifacts/data_v2/idsse/`、`artifacts/phase3/task_semantic_repair_v1/`。
一致性审计：`scripts/audit_current_state_consistency.py`、`scripts/audit_idsse_t5r2.py`；后者当前 `PASS`。
