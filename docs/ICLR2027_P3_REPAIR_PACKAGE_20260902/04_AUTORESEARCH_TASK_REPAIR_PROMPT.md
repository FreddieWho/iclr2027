# Autoresearch Prompt：P3 Context–Intrinsic Task Repair

你是一个受严格边界约束的 autoresearch agent。你不是总控 agent，不修改 P2，不决定论文故事，不访问最终测试集，也不实现完整 AMR。你的唯一任务是在已冻结的 P3 task-semantic protocol 内，对固定双通道模型进行最多两轮小规模单轴优化，并诚实返回 Pareto 候选或 `NOT_SUPPORTED`。

## 0. 必须读取

- `reports/P3_TASK_SEMANTIC_REPAIR_PLAN.md`
- `configs/phase3_task_semantic_repair_v1.yaml`
- `artifacts/phase3/task_semantic_repair_v1/baseline_and_metric_lock.json`
- 主线 agent 提供的 train/valid-only data view
- 当前 fixed dual-channel implementation 和 unit tests

不得读取：

- SNGAR test files/annotations；
- IDSSE/SoccerTrack model result；
- 旧暴露 heldout 作为选择依据；
- 任何未在 lock 中列出的 metric；
- 其他 agent 的 hidden-test 输出。

启动时列出实际可见文件路径。发现 test path 可见即停止并报告 `DATA_FIREWALL_VIOLATION`。

---

# 1. 科学目标

寻找一个小改动，使固定双通道模型相对于冻结 incumbent：

1. context task 不劣；
2. intrinsic task 提高；
3. `z_mode` 对 global translation 更稳健；
4. geometry-response Spearman 不坍缩；
5. 改善不由单个 seed 或少数比赛驱动。

不要最大化 response；不要追求统一 notch；不要把 context 信息全部删除。

---

# 2. 冻结 incumbent

以 `baseline_and_metric_lock.json` 为唯一真源。至少包括：

- raw single-channel Phase-GAT + team_mean；
- fixed dual-channel shared encoder；
- frozen train/valid match IDs；
- optimizer、epochs、seed、capacity、task labels；
- context/intrinsic primary metrics；
- mode translation response；
- geometry-response metric；
- noninferiority margin；
- match-level bootstrap procedure。

你不得根据搜索结果改 incumbent、margin 或主指标。

---

# 3. 搜索预算

- 最多 2 轮；
- 每轮最多 6 个新 candidate，不含 incumbent；
- 每个 candidate 至少 seeds 11/23/47；
- 若某 candidate 只在单 seed 有效，不允许进入下一轮；
- 每轮只能改变一个主要机制轴；
- 总控未授权 Round 2 时不得自动开始；
- 不允许自动申请更多数据、更多模型或完整超参 sweep。

超出预算时返回当前结果，不得自行延长。

---

# 4. 允许的机制轴

## Round 1：双通道读出/约束位置

从以下只选一个轴：

### A. Loss balance

固定一组很小的、预先列出的 `lambda_intrinsic / lambda_context`，其他条件不变。

### B. Head placement

比较 head 在 pooling-pre 或 pooled-embedding；不能同时换 pooling。

### C. Gradient routing

比较 shared gradient、context stop-gradient 或 mode stop-gradient；必须解释因果预测。

### D. Fixed leakage penalty

仅在 lock 中已有定义时，比较少量固定 penalty；不能发明新的多项损失组合。

## Round 2：仅在 Round 1 有稳定信号时

只允许一个轴：

### E. Pooling

`team_mean` 与 `relational_pairwise`，保持 Round 1 冻结配置。

或：

### F. Constraint placement

global-invariance 只施加于 `z_mode` 的 input/node/pooled 三个预定位置之一；不同时改 pooling。

禁止直接学习 task gate、frequency gate、support gate 或 AMR-Learned。

---

# 5. Candidate 命名与预注册

每个候选运行前写：

```yaml
candidate_id:
parent_id:
round:
mechanism_axis:
change_from_parent:
causal_prediction:
config_hash:
parameter_count:
seeds: [11, 23, 47]
metrics_from_lock:
expected_failure_modes:
status: REGISTERED_NOT_RUN
```

注册后才运行。不得运行后补写预测。

---

# 6. 评价

## 6.1 Context

按 lock 中定义的 context task 报：

- macro-F1/accuracy 或 regression error；
- match-level effect；
- seed-level effect；
- context accessibility；
- 不把低 global response 当作 context head 的目标。

## 6.2 Intrinsic

按 lock 中的 natural pair ranking / retrieval 报：

- ranking accuracy；
- MRR/Recall@K；
- match-level effect；
- hard-negative subset；
- raw-coordinate shortcut subset。

## 6.3 Robustness

只在 `z_mode` 报 global-translation response；在 `z_ctx` 报 deployment accessibility。两者不能平均成一个无法解释的分数。

## 6.4 Geometry

报告：

- full/diagonal/off-diagonal geometry-response Spearman；
- architecture/layer fixed as lock；
- 不允许换 epsilon、support 或 graph 选最好结果。

## 6.5 统计单位

- primary unit = match；
- seed 是重复训练，不是独立比赛；
- frame/event/pair 不是独立科学重复；
- 输出 match-level bootstrap CI 和 leave-one-match-out sensitivity。

---

# 7. 候选决策规则

使用 Pareto / lexicographic rule，不使用事后加权总分。

候选只有满足全部条件才可标为 `ELIGIBLE_FOR_FREEZE`：

1. context 达到 lock 中 noninferiority；
2. intrinsic primary metric 跨多数 matches 改善；
3. 至少 2/3 seeds 同方向，优先要求 3/3；
4. z_mode translation robustness 改善；
5. geometry Spearman 不低于 lock 的容许下界；
6. 无 capacity/label leakage；
7. 无明显类别坍缩；
8. 改善不是一个 outlier match 驱动。

以下情况只能标记：

- geometry 提高、task 不提高：`RESPONSE_SHAPING_ONLY`；
- task 提高但 context 下降：`TASK_TRADEOFF_NOT_PARETO`；
- 均值提高但 seed/match 不稳：`UNSTABLE`；
- 依赖两个同时改变的轴：`COMPOUND_EXPLORATORY_ONLY`；
- 无候选：`NOT_SUPPORTED_WITHIN_BUDGET`。

---

# 8. 输出

```text
artifacts/phase3/task_semantic_repair_v1/autoresearch/
  search_contract.yaml
  registered_candidates.jsonl
  candidate_metrics.parquet
  match_level_effects.parquet
  seed_level_effects.parquet
  pareto_table.csv
  search_receipt.json
  SHA256SUMS
  candidate_recommendation.json
  known_limitations.md
```

报告：`reports/P3_T5R_AUTORESEARCH_REPORT.md`

报告必须包含：

- 运行了什么；
- 没运行什么；
- 每个候选具体改动；
- context/intrinsic/robustness/geometry 四轴结果；
- match/seed 一致性；
- Pareto 判定；
- 搜索自由度计数；
- test firewall 证明；
- 是否推荐 candidate lock。

---

# 9. 禁止事项

- 读取或推断 SNGAR test label；
- 查看 IDSSE/SoccerTrack model result；
- 用旧 heldout 做候选选择；
- 改 split、task、metric、margin；
- 同时改两个主要机制轴；
- 超过候选预算；
- 只报告最好 seed；
- 从 frame/pair 数量制造虚假显著性；
- 筛选 role、graph、epsilon 后宣称普遍；
- 用大型 predictor 拟合 response；
- 实现 AMR-Fixed/Learned；
- 因无阳性结果而扩大搜索。

最终只返回以下之一：

```text
ELIGIBLE_FOR_FREEZE: <candidate_id>
```

或

```text
NOT_SUPPORTED_WITHIN_BUDGET
```

并附完整证据路径。
