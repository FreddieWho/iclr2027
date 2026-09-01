# 主线研究 Agent Prompt：P3 Task-Semantic Repair 与独立数据确认

你是 `FreddieWho/iclr2027` 的总控研究 agent。你的任务是在不重写 P2/P3 历史、不新增 Phase、不提前进入 AMR 的前提下，完成一个有界的 P3 task-semantic repair，并给出明确的 P4 go/no-go。

## 0. 开工前必读

依次读取：

1. `STATUS.md`
2. `DECISIONS.md`
3. `CLAIM_LEDGER.md`
4. `reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md`
5. `reports/P3_CANDIDATE_SEARCH_REPORT.md`
6. `configs/phase3_support_geometry_v1.yaml`
7. `docs/01_SCIENTIFIC_BLUEPRINT.md`
8. `docs/02_METHOD_SPEC_AMR.md`
9. `docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md`
10. `docs/04_DATA_AND_ENVIRONMENT_GUIDE.md`
11. `docs/05_AGENT_EXECUTION_MANUAL.md`
12. 本包的 `01_DATASET_ACQUISITION_AND_SPLIT_PLAN.md`
13. 本包的 `02_EXISTING_PLAN_UPDATE_INSTRUCTIONS.md`

首先记录当前 git HEAD、dirty state、P2/P3 authoritative artifact hashes。任何冻结资产 hash 不匹配时 fail closed。

---

# 1. 当前科学事实

必须按以下边界工作：

- P2 已冻结为 `mixed_or_graph_specific`；
- complete organization blindness 和 universal notch 不成立；
- ordinary spectral power 不能解释全部 response；
- support-conditioned local geometry 能强预测旧 response 和同资产 prospective response；
- diagonal sensitivity 为主，off-diagonal 有架构/层依赖的额外贡献；
- pooling switch 能塑造 geometry/response；
- 旧 task repair 没有得到稳定任务—鲁棒性 Pareto；
- 旧 heldout 已在候选搜索中暴露；
- AMR 尚未测试，P4 blocked；
- “模型敏感”不等于“模型理解语义”。

你不得修改这些历史结论来制造更整齐的故事。

---

# 2. 本轮唯一科学问题

> 当 context signal 与 intrinsic-structure nuisance 被正确分离后，support-conditioned local geometry 是否能指导一个固定、可审计、matched-capacity 的双通道表示，在独立比赛上同时保留 context task、改善 intrinsic task，并保持 geometry-response 可解释性？

次问题：

> 用动态事件定义的真实协同 support，是否比静态 role 更能形成稳定的 semantic-vs-matched-random 差异？

次问题可并行，但不得阻塞主 task-semantic repair。

---

# 3. 阶段边界

当前仍是 `P3_CAUSAL_MECHANISM`。使用任务编号：

- `P3-T5R0`：protocol/document repair
- `P3-T5R1`：data acquisition and canonical conversion
- `P3-T5R2`：task construction and baseline lock
- `P3-T5R3`：fixed dual-channel sanity
- `P3-T5R4`：bounded autoresearch
- `P3-T5R5`：candidate lock and single hidden test
- `P3-T5R6`：external confirmation and P4 decision

禁止创建 P3.5、P3b、P4-pre 或任何新 Phase。

---

# 4. P3-T5R0：协议与文档修复

## 必须完成

1. 按 `02_EXISTING_PLAN_UPDATE_INSTRUCTIONS.md` 原位更新全部 canonical docs/configs；
2. 旧 heldout 标记为 `EXPOSED_DURING_CANDIDATE_SEARCH`；
3. 创建 `reports/P3_TASK_SEMANTIC_REPAIR_PLAN.md`；
4. 创建 `configs/phase3_task_semantic_repair_v1.yaml`；
5. 创建 current-state consistency audit；
6. 在运行数据或模型前提交独立 commit。

## 验收

- P2/P3 frozen hash 未变化；
- README、MASTER prompt、STATUS、DECISIONS、CLAIM_LEDGER、source docs 与 configs 状态一致；
- P4=`blocked_pending_p3_t5r_gate`；
- P5 calligraphy=`blocked_pending_sports_prediction_lock`；
- 所有未运行 claim 为 WORKING_HYPOTHESIS/NOT_ESTABLISHED。

---

# 5. P3-T5R1：数据获取与转换

## 数据优先级

### 主开发

`OpenSportsLab/SNGAR-Action-Spotting-Tracking`

- 只下载 train+valid；
- test 不得下载；
- pin dataset revision；
- 校验 `MANIFEST.sha256`；
- 时间索引使用 `videoTimeMs`；
- 每场生成 conversion receipt。

### 现有动态 support 开发

SkillCorner current open data：复用 dynamic events + phases，记录 source commit。

### 外部确认

IDSSE：parser 可开发到 schema parity，但在 candidate lock 前不得运行模型选择或查看对比结果。

### Acquisition shift

SoccerTrack v2：最多两场用于 parser smoke；其余保留。

## Canonical conversion

实现 source adapters 输出统一 `CanonicalMatch/CanonicalFrame/DynamicSupportRecord`。不得修改现有 P1 canonical assets；新数据放入 versioned directory，例如：

```text
artifacts/data_v2/
  sngar/
  idsse/
  soccertrack_v2/
  conversion_manifests/
```

## 数据 QC

逐 match 报告：

- frames、duration、time gaps；
- node count；
- player identity/role coverage；
- ball coverage；
- out-of-pitch rate；
- long-gap interpolation policy；
- event alignment error；
- canonical coordinate parity；
- checksum。

禁止通过丢弃表现差的比赛提高结果；所有排除必须基于 response-blind QC rule。

---

# 6. P3-T5R2：任务构造与 baseline lock

## 6.1 Context tasks

至少两项：

1. phase/deployment classification；
2. absolute team-centroid / field-zone prediction。

它们由 raw-coordinate context head 读取。global translation response 不作为越低越好的统一指标；应报告 context accessibility 和 translation equivariance/sensitivity。

## 6.2 Intrinsic tasks

必须至少有一个自然、非 intervention-label 主任务：

### 首选：Natural pair ranking

构造规则在模型评估前冻结：

- positive：centered/Procrustes internal geometry 相近、absolute centroid 距离大；
- hard negative：absolute centroid 相近、internal geometry 距离大；
- pair 不跨 split；
- query/positive/negative 的 match 关系和泄漏规则明确；
- 主要指标为 match-grouped ranking accuracy、MRR/Recall@K 或 triplet ranking margin。

### 次选：Intrinsic formation retrieval

- 使用 natural frames；
- target 由 internal geometry/phase-stable window 定义；
- 不把合成 global translation copy 作为唯一 positive；
- 做 raw-coordinate shortcut audit。

## 6.3 Dynamic support task

调度 `05_AUTORESEARCH_DYNAMIC_SUPPORT_PROMPT.md` 的 agent。其输出进入附加证据，不用于选择主 task-repair 超参数。

## 6.4 Baseline lock

必须冻结以下 baseline：

1. raw single-channel Phase-GAT + team_mean；
2. centered single-channel；
3. raw relational pooling；
4. fixed dual-channel（主候选）；
5. geometry-free simple baseline：raw coordinates / Procrustes / pairwise distances。

冻结：

- train/valid matches；
- labels；
- seeds；
- epochs/optimizer；
- parameter matching；
- primary metrics；
- noninferiority margin；
- bootstrap unit；
- candidate budget。

生成 `baseline_and_metric_lock.json` 并提交后才能训练候选。

---

# 7. P3-T5R3：固定双通道最小检验

## 模型

先使用最简单、共享权重的两视图结构：

```text
shared encoder E
raw view:       X          → E → Pool → z_ctx
centered view:  X - mean(X)→ E → Pool → z_mode
context head:   h_ctx(z_ctx)
intrinsic head: h_mode(z_mode)
```

约束：

- 不实现频段 gate；
- 不实现 AMR-Learned；
- 不增加模型动物园；
- 与 baseline matched capacity；
- 若两个 head 增加参数，为 baseline 添加等容量 dummy/head control；
- role 不作为必须输入；
- train/valid 只使用 SNGAR train/valid；
- 至少 seeds 11/23/47，预算允许时增至 5 seeds；
- 统计单位 match。

## 交叉泄漏审计

同时报告：

- context head 读取 intrinsic label；
- mode head 读取 context label；
- z_mode 对 global translation response；
- z_ctx 对 absolute deployment accessibility；
- mode/context representation covariance；
- geometry-response Spearman。

目的不是强制完全解耦，而是确认任务所需信息是否可以分别访问。

## Sanity go/no-go

只有同时满足以下方向，才调度 autoresearch：

- context task 相对 raw baseline 不出现稳定下降；
- intrinsic task 有跨 seed/match 的正向信号；
- z_mode global translation response 下降；
- geometry-response predictive relation 不崩溃；
- 结果不由单个 seed 或单场比赛驱动。

若固定双通道完全无信号，停止候选搜索，直接进入 diagnostic paper 路线。

---

# 8. P3-T5R4：有界 Autoresearch

使用 `04_AUTORESEARCH_TASK_REPAIR_PROMPT.md`，总控必须通过只读 data views 与 test firewall 调度。

硬预算：

- 最多 2 rounds；
- 每轮最多 6 candidates；
- 每 candidate 至少 3 seeds；
- 一次只改一个主要机制轴；
- 不访问 test/IDSSE result；
- 不改变 metrics/split/task；
- 不进行 architecture zoo；
- 不增加超过 matched capacity tolerance 的参数；
- 不允许事后改 Pareto 权重。

Round 1 可探索：

- context/mode loss balance；
- head placement；
- stop-gradient placement；
- fixed fusion/readout separation。

Round 2 只有 Round 1 有稳定信号才允许探索：

- team_mean vs relational pooling；或
- fixed recoverability/invariance placement。

不得同轮同时改 pooling 与 constraint。

总控负责审计 agent 的每次读取路径，确保 test labels 从进程和 metadata 中物理隔离。

---

# 9. P3-T5R5：候选冻结与一次性测试

## Candidate lock

选择一个候选后，生成符合 `templates/candidate_lock.schema.json` 的：

```text
artifacts/phase3/task_semantic_repair_v1/candidate_lock.json
```

包含：

- git commit；
- data revision/hashes；
- architecture/config；
- seed list；
- checkpoint hashes；
- tasks；
- metrics；
- noninferiority margins；
- expected failure conditions；
- figure scripts；
- valid results summary；
- explicit declaration that test was not read。

提交 candidate lock。其 commit 之后才允许下载 SNGAR test。

## 一次性 test

- test 不用于早停、阈值选择、class merging 或 figure selection；
- 一次执行所有 frozen candidates：incumbent + fixed dual-channel + 最终候选；
- 失败不得重新调参后重跑同一 test；
- 输出按 match 的 effect、bootstrap CI 和 seed consistency；
- 保存 complete receipt/hash。

---

# 10. P3-T5R6：外部确认

在候选 test 结果读取后，不再改模型。运行：

1. IDSSE 7 场 external confirmation；
2. 可选 SoccerTrack remaining matches acquisition-shift；
3. 新比赛上 geometry-response prediction；
4. context/intrinsic task 方向；
5. dynamic support descriptive result。

允许 provider-specific coordinate/time parser，但不允许 provider-specific model tuning。若 label ontology 不一致，只评估可映射的预先列出的指标。

---

# 11. 最终 P4 gate

## 进入 P4 AMR-Fixed

必须同时有：

1. SNGAR test 上 geometry 增量复现；
2. context task noninferior；
3. intrinsic task 改善；
4. z_mode nuisance robustness 改善；
5. 多 seed、多 match 一致；
6. external data 方向不崩溃；
7. 无 test leakage；
8. 无 role oracle/capacity shortcut；
9. 一条可从 P3 机制推出的固定 routing rule。

此时只允许进入 AMR-Fixed，不直接 AMR-Learned。

## 不进入 P4

任一核心条件持续失败：

- 冻结 P3-T5R；
- 不再搜索；
- 论文收缩为：Action-Mode Probe + organization beyond power + support-conditioned local geometry + architecture anatomy + response-shaping boundary；
- task-repair 阴性结果进入限制/附录；
- calligraphy 仅作 exploratory，不承担确认性 claim。

---

# 12. 产物规范

每个 task 输出：

```text
artifacts/phase3/task_semantic_repair_v1/<task>/
  execution_contract.yaml
  input_provenance.json
  manifest.json
  receipt.json
  SHA256SUMS
  match_level_results.parquet
  seed_level_results.parquet
  summary.json
  known_limitations.md
  figure_data/
```

报告：

- `reports/P3_TASK_SEMANTIC_REPAIR_PLAN.md`
- `reports/P3_TASK_SEMANTIC_REPAIR_RESULTS.md`
- `reports/P3_DYNAMIC_SUPPORT_REPORT.md`
- `reports/P3_T5R_FINAL_DECISION.md`

状态同步：

- `README.md`
- `STATUS.md`
- `DECISIONS.md`
- `CLAIM_LEDGER.md`
- canonical docs/configs
- `PROJECT_PACKAGE_CONSOLIDATED.md`

---

# 13. 禁止事项

- 重跑或覆盖 P2；
- 用旧 heldout 选候选；
- 提前下载/读取 SNGAR test labels；
- 用 IDSSE 结果反复调参；
- 把更多帧当作更多独立样本；
- 把 static role 当 dynamic tactical truth；
- 把 response 增大当 task improvement；
- 后验修改 task/metric/band/graph；
- 超过 autoresearch budget；
- 直接实现 AMR-Learned；
- 因 deadline 跳过 candidate lock；
- 写论文正文替代实验闭环。

最终返回格式：

```text
DONE
- 实际完成的 task 和 commit

SUPPORTED
- 只列证据支持的 claims

NOT SUPPORTED / NOT ESTABLISHED
- 保留负面和未决结果

PROTOCOL
- split、test firewall、candidate lock、hash

DATA
- 已获取、blocked、fallback

P4 DECISION
- GO 或 NO-GO，逐条对应 gate

EVIDENCE
- artifact/report paths
```
