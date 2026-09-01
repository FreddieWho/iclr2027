# 现有方案原位更新规范

## 任务对象

仓库：`FreddieWho/iclr2027`

你的职责不是新建另一套路线图，而是把本文件中的修正**原位合并**到当前 canonical plan、状态文件、机器配置和总控 Prompt 中，并保留 P0–P3 的真实历史。

## 最高原则

1. 不重写或删除 P2/P3 阴性结果；
2. 不新增 Phase 节点；
3. 当前 active phase 仍为 `P3_CAUSAL_MECHANISM`；
4. 新工作编号为 `P3-T5R` 系列 task，不是 P3.5/P4；
5. P4 AMR 继续 blocked，直到新 task-semantic gate 闭合；
6. 旧 heldout 标记为已暴露，只保留 exploratory audit；
7. 先更新方案和 protocol，再运行新实验；
8. 每次更新后执行全仓库 current-state consistency audit。

---

# 1. 当前必须保留的冻结事实

不得修改含义：

- P2 route：`mixed_or_graph_specific`；
- exact-spectrum 说明普通谱功率不足以解释全部 response；
- role-defined coalition 不是普遍同方向效应；
- P3 full geometry 回顾性 mean Spearman 约 0.8466，旧 frequency/residual baseline 约 0.1222；
- 同资产 response-blind prospective mean Spearman 约 0.8526，baseline 约 0.1294；
- diagonal 为主要贡献，off-diagonal 有架构/层依赖的增量；
- Phase-GAT pooling switch 改变 geometry 和 response；
- 旧 phase task 没有稳定 macro-F1 改善，context response 反而上升；
- synthetic support localization 仅略高于 uniform；
- 旧 P4 gate 未满足；
- AMR 未正式测试。

不得把上述最后三项改写成“趋势支持方法”。

---

# 2. 新增的科学判断

## 2.1 旧 T5 的阴性结果仍有效，但只针对旧任务定义

新增明确表述：

> `NOT_SUPPORTED_UNDER_OLD_CONFLATED_TASK_GATE`：在把 phase/deployment macro-F1 与全局平移低响应作为同一个 embedding 的联合目标时，简单 pooling/centering 候选没有形成稳定 Pareto 改善。

这不是对下列更一般假设的证伪：

> 分离 context 与 intrinsic structure 后，support-conditioned geometry 能否指导有用的 task-conditioned representation。

## 2.2 标记 heldout 暴露

旧 candidate search 在候选循环中计算/打印/保存了 heldout。虽然排序没有显式使用它，但该 heldout 已不再是未见确认集。

必须写入：

```text
legacy_heldout_status: EXPOSED_DURING_CANDIDATE_SEARCH
allowed_use: exploratory_audit_only
forbidden_use: final_confirmatory_claim_or_candidate_selection
```

## 2.3 授权有界 P3-T5R

建议 current state：

```text
active_phase: P3_CAUSAL_MECHANISM
current_checkpoint: P3_T5R_PROTOCOL_AND_DATA_EXPANSION_AUTHORIZED
current_route: support_conditioned_geometry_with_task_semantic_repair
p4_status: blocked_pending_p3_t5r_gate
```

P3-T5R 只解决：

- task semantics；
- independent-match sample size；
- heldout protocol；
- minimal fixed dual-channel test；
- bounded autoresearch；
- independent confirmation。

不允许借此重新扫描统一 notch 或扩大完整 causal matrix。

---

# 3. 文件级更新要求

## `STATUS.md`

替换“冻结 P3 并永久停止机制复杂度”为：

- 当前 P3 主要几何与 response-shaping 结论冻结；
- 允许一个有限的 T5R task-semantic repair；
- 旧 task repair 仍为 NOT SUPPORTED；
- P4 继续 blocked；
- 新数据和未见 split 见 data manifest v2；
- 当前下一动作：更新协议、申请/下载数据、构建 context/intrinsic tasks；
- 旧 heldout 已暴露。

## `DECISIONS.md`

追加至少以下决策：

### D-P3-011：旧 heldout 暴露降级

记录脚本行为、允许用途和禁止用途。

### D-P3-012：拆分任务语义

- context task 允许依赖 absolute deployment；
- intrinsic task 应对 global translation 稳健；
- 不能再以单一 global-response 指标约束两个任务。

### D-P3-013：数据扩展和冻结 split

- SNGAR train/valid 用于开发；test lock；
- IDSSE external confirm；
- SoccerTrack acquisition shift optional；
- SkillCorner historical exploratory。

### D-P3-014：P3-T5R 有界优化

- 最多两轮 autoresearch；
- 每轮最多六个候选；
- 一次只改一个主要轴；
- P4 不提前启动。

### D-P3-015：动态 support 状态

- static role universal effect 保持 NOT SUPPORTED；
- dynamic semantic support 是新的、尚未建立的 operationalization；
- support construction 必须 response-blind。

## `CLAIM_LEDGER.md`

保留旧 claim，新增：

| Claim ID | Claim | 初始状态 |
|---|---|---|
| P3-R1 | 旧 T5 task gate 混淆了 context signal 与 intrinsic nuisance | DESIGN_DIAGNOSIS_SUPPORTED |
| P3-R2 | 固定 context/mode 双通道能改善正确任务 Pareto | WORKING_HYPOTHESIS |
| P3-R3 | geometry-response 关系可推广到未见比赛 | NOT_ESTABLISHED |
| P3-R4 | 动态协同 support 相对 matched random 有稳定效应 | NOT_ESTABLISHED |
| P3-R5 | P3-T5R 为 AMR-Fixed 提供非后验目标 | BLOCKED_PENDING_T5R |
| DATA-C1 | 旧 heldout 可用于最终确认 | FORBIDDEN |

## `README.md`

在顶部 current status 中：

- 区分 original ambition、current supported results 和 next bounded repair；
- 把旧 P0 bootstrap 命令移到 historical/reproduction section；
- 当前入口改为 data access + protocol update；
- 明确 AMR 是 target method，不是当前获准施工；
- P5 calligraphy 标为 pending sports prediction lock。

## `MASTER_AGENT_PROMPT.md`

替换当前“只归档，不继续”的启动动作：

1. 读取本次 update spec；
2. 先完成 current-state 文档迁移；
3. 标记旧 heldout；
4. 获取/转换 SNGAR train+valid；
5. 构建 context/intrinsic tasks；
6. 运行固定 dual-channel sanity；
7. 仅在 sanity 支持后调度 bounded autoresearch；
8. candidate lock 后一次性 test/external；
9. 按 gate 决定 P4 或诊断论文。

## `docs/01_SCIENTIFIC_BLUEPRINT.md`

保留 H1–H9 历史表，增加：

- universal notch 与 complete organization blindness 为 `FALSIFIED_AS_STATED`；
- organization beyond power、local geometry、response shaping 为条件支持；
- task usefulness、dynamic semantic support、新比赛泛化为未建立；
- 正确中心问题：

```text
operator × frequency × support/relationship × task semantics
```

- 明确“敏感”不等于“理解”；非线性语义编码尚未证明。

## `docs/02_METHOD_SPEC_AMR.md`

保持 AMR 未启动。增加 gate 前最小模型：

```text
shared encoder E
z_ctx = Pool(E(X))
z_mode = Pool(E(Center(X)))
context head only reads z_ctx
intrinsic head only reads z_mode
```

可选 cross-readout 仅用于 leakage audit，不作为主模型。

写清：

- context task 不要求 global invariance；
- mode task 才要求 global nuisance robustness；
- T5R 成功后，AMR-Fixed 才将固定双通道路由扩展到 frequency × support routing；
- 不允许用 P2 response 后验学习 gate。

## `docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md`

在现有 P3 下增加 T5R，不改 Phase 编号：

- T5R-0 protocol repair；
- T5R-1 data conversion；
- T5R-2 tasks；
- T5R-3 fixed dual-channel；
- T5R-4 bounded autoresearch；
- T5R-5 candidate lock/test；
- T5R-6 external confirmation/go-no-go。

更新 Figure contract：

- Figure 3：conditional spectrum/local geometry，不再画 universal notch；
- Figure 4：layer/block anatomy；
- Figure 5：fixed dual-channel task semantics；
- Figure 6：independent-match task/robustness Pareto；
- calligraphy figure 只有在 sports prediction lock 后启用。

## `docs/04_DATA_AND_ENVIRONMENT_GUIDE.md`

合并 `01_DATASET_ACQUISITION_AND_SPLIT_PLAN.md` 的来源、命令、许可与 split；更新 `configs/data_manifest.yaml`；加入 gate access 和 fallback。

## `docs/05_AGENT_EXECUTION_MANUAL.md`

更新 worker：

- Data worker：SNGAR/IDSSE/SoccerTrack conversion；
- Task worker：context/intrinsic/dynamic support；
- Method worker：fixed dual-channel，禁止先做 AMR；
- Autoresearch worker：受 candidate budget 和 heldout firewall 约束；
- Repro worker：candidate lock、test firewall、hash、license。

## `configs/project.yaml`

- 不新增 schedule phase；
- P3 status 改为 task-semantic repair authorized；
- P4 status 仍 blocked；
- P5 status 改为 blocked/pending sports prediction lock；
- 添加硬截止日前的压缩日程，但不得把日期当科学 gate。

## `configs/experiment_matrix.yaml`

新增 `p3_task_semantic_repair`：

- datasets and split roles；
- tasks；
- baselines；
- dual-channel；
- metrics；
- autoresearch budget；
- candidate lock；
- final gate。

## `configs/data_manifest.yaml`

合并 `configs/dataset_acquisition_manifest_v2.yaml` 中的数据项和 access state。不要把 gated dataset 写成已下载。

## `PROJECT_PACKAGE_CONSOLIDATED.md`

顶部增加新的 synchronization addendum，声明其优先级高于历史正文；不要删除历史路线或 P2/P3 失败记录。

## `reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md`

追加 revision note：

- 原结果不变；
- 旧 T5 的结果边界是旧 conflated gate；
- T5R 是独立 protocol-corrective follow-up；
- 不用 T5R 重算 P2/T1/T2；
- 旧 heldout 暴露。

## 新建 `reports/P3_TASK_SEMANTIC_REPAIR_PLAN.md`

这是当前 P3 的执行报告，不是另一套科学蓝图。应记录：

- task definitions；
- data split；
- baseline lock；
- autoresearch budget；
- candidate lock；
- gate；
- current status and artifacts。

---

# 4. 新任务定义

## Context task

最少包括：

- phase/deployment classification；
- absolute team centroid/field zone prediction。

主表示：`z_ctx`。不以 global translation invariance 为优点。

## Intrinsic task

至少一个非 intervention-label 任务：

- intrinsic formation retrieval；或
- natural pair ranking。

主表示：`z_mode`。要求 global translation robustness。

## Dynamic support task

可并行：

- dynamic support type classification；
- natural support localization；
- dynamic semantic vs matched random response。

它不应阻塞 fixed dual-channel sanity。

---

# 5. 新 gate

进入 P4 必须同时满足：

1. 新比赛上 full geometry 仍优于 frequency/residual baseline；
2. `z_ctx` 的 context task 不劣于 raw single-channel baseline；
3. `z_mode` 的 intrinsic task 优于 raw baseline，并对 global translation 更稳健；
4. 改善在多数 seed 和多数比赛同方向；
5. candidate lock 前未读取 test；
6. SNGAR test 一次性确认保持方向；
7. 至少一个外部来源数据不出现方向性崩溃；
8. 不依赖静态 role oracle、容量增加或单 seed；
9. geometry predictive relation 未坍缩；
10. 主要 claim 与统计单位为 match。

未满足时：

- 不启动 AMR；
- 将论文冻结为 diagnostic/mechanistic version；
- 保留 negative task-repair boundary。

---

# 6. 文档迁移验收

必须生成 `reports/P3_T5R_DOCUMENT_MIGRATION.md`，逐文件列出：

- 原状态；
- 修改内容；
- 仍保留的历史；
- 冲突检查；
- commit/hash；
- 尚未运行的实验。

执行检查：

```bash
python -m compileall scripts tests
pytest -q
python scripts/audit_current_state_consistency.py
```

若没有 consistency audit 脚本，创建一个只检查 current-state keys、heldout status、P4 status、P5 status 和关键路径的轻量脚本。

## 禁止做法

- 全仓库无语义 search/replace；
- 删除旧阴性报告；
- 用 addendum 掩盖 source docs 不一致；
- 未运行实验就把 claim 标为 supported；
- 把数据申请状态写成 downloaded；
- 把 T5R 编成新 phase；
- 因 deadline 放宽 heldout firewall；
- 先实现 AMR 再反推 gate。
