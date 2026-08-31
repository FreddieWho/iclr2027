# P2 实现方案：组织效应是否超出普通频率偏置

日期：2026-08-31

> 当前 matching 优化、P2-C 接入和资源门的细化方案见
> `docs/08_PHASE2_OPTIMIZATION_AND_NEXT_TASKS.md`。本文件保留 P2 总体科学设计；细化方案不会覆盖已执行的 v1 smoke。
>
> 2026-08-31：上述实现已执行，结果与剩余边界见
> `docs/09_PHASE2_IMPLEMENTATION_RESULTS.md`。
>
> 2026-08-31：rigid v2 得到 `mixed_or_graph_specific` 后，fracture continuity
> 改为独立的全精度两臂补充诊断；当前实施方案见
> `docs/10_P2_FRACTURE_CONTINUITY_PLAN.md`。本文件第 4 节关于直接复用 P1
> fracture delta/response 的早期设想不再作为正式实施方案。

## 1. 阶段定位

P2 对应 `P2_NOVELTY_DISCRIMINATOR / Exploration Checkpoint 2`。它不训练新方法，也不扩大到书法或视觉模型；唯一核心问题是：

> 在支持大小、总能量、位移方向和图频率尽量相同后，角色定义的真实联盟是否仍与随机联盟产生不同的表示响应？

P2 继续采用探索性研究模式。控制定义、频段和匹配参数可以根据不含模型响应的几何诊断迭代，但每次实际版本、修改理由和结果都必须记录；不设置预注册或类似的事前约束。

## 2. 当前证据与必须补的缺口

P1 点集主线已经提供：

- 250 个样本、10 场 SkillCorner 比赛；
- 6000 条 `semantic_coalition` 干预，其中 5689 条有效；
- DeepSets-AE、GAT-AE、Phase-GAT，各 3 seeds；
- 固定的 baseline graph、模型 checkpoint、baseline embedding 和响应表。

P1 的 `random_coalition` 只匹配了支持节点数，没有严格匹配：

- Rayleigh quotient；
- 六频段能量分布；
- 支持连通性；
- 位移向量/方向分布；
- 成对边界有效性。

此外，P1 的 `semantic_coalition` 在角色组内给每个节点施加独立随机位移，它测量的是 role-defined structural fracture，不是“整条防线共同平移”。P2 必须把这两类 operator 分开，不能用同一个 semantic 名称混合解释。

因此 P1 的 common/semantic ERRR 和 notch 只能作为候选现象，不能证明“组织效应超过频率”。

## 3. 输入与复用原则

P2 直接读取而不改写：

- `artifacts/phase1/canonical_samples.npz`；
- `artifacts/phase1/canonical_samples.jsonl`；
- `artifacts/phase1/intervention_manifest.parquet`；
- `artifacts/phase1/point_mainline/model_manifest.json`；
- `artifacts/phase1/point_mainline/spectrum_response.parquet`；
- `artifacts/phase1/embeddings/*_baseline.npy`；
- `artifacts/phase1/models/*.pt`。

启动时校验 checkpoint SHA-256、样本 ID、split、节点顺序和 P1 artifact schema。P2 不重新选帧、不重新训练模型，也不改写 P1 历史语义锚；但 P2 主分析会为所有可用的 `(team_slot, role_group)` 重新生成定义更清楚的 rigid subgroup translation。

## 4. 对照设计

P2 主分析使用 role-defined known coalition，而不把静态角色元数据称为动态战术真值。对每个支持大小至少为 2 的 `(team_slot, role_group)`，定义刚性子群平移：

\[
\delta_i=
\begin{cases}
\varepsilon v/\sqrt{|S|}, & i\in S,\\
0, & i\notin S,
\end{cases}
\]

其中 `v` 使用四个固定方向 `+x/-x/+y/-y`。semantic 和它的全部 controls 共用相同方向；任何一侧越界时，整个 matched set 对该方向无效，禁止 clipping。

P1 的 independent fracture operator 保留为次要连续性分析：使用原 P1 semantic delta，并把完全相同的非零位移向量 multiset 重新分配到随机支持。它不能与 rigid translation 合并为一个效应。

### 4.1 共同的精确匹配条件

稀疏随机对照必须与 rigid semantic anchor 精确共享：

- `source_sample_id`、team、seed、epsilon；
- 支持节点数；
- 总 Frobenius 能量；
- operator 和固定方向；
- baseline graph 和 baseline spectral basis；
- pitch boundary 合法性。

对 P1 fracture 连续性分析，额外要求非零节点位移向量的 multiset 完全相同。

### 4.2 三类必需控制

1. `arbitrary_same_team`：同队、同支持大小，均匀选择随机支持并排除 semantic 原集合，衡量基础 size/team confounding。
2. `topology_frequency_matched`：同队、同支持大小，并匹配 semantic support 的连通分量数、内部边密度、cut weight、Rayleigh quotient 和六频段功率。semantic 支持若不连通，不强行改成连通控制。
3. `spectrum_exact_sign_randomized`：保留每个 eigenmode 的功率、总能量、Rayleigh quotient 和 DC 系数，仅对非 DC 谱系数做确定性随机符号翻转后逆变换。它允许支持变为稠密，是排除普通频率解释的关键对照。

前两类回答“同队同大小以及拓扑/频率匹配后，角色组织是否特殊”；第三类回答“每个模态功率完全相同，仅改变谱相位/空间组织时，响应是否改变”。三条证据缺一时，对应的替代解释必须保持未排除。

## 5. 匹配算法

### 5.1 候选生成

每队固定 10 个节点，支持大小通常为 3–5，因此可以枚举同队所有同大小支持，而不依赖不透明的随机候选池：

1. 枚举同队所有大小为 `k` 的支持集；
2. 排除 semantic 原集合，并记录与 semantic 的节点重叠数；
3. 施加与 semantic 相同的 rigid operator 和方向；fracture lane 则重排同一个位移向量 multiset；
4. 拒绝越界或支持重复的候选；
5. 计算完整二维谱功率、Rayleigh quotient、六频段功率、连通分量数、内部边密度、cut weight 和 boundary slack。

### 5.2 响应盲匹配

匹配器只能读取几何、图和干预字段，不能读取任何模型响应。`topology_frequency_matched` 首先要求连通分量数与 semantic 精确一致，再按以下距离选择最佳控制：

- Rayleigh quotient 差；
- 六频段能量比例的 L1 距离；
- induced-edge density、cut weight 和 baseline boundary slack 差。

主分析每个 anchor/control type 选择一个确定性的最佳控制；`arbitrary_same_team` 使用稳定 hash 从合格集合抽取，并可保留多个 control draws 做 Monte Carlo 敏感性。caliper 和 control draw 数写入 `configs/phase2.yaml`；可以根据 matching-balance smoke 调整，但不能根据 SCG 或模型响应调参。若没有合法匹配，保留 `UNMATCHED`，不伪造或降级为仅支持大小匹配。

### 5.3 图构造敏感性

最低使用两个性质不同的 probe graph：

- `G0`：P1 weighted symmetric kNN-4；
- `G1`：weighted Delaunay graph，使用与 G0 相同的距离权重核。

probe graph 只用于频率定义和匹配敏感性。GAT 推理仍使用其 P1 训练时的固定 kNN-4 adjacency，避免把“改变模型输入图”混入 P2。Delaunay 退化样本标记 `GRAPH_INVALID`，不能静默退回 kNN。若效应 graph-sensitive，再增加 kNN-3/kNN-5 或 team-aware graph 作为后续诊断。

## 6. 干预与匹配校验

工程 validity gate 必须检查：

- source artifact/checkpoint hash 一致；
- `matched_set_id` 和 intervention ID 唯一；
- 稀疏对照的 sample/team/support size/operator/direction 精确相同；
- topology control 的 component count 精确相同，density/cut/RQ/band-power 平衡可审计；
- fracture lane 的位移向量 multiset 精确相同；
- 能量误差不超过 `1e-7`；
- 所有有效坐标位于 pitch boundary；
- exact-spectrum control 的每模态功率、总能量、RQ 和 DC 系数在数值容差内一致；
- 每种 control、epsilon、match 的匹配率和失配原因完整记录；
- 匹配后的 Rayleigh quotient、band-energy 和 boundary-slack 平衡可审计。

若 matching gate 失败，停止模型推理并将 P2 保持为 `INCOMPLETE`。

四个 P1 energies 均保留。`0.25/0.5/1.0` 作为较线性的主区域；`2.0` 作为高能量边界压力测试，单独报告较高的 unmatched/invalid 风险，不静默删除。

## 7. 模型响应计算

P2 复用 P1 的 9 个点集模型：

- DeepSets-AE：3 seeds；
- GAT-AE：3 seeds；
- Phase-GAT：3 seeds。

视觉网络不进入 P2 主线。Phase-GAT 的 P1 任务性能较弱，因此作为辅助任务训练族，不作为支持结论的唯一模型族。

执行方式：

- baseline embedding 直接读取 P1；
- rigid semantic anchors 和全部新 controls 重新计算 embedding；
- P1 fracture continuity lane 可直接复用旧 semantic response，但其新 controls 仍需推理；
- checkpoint hash 与 model manifest 必须一致；
- GAT 使用固定 baseline adjacency，不在干预后重构图；
- 分块读取 control manifest、分块推理、分块写 Parquet，避免在内存中保留全部 delta 和 embedding；
- 主指标继续使用 baseline/intervention embedding 的 normalized cosine distance。

## 8. 统计分析

### 8.1 主 estimand

对每个 matched set：

\[
\Delta_{SCG}=d_z(semantic)-d_z(control).
\]

分别报告 arbitrary、topology-frequency 和 exact-spectrum control，不先混成一个平均数；rigid translation 与 P1 fracture 也分开报告。

### 8.2 聚合与不确定性

- 先在 sample/coalition/seed/epsilon 内计算 paired difference；
- 再聚合为 match-level effect；
- bootstrap 单位是 match，不是 frame 或 intervention row；
- architecture 汇总先在 seed 内计算，再报告 seed spread；seed 不当作新的独立比赛；
- 报告均值、中位数、effect size、95% group-bootstrap interval、逐比赛方向和 leave-one-match-out 敏感性；
- train/dev/heldout 分开描述；只有两个 heldout matches 时不夸大 heldout 精度。

### 8.3 次要分析

- 当前 P1 ERRR 作为背景指标保留，不作为 C2 的主证据；
- 用 isotonic/平滑的单调谱响应作为低通基线，检查 notch 是否仍有非单调残差；
- 用 response 对 Rayleigh quotient、band energy、support size 和 organization indicator 的响应盲协变量模型做敏感性分析；
- 检查结论是否依赖 epsilon、graph k 或单个 match。

P2 是探索性检查点，不设置单一机械显著性阈值。科学判断综合 effect size、区间、跨比赛方向、跨模型族一致性和图构造敏感性。

## 9. 代码与配置边界

计划新增：

- `configs/phase2.yaml`：实际 control draws、匹配距离、caliper、graphs、energies、模型列表和 chunk size；
- `scripts/p2_matched_controls.py`：role support、probe graph、三类 control 和 balance gate；
- `scripts/p2_statistics.py`：matched effects、match bootstrap、notch null 和结果表；
- `scripts/run_phase2_pipeline.py`：source hash、stage orchestration、冻结 checkpoint 推理、shard receipt 和 resume；
- `tests/test_phase2_pipeline.py`：确定性、匹配、频谱、边界、pairing、hash 和 group-bootstrap 测试；
- `reports/EXPLORATION_CHECKPOINT_2.md`：C2 报告。

不重构到新的大型 `src/` 包，不修改 P1 历史 artifact，不把 P2 逻辑塞回 P1 脚本。

## 10. 计划产物

```text
artifacts/phase2/
  execution_contract.json
  source_provenance.json
  graph_manifest.parquet
  matched_set_manifest.parquet
  matching_balance.parquet
  intervention_validation.json
  model_manifest.json
  response.parquet
  match_level_effects.parquet
  scg_summary.parquet
  graph_sensitivity.parquet
  pipeline_summary.json
  logs/run.jsonl
  receipts/
  figures/
    matching_balance.png
    scg_by_model.png
    match_effect_forest.png
    graph_sensitivity.png
    notch_residual.png
```

大型 embedding 不重复保存；如调试需要，只保留分块临时文件并在正式 response 完整写盘后由明确流程清理。

`matched_set_manifest.parquet` 至少保存 sample/match/split、team/role/support、`probe_graph_id`、固定的 `encoder_graph_id`、semantic/control 身份、operator、direction、target/actual energy、完整谱功率、band power、RQ、component count、density、cut weight、validity/failure reason，以及相互独立的 `control_seed` 和 `model_seed`。`matched_set_id` 用于 semantic-control pairing，不能复用 P1 仅表示 baseline-intervention 关系的 `pair_id`。

正式输出按 run-specific 目录和 model/graph shard 写 completion receipt；只有 source/config hash 完全一致且状态为 `completed` 的 shard 才可恢复，绝不覆盖 P1 artifact。

## 11. 实施顺序

### P2-A：匹配器与单元测试

- 实现 control generation 和 matching；
- synthetic graph 上验证 exact constraints、连通性、band-energy preservation 和确定性；
- 不加载任何模型。

### P2-B：matching-only smoke

- 1 场比赛、少量 anchors、一个 energy；
- 输出候选池、匹配率和平衡图；
- 只根据几何平衡调整配置版本。

### P2-C：单模型响应 smoke

- 复用一个 DeepSets checkpoint；
- 验证 baseline 复用、分块推理、pairing、有限数值和 response schema；
- 在 fracture continuity lane 对同一个 P1 semantic anchor 检查复算响应与 P1 记录一致。

### P2-D：正式点集运行

- 250 samples、10 matches、4 energies、2 probe graphs、9 models；
- 不重新训练；
- 运行前根据 smoke 的 wall time/RAM 再判断本地 CPU 是否足够。只有 GPU 能显著缩短已确认的推理瓶颈时才提出租用。

### P2-E：C2 汇总

- 生成 match-level paired effects、bootstrap、graph/model/energy sensitivity；
- 写 `EXPLORATION_CHECKPOINT_2.md`；
- 更新 QA、claim ledger 和 artifact checksums。

## 12. C2 结束条件与结果分流

只有以下工程条件全部满足，P2 才可标记 `completed`：

- source provenance 和 hash 校验通过；
- matching/intervention validity gate 通过；
- arbitrary、topology-frequency、exact-spectrum 三类 controls 均有明确的 matched/unmatched 结果；
- 9 个 P1 点集模型均完成或在主结论中醒目标记具体 `NOT_RUN`；
- response rows 完整、有限且一一配对；
- match-level 统计和图构造敏感性完成；
- C2 报告明确区分证据、替代解释和未完成项。

科学分流：

1. `organization_signal_survives`：topology-frequency 与 exact-spectrum contrasts 在至少两个主要模型族、多个比赛和两个 probe graphs 下给出一致且有意义的组织残差，进入 P3 因果机制。
2. `frequency_bias_only`：控制频率后 SCG 接近零，项目 pivot 为关系频谱偏置，不再声称组织盲区。
3. `mixed_or_graph_specific`：效应仅出现在单一模型/图/energy，先做最小诊断，不直接进入大规模 AMR。
4. `matching_inadequate`：匹配率或平衡不足，保持 P2 未完成，改进控制而不是解释响应。

这些是探索性解释标签，不是预注册阈值。

## 13. 明确后置

P2 不包含：

- 新视觉模型或 minimap 计算；
- 书法、篮球或自然图像复现；
- AMR/CAP 训练；
- 大模型扫描；
- 新外部数据下载；
- 生信数据、索引或工作流。

它们分别属于 P3–P6 或独立后续决策。
