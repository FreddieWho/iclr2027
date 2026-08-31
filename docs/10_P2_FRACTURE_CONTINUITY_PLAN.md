# P2 fracture continuity 实现方案

日期：2026-08-31  
所属阶段：`P2_NOVELTY_DISCRIMINATOR`  
任务性质：rigid 正式分支完成后的补充诊断  
当前状态：`P2_FRACTURE_C2_COMPLETE`

## 1. 为什么仍属于 P2

P2 rigid v2 已完成，但结论为 `mixed_or_graph_specific`。下一步仍需回答：在频率和拓扑尽量相近时，模型是否对“哪些端点共同发生独立变化”敏感。这仍是排除普通频率偏置的 novelty discriminator，不是 P3 的训练机制实验。

本任务完成后才决定是否进入 `P3_CAUSAL_MECHANISM`。不启动 AMR、书法复现、视觉网络或新模型扫描。

## 2. 科学问题与最小 estimand

对同一足球样本、同一队、同一角色支持集和同一组二维位移向量，比较：

1. `semantic_fracture_anchor`：位移向量作用于原角色支持集；
2. `topology_frequency_same_vector_reassignment`：完全相同的非零位移向量 multiset 被一一重排到同队另一组端点。

主 estimand 为：

\[
\Delta_{fracture}=\frac{d_z(anchor)-d_z(reassigned)}{\varepsilon}.
\]

正值、负值和接近零都保留解释价值，不设置机械显著性门槛。

## 3. 为什么不用旧 P1 anchor 直接闭合

P1 的 `semantic_coalition` 是有效线索，但不作为本次正式 anchor 数值来源：

- P1 每档能量独立抽取位移方向，不能解释为同一扰动路径上的连续能量变化；
- P1 只选择每队第一个人数足够的角色组，实际 6,000 条记录全部为 defender；
- P1 保存的 `delta` 经过十位小数 JSON 表示，且旧谱字段只使用了 x 坐标系数；
- 当前 rigid 结果已经显示 defender、midfielder、forward 方向不同，只重复 defender 不足以解释异质性。

因此本任务重新生成全精度 anchor，但沿用 P1 的 node-independent 二维 fracture 语义。每个 `(sample, team, role, draw)` 先生成一个单位能量基础向量场，再按 `0.25/0.5` 缩放，使两档能量位于同一条扰动路径。

## 4. 正式范围

- 250 个 canonical samples、10 场比赛；
- 1,425 个支持大小至少为 2 的角色组：defender 500、midfielder 450、forward 475；
- 支持大小分布：2 节点 425、3 节点 525、4 节点 375、5 节点 100；
- 3 个独立 fracture draws；
- 两档低能量：`0.25/0.5`；
- 两个 probe graph：`knn4/delaunay`；
- 9 个冻结点集模型：DeepSets-AE、GAT-AE、Phase-GAT，各 3 个模型 seed；
- 不重新训练，不运行视觉模型；GAT 推理继续使用 P1 固定 kNN-4 adjacency。

预计正式规模：

- 计划 matched sets：`1425 × 3 × 2 × 2 = 17,100`；
- 两臂 slots：34,200；
- 全部配对成功时，九模型 response 上限约 307,800 行；
- 无法匹配或越界的 set 保留明确 failure code，不补造控制。

## 5. 控制生成与响应盲匹配

### 5.1 anchor

- 新 operator ID：`p2_fracture_endpoint_reallocation_v1`；
- 位移只落在一个 `(team_slot, role_group)` 支持集；
- 支持内各节点获得独立二维位移，再整体归一到单位能量；
- 同一个基础向量场只按 epsilon 缩放，不重新抽方向；
- 随机数由配置中的 seed root、sample、coalition 和 draw index 稳定派生。

### 5.2 reassignment control

- target support 与 source support 同队、同大小且完全不重叠；
- 枚举全部合法 target supports，并枚举非零向量到 target endpoints 的全部双射；
- 支持大小 2–5 时，每个 anchor 的双射候选数量可控；
- 每个候选复制 anchor 向量，不重新归一化、不裁剪、不反射；
- 对每个 probe graph 要求 induced component count 与 anchor 相同；
- 在所有边界合法候选中，响应盲地选择 topology/frequency 最接近者；
- 使用完整二维谱功率，同时记录 RQ、六频段 L1、density、cut weight、boundary slack 和 runner-up gap。

不设置频率残差的预注册式通过阈值。先选择几何上最近的合法控制，再完整报告残差与效应的关系；若残差仍足以解释结果，科学状态为 `matching_inadequate`。

### 5.3 不进入主臂的候选

- `arbitrary_same_team` 只作为候选池覆盖描述，不额外做模型响应；它不能排除频率解释。
- `within_support_permutation` 不进入首轮；它回答 vector-node assignment，而不是 endpoint-support relocation。只有首轮结果显示 assignment fragility 时才作为独立后续诊断。
- 不新增 exact-spectrum arm；该问题已由 rigid v2 回答，而且 exact-spectrum 不保持稀疏向量 multiset。

## 6. 数据与证据链

旧 `p2_rigid_formal_v2` 的计算代码、配置和科学结果不改写；持续更新的全局 QA
只同步其引用哈希。fracture 使用全新的运行目录与证据链。新增：

- `configs/phase2_fracture_continuity_v1.yaml`；
- `scripts/p2_fracture_controls.py`；
- `scripts/run_phase2_fracture_formal.py`；
- `scripts/p2_fracture_statistics.py`；
- `tests/test_p2_fracture_continuity.py`；
- `tests/test_phase2_fracture_formal.py`；
- `reports/EXPLORATION_CHECKPOINT_2_FRACTURE.md`；
- `artifacts/phase2/p2_fracture_continuity_v1/`。

输入配置锁定 canonical samples、role metadata、九个 checkpoint、baseline embedding、模型 manifest、源码和环境版本。全精度 `delta` 使用可无损往返的 float64 列表或二进制列，不再用十位小数 JSON 作为精确性依据。

每个 stage 按 match 或 `model_id × match_id` 分片，原子写入 Parquet、receipt 和 manifest。已完成 stage 在 `--resume` 下只验证、不改写；顶层 summary 必须单调保持最高完成状态。失败尝试写入独立 `attempts/`。

全局 `QA.md` 是持续更新文档；正式闭合时保存独立 QA snapshot，避免以后追加问答破坏已冻结 run 的 checksum。

## 7. 实施顺序

### P2-FC-A：operator 与单元测试

- 实现全精度 anchor、向量 multiset ID、端点重排和穷举双射；
- 测试同队、同大小、全不重叠、能量相同、multiset 精确相同、确定性和边界失败；
- 测试 x/y 两个坐标共同进入谱功率；
- 不加载模型。

结束状态：`P2_FC_OPERATOR_READY` 或明确 `BLOCKED_*`。

### P2-FC-B：matching-only smoke

- 1 场比赛、低能量、1 个 draw、两个图；
- 不读取任何 response 字段；
- 检查候选数量、边界覆盖、component match、频率/拓扑残差和角色差异；
- 只允许根据响应盲几何诊断调整匹配 score，并保留配置版本。

结束状态：`P2_FC_MATCHER_SELECTED`、`MATCHING_INADEQUATE` 或明确 `BLOCKED_*`。

### P2-FC-C：单模型 response smoke

- 使用 `deepsets_ae_seed11`；
- 验证 250 个 baseline parity 或在 canary 后进入完整 parity；
- 检查固定样本顺序、delta round-trip、配对、有限响应、receipt 和 resume immutability；
- smoke 只证明接线，不产生科学结论。

结束状态：`WIRING_ONLY_NOT_SCIENTIFIC` 或明确 `BLOCKED_*`。

### P2-FC-D：正式全量运行

- 10 个 matching shards；
- 90 个 response shards；
- 九模型全部完成，任何模型缺失都在主状态中显式标记，不能用部分模型替代；
- 不在全量内存中保存候选 tensor，只保留 selected、runner-up 和聚合失败诊断。

### P2-FC-E：统计与报告

- 先在 anchor 内配对，再聚合到 match；
- model seed 先在 architecture 内汇总，不能当作独立比赛；
- 报告 match bootstrap、逐比赛方向、leave-one-match-out、graph/energy/role 分层；
- 检查效应是否随 frequency/topology residual 改变；
- train/dev/heldout 分开描述，两个 heldout match 不做过度推广。

## 8. 测试与验收

必须覆盖：

- multiset hash 与 Parquet round-trip；
- support disjointness、team、size、能量、边界和 component count；
- exhaustive bijection 数量与稳定 tie-break；
- matcher 无权读取模型响应；
- intervention/matched-set ID 唯一；
- checkpoint、输入、源码和环境 hash 漂移 fail-closed；
- baseline parity、固定 GAT adjacency 和 response pairing；
- match-level 聚合与 model-seed 非独立性；
- receipt/output tamper detection；
- 完成后重复 `--resume` 全树 hash 不变，包含顶层 summary。

实现完成后由独立 tester 验证，随后由 reviewer 检查 provenance、伪重复、数据泄漏和 rigid/P1 结果污染。

## 9. 完成 checkpoint 与结果分流

工程 checkpoint：`P2_FRACTURE_C2_COMPLETE`。

只有 operator、matching、response、statistics、report 五段证据链全部闭合，九模型正式结果完成，且所有 planned sets 均有有效配对或明确失败原因，才可进入该状态。smoke、部分模型或缩小数据不能代替。

科学分流不使用固定阈值：

- `organization_signal_survives`：效应不局限于单场、单图、单能量或单架构，LOMO 后仍可解释，且不随匹配残差消失；进入 **P3 的最小 2×2 因果机制矩阵**。
- `frequency_bias_only`：匹配越好，fracture gap 越接近零；停止该组织机制主张，不进入对应 P3。
- `mixed_or_graph_specific`：方向主要随 graph、role 或 architecture 改变；仍停留 **P2**，只做一个有界异质性诊断。
- `matching_inadequate`：覆盖或平衡不足、结果跟随残差；P2 fracture 保持未完成，不用普通随机臂或放宽规则替代。

## 10. 实际执行结果

本方案已按独立 operator 完成。正式 run 使用 250 个样本、10 场比赛、17,100 个 planned sets、13,477 个 complete sets、34,200 个 arm slots、9 个冻结点模型和 90 个 response shards；有效两臂 pair effect 共 121,293 条。控制 residual 被单独保留，frequency residual 平均约 0.411、topology residual 平均约 1.399；residual-effect correlation 未显示明显单调关系，但控制并非 exact-spectrum replica。

实际科学分流为 `mixed_or_graph_specific`：中场角色在 DeepSets-AE/GAT-small-AE 中多为正，防守角色多为负，前锋和 Phase-GAT 更依赖模型、图和条件。该结果不能支持统一的 endpoint-organization 机制，因此不进入 P3 causal-mechanism matrix；完整结果见 `reports/EXPLORATION_CHECKPOINT_2_FRACTURE.md`。

统计实现曾发现并修正 residual 读取错误；旧统计结果保留在 run 目录的 `statistics_pre_residual_fix/` 作为审计副本，不作为结论。

## 11. 资源预算

该任务与主线直接相关，但不需要训练。预计约 17,100 个 matched sets、最多约 307,800 条模型响应，明显小于 rigid v2 的 1,184,742 条响应。

- 推荐：本地 CPU 或 `8c16g`；`16c32g` 可缩短 matching；
- 内存设计目标：低于 4 GB，按 match 流式处理；
- 磁盘新增量：预计低于 2 GB，不保存全候选 tensor；
- 初步总耗时：`16c32g` 约 30–90 分钟，`8c16g` 约 1–3 小时；以 P2-FC-B smoke 实测为准；
- GPU：预计无明显收益，smoke 证明模型推理成为主要瓶颈前不租用。

以上时间是工程估算，不是运行承诺。正式计算前必须重新报告实测 smoke、预计剩余时间和资源选择。

## 12. P2-H1 后续诊断

2026-09-01 已基于冻结的 fracture statistics 完成有界异质性诊断，状态为
`P2_H1_HETEROGENEITY_DIAGNOSIS_COMPLETE`。该诊断只重用已有的
match-level、model-match、LOMO 和 pair-effect 表，新增架构/图/角色/能量因素对比、
频率与拓扑 residual 敏感性及简单 residual adjustment；没有重新运行模型。

H1 仍保留 `mixed_or_graph_specific` 路由：部分条件具有跨比赛方向稳定性，但 seed、
架构、图和角色并不完全一致，且个别条件与 topology residual 有中等相关。因此不直接
进入 P3 causal-mechanism matrix；后续如进入 P3，应先定义一个由 H1 结果支持的最小机制，
而不是扩大模型或数据范围。详细结果见
`reports/P2_HETEROGENEITY_DIAGNOSIS.md`。
