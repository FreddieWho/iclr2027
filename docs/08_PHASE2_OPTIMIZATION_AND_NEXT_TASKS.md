# P2 匹配优化与下一步实施计划

日期：2026-08-31  
当前状态（历史实施前记录）：matcher=`MATCHER_VERSION_SELECTED`，P2-C=`WIRING_ONLY_NOT_SCIENTIFIC`，当时正式 P2=`NOT_RUN`
适用范围：P2 matching revision、P2-C 单模型 wiring smoke、P2-D 运行前资源门

> 2026-08-31 执行结果见 `docs/09_PHASE2_IMPLEMENTATION_RESULTS.md`。本文件保留实施前的设计依据与验收边界。
>
> 2026-09-01：fracture continuity 已闭合，随后完成 P2-H1 有界异质性诊断；当前状态见
> `reports/EXPLORATION_CHECKPOINT_2_FRACTURE.md` 和
> `reports/P2_HETEROGENEITY_DIAGNOSIS.md`。

## 1. 目标与边界

本轮优化解决两个独立问题：

1. topology-frequency 控制的共同支持不足；
2. 固定方向刚性平移在球场边界附近不可行。

优化不能读取模型 response、embedding distance、SCG 或任何下游效应。它是探索性 matcher 改进，不是预注册，也不设置机械通过阈值。每个版本保留配置、输入 hash 和选择理由，允许后续继续生成新的探索版本，但不同版本不能混写为同一次运行。

本计划不包含视觉网络、不重新训练 P1 模型、不租用 GPU、不访问或更新任何生信索引。

```text
P1 canonical samples
        │
        ▼
候选支持与边界诊断 ──► 覆盖率—频率平衡曲线
        │                         │
        └──────── matcher版本选择 ┘
                                  │
                                  ▼
                      P2-C DeepSets wiring smoke
                                  │
                                  ▼
                       P2-D 运行前资源与科学门
```

## 2. 已知阻塞与设计决定

### 2.1 主阻塞：topology-frequency 共同支持

首轮 576 个 set 中有 433 个 topology-frequency 控制无候选。当前 reference caliper 为：

- normalized RQ 差不超过 `0.05`；
- 六频段功率比例 L1 不超过 `0.20`；
- semantic/control 的连通分量数一致；
- 同队、同支持大小，语义节点重叠不超过 0.5。

每场比赛取 1 帧的 response-blind 检查显示，保持重叠上限 0.5 时，band L1 从 `0.20` 试探到 `0.30` 可将 topology 候选覆盖从 39/114 提高到 87/114，但残余 band 差异也随之增大。因此不能直接把 `0.30` 写成最终答案，必须先生成完整覆盖—平衡曲线。

主方案继续保持重叠上限 0.5。允许重叠到 1.0 只作为敏感性诊断，因为它可能让控制支持仍包含大部分原角色组。

### 2.2 次阻塞：边界可行性

`epsilon=2.0` 的 anchor 有效率只有 19.4%，无法通过调整频率 caliper 修复。处理原则：

- `0.25/0.5` 作为 P2-C wiring smoke 的低能量输入；
- `1.0` 保留为后续中等能量分析；
- `2.0` 保留为压力测试并完整报告 invalid，不进入低能量完整性叙述；
- 不使用 clip、reflect、wrap 或事后缩放，因为这些操作会改变刚性平移和频谱语义。

exact-spectrum 控制允许最多 32 次确定性谱符号重采样。每次都保持 DC 规则和逐模态功率；找到首个边界合法结果后停止，并记录 seed、attempt 和失败历史。32 是计算安全上限，不是科学阈值；耗尽后保持 `UNMATCHED_BOUNDARY`。

### 2.3 当前样本选择问题

首轮 `--max-samples 3` 读取的是同一场比赛开头的连续样本。下一轮改为显式分层选择：每场比赛按时间位置确定性抽取 2 帧，共 20 个样本。样本列表写入执行契约，不能用文件顺序隐式决定。

## 3. 实现包 A：response-blind matching revision

### A1. 配置版本

新增 `configs/phase2_matching_v2.yaml`，保留 `configs/phase2.yaml` 作为已执行的 v1 输入。v2 至少包含：

- source artifact 路径和 SHA256；
- 分层 sample selector；
- energies、directions、probe graphs；
- overlap 规则和 component-count 规则；
- RQ/band reference scales；
- exact-spectrum 最大重采样次数；
- candidate reuse policy；
- chunk size 和输出目录。

reference scales 只用于构造曲线横轴，不自动构成通过线。

### A2. 候选诊断数据模型

扩展 `scripts/p2_matched_controls.py`，增加：

- `boundary_slack`：干预后到最近坐标边界的距离；
- `candidate_id`：sample、graph、anchor support、candidate support 的稳定 hash；
- `semantic_overlap_count/fraction`；
- component、density、cut、RQ 和 band 差异；
- `frequency_ratio = max(rq_diff / 0.05, band_l1 / 0.20)`；
- 互斥 primary failure 和可并存 secondary flags；
- exact-spectrum bounded retry 与 attempt receipt。

候选特征按 `sample × graph × coalition × candidate_support` 计算一次。能量和方向只影响边界可行性，不能重复计算不变的谱特征。

候选筛选顺序：

1. 同队、同大小、非原集合；
2. overlap 不超过 0.5；
3. component count 一致；
4. 当前 energy/direction 下边界合法；
5. 按 frequency ratio、density/cut 次级距离、稳定 candidate ID 排序。

继续允许同一 candidate 被不同 anchor 使用，以保持 v1 的最小变更；必须记录 reuse count。无复用的全局 assignment 留作后续敏感性，不与本轮 caliper 优化同时改变。

### A3. 独立诊断器

新增 `scripts/p2_matching_diagnostics.py`：

- 只接受 canonical sample、graph 和 candidate 字段；
- 对 `model_id`、`embedding`、`response`、`SCG` 等列 fail-closed；
- 从所有候选的 frequency ratio 生成嵌套覆盖曲线；
- 按 match、graph、role、support size、energy 和 direction 输出覆盖与残余平衡；
- 输出 complete/incomplete 的边界余量和分布差异；
- 不自动选定最终 matcher。

最终 matcher 版本由覆盖、RQ/band 残差、semantic overlap、density/cut 和跨比赛分布共同决定，并在决策记录中写明理由。若没有可解释的折中，状态保持 `INCOMPLETE_COMMON_SUPPORT`，不得强行推进科学解释。

### A4. 管线与产物

修改 `scripts/run_phase2_pipeline.py`：

- 新增显式 sample manifest 或 `--samples-per-match`；
- 先写 graph/candidate shard，再生成匹配结果；
- exact-spectrum 使用 bounded retry；
- 区分 `VALID`、`UNMATCHED_FREQUENCY`、`UNMATCHED_TOPOLOGY`、`UNMATCHED_BOUNDARY` 和 `GRAPH_INVALID`；
- 不覆盖 v1 smoke。

计划输出：

```text
artifacts/phase2/matching_v2_stratified_smoke/
  execution_contract.json
  sample_manifest.parquet
  graph_manifest.json
  candidate_geometry.parquet
  candidate_feasibility.parquet
  matched_set_manifest.parquet
  matched_set_summary.parquet
  coverage_balance_curve.parquet
  balance_by_stratum.parquet
  exact_spectrum_attempts.parquet
  matching_validation.json
  pipeline_summary.json
```

若 candidate feasibility 体积增长，按 sample shard 流式写入，不在内存保留全量 Python dict。

## 4. 实现包 B：P2-C 单模型响应 smoke

### B1. 模型恢复方式

P1 的模型类由 `scripts/run_phase1_pipeline.py::build_torch_models()` 构造，checkpoint 是 `state_dict`。P2 不复制一份 DeepSets 架构，也不修改历史 checkpoint。

新增 `scripts/p2_point_model_adapter.py`：

- 通过 P1 构造函数实例化模型；
- 校验 P1 脚本 hash、model manifest 和 checkpoint SHA256；
- 首个模型固定为 `deepsets_ae_seed11`；
- 加载 `artifacts/phase1/models/deepsets_ae_seed11.pt`；
- 使用 `torch.load(..., weights_only=True, map_location="cpu")`；
- 提供统一 `encode(positions, team_slots, adjacency=None)` 接口；
- 明确 `eval()`、`inference_mode()`、CPU 和批大小。

本轮不把模型类迁出 P1，也不修改 P1 历史脚本。直接调用 P1 的唯一模型工厂并记录其 source hash，是 P2-C 的最小风险路径；若未来要抽取共享模块，应作为独立等价性重构处理，不能和 matcher 优化混在同一次变更中。

### B2. baseline 复用门

P1 已保存 250×128 的 DeepSets baseline embedding。P2-C 读取：

- `artifacts/phase1/point_mainline/embedding_manifest.parquet`；
- `artifacts/phase1/embeddings/deepsets_ae_seed11_baseline.npy`。

复用前必须检查 sample ID、row index、embedding dimension、preprocess、pooling 和 checkpoint hash。再对 smoke 样本重新计算 baseline，并与存量 baseline 比较。若不一致，状态为 `BLOCKED_BASELINE_PARITY`，不能静默改成全量重算。

现有 embedding manifest 没有 baseline `.npy` 自身的 hash，因此 P2-C 的 source provenance 必须补算并记录该文件 SHA256。point-mainline manifest 是从完整 P1 确定性筛选得到的视图，实际路径仍指向 `artifacts/phase1/models/` 和 `artifacts/phase1/embeddings/`。

### B3. response smoke

新增 `scripts/run_phase2_response_smoke.py`：

- 只读取已选定且带配置 hash 的 matching manifest；
- 仅选择 `epsilon=0.25/0.5` 的完整四臂 set；
- 一个 set 的四臂始终一起进入或一起排除；
- 分块计算 intervention embedding；
- response 继续使用 P1 的 normalized cosine distance；
- 不训练、不优化模型、不输出 SCG 科学判断；
- 状态固定为 `WIRING_ONLY_NOT_SCIENTIFIC`，即使所有工程测试通过也不写 P2 完成。

计划输出：

```text
artifacts/phase2/p2c_deepsets_seed11_smoke/
  execution_contract.json
  source_provenance.json
  model_receipt.json
  baseline_parity.json
  response.parquet
  response_validation.json
  runtime.json
  pipeline_summary.json
```

## 5. 测试计划

本次属于 behavioral/cross-module change，按定向 TDD 实施。

### 5.1 Matching 单元测试

- stratified selector 覆盖 10 个 match 且确定性；
- candidate ID、排序和输出 hash 可重复；
- overlap、team、support size 和 component invariants；
- coverage curve 随 ratio 单调不减；
- diagnostic schema 拒绝任何 response/model 列；
- boundary slack 与直接坐标检查一致；
- exact-spectrum 重采样保持每个 mode power、总能量、RQ 和 DC 规则；
- 32 次耗尽后显式失败；
- primary failure 互斥、secondary flags 可共存；
- incomplete set 不得被标记 PASS。

### 5.2 P2-C 集成测试

- checkpoint hash 不符时 fail-closed；
- P1 baseline 行映射和重新计算结果一致；
- baseline `.npy` SHA256 写入 source provenance；
- DeepSets 输出形状为 `(batch, 128)` 且有限；
- 同一输入重复推理完全确定；
- 只消费完整四臂 set；
- response pairing 唯一、有限且基线索引正确；
- summary 始终为 `WIRING_ONLY_NOT_SCIENTIFIC`；
- 不写入 P1 artifact。

为 P2-D 预留的 GAT 测试还要确认：无论 perturbation 后坐标如何变化，encoder 输入始终使用 P1 baseline weighted kNN-4 adjacency，不能重建图。

测试文件：扩展 `tests/test_phase2_pipeline.py`，新增 `tests/test_phase2_response_smoke.py`。完成后运行定向测试，再运行当前 `tests/` 全集。

## 6. 执行顺序与 checkpoint

### Task 1：候选诊断与 bounded retry

先写测试，再实现 candidate schema、boundary slack、失败分类和 exact-spectrum retry。

结束状态：所有匹配不变量通过；v1 结果未覆盖；尚无模型计算。

### Task 2：10 场比赛分层 matching smoke

每场比赛取 2 帧，运行两种 probe graph、四个方向和全部能量，生成覆盖—平衡曲线。只查看 response-blind 字段。

结束状态二选一：

- `MATCHER_VERSION_SELECTED`：存在可解释的覆盖—平衡折中，并记录 matcher 配置/hash；
- `INCOMPLETE_COMMON_SUPPORT`：只能通过过度放宽频率或语义重叠获得覆盖，继续改设计而不解释模型响应。

这里没有预设百分比门槛；决定依据和不确定性必须写入报告。

### Task 3：P2-C DeepSets wiring smoke

只有 Task 2 产生明确的 matcher 版本后才运行。使用本地 CPU 和低能量完整集，完成 baseline parity、embedding 和 response schema 检查。

结束状态：`WIRING_ONLY_NOT_SCIENTIFIC` 或明确的 `BLOCKED_*`；不能产生 C2 结论。

### Task 4：P2-D 资源门

根据 P2-C 的实际吞吐、RAM 和输出大小外推 250 样本×9模型。外推和一个有界 benchmark 均记录后，再判断 CPU 或 GPU。未经新的相关性/成本说明，不租用远程实例。

## 7. 资源估计

- Task 1/2：本地 CPU，建议 4–8 cores、低于 4 GB RAM；20 样本 matching smoke 预计分钟级。
- Task 3：本地 CPU 足够，20 节点 DeepSets、单 checkpoint、低能量 smoke 预计低于数分钟，RAM 低于 2 GB。
- Task 4：暂不估计 GPU 租用时长；必须使用 P2-C 实测数据。

这些是工程估计，不是已测性能承诺。

## 8. 主要风险与处理

- **放宽 caliper 改变“同频率”含义**：保留完整曲线和所有残差，不按覆盖率单独选择。
- **高 overlap 污染控制语义**：主方案保持 0.5，上限放宽只做敏感性。
- **边界筛选改变研究对象**：按 match/role/direction 报告 complete 与 incomplete 分布，保留压力测试能量。
- **P1 模型定义漂移**：记录 P1 脚本 hash、checkpoint hash，并做 baseline parity。
- **同一样本内伪重复**：P2-C 只做 wiring；正式统计仍先聚合到 match。
- **模型结果反向影响 matcher**：诊断器拒绝 response/model 字段，matcher 版本与模型运行使用独立产物目录。
- **OOM 重演**：候选和 response 均按 sample/chunk 流式写，禁止全量 delta/embedding 常驻内存。
- **项目污染**：不触碰 `infra/bioinf-data-index/` 或其他项目数据。
