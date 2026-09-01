# Agent 施工手册

> 一句话摘要：所有 agent 围绕同一数据接口和同一 Action-Mode Probe 协作；允许并行实现，也允许由结果自然生长出邻近探索。

## 1. 推荐组织方式

### 1.1 总控 Agent

职责：

- 维护唯一科学目标；
- 分派任务和接口；
- 汇总阶段性检查点；
- 合并结果；
- 识别任务与主线的关系；
- 决定是否申请 GPU；
- 保持 `STATUS.md`、`DECISIONS.md`、`CLAIM_LEDGER.md`。

总控不应亲自实现所有代码，但应查看每个 worker 的：

- 输出 manifest；
- 测试结果；
- 核心图；
- 失败说明；
- commit diff。

### 1.2 Worker 角色

#### W1 — Sports Data

- SkillCorner/Metrica parser；
- frame/window sampling；
- graph construction；
- natural pairs；
- sports task labels。

#### W2 — Intervention and Prospective Validation

- 复用 P2 的合法 operator 与 exact-spectrum/matched-support 接口；
- 在 T1 公式和 layer 冻结后生成 response-blind prospective reallocations；
- 保存 intervention manifest、seed、support 合法性、checksum 和独立 receipt；
- 不查看 response 后筛选 support，不补数据直到预测成功。

#### W3 — Local Geometry and Layer Diagnostics

- normalized-embedding Jacobian/JVP 与有限差分 parity；
- full、diagonal-only、off-diagonal、team/edge/support block 分解；
- 输入、message-passing、pooling 前和 pooled 层 hooks；
- 低自由度频率/residual/support 基线、match-grouped prediction 与失败分析。

#### W4 — Gate-conditional AMR Method

- 在 P3 gate 通过前不训练 AMR；
- 只实现证据选出的一个 matched-capacity causal switch；
- 通过后实现 AMR-Fixed，再由任务证据决定是否进入 AMR-Learned；
- 保持 CAP、canonicalization、relational pooling 等强基线和容量记录。

#### W5 — Calligraphy

- Make Me a Hanzi parser；
- stroke/component graph；
- vector renderer；
- open/MCCD image loaders；
- cross-domain prediction test。

#### W6 — Reproducibility and Figures

- experiment registry；
- match-level CV、bootstrap 和 confidence intervals；
- manifest/receipt/checksum 与 figure data contracts；
- table/figure generation；
- license and source ledger。

W6 可在早期由总控兼任。

## 2. 并行协作原则

优先避免：

- W1 自己做一篇阵型分类论文；
- W5 自己做一篇书法识别论文；
- W4 发明与 spectrum 无关的大模型；
- W3 因模型下载困难改成完全不同 benchmark；
- 同一指标由多个 worker 各写一个不兼容版本。

并行工作最好共享以下接口；若结果产生新的合理问题，可以登记为邻近探索：

```text
CanonicalSample
  -> InterventionRecord
  -> EmbeddingRecord
  -> SpectrumRecord
  -> Exploration Checkpoint Report
```

## 3. 建议代码仓库结构

```text
repo/
  README.md
  pyproject.toml
  configs/
  data/
  src/actionmode/
    data/
      skillcorner.py
      metrica.py
      trackid3x3.py
      makemeahanzi.py
      calligraphy_images.py
      mathwriting.py
      schemas.py
    graphs/
      construction.py
      laplacian.py
      filters.py
      matching.py
    interventions/
      explicit.py
      vector_stroke.py
      render.py
      energy.py
      controls.py
    probes/
      transfer_function.py
      reversal.py
      accessibility.py
      nullspace.py
      semantic_gap.py
    models/
      deepsets.py
      set_transformer.py
      gnn.py
      image_backbones.py
      canonicalization.py
      cap.py
      amr.py
    tasks/
      formation_retrieval.py
      context_prediction.py
      localization.py
      calligraphy_matching.py
    eval/
      bootstrap.py
      metrics.py
      registry.py
    figures/
      fig1_smoking_gun.py
      spectrum.py
      causal.py
      method.py
      cross_domain.py
  scripts/
  tests/
  reports/
  artifacts/
```

## 4. 数据契约

### 4.1 CanonicalSample

实现为 dataclass 或 Pydantic model：

```python
@dataclass(frozen=True)
class CanonicalSample:
    sample_id: str
    domain: str
    node_positions: np.ndarray       # [N,2]
    node_features: np.ndarray        # [N,D]
    node_mask: np.ndarray            # [N]
    adjacency: np.ndarray            # [N,N]
    context: dict[str, Any]
    labels: dict[str, Any]
    image_path: str | None = None
    support_metadata: dict[str, Any] | None = None
```

必须检查：

- shape；
- finite values；
- 图对称性；
- 有效节点；
- 坐标单位；
- split 信息。

### 4.2 InterventionRecord

必须包含生成前后的 checksum，防止重复或错配。

### 4.3 EmbeddingRecord

```text
model_id
model_commit
layer_id
sample_id
intervention_id
embedding_path
normalization
```

大型 embedding 不直接存入 CSV；保存 `.npy/.pt`，表中只存路径与 checksum。

## 5. 实验记录

建议每个实验有一份 YAML 记录：

```yaml
experiment_id: P1_FOOTBALL_DINO_EQUAL_ENERGY_001
scientific_question: "Does matched-energy common motion outrank structural modes?"
hypothesis: H1
status: exploring
owner: W3
dataset:
  name: skillcorner
  split: match_holdout_v1
model:
  name: dinov2_vits14
  frozen: true
intervention:
  energy: 1.0
  bands: 6
  matching: total_l2
metrics:
  - errr
  - transfer_function
  - semantic_coalition_gap
seeds: [11, 23, 47]
working_expectation: "common-mode and structural-mode responses may differ"
exploration_updates: []
```

实验完成时补充实际状态和产物：

```yaml
status: completed
artifacts:
results_hash:
notes:
```

可以保留历史版本，并根据新证据修订工作假设、参数或路线。

## 6. Phase 工作包

## Phase 0 — Bootstrap

### 总控任务

- 初始化 git；
- 运行下载脚本；
- 创建 `STATUS.md`；
- 创建 `reports/exploration_log.yaml`；
- 建立初始 schema，并允许在实现中迭代。

### Worker 输出

W1：一场比赛的 canonical samples。  
W2：100 个 equal-energy interventions。  
W3：一个 baseline embedding。  
W5：10 个汉字的矢量渲染。  
W6：数据检查和样本图。

### 初始检查

- `pytest -q` 通过；
- smoke 脚本退出码 0；
- energy matching 偏差 <1%；
- 所有输出有 manifest。

## Phase 1 — Phenomenon

W2/W3 并行：

- exact spectrum；
- band spectrum；
- coordinate models；
- frozen image models；
- support sweep。

总控先看原始曲线，再决定是否扩大 AMR 探索。

输出：`reports/EXPLORATION_CHECKPOINT_1.md`。

## Phase 2 — Novelty discriminator

W1 提供 semantic coalitions；W2 生成 matched random controls；W3 复测模型。

主要问题：组织效应是否超出频率。

输出：`reports/EXPLORATION_CHECKPOINT_2.md`。

## Phase 3 — Mechanism

P2 已收口为 `mixed_or_graph_specific`。P3 不增加阶段节点，任务编号为 T0–T5，且采用证据自适应的最小路线。

### P3 worker contract

| 任务 | owner | 输入 | 输出 | 必须测试/manifest | 停止条件 |
|---|---|---|---|---|---|
| T0 接口与 parity | W3 | 250 canonical samples、9 个冻结 checkpoint、P1 factory、P2 adapter | normalized embedding、层接口、JVP smoke、parity summary | forward parity、JVP/finite difference、二阶近似、checkpoint hash | parity 或 autograd 不闭合则停在 T0 |
| T1 局部几何预测 | W3/W6 | T0 锁定层、P2 response 和 matching manifest | 单臂 q、pair Δq、低自由度基线、grouped CV、figure tables | match split 无泄漏、bootstrap、summary/table 一致 | geometry 不优于基线时不引入大预测器 |
| T2 prospective | W2/W6 | T1 冻结公式、层、指标和新 seed | response-blind intervention manifest、独立 response receipt、prospective report | 生成器不读 response、seed/hash/draw 可复现 | 失败则降低 mechanism claim，不补选干预 |
| T3 block/layer | W3 | T0/T1 输出和可比层 | block 分解、layer-wise 定位、失败条件 | 恒等式、permutation/order、维度归一化 | off-diagonal 无额外信息则收缩为 diagonal claim |
| T4 单一 causal switch | W4 | T3 选择的一条 route | matched-capacity 3-seed switch 结果 | 机制量、prospective response、任务、robustness 同时记录 | 未满足四项最低证据则不进入 AMR |
| T5 任务意义 | W1/W4/W6 | 预先冻结的任务定义、已有标签或自然 pair | localization、客观任务、geometry-task alignment | pair/match 隔离、非 intervention-label、Pareto | 只有 response 改变而任务无收益时称表示塑形 |

所有任务使用 `artifacts/phase3/` 下不可覆盖的版本化目录，并在主报告和 claim ledger 中区分事实、解释、未测试和失败条件。完整 augmentation × pooling × constraint 矩阵仅为 gate 后 optional extension。

## Phase 4 — AMR

W4 实现 M0/M1/M2；W1/W5 提供任务；W6 做 matched capacity。

建议顺序：

1. CAP；
2. AMR-Fixed；
3. AMR-Learned；
4. AMR-Implicit optional。

AMR-Implicit 可在 AMR-Fixed 稳定前做小规模探索，但不应取代对基础方法的清晰比较。

## Phase 5 — Calligraphy prediction

总控记录当前 sports observations；W5 可比较多个 band mapping 版本，并保留选择与变化记录。

若 MCCD 未到位：

- Make Me a Hanzi controlled；
- zhuojg/kirosc natural；
- CCSE pseudo-support。

## Phase 6 — Consolidation

- 默认 3 seeds，按预算和稳定性调整；
- match/character bootstrap；
- figure data consolidation；
- license audit；
- anonymization；
- paper claim ledger。

## 7. 探索检查点报告建议

每个主要检查点报告建议回答：

1. 原问题是什么？
2. 当前工作假设及其如何变化？
3. 数据与模型是什么？
4. 结果如何更新当前解释？
5. 最危险替代解释是什么？
6. 哪些 claim 得到支持、哪些仍不确定？
7. 下一步有哪些继续、扩展、pivot 或暂停选项？
9. 所有结果路径与 commit 是什么？

报告应提供具体证据，而不只写“实验完成，效果不错”。

## 8. Claim Ledger

维护表：

| Claim ID | Claim | 证据状态 | 支撑证据 | 注意事项 |
|---|---|---|---|---|
| C1 | matched-energy reversal exists | pending | checkpoint 1 | 避免泛化到所有模型 |
| C2 | mid-frequency notch is non-monotonic | pending | checkpoint 2 | 避免直接称为 universal law |
| C3 | organization exceeds frequency | pending | checkpoint 2 | 不等同于 understands tactics |
| C4 | augmentation causes notch | pending | checkpoint 3 | 区分相关性与机制证据 |
| C5 | AMR improves frontier | pending | checkpoint 4 | 不等同于 solves relational reasoning |
| C6 | sports informs calligraphy | pending | checkpoint 5 | 说明映射版本和适用范围 |

摘要式结论应标明 pending、支持和不确定的证据状态。

## 9. 文献与 novelty 审计

W3 或独立 literature worker 建议建立：

```text
literature/
  invariant_equivariant.md
  learned_equivariance_measurement.md
  canonicalization.md
  graph_spectral_bias.md
  sports_representation.md
  calligraphy_structure.md
  method_novelty_matrix.csv
```

每篇近邻记录：

- 核心问题；
- 数学对象；
- 数据条件；
- 方法；
- 与 Action-Mode Spectrum 的相同点；
- 真正差异；
- 是否应进入 baseline。

至少覆盖：Lie derivative/LEE、PooDLe、SER、learned canonicalization、frame averaging、oversmoothing/spectral bias、Kendall/Procrustes、关系/组合性基准。

## 10. 测试规范

### Unit tests

- graph Laplacian PSD；
- common mode eigenvalue ≈0；
- equal-energy normalization；
- band purity；
- permutation equivariance；
- intervention reversibility；
- SVG re-render consistency；
- no raw-data mutation。

### Integration tests

- one football frame -> graph -> intervention -> render -> embedding -> spectrum；
- one character -> strokes -> graph -> intervention -> render -> embedding -> spectrum；
- AMR forward/backward on CPU；
- experiment registry writes deterministic outputs。

### Statistical tests

测试代码使用 group bootstrap，不把每个 10 fps frame 当独立样本。

## 11. 计算资源策略

GPU 是否使用由当前瓶颈和信息增益决定，优先用 CPU/MPS 完成低成本探索。

申请 GPU 时记录：

- 视觉 embedding 或 AMR 训练的当前瓶颈；
- CPU/MPS 基准和预计收益；
- 最低机器规格、预算和可替代路线。

## 12. Git 与产物纪律

- 每个 worker 独立 branch；
- 每个 PR 尽量完成一个可验收任务；
- 不提交原始大数据和受限数据；
- 大产物进入 `artifacts/` 并记录 checksum；
- 配置、代码、图数据分离；
- 主分支任何时候可运行 smoke test。

提交格式：

```text
phase(component): concise action

Scientific question:
Implementation:
Tests:
Artifacts:
Known limitations:
```

## 13. 状态更新格式

每次 worker 返回推荐保持一页以内：

```text
DONE
- ...

FOUND
- ...

BLOCKED
- ...

EVIDENCE
- paths / metrics / confidence intervals

NEXT
- one concrete action
```

尽量避免长篇复述项目背景。

## 14. 总控探索决策

总控根据证据决定继续、扩展、改变假设、pivot、暂停或停止，并记录理由、共享资产和放弃的解释。停止某条路线不等于项目失败；它只是把资源转向更有信息量的探索。

## 15. P3 执行收口（2026-09-01）

原 T0–T5 的实际 receipt、manifest、checksum 和结果已写入 `artifacts/phase3/`。T1/T2 支持局部几何预测，T3 支持条件性 layer/block anatomy，T4/T5 的唯一开关只支持 representation shaping。由于旧客观任务与 context robustness gate 未闭合，W4 仍不得实现 AMR-Fixed 或完整 causal matrix；截至 2026-09-02，补丁包授权在同一 P3 内执行 T5R task-semantic repair，W6 继续保存旧失败边界并为新 candidate lock 建立可复现 contract。

## 16. P3-T5R worker 分工与协议（2026-09-02）

P3-T5R 仍属于 `P3_CAUSAL_MECHANISM`：

| Worker | T5R 职责 | 必须输出 |
|---|---|---|
| Data worker | SNGAR train/valid、SkillCorner dynamic-support、canonical conversion | source revision、split manifest、conversion receipt、checksum、QC |
| Task worker | context/intrinsic task、natural pair/ranking、fixed dual-channel baseline | frozen task definitions、label coverage、leakage audit、baseline lock |
| W2 | response-blind support/intervention复用与验证 | support manifest、合法性、match-level response、receipt |
| W3 | geometry/JVP、layer/block 与 task-geometry diagnostics | fixed-layer metrics、match bootstrap、失败条件 |
| W4 | T5R gate 通过前不得实现 AMR；通过后才可实现 AMR-Fixed | matched-capacity diff、训练 receipt、候选 gate |
| Autoresearch worker | 最多两轮、每轮最多六候选、单轴搜索 | registered candidates、candidate metrics、Pareto 表、firewall 证明 |
| W6/Repro worker | candidate lock、test firewall、hash、figure contract | lock schema、receipts、SHA-256、known limitations |

每个任务必须标明 input、output、tests、manifest 和 stop condition。test/IDSSE 结果在 candidate lock 前必须物理不可见；旧 heldout 不得选候选。动态 support 可并行，但不能反向选择主 task-repair 候选。
