# 实验、探索检查点与 Figure 规划

> 一句话摘要：先用等能量探针确认“模型把谁排错了”，再区分普通频率偏置与组织盲区，随后用因果矩阵定位机制，最后检验 AMR 是否能塑造作用谱并跨域预测书法失效。

## 1. 实验原则

### 1.1 等能量

比较任意两个作用模式前，必须满足：

\[
\|\delta_a\|_F=\|\delta_b\|_F=\varepsilon.
\]

不能拿“所有球员各走 2 米”和“单个球员走 2 米”直接比较。推荐两种解释并同时报告：

- **总能量匹配**：所有节点位移平方和相同；
- **感知变化匹配**：渲染后像素/LPIPS 变化相近。

### 1.2 同频对照

语义子群与随机子集必须配平：

- 支持大小；
- 总能量；
- Rayleigh quotient；
- 位移方向分布；
- 边界合法性。

### 1.3 动态假设与探索记录

研究过程中可以提出、修改、合并或放弃工作假设。建议记录每轮探索关注的内容：

- 哪些模型预计 common mode 更强；
- 哪些增强预计压低哪些频段；
- 哪种 pooling 预计产生更深凹陷；
- 体育端预计哪个归一化频段最弱；
- 书法端的对应预测。

记录实际使用的模型、band、参数、选择理由、结果和路线变化，保留版本历史以便解释探索过程。

## 2. 统一数据对象

所有域转换成：

```text
CanonicalSample
  sample_id
  domain
  node_features[N,D]          # 若显式
  node_positions[N,2]
  node_mask[N]
  adjacency[N,N]
  raw_context                 # 球场绝对位置、页面姿态等
  image_path                  # 若像素输入
  task_labels
  support_metadata
```

干预记录：

```text
InterventionRecord
  source_sample_id
  operator                    # translation / rotation / scale
  energy
  band_id
  spectral_coefficients
  support_mask
  support_size
  rayleigh_quotient
  semantic_coalition_id       # null for random
  renderer
  artifact_control_id
```

## 3. Phase 0：数据与探针 smoke test

### 目标

- 一场足球比赛可解析；
- 生成 1,000 个合法 frame samples；
- 构图和谱分解稳定；
- 同一帧可生成 common、单模式、语义联盟和随机联盟干预；
- 渲染 minimap；
- 一个小模型和一个冻结图像编码器可输出 embedding；
- Figure 1 雏形能运行，不要求有阳性结果。

### 输出

- `artifacts/phase0/sample_grid.png`
- `artifacts/phase0/intervention_manifest.parquet`
- `artifacts/phase0/spectrum_sanity.csv`
- `reports/PHASE0_REPORT.md`

### 探索检查点 0

检查能量误差、目标频段纯度、越界情况、节点保留和重复生成的一致性；若这些基础量不稳定，先修数据与干预实现，再解释模型差异。

## 4. Phase 1：Smoking-gun 现象

### 4.1 模型族

最低覆盖三类：

1. 坐标模型：MLP/DeepSets、Set Transformer、GNN；
2. 图像预训练模型：DINOv2、OpenCLIP、MAE/ViT；
3. 任务训练模型：formation/phase baseline。

不需要扫描十几个大模型；每类 1–2 个代表即可。

### 4.2 干预集合

- 精确 eigenmode 或频段干预；
- common/DC；
- 真实战术子群：后卫线、中场线、弱侧单元；
- 同频随机子集；
- 单人局部；
- 支持域滑动：\(k/n\) 从 1 到最小支持。

### 4.3 主要指标

#### Equal-energy Reversal Rate

\[
ERRR=P\left[d_z(common)>d_z(semantic)\right].
\]

#### Structural Transfer Function

\[
H_f(b)=\mathbb E[d_z(f(x'),f(x))/\varepsilon].
\]

#### Notch Depth

\[
ND=1-\frac{H(mid)}{\frac12(H(low)+H(high))}.
\]

仅在曲线呈非单调时使用，不要强行制造。

#### Semantic Coalition Gap

\[
SCG=H(semantic\ coalition)-H(random\ matched).
\]

符号方向由任务定义；重点是显著非零。

#### Mode Accessibility

线性 probe 从单样本或表示差恢复频段系数、内部构型和上下文：报告 \(R^2\)、circular MAE 或 retrieval。

### 探索检查点 1：现象初筛

优先观察以下证据，不设统一的机械阈值：

- 不同模型族的 ERRR、作用谱和不确定性；
- 是否出现稳定的中频凹陷或其他非单调结构；
- common-mode 与语义谱的错序是否与自然任务错误相关。

若候选现象不出现，记录阴性结果，并调整问题表述、模型范围或后续探索；不通过方法训练制造失败。

## 5. Phase 2：排除“只是过平滑/谱偏置”

### 5.1 核心判别实验

在相同频率带内比较：

- 真实战术联盟；
- 连通随机子图；
- 非连通随机子集；
- 相同支持大小的随机节点；
- phase-randomized 频谱匹配扰动。

### 5.2 判决

#### 新现象存活

SCG 在多个图构造、多个比赛和至少两类模型中稳定非零，或中频凹陷对组织语义具有选择性。

#### 退化为普通频率偏置

响应几乎只由 Rayleigh quotient/频率解释，语义与随机联盟无差异。

这种结果不是“完全没价值”，但当前论文需要改写为关系频谱偏置分析，不能继续声称模型看不见组织。

### 探索检查点 2

综合效应量、group-bootstrap 区间、图构造敏感性和模型差异，判断组织效应是否值得继续深入；可以根据结果增加或调整匹配方案。

## 6. Phase 3：因果机制

P2 已收口为 mixed_or_graph_specific。P3 在同一阶段编号内采用证据自适应的最小机制路线，不默认运行 augmentation × pooling × constraint 全矩阵，也不直接训练 AMR。P3-T0–P3-T5 是本阶段任务编号，不是新阶段节点。

### 6.1 P3-T0：接口、provenance 与数值等价

输入固定为 P1 的 250 个 canonical samples、9 个冻结点集模型、原始 baseline adjacency 和 P2 的 response-blind intervention 接口。必须：

- 通过 P1 唯一的 build_torch_models() 恢复 checkpoint，使用 eval()、无 dropout、固定 dtype/device；
- 复现 P2/P1 baseline embedding；
- 暴露输入/初始节点编码、每个 message-passing block 后、pooling 前节点表征和 pooled embedding；
- 对 normalized embedding 计算 autograd JVP，并验证中心有限差分；
- 在小 epsilon 下检查 \(1-\cos\mathrel{\approx}\frac12\|J\delta\|_2^2\) 的误差收敛；
- 验证 checkpoint、P2 输入文件和 P2 artifact 不被修改。

T0 只产生工程 parity 和 smoke 证据，不产生机制结论。

### 6.2 P3-T1：冻结局部几何预测

主估计量为：

\[
q_f(X,\delta)=\frac12\|J_f(X)\delta\|_2^2.
\]

对每个 anchor/control pair 计算 \(\Delta q\)，至少预测单臂 raw representation response、anchor-control raw pair effect、pair-effect 方向，以及 match-level 聚合的方向和大小。

必须并列比较 spectral power/band power、Rayleigh quotient、support size、role/graph/architecture/epsilon、frequency residual、topology residual、简单线性模型、diagonal-only geometry、full geometry 和 off-diagonal contribution。禁止用大型可学习预测器拟合 P2 response。

评估按 match 分组，使用 leave-one-match-out 或明确 grouped CV；超参数只能在训练 match 内选择。报告 Spearman、MAE、\(R^2\)、方向准确率和 calibration，并以 match-level bootstrap 比较 full geometry 与最佳非几何基线。seed 只在 architecture 内分层，不能伪造比赛重复。

先检验 tangent metric。只有小 epsilon 明显优于大 epsilon，且残差有清楚路径非线性时，才允许在同一机制内增加预先固定的 3 点 integrated-Jacobian/path-energy 估计。

### 6.3 P3-T2：prospective 同资产验证

在 T1 后冻结公式、层、指标、统计单位、失败分析和允许的有限扩展；随后只用相同 250 个样本、相同 9 个模型、相同 operator、主 epsilon 范围和新的 intervention seed/合法 support reallocation 生成 response-blind prospective set。设计文件、seed、manifest 和 checksum 必须在读取新 response 前锁定。

禁止新增数据域、新训练模型、查看 response 后选 support 或 layer、用 prospective response 调阈值，以及只保留预测正确的 intervention。prospective 失败时降低机制 claim，不补数据直到成功。

### 6.4 P3-T3：block 分解与 layer-wise 定位

至少报告 full、diagonal-only、off-diagonal，以及 team 内/间、support 内/外、graph edge/non-edge 的贡献。回答是哪些节点更敏感，还是哪些节点之间被耦合，并比较 DeepSets、GAT-small 和 Phase-GAT 的架构差异。role 只作描述性分层。

对可比层报告 response prediction、diagonal/off-diagonal contribution、directional Jacobian rank、mode/support accessibility 和任务相关信息：

1. 输入/初始节点 embedding；
2. 每个 interaction/message-passing block 后；
3. pooling 前节点或 token 表征；
4. pooled embedding；
5. 若 P2 使用，projection head 后。

不同维度层不直接比较绝对 Frobenius norm；使用规范化指标、matched directions 或明确的可比尺度。若 off-diagonal 没有额外解释力，收缩为节点敏感度各向异性；若信息只在 pooling 前存在，才考虑 pooling accessibility 路由。

### 6.5 P3-T4：只选择一个因果开关

根据 T3 只选择一条最小路线：

- pooling accessibility：pooling 前有信息、pooling 后丢失或反号时，对比 unary mean/sum 与 matched-capacity relational pairwise pooling；
- interaction coupling：差异在 message passing 中形成时，对比无跨节点交互与一个最小的局部/relational interaction；
- constraint placement：结构信息存在但 pooled objective 压低时，对比 pooled-only 与 node-level mode-preserving/recoverability objective。

不并行扫描完整矩阵。因果开关必须同时满足：机制量按预测改变、prospective response 按预测改变、至少一个客观任务指标改善、没有通过牺牲 global context/robustness 获得虚假提升，并完成 matched capacity 与 3 seeds 或明确稳定性分析。只改变 response 而任务无收益时，称为表示塑形，不称为修复。

### 6.6 P3-T5：任务意义

任务定义和 pair 规则必须在查看方法结果前冻结。至少完成 controlled perturbation localization，以及一个已有且非 intervention-label 的客观任务，优先使用已有 phase/deployment 标签、自然 pair ranking，或 intrinsic formation retrieval/context prediction。

定义 \(S_\tau(X,\delta)=\Delta\ell_\tau(X,X\oplus\delta)\) 或等价的 oracle 几何/排序变化，检验 \(q_f\) 与 \(S_\tau\) 的排序、校准和 alignment；在相同 context robustness 下评价结构任务 Pareto。表示 response 大小本身没有正确方向。
### P3 路由

- full geometry 在 retrospective 和 prospective 均胜过最佳频率/residual 基线，且 layer/block 与一个因果开关、任务指标闭合：允许进入 P4 AMR-Fixed；
- 只有 diagonal 有效：限定为节点/支持敏感度各向异性；
- 仅一个 architecture 有效：限定为该架构；
- retrospective 有效而 prospective 失败：只保留描述性解剖，不进入 AMR；
- response 可控而任务无收益：保留 probe/机制分析，不称方法修复；
- geometry 和 integrated metric 均不能预测：停止当前机制复杂度，不直接进入 AMR，保留 P2 条件异质性结论。

### 6.7 历史候选因子矩阵（不作为默认施工路径）

以下矩阵保留为 P3 的 optional_after_minimal_mechanism_gate 扩展，只有 T0–T5 的证据支持后才可取其中一个最小对照；不再是当前默认路线。

| 因子 | 条件 |
|---|---|
| 增强耦合 | 全构件共同 / 子群 / 独立构件 |
| 约束类型 | invariance / equivariance / 无约束 |
| 聚合 | unary sum/mean / CLS / pairwise relational / attention |
| 图处理 | 无图 / 固定图 / 谱频段 |
| 训练信号 | 自然分布 / 谱白化 |

先做 2×2 主矩阵：

1. global-coupled augmentation + unary pooling；
2. global-coupled + relational；
3. independent augmentation + unary；
4. independent + relational。

随后只加最必要的 constraint-type 对照。

### 6.8 历史机制预测（待检验）

- independent part invariance 会扩大低敏感度子空间；
- unary pooling 更难保留相对 phase；
- relational interaction 应恢复部分中频响应；
- 把中频加入 invariance 会加深凹陷；
- 把同一中频加入 equivariance 会填平凹陷。

### 6.9 Layer-wise 补充指标

每层报告：

- \(H_f(b)\)；
- mode coefficient probe；
- token/node level vs pooled embedding；
- Jacobian directional rank；
- context accessibility。

### 探索检查点 3

比较训练和架构开关是否改变凹陷或错序，并区分相关性、可重复性和潜在机制；结果决定继续机制实验、换模型或转向其他解释。

## 7. Phase 4：方法实验

### 7.1 比较方法

#### 低成本几何基线

- raw coordinates；
- centering；
- Procrustes/Kendall shape；
- fixed canonicalization；
- learned canonicalization；
- frame averaging（可行时）。

#### 表征基线

- DeepSets；
- Set Transformer；
- GNN/GAT；
- relational pooling；
- invariant/equivariant auxiliary loss；
- CAP。

#### 主方法

- AMR-Fixed；
- AMR-Learned；
- AMR-Implicit（条件性）。

### 7.2 体育任务

#### Intrinsic formation retrieval

相同内部构型、不同球场位置应检索为近邻。

#### Context/deployment prediction

保留球队绝对位置、高/中/低 block 或 phase 标签。

#### Structural perturbation localization

识别哪个频段或哪个节点联盟发生改变。

#### Natural pair ranking

从真实比赛中寻找：

- 内在构型近、绝对位置远；
- 绝对位置近、内在构型远。

不依赖合成干预。

#### Optional short-horizon task

只在不引入视频模型时，使用短窗预测 phase/change point；不是主贡献。

### 7.3 书法任务

全部使用客观结构目标：

- 同字符跨页面姿态 retrieval；
- reference–candidate 结构匹配；
- component/stroke perturbation localization；
- 不同任务下全局姿态与内部构型的分别读出；
- 自然书法图像与矢量受控模板之间的结构一致性。

不以审美 scalar 为主任务。

### 7.4 方法评价

- robustness–structure Pareto；
- semantic spectrum alignment；
- quotient retrieval；
- context 与 intrinsic 双读出；
- natural task performance；
- learned gate 可解释性。

### 探索检查点 4

重点比较 AMR 与基线的：

1. 相同 global robustness 下的内部模式可访问性；
2. 自然任务表现及其失败案例；
3. 任务门的差异、稳定性与可解释性；
4. 参数量、运行成本和容量匹配。

若 AMR-Learned 暂时不超过 Fixed，可保留两者继续探索，并把“自动发现边界”表述为尚未验证的工作假设。

## 8. Phase 5：体育到书法的跨域探索

### 8.1 建立映射

体育图大小与书法 stroke 数不同，不直接比较 eigenvalue 数值。使用：

- 归一化谱 rank \(q=k/(n-1)\)；
- 频段累计谱质量；
- 或支持域比例和图割尺度。

体育端在每轮分析后记录当前观察到的：

- 最弱频段；
- 支持域边界；
- 语义联盟与随机联盟差异方向；
- 哪类模型最严重。

### 8.2 书法检验

围绕当前映射版本测试，并允许根据新结果调整映射：

- 偏旁级干预是否落在对应频段；
- 关键模型的弱点排序是否复现；
- AMR 是否在对应模式上优先修复；
- 自然书法结构匹配是否随谱改善。

### 探索检查点 5

比较不同 band、谱 rank、支持域比例和图割尺度的映射表现，记录命中、失败和映射调整；跨域结果用于更新研究解释，不设单一命中阈值。

## 9. 模型规模实验

只做便宜的尺度轴：

- 小/中 backbone；
- DINOv2 S/B 或 CLIP B/L 的冻结表征；
- 坐标模型 hidden 64/128/256。

目标不是拟合 scaling law，而是回答：缺陷是否会被规模自然消除。

## 10. Figure 规划

### Figure 1：Smoking gun — Who moved together?

优先使用实测结果，而不是只画概念插画。

- Panel A：足球原阵型、等能量整队位移、等能量结构破坏；
- Panel B：现有模型距离排序；
- Panel C：书法原字、页面变化、偏旁/单笔结构变化；
- Panel D：同一模型的排序复现。

首选图注：

> Equal energy, different coalition: modern representations can react more to common displacement than to structural fracture.

若该候选结果不成立，就改画更符合数据的现象；不能通过调幅度制造视觉效果。

### Figure 2：Action-Mode Probe

- 关系图和 Laplacian 模式；
- 等能量扰动如何覆盖频谱；
- \(H_f(b)\) 和任务语义谱 \(S_\tau(b)\)；
- common/subgroup/local 只作为示意标签。

### Figure 3：核心规律

不再预设统一中频凹陷。核心图优先展示 observed vs geometry-predicted pair effect，并同时给出频率、Rayleigh/residual 与 support-conditioned geometry 基线；若只有特定 architecture、graph 或 role 条件成立，按条件分面展示，不合并成统一同号结论。主图宜包含同频随机对照，以便区分组织效应与 oversmoothing。

- Panel A：observed pair effect 与 full/diagonal-only/off-diagonal geometry prediction；
- Panel B：geometry 相对于 frequency/residual baseline 的 match-level 增量；
- Panel C：失败条件、方向反转和 epsilon 敏感性。

### Figure 4：因果开关

- 只展示一个由 T3 证据选择的开关；
- 预测的 block/layer 变化 → 实际操作 → geometry、prospective response 与任务指标；
- 若路由未通过 gate，展示失败链而不是补画完整 causal matrix。

最好形成“预测—操作—几何—任务”的四列，而不是消融大表。

### Figure 5：AMR 方法

- context channel；
- graph spectral filter bank；
- task-conditioned gates；
- frequency × support/relationship-conditioned fixed routing；
- invariant/equivariant branches；
- learned gate heatmap，展示不同任务/域的模式指纹。

### Figure 6：方法效果

- robustness–structure Pareto；
- AMR vs CAP/canonicalization/relational；
- 至少一个自然任务；
- 参数量/算力匹配。

### Figure 7：跨域预测

左：体育端当前探索得到的弱频段。  
中：跨域映射及其版本变化。  
右：书法端命中或失败，以及 AMR 修复。

这张图负责把“双域验证”升级为科学预测。

### 可选 Figure 8：规模与层级

- 模型尺度；
- 信息在哪层消失；
- token/node 中存在、pooling 后消失，或 encoder 早期即消失。

正文最终保留 5–6 张主图，其他进入附录。此文档只规划候选，不规划正文顺序。

## 11. 表格规划

### Table A：核心模型与数据矩阵

只列代表模型，不做模型动物园。

### Table B：方法主结果

同时报告：

- global robustness；
- intrinsic retrieval；
- mode localization；
- context prediction；
- compute/parameters。

### Table C：跨域结果

书法只报告客观结构任务和预测命中，不用专家审美作主要数值。

## 12. 统计与复现

- 默认使用 3 seeds；根据计算预算和结果稳定性可调整，并记录实际设置；
- 数据 split 按比赛/书法家隔离，避免帧级泄漏；
- bootstrap 单位是 match/character 或更高层样本，不是逐帧假独立；
- 报效应量和 CI，不只报 p 值；
- 阈值、band 切分和图定义可在探索中迭代，并报告各版本及敏感性；
- 在多种合理图构造下检查结果方向和失败模式。

## 13. 结果分流与 pivot

不同结果可以引出不同的研究方向：

- 等能量后没有稳定失效：研究模型正确响应结构模式的条件；
- 同频语义联盟效应较弱：聚焦关系频谱偏置或 pooling accessibility；
- 结果受像素伪影影响：优先改进渲染和 artifact control；
- 方法暂未超过简单 baseline：保留诊断结果，分析 AMR 的适用边界。

### 邻近 pivot A：纯关系频谱偏置

若只有单调曲线，可以改研究架构/增强如何决定多体关系频谱，收缩组织盲区和中频凹陷表述。

### 邻近 pivot B：Pooling accessibility

若 token 有信息而 pooled 表征丢失，主张收紧为：关系信息在聚合后不可线性访问；AMR 作为关系路由聚合方法。

### 邻近 pivot C：Augmentation overclosure

若开关增强即可控制，重点转为增强谱与不变性边界；AMR 作为 constraint-type routing。

这些方向共用数据、探针、图和大部分代码，可按探索结果灵活切换。

## 12. P3 实际结果与 Figure source contract（2026-09-01）

P3-T0–T5 已执行完毕；当前结论为 response shaping only，P4 gate 未满足。retrospective full-vs-baseline mean Spearman 为 `0.8466` vs `0.1222`，prospective 为 `0.8526` vs `0.1294`，统计单位为 `source_match_id`。

- Figure 3 使用 `artifacts/phase3/support_geometry_v1/pair_predictions.parquet`、`prediction_metrics.parquet` 和 `match_bootstrap_comparisons.parquet`，并分开显示 `artifacts/phase3/support_geometry_prospective_v1/prospective_pair_predictions.parquet` 与 prospective metrics。
- Figure 4 使用 `layerwise_block_summary.parquet`、`layerwise_prediction_metrics.parquet` 和 `layerwise_jacobian_rank.parquet`，使用规范化/匹配方向指标，不比较不同维度层的原始 Frobenius norm。
- Figure 5 使用 `selected_causal_switch_v1/switch_seed_summary.parquet`、`switch_task_metrics.parquet`、`switch_context_robustness_by_seed.parquet` 和 `node_sensitivity_localization_v2_summary.parquet`，必须同时画出 response shaping 与 task/robustness failure boundary。
- 本轮 Figure 只支持“预测—定位—response shaping”链，不能标为任务修复或 AMR 成功图。完整 causal matrix 由于 gate 失败不启动。
