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

## 6. Phase 3：因果机制矩阵

不先大规模巡检模型，而做匹配容量的受控矩阵。

### 6.1 因子

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

### 6.2 机制预测

- independent part invariance 会扩大低敏感度子空间；
- unary pooling 更难保留相对 phase；
- relational interaction 应恢复部分中频响应；
- 把中频加入 invariance 会加深凹陷；
- 把同一中频加入 equivariance 会填平凹陷。

### 6.3 Layer-wise anatomy

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

候选二选一：

1. 中频凹陷跨模型/层出现；
2. 若无凹陷，则展示更强的同频语义联盟效应。

主图宜包含同频随机对照，以便区分组织效应与 oversmoothing。

### Figure 4：因果开关

- 增强谱能量 vs 表示响应；
- 中频 invariance 加深凹陷；
- 中频 equivariance 填平/移动凹陷；
- unary 与 relational pooling 的差异。

最好形成“预测—操作—响应”的三列，而不是消融大表。

### Figure 5：AMR 方法

- context channel；
- graph spectral filter bank；
- task-conditioned gates；
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
