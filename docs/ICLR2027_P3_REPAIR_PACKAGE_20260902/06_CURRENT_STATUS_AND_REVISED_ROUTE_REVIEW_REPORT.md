# Action-Mode Spectrum 项目当前状态与修订路线报告

**用途**：第三方科学与投稿评审  
**日期**：2026-09-02  
**目标会议**：ICLR 2027  
**仓库**：`https://github.com/FreddieWho/iclr2027`  
**当前公开主分支最近状态**：P3 geometry/mechanism 已完成，task repair 未闭合，额外候选搜索为阴性  
**报告性质**：项目事实整理、独立判断与后续方案。本文不修改仓库，不声称第三方复现实验已经完成。

---

# 1. 执行摘要

## 1.1 项目原始问题

项目研究多构件系统中，同一个几何算子作用于不同构件集合时，模型是否以任务正确的方式表征这种变化。中心索引为：

\[
\text{operator}\times\text{support/coalition}\times\text{task}.
\]

足球中的例子包括全队共同移动、防线共同移动和单个球员脱离；书法中的例子包括整页移动、偏旁移动和单笔移动。

原始强故事押注：

1. 现有模型可能出现等能量排序反转；
2. 多类模型可能共享中频组织盲区；
3. 真正语义小组相对同频随机小组具有稳定差异；
4. augmentation/pooling/constraint 能控制该规律；
5. AMR 能改善 robustness–structure frontier；
6. 体育端规律可预先预测书法端失败。

## 1.2 当前最重要的已支持结果

当前最坚实的结果不是“所有模型都有中频盲区”，而是：

> **图频率或逐模态功率不足以决定多构件表示对干预的响应；位移落到哪些节点、这些节点之间如何耦合，以及模型在哪一层聚合，会系统性改变响应。**

对归一化 embedding 定义：

\[
q_f(X,\delta)=\frac12\|J_f(X)\delta\|_2^2,
\qquad
G_f(X)=J_f(X)^\top J_f(X).
\]

该局部几何在当前 250 个样本、10 场比赛和 9 个冻结点集模型上，对 P2 fracture pair response 有很强预测力：

| Predictor | Retrospective mean Spearman | Response-blind prospective mean Spearman |
|---|---:|---:|
| frequency/residual baseline | 0.1222 | 0.1294 |
| diagonal geometry | 0.6528 | 0.6580 |
| off-diagonal geometry | 0.3886 | 0.3815 |
| full geometry | **0.8466** | **0.8526** |

P3 还发现 geometry-response 关系随网络层逐渐增强，并且不同架构形成跨节点耦合的路径不同。

## 1.3 当前最大阻隔

P3 已证明“可解释”和“可塑形”，但没有证明“有用”。原因至少包括：

1. 旧 T5 把需要保留绝对位置的 phase/deployment task，与要求对全局平移不敏感的 robustness 指标放进同一个 gate；
2. 标签样本仅 train 113 / dev 37 / heldout 32，且 heldout 已在候选循环中被计算和展示；
3. Phase-GAT 原始 task performance 本身较弱且 seed 不稳定；
4. prospective 只更换同一批比赛上的 intervention，不是新比赛泛化；
5. 当前只测试了一个 Phase-GAT pooling switch 和少量 centering/pooling 候选；
6. AMR 没有正式运行，不能被称为失败。

因此旧任务下的结论是：

> **这些简单开关在旧 conflated gate 下没有形成稳定 task repair。**

不是：

> **不存在可用的 task-conditioned repair。**

## 1.4 本报告建议

在既有 P3 内追加一个严格有界的 task-semantic repair：

- 标记旧 heldout 已暴露；
- 引入 64 场完整 tracking 数据作为开发/未见测试；
- 将 context task 与 intrinsic task 分开；
- 先测固定 raw/centered 双通道；
- 只在出现稳定信号后运行最多两轮有限 autoresearch；
- 冻结候选后一次性运行未见 test 和 IDSSE 外部确认；
- 成功才进入 AMR-Fixed；失败则冻结为诊断/机制论文。

---

# 2. 证据状态：哪些被否定，哪些成立，哪些未决

## 2.1 `FALSIFIED_AS_STATED`：原命题的强版本与数据冲突

### A. 所有模型共享统一中频盲区

P1 中点集模型出现较深 notch candidates，但视觉模型 DINOv2/OpenCLIP 的候选深度明显较弱；P2 又显示 response 强依赖 architecture、graph、role/support 和 energy。故“跨模型统一的一维中频 notch”不再成立。

这不等于任何模型都没有 notch；合理的新问题是：

> 哪些架构、层和 support 条件产生 notch？

### B. 模型完全看不见组织

P2 exact-spectrum control 保持逐模态功率，仅改变空间排列，模型仍有稳定不同 response。字面意义上的 complete organization blindness 被否定。

更准确的结论是：

> 模型能感知 support/organization，但其敏感性未必与任务语义对齐。

### C. 静态角色组普遍产生统一方向的语义效应

P2/H1 中 defender、midfielder、forward 方向明显不同，并随 architecture/graph 变化；LOMO 显示许多方向跨比赛稳定。因此不是简单功效不足，而是 universal static-role claim 有稳定反例。

## 2.2 `SUPPORTED_CONDITIONALLY`：规律存在，但受条件限制

### A. Organization beyond spectral power

逐模态功率相同仍有 response 差异，说明频谱功率不是充分统计量。

### B. Support-conditioned response

保持非零 displacement-vector multiset，只重新分配 endpoint support，response 发生可预测变化。

### C. Local geometry predicts response

full local geometry 在 retrospective 和 response-blind prospective 均显著超过低维频率/residual baseline；结论边界为当前模型和同一批比赛/新干预。

### D. Layer/architecture anatomy

geometry-response 预测相关从 input 到 message passing、pooling 逐步增强；DeepSets 与 GAT/Phase-GAT 形成 off-diagonal interaction 的路径不同。

### E. Pooling is a causal shaper

Phase-GAT 的 team mean 与 relational pairwise pooling 在 matched capacity、3 seeds 下改变了 off-diagonal/full ratio、geometry-response correlation 和实际 response。

## 2.3 `NOT_ESTABLISHED`：目前证据不足，不能下肯定或否定结论

- 动态真实战术协同小组是否相对 matched random support 有稳定效应；
- organization information 是否只是以非线性方式可解码；
- geometry shaping 是否能改善语义正确的 intrinsic task；
- geometry-response 关系能否推广到新比赛、新 provider；
- 体育机制是否能预测书法自然风格/结构任务。

## 2.4 `NOT_TESTED`

- AMR-Fixed；
- AMR-Learned；
- scale is not a cure；
- 完整 augmentation × constraint causality；
- 独立体育→书法冻结预测。

---

# 3. 当前数据与模型

## 3.1 当前足球数据

- 10 场 SkillCorner A-League 比赛；
- 250 canonical samples，每场 25 个采样帧；
- 每帧约 20 名外场球员、team slots、角色、坐标、图；
- phase labels 来源于 SkillCorner phases of play；
- P2/P3 的主要独立统计单位为 source match。

## 3.2 当前模型

九个冻结模型：

- DeepSets-AE × 3 seeds；
- GAT-small-AE × 3 seeds；
- Phase-GAT × 3 seeds。

latent dimension 为 128。Phase-GAT 以七类 phase labels 训练，包括 high/medium/low block、transition、set play 等。

## 3.3 当前 intervention

### Rigid

比较 role-defined rigid support 与多类 matched controls，控制总能量、支持大小、图/频率特征等。

### Exact-spectrum

保持每个 Laplacian mode power 相同，通过相位/符号改变空间排列。

### Fracture endpoint reallocation

保持非零 displacement-vector multiset 不变，将向量重新分配给同队节点，检查 support organization 的增量。

---

# 4. P1：现象候选

P1 完成：

- 250 samples、10 matches；
- 42,500 interventions，39,328 valid；
- 9 个点集模型和辅助视觉模型；
- 等能量误差和目标 mode purity 均通过。

初步现象：

- 多数模型 common-over-semantic ERRR 很高；
- 点集模型出现深 notch candidates；
- DINOv2/OpenCLIP notch 较弱；
- P1 明确没有将候选写成正式 claim，要求 P2 controls。

P1 的价值是建立测量，不是证明 universal law。

---

# 5. P2：频率与组织判别

## 5.1 Rigid formal v2

关键结果：

- exact-spectrum control 下，低能量组合在多个 architecture/graph 上均显示空间排列会影响 response；
- topology-frequency matched control 下，整体 effect 对 architecture、graph、role 和 matching strictness 敏感；
- strict matching 后许多区间跨 0；
- role heterogeneity 明显，defender/midfielder/forward 不能合并成统一平均。

结论：

> 普通频谱功率不能解释全部 response，但“静态角色联盟普遍更特殊”不成立。

## 5.2 Fracture continuity

- 10 matches；
- 9 models；
- 121,293 complete pair effects；
- vector multiset 在 anchor/control 间严格相同；
- control frequency/topology residual 被显式报告。

结果：

- midfielder 在多种模型中偏正；
- defender 多数偏负；
- forward mixed；
- architecture 和 graph 进一步影响方向；
- residual correlations 整体小但部分 strata 不可忽略。

## 5.3 H1 异质性诊断

36 个 architecture × graph × role × epsilon 条件：

- 20 positive；
- 16 negative；
- LOMO 同号比例中位数 1.0；
- 30/36 条件 LOMO 与 full direction 完全一致；
- 83.3% 条件在 3 seeds 方向一致；
- role 和 architecture 的 descriptive range 大于 graph/epsilon。

这说明 P2 mixed result 不是纯噪声，而是结构性条件效应。

---

# 6. P3：Support-conditioned local geometry

## 6.1 数学对象

对任一可比层的归一化表示：

\[
\hat z_f(X)=\frac{z_f(X)}{\|z_f(X)\|_2},
\qquad
J_f(X)=\frac{\partial\hat z_f(X)}{\partial\operatorname{vec}(X)},
\qquad
G_f(X)=J_f(X)^\top J_f(X).
\]

局部预测量：

\[
q_f(X,\delta)=\frac12\delta^\top G_f(X)\delta.
\]

对二维节点坐标，G 可分解为节点 diagonal blocks 与跨节点 off-diagonal blocks。

## 6.2 T0：数值与 provenance

- 250 × 9 baseline embedding parity：9/9 PASS，最大差 0；
- JVP 与 full Jacobian 一致；
- centered finite difference 误差在可接受范围；
- block decomposition identity residual 接近数值精度；
- P1/P2 frozen inputs 和 checkpoints 前后 hash 一致。

## 6.3 T1：retrospective prediction

- 272,169 arm rows；
- 121,293 pair rows；
- leave-one-source-match-out；
- 无大型 learned predictor；
- 对比 frequency、Rayleigh、support、metadata/residual 与 geometry families。

结果：

| Predictor | Spearman | MAE | R² | Direction accuracy |
|---|---:|---:|---:|---:|
| frequency/residual baseline | 0.1222 | 0.000279 | 0.0145 | 0.5398 |
| diagonal geometry | 0.6528 | 0.000182 | 0.4754 | 0.7248 |
| off-diagonal geometry | 0.3886 | 0.000242 | 0.2760 | 0.6072 |
| full geometry | 0.8466 | 0.000114 | 0.7495 | 0.8109 |

## 6.4 T2：response-blind prospective

先冻结公式、层、指标和统计单位，再生成新 intervention seed 20260901：

- 5,700 sets；
- 4,475 complete sets；
- 90,432 arm rows；
- 40,275 pair rows；
- 同一批 samples/models/operator，仅新 support reallocations。

结果：

| Predictor | Spearman | MAE | R² | Direction accuracy |
|---|---:|---:|---:|---:|
| frequency/residual baseline | 0.1294 | 0.000262 | 0.0139 | 0.5449 |
| diagonal geometry | 0.6580 | 0.000175 | 0.4584 | 0.7224 |
| off-diagonal geometry | 0.3815 | 0.000236 | 0.2224 | 0.6026 |
| full geometry | 0.8526 | 0.000109 | 0.7416 | 0.8113 |

这证明不是对旧 pair 的简单后验拟合，但不是新比赛泛化。

## 6.5 T3：layer/block anatomy

总体 geometry-response Spearman：

- input：约 0.2125；
- message passing 1：约 0.4420；
- message passing 2：约 0.5740；
- pooling-pre：约 0.7578；
- pooled：约 0.8407。

机制解释：

- diagonal node sensitivity 是主要贡献；
- off-diagonal coupling 提供额外信息；
- DeepSets 的 cross-node terms 主要在 pooling 聚合中出现；
- GAT/Phase-GAT 在 message passing 中形成 interaction，并在 pooling 后保留；
- 没有单一 universal stage 可以解释所有架构。

## 6.6 T4/T5：pooling switch

Phase-GAT：`team_mean` vs `relational_pairwise`，3 seeds，matched parameter count 149,639。

| Metric | team_mean | relational_pairwise |
|---|---:|---:|
| mean abs pair effect | 0.002935 | 0.002066 |
| mean abs delta q full | 0.001479 | 0.000883 |
| off/full abs ratio | 0.4929 | 0.5792 |
| geometry-response rho | 0.7383 | 0.8453 |
| heldout accuracy | 0.3542 | 0.3854 |
| heldout macro-F1 | 0.1861 | 0.1839 |
| context response | 0.01324 | 0.01571 |

结论：

- geometry 与 response 被可预测地塑形；
- task macro-F1 未稳定改善；
- context response 变差；
- controlled support localization 仅 0.1546 vs uniform 0.1458；
- 只能称 `RESPONSE_SHAPING_ONLY`。

---

# 7. 额外 candidate search

候选：

- team_mean；
- relational_pairwise；
- centered_team_mean；
- centered_relational；
- team_centered。

开发结果：

- centered_team_mean：dev 较高、context 近 0，但 geometry 略降且旧 heldout 明显差；
- relational_pairwise：geometry 提高，但 context response 上升；
- centered_relational：表面三指标过线，但同时改两个轴，且主要由单 seed 驱动；
- team_centered：任务与 geometry 坍缩。

判定：`NOT_SUPPORTED`，保留 incumbent。

## 协议风险

脚本虽然没有把 heldout 放入排序公式，但在每个 candidate 循环中计算、打印和保存 heldout。故：

```text
旧 heldout = exposed exploratory audit
不是 final confirmatory set
```

另外，候选搜索代码使用 runtime monkey-patch 和临时分支逻辑，适合探索，不适合最终复现入口。

---

# 8. 为什么旧 Task Repair 不能视为一般性证伪

## 8.1 任务语义冲突

phase labels 包括 high/medium/low block，这些标签需要绝对球场位置；而旧 gate 同时要求 global translation response 越低越好。

因此它同时要求模型：

- 记住部署位置；
- 对部署位置变化不敏感。

这与原始 AMR 的 context/mode 双通道原则相冲突。

## 8.2 样本量与标签稳定性

旧有标签样本：

- train 113；
- dev 37；
- heldout 32；
- 7 classes。

Phase-GAT 原始 dev macro-F1 本身约 0.075–0.144，表明任务不稳。一个 seed 或一个 match 足以翻转均值。

## 8.3 修复空间很窄

正式 causal switch 只覆盖一个 architecture 和 pooling axis；额外 search 只覆盖少量 centering/pooling 变体。可以否定这些具体候选，不能否定所有 task-conditioned routing。

## 8.4 AMR 没有运行

AMR-Fixed/Learned 仍是 NOT TESTED，而不是 NOT SUPPORTED。

---

# 9. 数据扩展方案

## 9.1 主开发：SNGAR whole-match tracking

地址：`https://huggingface.co/datasets/OpenSportsLab/SNGAR-Action-Spotting-Tracking`

- 64 matches；
- 45/9/10 match split；
- 11.85M tracking rows；
- 87,939 events；
- Parquet，约 3.0 GB；
- player roles、positions、ball、event fields；
- gated research access。

价值：扩大独立比赛、提供自然 task、恢复未见 test。

## 9.2 Dynamic support：SkillCorner dynamic events

地址：`https://github.com/SkillCorner/opendata`

可使用 pressing chain、simultaneous engagements、off-ball runs、passing options、line-breaking 和 phase fields 构造动态 support。由于比赛已经用于 P1–P3，只用于规则开发和历史复现。

## 9.3 外部确认：IDSSE

地址：`https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177`

- 7 complete Bundesliga matches；
- 25 Hz tracking + synchronized events；
- CC BY 4.0；
- 不同 provider/league。

## 9.4 Acquisition shift：SoccerTrack v2

地址：`https://atomscott.github.io/SoccerTrack-v2/`

- 10 full matches；
- panoramic full-pitch acquisition；
- per-frame GSR + 12 BAS labels；
- CC BY 4.0；
- HF gated download。

## 9.5 书法

- Make Me a Hanzi：矢量受控 instrument；
- HCSU：自然作者/书体/结构属性；
- MCCD：更大规模作者/书体多属性，需申请。

书法只在体育 prediction lock 后使用。

---

# 10. 修订后的任务设计

## 10.1 Context channel

输入 raw positions，任务包括：

- phase/deployment；
- team centroid / field zone；
- absolute spatial context。

应保留绝对位置可访问性。

## 10.2 Mode channel

输入 globally centered positions，任务包括：

- intrinsic formation retrieval；
- natural pair ranking；
- natural dynamic support localization；
- local structural change。

应对整体平移稳健，但保留内部构型与 support information。

## 10.3 最小固定双通道

\[
z_{ctx}=P(E(X)),
\qquad
z_{mode}=P(E(X-\bar X)).
\]

共享 encoder，不先学习 AMR gate。context head 只读 z_ctx，intrinsic head 只读 z_mode；cross-readouts 用于 leakage audit。

该检验回答：

> 旧失败是表示无法保留两类信息，还是评价把两类语义错误绑在一起？

---

# 11. 修订后的实验与统计协议

## 11.1 Split

- SNGAR train 45：训练；
- valid 9：候选搜索；
- test 10：candidate lock 后一次性 run；
- IDSSE 7：external confirm；
- SoccerTrack parser smoke 与 final matches 分离；
- 旧 SkillCorner heldout 不再用于确认。

## 11.2 Candidate search

- 最多两轮；
- 每轮最多六个候选；
- 一次只改一个机制轴；
- 至少三个 seeds；
- primary unit = match；
- no test access；
- no metric/split changes；
- no architecture zoo。

## 11.3 Candidate lock

候选配置、数据 hash、任务、指标、seeds、failure rules 和 figure scripts 在 test 下载前提交并冻结。

## 11.4 P4 gate

进入 AMR-Fixed 需要：

1. 新比赛上 geometry prediction 复现；
2. context task noninferior；
3. intrinsic task improved；
4. z_mode translation robustness improved；
5. 多 match/seeds 一致；
6. external data direction preserved；
7. no leakage/capacity/role-oracle artifact；
8. 可给出固定、非后验 routing rule。

---

# 12. 动态语义 Support 的重新检验

当前 static role 不是动态战术真值。建议新增独立、非阻塞分析：

- pressing chain；
- simultaneous defensive engagement；
- off-ball-run / passing-option unit；
- event-window movement coherence。

每类 support 规则先注册、再人工小审计、再匹配 random support、最后计算 response。该任务可能得到三类结果：

1. dynamic support 比 static role 更稳定：说明此前 proxy 太粗；
2. dynamic support 仍 mixed：说明 organization effect 本身条件化；
3. dynamic support 无增量：进一步收缩 semantic-organization claim。

无论哪种都不应通过阈值搜索追求所有条件同号。

---

# 13. 投稿形态与创新性

## 13.1 相比最初理想方案

机会下降的部分：

- universal notch 不成立；
- 简洁的“组织盲”故事不成立；
- AMR/task repair 尚无证据；
- 跨域确认未完成。

## 13.2 相比 P2 刚收口

机会明显增加的部分：

- mixed effects 有了统一的条件化数学解释；
- retrospective + response-blind prospective；
- layer/block anatomy；
- matched-capacity causal shaping；
- 失败边界清晰、provenance 完整。

## 13.3 当前可投稿版本的主要 reviewer 风险

1. 用 Jacobian 预测局部 response 被认为近乎定义性；
2. 没有证明 geometry 与有意义任务的联系；
3. 同资产 prospective 不等于新数据泛化；
4. 当前只有小型点集模型；
5. 方法贡献缺失；
6. 书法尚未承担冻结预测；
7. 旧 heldout 暴露；
8. semantic coalition 使用 static role 太粗。

## 13.4 最有价值的补强

优先级：

1. 正确拆分 context/intrinsic tasks；
2. 独立比赛；
3. fixed dual-channel Pareto；
4. candidate-lock + test firewall；
5. IDSSE external；
6. dynamic semantic support；
7. 体育预测冻结后的 HCSU/Make Me a Hanzi 验证；
8. 成功后才 AMR-Fixed。

---

# 14. 建议 Figure 结构

## Figure 1：Equal energy, different support

- 原始阵型；
- exact-spectrum / vector-multiset control；
- 频率相同但 response 不同；
- 不再画 universal notch 作为既定事实。

## Figure 2：Action-Mode Probe 与 P2 条件异质性

- transfer curves；
- exact-spectrum organization beyond power；
- role/architecture conditional heatmap。

## Figure 3：Support-conditioned local geometry

- J、G block；
- baseline vs diagonal/off/full prediction；
- retrospective/prospective。

## Figure 4：Layer/architecture anatomy

- input → message passing → pooling；
- DeepSets vs GAT/Phase-GAT；
- diagonal/off-diagonal formation。

## Figure 5：Task semantics and fixed dual channel

- z_ctx / z_mode；
- context accessibility；
- intrinsic retrieval；
- mode robustness；
- cross-readout leakage。

## Figure 6：Independent confirmation / Pareto

- SNGAR valid/test；
- IDSSE external；
- match-level effects；
- candidate-lock protocol。

## Figure 7：Calligraphy，仅在成功后

- 体育端 frozen prediction；
- Make Me a Hanzi controlled support；
- HCSU natural style/structure；
- 命中或失败均完整报告。

---

# 15. 决策树

## Outcome A：双通道 + 独立数据成功

进入 P4 AMR-Fixed：

- fixed context/mode routing；
- frequency × support routing；
- no learned gate initially；
- 与 centering、relational pooling、canonicalization、CAP 比较。

## Outcome B：开发有效，test/external 失败

不进入 AMR。论文定位：

- strong diagnostic；
- local geometry anatomy；
- domain/selection boundary；
- 不声称 general repair。

## Outcome C：固定双通道开发阶段就失败

立即停止 autoresearch。保留：

- organization beyond power；
- support-conditioned local geometry；
- architecture anatomy；
- causal response shaping；
- negative task-repair result。

## Outcome D：dynamic support 成功但 task repair 失败

可增强语义分析，但不能冒充方法成功。论文仍以诊断/机制为主。

---

# 16. ICLR 时间约束

官方 Author Guidelines/Call for Papers 当前列出：

- abstract：2026-09-18 11:59 PM AOE；
- full paper：2026-09-25 11:59 PM AOE。

来源：

- `https://iclr.cc/Conferences/2027/AuthorGuidelines`
- `https://www.iclr.cc/Conferences/2027/CallForPapers`

建议压缩日程：

| 日期 | 必须完成 |
|---|---|
| Sep 2–4 | 文档迁移、数据申请/下载、test firewall、parser |
| Sep 4–7 | context/intrinsic task baseline、fixed dual-channel |
| Sep 7–10 | 有信号才进行 bounded autoresearch |
| Sep 10–12 | candidate lock + SNGAR test |
| Sep 12–14 | IDSSE external；P4 go/no-go |
| Sep 14–18 | figures、claim ledger、真实 abstract |
| Sep 18–25 | full paper、supplement、anonymity/repro audit |

科学 gate 不能因日期被放宽。无法闭合时宁可提交诚实的机制版本，也不要通过 test reuse 制造阳性。

---

# 17. 建议第三方评审重点回答的问题

1. `q=1/2||Jδ||²` 相对局部泰勒展开的定义性成分之外，当前结果还有多大独立科学新意？
2. exact-spectrum 与 endpoint-reallocation 是否足以证明 frequency is insufficient？还有哪些更强 alternative controls？
3. diagonal/off-diagonal 分解是否有更合适的 invariant parameterization？
4. layer-wise Spearman 上升是否可能由表示维数、归一化或 pooling scale 自动产生？
5. fixed dual-channel 是否只是普通 centering + multi-head，怎样才能成为 AMR 的必要前置而非方法贡献本身？
6. context/intrinsic tasks 是否真正对应不同 transformation semantics，还是人为构造？
7. natural pair ranking 的 target 是否过于由输入几何定义，导致 circularity？
8. dynamic support 的 weak labels 是否足够可信，人工审计最低要求是什么？
9. SNGAR/IDSSE/SoccerTrack 的 provider/domain shift 是否足以支撑一般化？
10. 若 task repair 失败，诊断/机制论文是否仍达到 ICLR 的 importance 门槛？需要补什么理论或基准？
11. 书法作为跨域确认能否真正提升 generality，还是增加复杂度？
12. 当前主 claim 应该是 Action-Mode Spectrum、support-conditioned geometry，还是 task-conditioned transformation semantics？
13. 哪些相关工作最可能一票否决 novelty：equivariance、local invariance、Jacobian geometry、GNN spectral bias、relational pooling？
14. 当前 9 个小模型是否足够；最少还需要哪一类现代结构模型？
15. 哪些结果必须进入主文，哪些应降为附录以控制九页叙事？

---

# 18. 建议允许的论文级表述

## 当前已经允许

> Under matched spectral power or an identical displacement-vector multiset, reallocating perturbation support changes the responses of frozen structured encoders. A support-conditioned local metric derived from the normalized-embedding Jacobian predicts these response differences substantially better than frequency and matching-residual baselines, with architecture-specific diagonal and cross-node contributions.

> A matched-capacity pooling intervention changes the predicted local geometry and finite response, establishing representation shaping within Phase-GAT, but does not yet establish task-level repair.

## 当前不允许

- all models exhibit a universal mid-frequency notch；
- models are blind to organization；
- role-defined coalitions are universally special；
- the model understands tactics；
- nonlinear decoding recovers semantic organization；
- AMR improves downstream tasks；
- sports findings generalize to calligraphy；
- prospective evidence is new-match validation；
- old heldout is confirmatory。

## T5R 成功后可能允许

> Separating context-preserving and intrinsic-structure readouts resolves the apparent conflict between deployment prediction and global-translation robustness, producing a match-consistent Pareto improvement that transfers to held-out games and an external tracking source.

此句必须由未见数据支持，不能提前写入 abstract。

---

# 19. 总体判断

## 科学上

项目没有崩溃，也不是简单失败。它从一个过强、戏剧化但不成立的 universal blindness 假设，转向了一个更准确的条件化机制：

> 结构化表示对“谁承担同样变化”的响应由节点敏感度、跨节点耦合、支持分配和网络聚合共同决定；一维频率摘要不足以描述这一现象。

这是实际发现，而不是为了保住项目发明的新词。

## 方法上

当前仅证明 response shaping，尚无 task repair。旧 T5 设计存在语义冲突，故值得进行一次 protocol-corrective follow-up；但必须有限、独立数据、test firewall。继续无限搜索不合理。

## 投稿上

- 相比最初理想方案：简洁性和方法闭环下降；
- 相比 P2 收口时：机制解释、证据纪律和可发表性明显提高；
- 当前直接投稿：更像扎实但 importance/utility 不足的机制分析；
- 若 T5R 在独立数据闭合：可重新形成“发现—解释—修复”的 ICLR 主线；
- 若 T5R 失败：应停止 AMR，集中强化理论边界、跨模型/跨数据验证和诊断工具。

本报告的核心建议是：

\[
\boxed{
\text{最后一次有界 P3 修复，重点修任务语义和独立数据，而不是继续搜 pooling。}
}
\]

---

# 20. 证据路径索引

仓库内优先检查：

- `STATUS.md`
- `DECISIONS.md`
- `CLAIM_LEDGER.md`
- `reports/EXPLORATION_CHECKPOINT_1.md`
- `reports/EXPLORATION_CHECKPOINT_2.md`
- `reports/EXPLORATION_CHECKPOINT_2_FRACTURE.md`
- `reports/P2_HETEROGENEITY_DIAGNOSIS.md`
- `reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md`
- `reports/P3_CANDIDATE_SEARCH_REPORT.md`
- `configs/phase3_support_geometry_v1.yaml`
- `artifacts/phase3/support_geometry_v1/`
- `artifacts/phase3/support_geometry_prospective_v1/`
- `artifacts/phase3/selected_causal_switch_v1/`
- `artifacts/phase3/candidate_search_v1/`
- `scripts/p3_support_geometry.py`
- `scripts/p3_causal_switch.py`
- `scripts/p3_candidate_search.py`
- `tests/test_p3_support_geometry.py`

外部数据：

- SNGAR tracking：`https://huggingface.co/datasets/OpenSportsLab/SNGAR-Action-Spotting-Tracking`
- SoccerNet-GAR：`https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR`
- SkillCorner：`https://github.com/SkillCorner/opendata`
- IDSSE：`https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177`
- SoccerTrack v2：`https://atomscott.github.io/SoccerTrack-v2/`
- HCSU：`https://huggingface.co/datasets/Tongji209/HCSU`
- MCCD：`https://github.com/SCUT-DLVCLab/MCCD`
- Make Me a Hanzi：`https://github.com/skishore/makemeahanzi`
