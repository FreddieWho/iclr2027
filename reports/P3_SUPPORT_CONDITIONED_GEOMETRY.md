# P3 Support-Conditioned Local Geometry

状态：T4_T5_COMPLETE_RESPONSE_SHAPING_ONLY_P4_GATE_NOT_MET
阶段：P3_CAUSAL_MECHANISM  
性质：post-P2 mechanistic follow-up，exploratory

## 当前结论边界

P2 的唯一冻结 route 为 mixed_or_graph_specific。本报告不重算、不覆盖 P2，也不把 P2 representation response 写成任务改善。P3 的待检验对象是归一化 embedding 的局部几何：

\[
q_f(X,\delta)=\frac12\|J_f(X)\delta\|_2^2,\qquad
G_f(X)=J_f(X)^\top J_f(X).
\]

Action-Mode Spectrum 继续作为该条件化对象的边缘汇总。T0–T5 已完成；P3-G1–G3 获得条件性支持，P3-G4 只支持 geometry/response shaping，P3-G5 未形成任务—鲁棒性方法收益。P4 AMR gate 未满足。

## 冻结施工路线

1. T0：复现 250 samples × 9 frozen models 的 P2 baseline embedding，暴露节点/interaction/pooling 层并验证 JVP、有限差分和二阶近似。
2. T1：在 P2 fracture response 上，以 match 为统计单位比较频率/residual/support 基线、diagonal geometry、full geometry 和 off-diagonal geometry。
3. T2：冻结 T1 公式、层、指标和统计单位后，生成 response-blind 的同资产 prospective reallocations。
4. T3：分解 diagonal/off-diagonal、team/support/edge block，并定位信息形成或丢失的层。
5. T4：只选择一条有证据支持的 matched-capacity causal switch。
6. T5：在方法结果前冻结任务定义，完成 perturbation localization 和一个非 intervention-label 的客观任务。

完整 causal matrix 只是 gate 后 optional extension；P4 AMR 在 gate 前不运行。

## 结果记录模板

正式运行时，本报告需回写：

- retrospective 与 prospective 分开；
- match-level Spearman、MAE、R²、方向准确率、calibration、效应量和 bootstrap 区间；
- architecture、seed、graph、role、epsilon 的失败条件；
- geometry 相对于最佳 frequency/residual baseline 的增量；
- layer/block 定位、因果开关的预测—操作—几何—任务链；
- manifest、receipt、SHA-256、figure source tables 和 known limitations。

若 T1/T2 不支持预测，或 T4 只有 response 无任务收益，必须在本报告保留失败结果并降低 claim；不得补选干预、扩大模型或直接训练 AMR。

## P3 实际执行与证据（2026-09-01）

### 冻结资产和 provenance

所有任务复用同一批 250 个 canonical samples、10 个 source matches 和 9 个冻结模型；没有新数据域、没有修改 P2 artifact，也没有访问或修改 `infra/bioinf-data-index/`。模型为 DeepSets、GAT-small 和 Phase-GAT，各含 seed 11、23、47。环境为 `/opt/anaconda3/bin/python`，Python 3.11.5、PyTorch 2.11.0、NumPy 1.26.4、pandas 2.3.3、SciPy 1.13.1、PyArrow 22.0.0、PyYAML 6.0.1、scikit-learn 1.5.1；默认 dtype 为 float32、device 为 CPU。

P2 freeze 的 Git HEAD 为 `f44be1e9526796a76b58ca2bd44dc562ab7841a3`；文档迁移提交为 `10d0711`。输入、9 个 checkpoint、P2 manifest/hash 以及实现源码 hash 锁定在 `configs/phase3_support_geometry_v1.yaml`。当前实现 hash 为 `scripts/p3_support_geometry.py`=`66e5e9cd3353870eb4499ad4cb86951964cee8b7a3f92dbf88da862d3aa9431f`，causal switch hash 为 `scripts/p3_causal_switch.py`=`d50dcc947562333c0c860d33a6055d417da78d02d88ce5ced4a5ba1bf9220514`。

P2 唯一权威输出仍是 `artifacts/phase2/p2_fracture_continuity_v1/`；P2-H1 diagnosis 为 `artifacts/phase2/p2_heterogeneity_diagnosis_v1/`。这些报告、manifest、checksum 和历史 claim 未被重写。

### T0：接口、数值等价与 immutability

`artifacts/phase3/support_geometry_v1/t0_summary.json` 记录 250×9 的 baseline embedding parity，9/9 PASS，embedding shape 为 250×128，最大绝对误差为 0。JVP 与 autograd Jacobian 的相对误差约为 `1.8e-7–2.5e-7`；中心有限差分在 `h=5e-4` 下误差约为 `4.5e-4–3.2e-3`。归一化 embedding 的二阶 cosine 近似在小 epsilon 下达到 float32 数值舍入地板；没有把该地板误读为科学效应。checkpoint 和冻结输入的前后 hash 一致。T0 receipt、source provenance 和 SHA-256 位于同一 artifact 目录。

### T1：retrospective grouped prediction

以 `source_match_id` 为分组单位，9 个模型共计算 272,169 条 arm rows 和 121,293 条 pair rows；预测目标是 observed anchor-control pair effect。没有使用大型可学习预测器，比较频率、Rayleigh、support、metadata/residual 低自由度基线与 diagonal/full/off-diagonal geometry。leave-one-source-match-out 和 match-level bootstrap 均已生成。

| predictor | mean Spearman | mean MAE | mean R² | direction accuracy |
|---|---:|---:|---:|---:|
| frequency/residual baseline | 0.1222 | 0.000279 | 0.0145 | 0.5398 |
| diagonal geometry | 0.6528 | 0.000182 | 0.4754 | 0.7248 |
| off-diagonal geometry | 0.3886 | 0.000242 | 0.2760 | 0.6072 |
| full geometry | 0.8466 | 0.000114 | 0.7495 | 0.8109 |

Full geometry 在 9 个模型上均为较强的 response predictor；GAT-small 的单模型 Spearman 约为 0.934–0.949，Phase-GAT 约为 0.810–0.930，DeepSets 约为 0.698–0.743。match bootstrap 的 n=10 较小，DeepSets 个别 seed 的 full-vs-baseline 区间包含 0，因此结论是“跨模型的预测增益和有限条件稳定”，不是对每个架构/seed 的普遍显著性宣称。

### T2：prospective 同资产验证

T1 的公式、pooled embedding 层、预测器、指标、分组单位和失败分析先写入 `t1_prediction_lock.json`，随后 response-blind 生成一次新 draw（seed `20260901`）。生成器没有读取 response path，也没有按预测正确与否筛选 intervention。5,700 个 intervention sets 中 4,475 组完整；9 个模型产生 90,432 条 arm rows 和 40,275 条 pair rows。

| predictor | mean Spearman | mean MAE | mean R² | direction accuracy |
|---|---:|---:|---:|---:|
| frequency/residual baseline | 0.1294 | 0.000262 | 0.0139 | 0.5449 |
| diagonal geometry | 0.6580 | 0.000175 | 0.4584 | 0.7224 |
| off-diagonal geometry | 0.3815 | 0.000236 | 0.2224 | 0.6026 |
| full geometry | 0.8526 | 0.000109 | 0.7416 | 0.8113 |

这支持“冻结的局部几何在同资产的新 support reallocation 上具有 prospective 预测力”。它不是新比赛或新数据域泛化；该边界保留在 claim ledger 中。

### T3：block decomposition 和 layer-wise 定位

整体 mean absolute block contribution（按模型和层平均）如下：

| layer | full | diagonal | off-diagonal |
|---|---:|---:|---:|
| input_node | 1.155e-4 | 1.149e-4 | 8.19e-6 |
| message_passing_1 | 2.070e-4 | 1.896e-4 | 9.41e-5 |
| message_passing_2 | 2.842e-4 | 2.539e-4 | 1.615e-4 |
| pooling_pre | 8.755e-5 | 7.710e-5 | 5.334e-5 |
| pooled_embedding | 8.409e-5 | 7.475e-5 | 5.256e-5 |

所有 block decomposition identity residual 均在约 `1e-18` 数值量级。geometry→response 的整体 Spearman 由 input `0.2125`、message passing 1 `0.4420`、message passing 2 `0.5740`、pooling-pre `0.7578` 增至 pooled `0.8407`。DeepSets 的 off-diagonal 主要在 pooling 后出现；GAT/Phase-GAT 的 off-diagonal 在 message passing 中增加，并在 pooling 后保留。因此 T3 支持“节点敏感度是主要量、跨节点耦合提供额外且架构依赖的信息”，不能压缩成单一 pooling 或单一 pairwise 机制。

### T4–T5：唯一因果开关和任务边界

依据 T3，只选择 `pooling_accessibility` 路线，限于 Phase-GAT，比较无新增参数的 `team_mean` 与 `relational_pairwise` pooling。3 个 seed 的参数量均为 149,639；每个变体训练 80 epochs、learning rate `0.001`。seed-resolved 结果如下：

| pooling | mean abs observed pair effect | mean abs delta q_full | off/full abs ratio | geometry-response rho | heldout accuracy | heldout macro-F1 | context response |
|---|---:|---:|---:|---:|---:|---:|---:|
| team_mean | 0.002935 | 0.001479 | 0.4929 | 0.7383 | 0.3542 | 0.1861 | 0.01324 |
| relational_pairwise | 0.002066 | 0.000883 | 0.5792 | 0.8453 | 0.3854 | 0.1839 | 0.01571 |

relational pooling 在三个 seed 均提高 off-diagonal/full 比例、提高 geometry-response Spearman，并降低 prospective pair response 与 `delta q` 的绝对均值；这满足 geometry 和 response 被开关改变。但 context response 三个 seed 均上升，heldout macro-F1 没有稳定改善（均值略降 0.1861→0.1839）。heldout accuracy 均值上升不足以抵消 macro-F1 与 robustness 失败，且 heldout 只有 32 个带标签样本。因此该结果只能称为 **representation shaping**，不能称为 repair、task improvement 或 AMR 机制成功。

T5 的 controlled-intervention localization 进一步用六个 switch checkpoint 的 nodewise Jacobian sensitivity 做 top-k support recall，k 等于声明的 support size，并以 `k/20` 为 uniform baseline；计算时明确没有读取 response 列，q-full 重算与 switch 表的最大绝对误差为 `8.9e-9`。relational pairwise 的 mean recall 为 `0.1546`，team mean 为 `0.1487`，uniform baseline 为 `0.1458`。该增益很小，不能作为有用 localization task 或 task repair 的证据；它只完成了受控支持区域的诊断性定位检查。原始 support-mass sanity 表因 support 与 delta 共定义而保留，但不作为结果。

### P3 假设状态和路由

- P3-G1：`SUPPORTED_CONDITIONALLY`。full local geometry 在 retrospective 和 prospective 同资产评估中超过频率/residual 基线；外推边界为当前资产和 10 个 source matches。
- P3-G2：`SUPPORTED_CONDITIONALLY`。diagonal 是主要贡献，off-diagonal 有额外信息但依赖架构和层；不宣称统一关系因果。
- P3-G3：`SUPPORTED_CONDITIONALLY`。message passing 与 pooling 的形成/保留模式可定位；不是唯一机制解释。
- P3-G4：`PARTIAL / RESPONSE_SHAPING_ONLY`。开关改变 geometry 和 response，但没有闭合客观任务与 context robustness 条件。
- P3-G5：`NOT_SUPPORTED_AS_METHOD_REPAIR`。nodewise support localization 只有接近 uniform 的弱增益，phase-label macro-F1 没有稳定改善，且没有 robustness–structure Pareto 改善。

因此 P3 的 T0–T5 施工与诊断均已执行，但强机制 gate 未闭合。P4 AMR-Fixed、AMR-Learned 和完整 causal matrix 均不启动；保留 Action-Mode Spectrum 作为条件化几何的边缘汇总，保留 `mixed_or_graph_specific` 作为 P2 的准确结果。

## 最终验收问题

1. **是否能跨比赛预测？** 在当前 10 个 source matches 的 grouped retrospective 与同资产 prospective draw 上能；不是新比赛泛化结论。
2. **支持分配增加多少信息？** 相对 baseline，retrospective Spearman 从 0.1222 增至 0.8466，prospective 从 0.1294 增至 0.8526；具体解释由 geometry 而非大预测器提供。
3. **主要机制在哪里？** 节点对角敏感度为主；off-diagonal 在 message passing 或 pooling 中按架构形成/保留，不能选出单一普适阶段。
4. **prospective 是否支持？** 支持冻结解释在同资产新 reallocation 上复现；边界不是独立比赛。
5. **因果开关是否同时改变任务？** 改变 geometry/response；controlled support localization 仅接近 uniform，phase-label macro-F1 没有稳定改善，context robustness 变差；答案是否定。
6. **Spectrum 如何继续成立？** 它是对 support、relationship 和 task 条件化响应的边缘汇总，不要求所有条件同号。
7. **AMR-Fixed 是否有明确目标？** 本轮没有非后验、兼顾 robustness 的修复目标；P4 blocked。
8. **哪些 claim 可进入文章？** 可写 P2 mixed heterogeneity、局部 geometry 的条件性预测、架构依赖的 block/layer anatomy 和 response-shaping boundary；support localization 的弱/近-uniform 结果、统一角色/endpoint 因果、任务修复和跨域迁移必须留在 limitation。
9. **canonical docs 是否同步？** 已同步 README、MASTER prompt、四个科学/施工文档、两个配置、QA、STATUS、DECISIONS、CLAIM_LEDGER 和 consolidated package；P2 历史文件只读。
10. **是否进入 P4？** 否。停止点是“geometry prediction + response shaping 已有证据，但任务—鲁棒性 gate 未闭合”；继续堆复杂度不具有当前证据支持。

## Artifact 索引

- T0/T1/T3：`artifacts/phase3/support_geometry_v1/`
- T2：`artifacts/phase3/support_geometry_prospective_v1/`
- T4/T5：`artifacts/phase3/selected_causal_switch_v1/`
- T5 localization：`node_sensitivity_localization_v2.parquet`、`node_sensitivity_localization_v2_summary.parquet` 及同名 receipt。
- 每个目录的 `receipt.json`、`SHA256SUMS`、`known_limitations.md` 与 machine-readable summary 是对应结果的权威证据；Figure source tables 见各目录中的 parquet 以及 `figure_data/README.md`。
