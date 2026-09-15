# Claim Ledger

更新时间：2026-09-05

| Claim ID | Claim | Evidence status | Allowed wording / boundary | Next evidence |
|---|---|---|---|---|
| P2-C1 | exact-spectrum 下普通谱功率不能解释全部表示响应 | SUPPORTED_P2 | 仅指 P2 测量对象和冻结资产 | 无需重跑；保留 P2 provenance |
| P2-C2 | fracture response 在部分条件跨比赛稳定，但方向依赖 architecture、graph、role、energy | SUPPORTED_P2 | 总结为 mixed_or_graph_specific | P3 条件化预测 |
| P2-C3 | role alliance 普遍更特殊 | NOT_SUPPORTED | 不得写成普遍机制 | 若研究，仅作诊断分层 |
| P2-C4 | 统一 endpoint-organization 因果机制 | NOT_SUPPORTED | 不得外推为统一理论 | P3 block/layer 和 prospective |
| P2-C5 | P2 response 改善下游任务 | NOT_TESTED | P2 不是任务性能 | P3-T5 客观任务 |
| P3-G1 | local geometry 可跨 match 预测 fracture/pair response | HISTORICAL_PRE_RESULT_HYPOTHESIS | 结果前假设，已由完成证据 P3-C1 替代；本行保留为历史，不得当作已验证 | T1/T2 grouped evaluation → P3-C1 |
| P3-G2 | diagonal/off-diagonal block 可解释异质性 | HISTORICAL_PRE_RESULT_HYPOTHESIS | 结果前假设，已由完成证据 P3-C3 替代；本行保留为历史 | T3 block decomposition → P3-C3 |
| P3-G3 | 异质性可定位到特定网络阶段 | HISTORICAL_PRE_RESULT_HYPOTHESIS | 结果前假设，已由完成证据 P3-C4 替代；本行保留为历史 | T3 layer-wise → P3-C4 |
| P3-G4 | 一个 matched-capacity 开关可改变 geometry、response、任务 | HISTORICAL_PRE_RESULT_HYPOTHESIS | 结果前假设，已由完成证据 P3-C5/C6 替代（仅 response shaping）；本行保留为历史 | T4 → P3-C5/P3-C6 |
| P3-G5 | geometry-task alignment 在旧 conflated task gate 下带来任务 Pareto 改善 | NOT_SUPPORTED_UNDER_OLD_CONFLATED_TASK_GATE | 仅否定旧任务定义下的修复；response 大小本身无方向意义 | P3-R2 |
| P4-C1 | AMR-Fixed 适合作为机制驱动方法 | BLOCKED_BY_P3_GATE | 只有 P3 强/条件支持后才可进入 | P3 gate |
| Domain-C1 | 书法结果可选择体育端机制 | FORBIDDEN | 书法 exploratory，不用于本轮选择 | 未来独立冻结预测后再评估 |

## P3 实际证据回写（2026-09-01）

| Claim ID | Claim | Evidence status | Allowed wording / boundary | Evidence |
|---|---|---|---|---|
| P3-C1 | normalized-embedding local geometry 可预测 P2 fracture pair effect | SUPPORTED_CONDITIONALLY | 在当前 250 samples、10 source matches、9 frozen models 及同资产 prospective draw 内成立；不是新比赛泛化 | `artifacts/phase3/support_geometry_v1/`、`artifacts/phase3/support_geometry_prospective_v1/` |
| P3-C2 | full geometry 超过频率/residual 低维基线 | SUPPORTED | retrospective mean Spearman 0.8466 vs baseline 0.1222；prospective 0.8526 vs 0.1294；按 match grouped | `prediction_metrics.parquet` |
| P3-C3 | diagonal geometry 有主要贡献，off-diagonal 提供额外信息 | SUPPORTED_CONDITIONALLY | 不能写成所有架构的纯 pairwise 因果；off-diagonal 在 DeepSets pooling 后、GAT/Phase-GAT message passing 中呈不同形成路径 | T3 layer/block tables |
| P3-C4 | 异质性可定位到网络阶段 | SUPPORTED_CONDITIONALLY | 输入→message passing→pooling 的可比层中，prediction 与 block composition 改变；不是唯一生物/战术解释 | T3 artifacts |
| P3-C5 | pooling accessibility 开关可改变几何和 response | SUPPORTED | Phase-GAT、3 seeds、matched parameter count 149,639；relational pairwise 改变 off-diagonal/full 比例和 prospective response | `selected_causal_switch_v1/switch_seed_summary.parquet` |
| P3-C6 | 开关修复了任务—鲁棒性 Pareto | NOT_SUPPORTED | 只能称 representation shaping；heldout macro-F1 均值略降，context response 三 seed 均增加 | T4/T5 seed summary |
| P3-C7 | AMR-Fixed 现在有非后验任务修复目标 | BLOCKED | P3 gate 未闭合；不启动 AMR、不扩大 causal matrix | D-20260901-P3-008 |
| P3-C8 | nodewise geometry 能有效定位 controlled support | WEAK_NOT_SUPPORTED_AS_TASK_GAIN | top-k recall relational 0.1546、team 0.1487，uniform 0.1458；只保留诊断性弱增益 | `node_sensitivity_localization_v2_summary.parquet` |

## 证据状态规则

SUPPORTED_P2 只来自既有 P2 artifacts/reports；WORKING_HYPOTHESIS 是 post-P2 mechanistic follow-up，不是预注册确认；NOT_RUN、NOT_TESTED、BLOCKED 不得用“趋势一致”替代。每次结果回写都必须同时记录 match-level 单位、失败条件、prospective 状态和 artifact provenance。

## P3-T5R 新增 claim（2026-09-02）

| Claim ID | Claim | Evidence status | Allowed wording / boundary | Next evidence |
|---|---|---|---|---|
| P3-R1 | 旧 T5 gate 混淆了 context signal 与 intrinsic nuisance | DESIGN_DIAGNOSIS_SUPPORTED | 仅解释旧任务设计，不等于已证明修复有效 | T5R-2/T5R-3 |
| P3-R2 | 固定 context/mode 双通道能改善正确任务 Pareto | SUPPORTED_ON_IDSSE_UNSEEN_MATCH_CONDITIONALLY | T5R5 单次 J03WQQ 确认：任务 gate 全过（pair −0.0026、geometry +0.0051、context 无退化）；边界 = 单一同源未见 match，不是独立 provider 确认 | T5R-6 |
| P3-R3 | geometry-response 关系可推广到未见比赛 | SUPPORTED_ON_INDEPENDENT_PROVIDER_CONDITIONALLY | T5R6：独立 provider（SoccerTrack-v2 大学联赛）8 场比赛同方向；边界 = 单一 provider，不是跨域泛化 | P4 decision |
| P3-R4 | dynamic semantic support 相对 matched random 有稳定增量 | NOT_ESTABLISHED | support construction 必须 response-blind 并经审计 | dynamic-support task |
| P3-R5 | P3-T5R 可为 AMR-Fixed 提供非后验目标 | GATE_CONDITIONS_MET_PENDING_P4_DECISION | 任务—鲁棒性—外部确认门已闭合（T5R6 CONFIRMED）；是否启动 P4 由事项 8 决定，不自动放行 | P4 decision |
| DATA-C1 | 旧 heldout 可用于最终确认 | FORBIDDEN | 只允许 exploratory audit；不得选择候选 | SNGAR test lock |
| P3-R6 | intrinsic objective 必须直接塑造 shared encoder 而非只训练末端 mode head | SUPPORTED_AS_ROUND1_MECHANISTIC_CONTROL | 边界 = IDSSE 开发集 / 当前固定结构；来自 Round 1 `routing_mode_head_only` 负对照 | T5R-5 candidate lock |
| P3-R7 | shared-encoder update allocation 实质影响 task–geometry Pareto（2:1 入选） | SUPPORTED_CONDITIONALLY_ON_IDSSE_UNSEEN_MATCH | 开发集方向在 J03WQQ 保持：任务保持＋干预 full 3 seeds 高于同 seed reference；不是通用最优 ratio 结论 | T5R-6 |
| P3-R8 | 归一化局部几何可预测未见 IDSSE 源比赛上的受控 support-reallocation 响应 | SUPPORTED_ON_SINGLE_UNSEEN_MATCH | 边界 = 一个未见源比赛、固定家族、冻结干预（eps 0.25、support 4）；2:1 mean full 0.63 vs 基线 ~0，不是独立 provider 确认 | T5R-6 |
| P3-R9 | 修复存在选择性 context 代价：辅助 match-half 探针 0/8 场反向（−30%） | SUPPORTED_AS_BOUNDARY_ON_REPAIR | 冻结 addendum 将该探针定位为 auxiliary（测 z_ctx 绝对上下文保留）；四项主指标 8/8 占优不受其影响；论文必须如实并列表述，用于界定“context 无退化”的边界——空间上下文改善、赛时上下文退化 | P4 decision |
| P3-R10 | 局部几何→响应排序预测存在有界有效域（ε≲0.25），域外按理论预测方式失效 | SUPPORTED_ON_DEV | ε-sweep（57 组、12 模型）：2:1 ρ 0.83→0.56→−0.001；相对误差 0.26→0.73；ε=1.0 协议合法性崩（11/57）；冻结工作点 0.25 在域内近边界；dev 证据，非外部确认 | 论文写作 |
| P3-R11 | support 敏感的响应现象是训练诱导的 | SUPPORTED_ON_DEV | 未训练对照 ΔR 量级 ~1e-6 vs 训练后 ~1e-1（差 5 个数量级）；未训练处于平凡线性域（q/a≈1.0），训练将映射推出平凡域；未训练 Spearman 绝对值不做过度解读（近 float32 精度边缘） | 论文写作 |
| P3-R12 | 几何预测的信息载体随联盟粒度从对角项向耦合项迁移 | SUPPORTED_ON_DEV | support 伸缩 v2（4 档×60 快照×6 模型）：full−diag 差距 0→+0.2~0.27（两族同向）；s=1 时 full≡diag（构造一致性）；“谁和谁一起动”的语义在 off-diagonal block 中且随联盟规模递增 | 论文写作 |
| P3-R13 | support 敏感与几何→响应关系跨 SSL 目标族复现 | SUPPORTED_ON_DEV_CROSS_OBJECTIVE | Barlow-Twins 对照（冻结 2:1 配方仅换损失）：干预 full ρ 0.47–0.67（均值 0.59≈对比族 0.56），full＞diag 三 seed 一致；边界 = 同架构同视图，架构通用性（EGNN/SetTransformer）未测 | 论文写作 |
| P3-R14 | Jacobian 几何超越无模型基线但幅度有限；谱基线失效对 k 稳健 | SUPPORTED_ON_DEV | 同集头对头：full 0.49 ＞ gyration 0.35 ＞ 谱基线 0.11（逐 seed 一致）；谱基线在 k=2/4/6/8 下 ρ∈[−0.05,0.19] 全近零；不许声称“只有微分几何能预测”——简单队形统计携带约一半信号 | 论文写作 |

| P3-R15 | update-allocation 曲线（dev 事后消融）：pair 随 intrinsic 配额单调升（0.804→0.944→0.951→0.967），zone 单调降、centroid 总体升，几何代理 ≥1:1 平台化（~0.70）；无占优配比点，2:1 为膝点；1:0 在随机 mode_head 读出下仍保留 pair 0.80/干预 ρ 0.53（结构在池化特征）；干预预测对所有已训配比存在 | SUPPORTED_DEV_ONLY | 配比扫掠 1:0/1:1（本节点）＋T5R4 Round-2 冻结值 | `artifacts/phase3/ratio_sweep_v1/`、`reports/P3_RATIO_SWEEP_REPORT.md` | dev-only，NOT selection；1:0 行带随机读出警示 |

| P4-A1 | 朴素谱带不变性压力（无防护实现）导致模态通道塌缩：3/3 seeds pair=0、干预退化；根因=路由损失尺度失衡＋塌缩同时满足未归一化 L_inv/L_eqv＋防塌缩不足 | SUPPORTED_DEV_ONLY（负结果） | M1 v1 训练（锁 v1） | `artifacts/phase4_amr/m1_v1/`、`reports/P4_N3_V1_COLLAPSE_REPORT.md` | dev-only；判决的是无防护实现，不是路由假设本身 |

| P4-A2 | 共享主干上的学习型不变性压力（v1 带尺度失衡、v2 归一化修复后）均导致系统性塌缩：v1 毁模态通道，v2 seed 11 经共享编码器毁 context 通道（z_ctx 平移响应 1.0）；机制：不变性目标在共享主干上的最小阻力路径是通道/主干塌缩 | SUPPORTED_DEV_ONLY（负结果） | M1 v2 训练（锁 v2） | `artifacts/phase4_amr/m1_v2/`、`reports/P4_N3_V2_FAILURE_AND_REDESIGN.md` | dev-only；与 P4-A1 合并构成"不变性压力需结构性防护"的证据 |

| P4-A3 | 移除全部不变性压力后（α 全零）塌缩依然复现 3/3：不变性压力不是必要破坏因子；残存驱动为对称路由梯度＋中间层 floor 旁路＋三元组对称不动点（与 Track B 文献诊断一致） | SUPPORTED_DEV_ONLY（负结果＋隔离结论） | M1 v3 训练（锁 v3） | `artifacts/phase4_amr/m1_v3/`、`reports/P4_N3_V3_FAILURE_AND_STOP.md` | dev-only；证伪 v2→v3 假设；支撑重设计方向 |

| P4-A4 | v4（精确谱＋Slepian＋最终嵌入 VICReg＋stop-grad）：模态通道 9 个失败种子后首次 3/3 存活（pair 0.63–0.86，干预 full ρ 0.82–0.99 为全项目 dev 最强，奇异谱铺展无零秩）；但 W_ROUTE=12 下路由赢得共享主干，context 通道 2/3 塌缩（zone 0.11–0.13，z_ctx 响应≈0）＋1/3 侵蚀（zone 0.57）——共享主干竞争的反向实现，H1 未通过 | MIXED_DEV（机制有效＋分配失败） | M1 v4 训练（锁 v4） | `artifacts/phase4_amr/m1_v4/`、`reports/P4_N4_V4_RESULTS.md` | dev-only；触发 Route-3 备用 v4b |
