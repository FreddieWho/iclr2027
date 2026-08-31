# Claim Ledger

更新时间：2026-09-01

| Claim ID | Claim | Evidence status | Allowed wording / boundary | Next evidence |
|---|---|---|---|---|
| P2-C1 | exact-spectrum 下普通谱功率不能解释全部表示响应 | SUPPORTED_P2 | 仅指 P2 测量对象和冻结资产 | 无需重跑；保留 P2 provenance |
| P2-C2 | fracture response 在部分条件跨比赛稳定，但方向依赖 architecture、graph、role、energy | SUPPORTED_P2 | 总结为 mixed_or_graph_specific | P3 条件化预测 |
| P2-C3 | role alliance 普遍更特殊 | NOT_SUPPORTED | 不得写成普遍机制 | 若研究，仅作诊断分层 |
| P2-C4 | 统一 endpoint-organization 因果机制 | NOT_SUPPORTED | 不得外推为统一理论 | P3 block/layer 和 prospective |
| P2-C5 | P2 response 改善下游任务 | NOT_TESTED | P2 不是任务性能 | P3-T5 客观任务 |
| P3-G1 | local geometry 可跨 match 预测 fracture/pair response | WORKING_HYPOTHESIS | 不得当作已验证 | T1/T2 grouped evaluation |
| P3-G2 | diagonal/off-diagonal block 可解释异质性 | WORKING_HYPOTHESIS | 若 off-diagonal 无增益，收缩为 diagonal claim | T3 block decomposition |
| P3-G3 | 异质性可定位到特定网络阶段 | WORKING_HYPOTHESIS | 不预设 pooling 或 message passing | T3 layer-wise |
| P3-G4 | 一个 matched-capacity 开关可改变 geometry、response、任务 | WORKING_HYPOTHESIS | 仅一条 evidence-selected route | T4，未通过则不做 AMR |
| P3-G5 | geometry-task alignment 带来任务 Pareto 改善 | WORKING_HYPOTHESIS | response 大小本身无方向意义 | T5 |
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
