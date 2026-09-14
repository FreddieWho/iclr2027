# 证据映射：手稿每一节的数字与措辞规则 → 冻结 artifact

手稿纪律：正文里每个数字必须能在此表找到唯一来源；措辞规则（WORDING RULE）
来自评审第四部分，写作时不可绕过。ledger ID 见 `CLAIM_LEDGER.md`。

| 章节 | 主张/数字 | 来源 artifact | ledger | 措辞规则 |
|---|---|---|---|---|
| 摘要/4 | 几何→响应 Spearman 0.5–0.85（跨 dev/保留场/外部） | `artifacts/phase3/support_geometry_v1/`、`support_geometry_prospective_v1/`、`t5r5_reserved_j03wqq_v1/`、`t5r6_soccertrack_v1/` | P3-C1/C2, R8, R3 | 给范围不给单点；标明有效域 |
| 4.1 | ε 有效域：ρ 0.83→0.56→−0.00；rel-err 0.26→0.73；ε=1 合法域崩 | `artifacts/phase3/epsilon_sweep_v1/` | P3-R10 | "有界有效域内的排序预测"；工作点近边界须说明 |
| 3 | 现象训练诱导：ΔR 差 5 个数量级 | `artifacts/phase3/epsilon_sweep_v1/summary.json` | P3-R11 | 未训练 Spearman 不解读绝对值（float32 边缘） |
| 3 | 跨目标族：BT full ρ 0.47–0.67（均值 0.59） | `artifacts/phase3/ssl_family_control_v1/` | P3-R13 | 边界=同架构；架构通用性进 Limitations |
| 4.2 | 头对头：full 0.49＞gyration 0.35＞谱 0.11；k∈{2,4,6,8} 谱全近零 | `artifacts/phase3/predictor_baselines_v1/` | P3-R14 | 禁"只有我们能预测"；说"超越最强无模型统计" |
| 4.3 | full−diag 随联盟 0→+0.2~0.27；s=1 时 full≡diag | `artifacts/phase3/support_scaling_v2/` | P3-R12 | "耦合项承载联盟语义" |
| 4.4 | 因果开关仅 response shaping；任务 Pareto 未修 | `selected_causal_switch_v1/`、T4/T5 | P3-C5/C6 | NOT_SUPPORTED 必须引为 Sec.5 动机 |
| 5.3 | 单次保留场：gate 全过；干预 0.63 vs ~0 | `t5r5_reserved_j03wqq_v1/` | P3-R2/R8 | z_mode 平移=构造保证＋sanity check，禁写成学到的性质 |
| 5.4/6.2 | match-half 0/8 反向（−30%） | `t5r6_soccertrack_v1/`、`reports/P3_HEADLINE_EFFECT_TABLE.md` | P3-R9 | 与正向结果并列呈现；界定 context 保持边界 |
| 6 | 外部 8 场：四主指标 8/8（CI 见 headline 表）；干预 8/8（+0.53） | `t5r6_soccertrack_v1/`、`artifacts/data_v2/soccertrack/` | P3-R3 | "小效应、全同向"；132877 最弱场保留 |
| 2/8 | 协议：五层锁、单次读取、SHA、审计 | `artifacts/phase3/task_semantic_repair_v1/*lock*.json`、`scripts/audit_*.py` | DATA-C1 等 | 写成方法论贡献＋威胁模型框，不是噱头 |

## 待决槽位（写作留位，不计入证据）

- 配比扫掠 1:0/1:1（§5.2 SLOT）——待用户批准；批准后仅 dev 事后消融
- AMR 定位（§6.3 SLOT）——待用户裁决 A/B/C；默认 A：outlook 一段，须区分 Partial G-CNN
- Figure 1 概念图（§1）——待生成
