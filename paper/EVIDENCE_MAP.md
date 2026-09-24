> **2026-09-24修订入口**：e1a933审计后的当前措辞以
> `reports/e1a933_review/AFFECTED_CLAIMS.csv` 与
> `reports/e1a933_review/PAPER_PATCH_AUDIT.md` 为准。此前final_closure
> ledger保留为原始结果来源；受本轮纠正的归因不继续沿用。

> ⚠️ **已取代（旧论文骨架的证据映射，2026-09-05）**：本表对应的是机制先行旧稿
> （AMR/T5R 叙事）。论文已于 2026-09-18 迁移到最终叙事（见
> `reports/final_closure/PAPER_MIGRATION_AUDIT.md`）。当前正文数字唯一来源：
> `reports/final_closure/MASTER_CLAIM_LEDGER.md`；证据表：
> `reports/final_closure/FINAL_EVIDENCE_TABLE.md`。本文件保留作历史。

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
| 4.5 | 证伪链：v1–v4b 共享主干 15 连败（通道/主干塌缩模式见报告） | `artifacts/phase4_amr/m1_v{1,2,3,4,4b}/` | P4-A1..A5 | 单变量隔离表述；禁调参不够暗示 |
| 4.5 | v5 解耦 H1 通过：zone 0.97–0.99、pair 0.52–0.82、干预 0.81–0.96 | `artifacts/phase4_amr/m1_v5/` | P4-B1 | seed-23 pair 弱＋seed-11 diag 负并列写 |
| 4.5 | CAP 打平：pair 0.61/几何 0.50/full 0.89；context 逐位相同 | `artifacts/phase4_amr/m1_v5cap/` | P4-B2 | 主动拥有无特异性；margin 薄代价保留 |
| 4.5 | v6 教师共享主干 H1 通过：pair 0.80–0.88、margin 0.15–0.17 | `artifacts/phase4_amr/m1_v6/` | P4-C1 | seed-47 分裂＋margin 仍薄并列写 |
| 4.5 | μ 阈值证伪：AUC 0.44–0.50，三池无单调 | `artifacts/phase4_amr/mu_threshold_v1/` | GOAL-MU01 | 一句 filed negative，不藏不洗 |
| 4.5 | R1 nomask 消融（待判据）：幽灵球员是否 load-bearing | `artifacts/phase4_amr/m1_v6nomask/` | 待入账 | 通过→简化故事；退化→新颖性 claim |

## 待决槽位（写作留位，不计入证据）

- 配比扫掠 1:0/1:1（§5.2 SLOT）——已完成（P3-R15：双向单调权衡＋2:1 膝点），待填 prose
- AMR 定位（§6.3 SLOT）——已走完 P4 全周期，定位＝§4.5 边界证据（非 outlook）；Partial G-CNN 区分仍须写
- Figure 1 概念图（§1）——待生成
