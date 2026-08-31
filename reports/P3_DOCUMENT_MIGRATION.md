# P3 文档迁移报告

日期：2026-09-01  
迁移提交目标：docs(p3): integrate support-conditioned geometry refinement

## 当前冻结事实

- Git HEAD：f44be1e9526796a76b58ca2bd44dc562ab7841a3。
- P2 rigid formal v2、independent fracture continuity 和 P2-H1 diagnosis 已闭合；当前 route 为 mixed_or_graph_specific。
- 250 个 canonical samples、9 个冻结模型和 P2 唯一权威输出路径已锁入 configs/phase3_support_geometry_v1.yaml。
- 当前环境：/opt/anaconda3/bin/python，Python 3.11.5；NumPy 1.26.4、pandas 2.3.3、SciPy 1.13.1、PyTorch 2.11.0、PyArrow 22.0.0、PyYAML 6.0.1、scikit-learn 1.5.1。
- 关键输入、checkpoint、manifest 和 source hash 见上述 config；9 个 checkpoint 的 SHA-256 逐一锁定。
- P2 输出只读：artifacts/phase2/p2_fracture_continuity_v1/；P2-H1 summary 只读：artifacts/phase2/p2_heterogeneity_diagnosis_v1/。
- 本轮未新增外部数据，未访问或修改 infra/bioinf-data-index/。

## Canonical 文件迁移

| 文件 | 衔接内容 |
|---|---|
| README.md | 将当前入口从旧 bootstrap/P2 后续改为 P3-T0；写明 P2 状态、mixed_or_graph_specific、local geometry 目标和唯一下一步。 |
| MASTER_AGENT_PROMPT.md | 将启动动作切换为冻结资产上的 T0；禁止继续调 P2、先训 AMR、response-driven prospective 选择。 |
| docs/01_SCIENTIFIC_BLUEPRINT.md | 保留原问题历史和 H1–H9；加入 P2 修正、边缘 spectrum/条件化 geometry 区分、P3-G1–G5 和假设状态。 |
| docs/02_METHOD_SPEC_AMR.md | 保留 AMR、谱滤波、谱白化和 context/mode；加入 normalized Jacobian/block 机制、frequency × support/relationship routing，M1/M2 gate 和差异边界。 |
| docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md | 保持 Phase 0–8 编号；把 P3 改成 T0–T5 最小证据路线，历史大矩阵降为 optional，并更新核心、分解、layer、causal 和方法 Figure contract。 |
| docs/05_AGENT_EXECUTION_MANUAL.md | 更新 W2/W3/W4/W6 职责，并写入每个 P3 任务的输入、输出、测试、manifest 和停止条件。 |
| configs/project.yaml | active phase 改为既有 P3_CAUSAL_MECHANISM，保留 Phase 0–8，不增阶段节点；更新 P3/P4 日期和 gate。 |
| configs/experiment_matrix.yaml | 保留 C1–C5 和 C2；在 C3/P3 增加 geometry、prospective、layer、causal switch、task alignment，完整矩阵标为 optional。 |
| PROJECT_PACKAGE_CONSOLIDATED.md | 以同步 addendum 汇总当前 P3 状态、公式、路线和 source-doc 优先级；旧合并正文保留为历史，不覆盖 P2。 |
| QA.md | 追加 P2 异质性不是降级、spectrum 仍保留、geometry 路线和不直接跑矩阵/AMR 的 QA。 |
| STATUS.md | 新建当前 phase、证据等级、唯一下一步、升级/停止条件和范围边界。 |
| DECISIONS.md | 新建日期化路线、机制对象、最小施工、统计边界和同步审计决策。 |
| CLAIM_LEDGER.md | 新建 P2 已支持/不支持/未测试与 P3 工作假设、P4 gate。 |
| configs/phase3_support_geometry_v1.yaml | 新建 P3 provenance、固定资产、层、公式、CV、prospective、artifact 和 gate 锁。 |

## 假设如何处理

- 保留：Action-Mode Spectrum 总体测量、graph spectral filter bank、equal-energy/spectral-whitened interventions、context/mode 双通道、任务条件 invariance/equivariance/recoverability、CAP/canonicalization/relational pooling 强基线。
- 收缩：统一中频盲区、角色联盟普遍特殊、统一 endpoint-organization 因果机制；这些不再是默认事实。
- 重写：原来一维/平均的 P3 机制问题重写为 frequency + support + relationship + task 的条件化几何预测；M1 变为 gate 后的 AMR-Fixed，M2 不先行。
- 工作假设：P3-G1–G5 全部仍为 WORKING_HYPOTHESIS；书法路线仍为 exploratory，不用于选择体育端机制。

## 当前状态矛盾审计（文档迁移时快照）

canonical source docs、README、MASTER prompt、两个 machine-readable config、STATUS、DECISIONS 和 claim ledger 的当前入口均一致：P2 已冻结，当前为 P3_CAUSAL_MECHANISM，下一步为 T0，AMR 被 P3 gate 阻塞。consolidated package 中的旧 Phase 3/旧 bootstrap 文字没有被删除；同步 addendum 将其标为历史并规定 source docs/config 优先，故不存在未标示的 current-state 冲突。P2 reports、P2 artifacts、P2 manifest/checksum 和历史 claim 未修改。

## 迁移边界

本报告是方案迁移验收，不是 P3 科学结果。T0 之前没有 geometry、prospective、causal switch 或任务性能结论；正式代码和计算从 Commit 2 开始。

## 结果回写后的同步复核（2026-09-01）

文档迁移后实际完成了 P3-T0–T5，但没有把迁移报告中的历史 `T0_NOT_RUN` 叙述当作当前状态。当前状态已在 README、MASTER prompt、四个 canonical docs、两个机器配置、PROJECT_PACKAGE_CONSOLIDATED、QA、STATUS、DECISIONS、CLAIM_LEDGER 和主 P3 报告中同步为：`T4_T5_COMPLETE_RESPONSE_SHAPING_ONLY_P4_GATE_NOT_MET`。P3 结果和失败边界只写入 `artifacts/phase3/` 与 `reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md`；本报告保留原始迁移验收事实，不覆盖 P2 历史，也不承担结果报告职责。

当前同步审计未发现未标记的 current-state 冲突：旧合并正文继续明确标为历史；P4 状态为 blocked；AMR 未实现。
