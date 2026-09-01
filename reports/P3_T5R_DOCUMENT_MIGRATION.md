# P3-T5R 文档迁移验收

日期：2026-09-02  
状态：`PASS`  
所属阶段：既有 `P3_CAUSAL_MECHANISM`

## 1. Provenance

- 迁移前 HEAD：`b0134d4a0c2338c5711fb50c37ef3d2fe55969ab`
- 迁移前工作区：仅补丁包 `docs/ICLR2027_P3_REPAIR_PACKAGE_20260902/` 未跟踪；既有 P3 产物无修改。
- 补丁包 manifest：`docs/ICLR2027_P3_REPAIR_PACKAGE_20260902/PACKAGE_MANIFEST.json`
- 补丁包 `SHA256SUMS`：10/10 文件验证通过。
- 目标阶段：`P3_CAUSAL_MECHANISM`；未新增 Phase。
- 目标任务编号：`P3-T5R0`–`P3-T5R6`。

## 2. Canonical file migration

| 文件 | 从旧路线到当前路线的衔接 |
|---|---|
| `README.md` | 将当前入口改为 T5R 协议/数据/任务修复；保留 P2/P3 历史和 AMR gate；旧 bootstrap 标为历史准备。 |
| `MASTER_AGENT_PROMPT.md` | 将启动动作改为包校验 → T5R0 → train/valid → fixed dual-channel → 有界搜索 → lock/test/external；禁止提前 AMR。 |
| `docs/01_SCIENTIFIC_BLUEPRINT.md` | 保留 Action-Mode Spectrum、P2 mixed 证据和 H1–H9；加入旧 conflated gate 限定、context/intrinsic 任务语义和 T5R。 |
| `docs/02_METHOD_SPEC_AMR.md` | 保留 AMR 作为 P4 条件目标；加入 raw `z_ctx` / centered `z_mode` 固定双通道前置检验。 |
| `docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md` | 保留 Phase 0–8 和历史 Figure contract；加入 T5R0–T5R6、双通道、独立比赛和 candidate-lock Figure 规划。 |
| `docs/04_DATA_AND_ENVIRONMENT_GUIDE.md` | 加入 SNGAR、SkillCorner、IDSSE、SoccerTrack 的 T5R 访问状态、许可和 test firewall。 |
| `docs/05_AGENT_EXECUTION_MANUAL.md` | 加入 Data/Task/Autoresearch/Repro worker 分工、预算、输入输出和停止条件。 |
| `PROJECT_PACKAGE_CONSOLIDATED.md` | 顶部同步状态和 2026-09-02 T5R addendum 与 source docs 对齐；历史正文保留。 |
| `QA.md` | 记录旧 task gate 的语义冲突、heldout 暴露和 T5R 数据边界。 |
| `configs/project.yaml` | active phase 仍为 P3；当前 checkpoint、route、P4/P5 状态和 T5R 时间/协议已更新。 |
| `configs/experiment_matrix.yaml` | 保留 C1–C5，新增机器可读 T5R task lane；未创建 C2.5 或新 Phase。 |
| `configs/data_manifest.yaml` | 升级为 version 2，加入 data policy、访问状态、split 和 candidate-lock 约束；gated 数据仍未标为 downloaded。 |
| `configs/phase3_task_semantic_repair_v1.yaml` | 新增 T5R 的 data/split/task/model/metrics/autoresearch/firewall/gate lock。 |
| `STATUS.md` | 当前状态切换为 T5R0 完成、T5R1 数据访问待验证；旧 task gate 和 P4 blocked 明确保留。 |
| `DECISIONS.md` | 新增 D-011 至 D-015，记录 heldout、任务拆分、数据、预算和 dynamic support 决策。 |
| `CLAIM_LEDGER.md` | 保留历史 claims，加入 P3-R1 至 P3-R5 和 DATA-C1。 |
| `reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md` | 追加 revision note；原 P3 结果不变，T5R 尚未产生结果。 |
| `reports/P3_TASK_SEMANTIC_REPAIR_PLAN.md` | 新增当前 T5R 执行计划、未运行项和 P4 gate。 |
| `scripts/audit_current_state_consistency.py` | 新增只读 current-state、firewall、配置和关键路径审计。 |

## 3. 假设状态迁移

- 保留：P2 `mixed_or_graph_specific`、ordinary spectral power 不充分、局部 geometry 的条件性预测、diagonal/off-diagonal 与 layer anatomy、response shaping。
- 收缩：universal mid-frequency notch、static role coalition、complete organization blindness、旧 T5 task repair；不得写成普遍机制或方法成功。
- 重写：旧 T5 的统一 task gate 改为 context task 与 intrinsic task 分离；状态为 `NOT_SUPPORTED_UNDER_OLD_CONFLATED_TASK_GATE`。
- 新增但未验证：fixed dual-channel Pareto、未见比赛 geometry 迁移、dynamic semantic support、T5R 对 AMR-Fixed 的目标性；均为 `WORKING_HYPOTHESIS` 或 `NOT_ESTABLISHED`。
- 禁止：旧 heldout 作为 confirmation；书法选择体育机制；candidate lock 前读取 SNGAR test/IDSSE 结果；先训练 AMR。

## 4. Consistency checks

- `python3 scripts/audit_current_state_consistency.py`：PASS（T5R0 complete / T5R1 data access pending）。
- 四个 YAML 配置解析：PASS。
- 补丁包 checksum：PASS（10/10）。
- 全仓库 Python AST parse：PASS（37 files）。
- canonical source edits 的 `git diff --check`：PASS；补丁包原文件保留其既有 Markdown hard-break 空白，未修改以维持 10/10 checksum。
- 默认 `compileall`：`NOT_TESTABLE_IN_READ_ONLY_ENVIRONMENT`，受限环境不能写 `__pycache__`；不代表语法失败。
- 全量 pytest：`NOT_TESTABLE_AS_FULL_SUITE`；收集阶段受环境/既有依赖阻断（缺少 `torchvision`、旧导入路径和 `GLIBCXX_3.4.29`）。针对性 `tests/test_p3_support_geometry.py`：9 passed。

## 5. 尚未运行

SNGAR train/valid 获取与转换、context/intrinsic task 构造、fixed dual-channel sanity、bounded autoresearch、candidate lock、SNGAR test、IDSSE external confirmation、dynamic-support response 检验和 P4 AMR 均尚未运行。

本报告随 T5R0 文档迁移提交；正式实验结果必须另建 versioned artifacts 和 receipt，不覆盖 P2/P3 历史产物。
