# ICLR 2027 项目方案包：Action-Mode Spectrum

> 一句话摘要：研究多构件表征是否看见了“谁和谁一起变化”；以足球/篮球阵型作为显式结构仪器，以中国书法作为隐式结构检验，发现并控制模型在共同、子群和局部作用模式上的敏感度错配。

## 0. 项目范围

- **核心科学对象**：Action-Mode Spectrum，作用模式谱。
- **核心现象候选**：等能量扰动下，模型对整体位移的反应大于对结构破坏的反应；或在子群协同对应的中频出现异常盲区。
- **核心方法贡献**：**AMR（Action-Mode Routing，作用模式路由）**。它在关系图谱坐标中，按任务学习“哪些模式应不变、哪些模式应等变/可恢复”，并保留独立的全局上下文通道。
- **主实验域**：足球阵型。
- **重复验证域**：3×3 篮球阵型。
- **跨域预测与旗舰案例**：中国书法。
- **技术 fallback**：手写数学公式 MathWriting。
- **明确不做**：体育视频跟踪算法、3D 姿态、审美打分、大型 VLM 训练、从零构建书法专家标注集。

## 1. 为什么值得做

现有表示学习通常按“变换算子”定义不变性，例如平移不变、旋转不变；但在多构件系统里，语义更依赖**变换如何分配给构件**：全队一起移动、后卫线一起移动、单人移动，虽然位移算子相同，含义完全不同。书法中的整页移动、偏旁移动、单笔移动也是同一个结构。

本文不主张删除全局信息。通用表征应把：

1. 全局上下文；
2. 内部构型；
3. 子群协同与局部偏离；

分成可被简单读出器分别访问的模式，而不是让训练目标提前把某些模式永久压掉。

## 2. 从哪里开始

建议依次阅读：

1. `docs/01_SCIENTIFIC_BLUEPRINT.md`：科学问题、假设、贡献边界。
2. `docs/02_METHOD_SPEC_AMR.md`：AMR 方法的完整规格。
3. `docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md`：实验、探索检查点和 Figure 规划。
4. `docs/04_DATA_AND_ENVIRONMENT_GUIDE.md`：数据下载、许可与环境。
5. `docs/05_AGENT_EXECUTION_MANUAL.md`：多 agent 施工协议。
6. `docs/06_REVIEW_DECISIONS.md`：五份评审建议的接受与拒绝。
7. `MASTER_AGENT_PROMPT.md`：可直接交给总控 agent 的启动 prompt。
8. `QA.md`：模型架构、技术路线和科学问题的持续问答记录。
9. `reports/p0-overview.html`：快速查看 P0 原始样本、干预和 embedding。

当前执行状态（2026-09-02）：P0/P1 已完成；provenance-locked 的 P2 rigid formal v2、独立 fracture continuity 和 P2-H1 有界异质性诊断均已闭合。P2 的冻结结论仍是 `mixed_or_graph_specific`：部分方向跨比赛稳定，但幅度和方向依赖 architecture、graph、role、energy，且它是 representation response，不是下游任务性能。项目仍在既有 `P3_CAUSAL_MECHANISM`；原 T0–T5 只得到“表示塑形”证据，旧 task gate 未闭合，且旧 heldout 已在候选循环中暴露，只能作 exploratory audit。当前授权的是有界的 `P3-T5R` task-semantic repair，不是新 Phase，也不是直接训练 AMR。其目标是把需要保留绝对部署信息的 context task 与要求全局平移稳健的 intrinsic task 分开，并在独立比赛上检验固定双通道表示。权威 P2 证据见 `artifacts/phase2/` 与 P2 reports；P3 证据见 `artifacts/phase3/`、`configs/phase3_support_geometry_v1.yaml` 和 `reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md`。

P3 不否定 Action-Mode Spectrum，而把它保留为跨条件的边缘汇总，并补上 support/relationship-conditioned local geometry 机制层：先检验归一化 embedding 的 Jacobian 与 \(G_f=J_f^\top J_f\) 能否预测 fracture response，再做 prospective、layer-wise 定位和一个证据选择的因果开关。只有出现预测、定位、因果开关和客观任务联系，才进入 P4 AMR；M1/M2 不得先行。

## 3. 当前启动动作：P3-T5R

当前唯一启动动作是先完成协议/文档迁移，再获取开发数据：

1. 读取 `docs/ICLR2027_P3_REPAIR_PACKAGE_20260902/` 全部说明；
2. 完成 `P3-T5R0`，更新 canonical docs/configs 和 current-state audit；
3. 只申请或下载 SNGAR train+valid，test 保持 firewall；
4. 构建 context/intrinsic task 与 fixed dual-channel baseline；
5. 只有 sanity gate 支持后，才允许最多两轮 bounded autoresearch。

旧 heldout 标记为 `EXPOSED_DURING_CANDIDATE_SEARCH`，不再作为确认集。P4 AMR 继续 blocked，书法不参与体育端机制选择。

## 4. 历史/基础准备

```bash
cd ICLR_ActionMode_Project_Pack_20260818
bash scripts/bootstrap_env.sh
bash scripts/download_public_data.sh --core
python scripts/verify_data.py --manifest configs/data_manifest.yaml
bash scripts/run_phase0_smoke.sh
```

该段命令属于历史基础准备，不是当前 T5R 启动动作。原 `--core` 只下载低成本、无需审批的数据；SNGAR、IDSSE 和 SoccerTrack 的当前访问、split 和 candidate-lock 规则以 `configs/data_manifest.yaml`、`configs/phase3_task_semantic_repair_v1.yaml` 及补丁包为准。不得把 gated 数据写成已下载。

旧 P3 结果继续使用同一批 250 个 canonical samples、9 个冻结点集模型和 P2 provenance 链；其 heldout 只能作为 exploratory audit。P3-T5R 允许在既有阶段内扩充独立比赛并修复任务语义，但不访问、修改或重建 `infra/bioinf-data-index/`。只有 T5R gate 闭合后，才重新考虑 P4 AMR。

## 5. 研究原则与探索空间

- **等能量扰动是测量条件**：比较不同模式时，应在坐标能量或像素感知能量上配平。
- 对单调低通、非单调凹陷和组织性差异分别进行同频语义联盟与随机联盟对照，避免把不同现象混为一谈。
- 假设、band 划分、模型和路线都可以随探索结果迭代；记录每轮实际采用的版本、选择理由与结果。
- 除 probe 外，结合 quotient retrieval、自然任务和表示距离排序，逐步判断信息是否真正可访问。
- 已知构件支持统一标为 intervention-supervised 或 known structural support，不将其描述为 self-supervised。
- CAP 作为 baseline；方法探索从 AMR-Fixed 开始，并根据结果决定是否推进 AMR-Learned。
- 算力按当前瓶颈和信息增益决定；需要 GPU 时记录 workload、成本和替代方案。

## 6. 时间边界

ICLR 2027 官方节点：

- 摘要：2026-09-18 AOE
- 全文：2026-09-25 AOE
- 主文：9 页

项目日程已写入 `configs/project.yaml`。延期时可根据实际发现动态重排阶段、扩展探索或收缩范围。

## 7. 最小成功形态

探索性研究的高价值结果不是“体育和书法都提高准确率”，而是：

1. 一个可复用的等能量 Action-Mode Probe；
2. 一个不能被普通谱偏置解释的反直觉规律；
3. 一个能开关或移动该规律的因果机制；
4. 一个有独立方法价值的 AMR 变体；
5. 体育端形成可解释规律，并在书法端检验其跨域可迁移性。

若某一候选规律不成立，就据此调整问题表述、模型范围或跨域路线；阴性结果本身也应作为探索性发现记录。
