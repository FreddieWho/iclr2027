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

当前执行状态（2026-08-31）：P0/P1 已完成；provenance-locked 的 P2 rigid formal v2 已在全部 250 个样本和 9 个冻结点集模型上闭合，状态为 `P2_RIGID_C2_COMPLETE`。当前探索性解释为 `mixed_or_graph_specific`：exact-spectrum 对照支持空间组织/谱相位敏感性，但 topology-frequency、严格频段子集、模型架构和角色分层并不一致。v1 标记为 `UNVERIFIED_CODE_PROVENANCE`，fracture continuity 仍为 `NOT_RUN_SEPARATE_OPERATOR`。详见 `reports/EXPLORATION_CHECKPOINT_2.md`。

## 3. 一键准备

```bash
cd ICLR_ActionMode_Project_Pack_20260818
bash scripts/bootstrap_env.sh
bash scripts/download_public_data.sh --core
python scripts/verify_data.py --manifest configs/data_manifest.yaml
bash scripts/run_phase0_smoke.sh
```

`--core` 只下载低成本、无需审批的数据：SkillCorner、Metrica、Make Me a Hanzi、开放书法小数据和 MathWriting excerpt。TrackID3x3、CCSE、MCCD、MathWriting full 需按文档中的可选参数或人工步骤执行。

## 4. 研究原则与探索空间

- **等能量扰动是测量条件**：比较不同模式时，应在坐标能量或像素感知能量上配平。
- 对单调低通、非单调凹陷和组织性差异分别进行同频语义联盟与随机联盟对照，避免把不同现象混为一谈。
- 假设、band 划分、模型和路线都可以随探索结果迭代；记录每轮实际采用的版本、选择理由与结果。
- 除 probe 外，结合 quotient retrieval、自然任务和表示距离排序，逐步判断信息是否真正可访问。
- 已知构件支持统一标为 intervention-supervised 或 known structural support，不将其描述为 self-supervised。
- CAP 作为 baseline；方法探索从 AMR-Fixed 开始，并根据结果决定是否推进 AMR-Learned。
- 算力按当前瓶颈和信息增益决定；需要 GPU 时记录 workload、成本和替代方案。

## 5. 时间边界

ICLR 2027 官方节点：

- 摘要：2026-09-18 AOE
- 全文：2026-09-25 AOE
- 主文：9 页

项目日程已写入 `configs/project.yaml`。延期时可根据实际发现动态重排阶段、扩展探索或收缩范围。

## 6. 最小成功形态

探索性研究的高价值结果不是“体育和书法都提高准确率”，而是：

1. 一个可复用的等能量 Action-Mode Probe；
2. 一个不能被普通谱偏置解释的反直觉规律；
3. 一个能开关或移动该规律的因果机制；
4. 一个有独立方法价值的 AMR 变体；
5. 体育端形成可解释规律，并在书法端检验其跨域可迁移性。

若某一候选规律不成立，就据此调整问题表述、模型范围或跨域路线；阴性结果本身也应作为探索性发现记录。
