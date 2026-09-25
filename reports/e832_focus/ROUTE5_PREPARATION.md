# 路线5预备审计（等待路线1–4结果）

日期：2026-09-25。本文件不把未完成实验写成结果。

## 当前已成立

- 主问题已经收束为 single-edit supervision → full joint correctness。
- 237/509/1229/159/178 银行口径已分开。
- source typed 函数的表达能力限制已修正，不再用它排除一般可学习关系能力。
- 视觉 ResNet18 已有四臂和 full-repair/migration 表，但不是结构消融。
- 新 source bank 的 sorted-vs-unsorted 范围句已在附录；不是跨任务或视觉确认。

## 路线5等结果时要改什么

### 必改

1. 主文只保留一个主问题、最强1–2个新增结论和一个视觉边界。
2. 新路线结果只有在 artifact、分母、父场景区间和失败边界齐全后才进入正文。
3. Figure 1 使用真实渲染和固定选择规则；不能使用示意图或按结果挑图。
4. 主表补齐 J3/J4、A/B/AB、atomic、full repair、migration、111 regression、parent/seed 单位和成本。
5. 把 source expressivity 限制、GPU不可用、CPU smoke 与正式结果分开。

### 不应再做

- 不启动 N02 长训练；它不解决当前结构主张。
- 不通过增加足球、VLM、辅助损失或随机种子来填主表。
- 不把“加了一层”或“排序有效”写成新方法。

## Figure 1候选数据

现有 `reports/e832_focus/PAPER_MIGRATION.md` 已规定从历史 bank_dev512、seed 11 选 endpoint-only repair 与 full repair 各一例，且不用于效应估计。新路线完成后，优先用路线1/3产生的新结构银行选一个真正结构干预正例；如果新银行尚无合法可辨识渲染，则保留旧规则并把新结构图放附录，不能伪造。

## 审稿攻击准备

- **已知不变性/特征工程：** 用 rich8 unsorted/sorted 的同信息对照和有限群/容量匹配基线回答。
- **关系模型表达力不足：** 用 interaction/additive 对照与四例表达审计回答，不用旧 typed 阴性替代。
- **视觉任务太简单：** 同时报告经典解析基线、原子感知误差和 full repair，不声称网络优于解析器。
- **算力混杂：** 视觉结构消融不与 N02 采样矩阵混合；MAC、训练步数、曝光分别记账。

## 依赖

路线5的正文改动依赖路线1–4的最终 handoff；在此之前只更新 ledger、图表生成器和可复现入口，不写 PENDING 结果的正文句子。
