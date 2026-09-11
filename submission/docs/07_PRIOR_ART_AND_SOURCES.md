# 07｜来源、查重与主张边界

## 附件来源与核验界限

六份 `sources/reviews/review_*.txt` 是用户提供的第三方LLM审计意见，不是彼此独立的实验、也不是接收率数据。带行号副本仅为引用定位，原文未修改。Action-Mode当前状态只依据旧比较材料/进度报告，**本包没有重新读取最新GitHub仓库、没有复现A的结果**。

本次已通过原始论文页面/HTML核验下列文献存在及列出的有限内容。没有完成穷尽式新颖性审计，因此“fixed-budget path从无人研究”仍不能声称。摘要存在不等于全部实验设定与本项目不同；agent需在T01检查方法细节、附录与代码。

## 核验表（2026-09-10）

| ID | 原始入口 | 本次核验的范围 | 本项目怎么处理 |
|---|---|---|---|
| EXT-RD | https://arxiv.org/html/2607.08032v1 | §14.2 比较不可逆摘要与可回读归档，并改变compaction频率；有统一预算框架 | 是直接近邻；不能卖“首次测多轮”；检查是否覆盖我们的分支对比 |
| EXT-CLIFF | https://arxiv.org/html/2608.22752v1 | 多轮compaction、知识类型保留及triage；正文有逐轮50%方案 | 不能卖“发现约束遗忘”或“首次按类型保护” |
| EXT-RESTORE | https://arxiv.org/html/2609.08279v1 | eviction的逐题restore-counterfactual及错误分解 | audit为借鉴/适配，不独占retention/accessibility |
| EXT-SSGM | https://arxiv.org/abs/2603.11768 | 标题/摘要关于evolving memory治理与稳定性框架 | semantic drift邻域；具体覆盖程度待全文核对 |
| EXT-GOV | https://arxiv.org/abs/2606.22528 | 标题/摘要关于compaction后的constraint治理退化 | 约束类不作唯一novelty |
| EXT-LME | https://arxiv.org/abs/2410.10813 | LongMemEval长期记忆问答、ICLR 2025 | 外部评估来源，不是新造数据集 |
| EXT-LME-CODE | https://github.com/xiaowu0162/LongMemEval | cleaned下载入口、history/QA/evidence字段及oracle区别 | 下载/schema参考，精确revision由agent保存 |
| EXT-SR | https://research.ibm.com/publications/successive-refinement-of-information | successive refinement原始研究入口 | 背景类比，不作为递归摘要的现成定理 |

本表只为快速定位，不大段复述原文。查重记录必须注明“读到哪里/没读哪里”。下载不可用是access问题，不是论文不存在。

## T01逐项判断，避免标题查重

在三篇直接近邻全文中逐项记录yes/no/not-established及section/code出处：
1. 同一完整H分叉，所有分支都不可回读H？
2. 相同最终**实际**长度或只相同cap/平均预算？
3. direct和多阶段不可逆摘要直接配对？
4. 同K变中间schedule？
5. 固定B原地rewrite对照？
6. future-query-blind写入？
7. 跨信息结构/reader分离？
8. 相同预算下有可实施策略含义？

一个“no”不自动构成新颖；必须解释该设计差别改变了什么未知结论。若全部实质覆盖且没有新机制/边界，按NO_GO_NOVELTY，不用继续大规模调用。查重可以与合法小pilot并行，不能要求“证明文献不存在”才开工。

## 不采纳为事实的审计说法

- Sonnet把2026年材料一概当未来模拟：今天为2026-09-10，上表多篇原文已核验；日期推断错误不采纳，其他设计批评单独评价。
- Kimi“support/coalition确认无人占据”：有限抽查不能支持穷尽性否定，A相关工作仍待检查。
- 各评审中稿概率：无校准样本，不进入预算或go/no-go公式。
- 多次有损压缩一定使direct优于staged：不是DPI的结论，见docs/01。

## A的待查文献线索（来源：review_kimi；非本包已完成全谱系核验）

Learning Partial Equivariances From Data；Augerino；Residual Pathway Priors；SER；PooDLe；Geiger等causal abstraction/interchange interventions。任务是区分conditioning对象、干预层级和实际任务后果，不把列出名称当完成related work。未核验的年份、venue、公式不要继续抄写成事实。
