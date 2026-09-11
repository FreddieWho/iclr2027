# 01｜研究定义与可检验假设

## 1. 研究什么，不研究什么

这里的 memory 是把历史文本压缩成可再输入 LLM 的文本状态，不是模型参数记忆，不是 KV cache 量化。H 为同一完整历史；Q 为压缩后才对 reader 开放的未来问题；C 为固定模型、prompt 和采样参数的写入器；R 为固定 reader。

D_B = C_B(H)

P_s = C_B(C_{b_(K-1)}(...C_{b_1}(H)))

s=(b_1,...,b_(K-1),B)，K 计入第一次 H→memory 的写入。每条历史和随机重运行均与自己其他路径配对。最终 utility U=score(R(memory,Q),gold)，主单位是独立 history，历史内问题和 seeds 是重复测量。

主问题：同一预算约束下，D_B 与 P_s 是否存在稳定、实践上值得在意的差异？在 K 相同时改变中间预算会怎样？效果集中在哪些信息结构、写入器、reader 和预算范围？

原始 long context 是参照：它同时改变了可读信息量和 reader 输入长度，不是固定 B 的竞争臂。原始历史不一定是任务分数上限；oracle evidence 也只是经选择的诊断上界参照，不是可部署基线。

## 2. 五个假设与优先级

| 编号 | 假设 | 状态 | 核心对比 |
|---|---|---|---|
| H1 | 在相同最终预算下 direct/staged utility 不等 | 未验证，不预设符号 | D vs S3 |
| H2 | 相同调用次数下，中间预算 schedule 改变 utility | 未验证，较强区分点 | S2-wide vs S2-tight；S3 vs W3 |
| H3 | path effect 随预算或信息类型变化 | 未验证，不要求符号反转 | Δ_lowB−Δ_highB；Δ_update−Δ_atomic |
| H4 | 相同 B 重写所造成的漂移与中间预算收紧不同 | 未验证，不预设独立贡献 | R3 vs D；S3 vs R3；W3 vs S3 |
| H5 | 存储损失、失真与 reader 使用失败可区分，并解释任务后果 | 测量目标，非首次提出此拆分 | final memory 证据审计、highlight/restore |

H2/H3 可在 H1 总体均值很小时成立；不得先按全局均值过滤它们。H5 单独成功一般不足以证明本项目新颖，见 docs/07。

## 3. 信息论应如何使用

对无外部回读、固定参数的 Markov 压缩链，条件于模型参数/实验约定，可以用 DPI 得到 I(Y;M_(k+1))≤I(Y;M_k)。但 direct 和 staged 是从 H 出发的两条分支，DPI 不给两条分支最终 memory 的信息量排序，更不保证有限 reader 的准确率单调。

“最终 token 数一样”也不等于“熵相同”或“有用信息相同”。本项目测试的是有限模型+给定编码程序，不是求 Shannon 最优率失真函数。经典 successive refinement 可作相关理论背景 [EXT-SR]，但它通常讨论渐进增加描述；本项目递归再摘要不能被直接宣称为同一定理的应用。

“path-dependent”不是物理学迟滞。只有多个 compression schedule 分数不同，不足以用 hysteresis、phase transition 或 non-Markovian 当结论。离散的 2–3 个预算点也无法支持相变。

## 4. 两个预算估计量必须区分

**cap-matched policy effect**：各臂同样允许 B tokens，比较真实输出。模型未充分用预算本身可以是系统现象，但不能据此声称“实际最终长度相等仍有差异”。

**realized-length sensitivity**：记录同一指定 canonical tokenizer 的实际长度，控制明显差异后再检验。优先使用相近长度结果的预设分层和另做所有臂共同 target 的无模型截断敏感性。不得根据正确率选择截断点或删样本；必须保留原始 cap 分析及全体不合规率。

无填充、无隐藏 scratchpad memory、不通过补原文把某臂“修”回预算。若正式 memory 长度差异无法排除，NO_GO_EVIDENCE（或 GO 只能声明 cap-level 现象，不能通过本包严格 path 主线门槛）。

## 5. 静态与流式边界

主 pilot 为静态：所有路径从相同 H 出发，之后不可再看 H。流式 M_t=C(M_(t−1),chunk_t) 会多出事实到达时间、最近窗口和原文暴露次数，必须独立作为扩展，不合并 H1 估计。

本包不新增 RAG 系统训练，不建设多 agent 记忆系统，不做用户私人对话采集，不用“安全规则遗忘”一种类型代表所有 memory。规则任务使用无害虚构工作流，避免引入医学/合规知识正确性混杂。
