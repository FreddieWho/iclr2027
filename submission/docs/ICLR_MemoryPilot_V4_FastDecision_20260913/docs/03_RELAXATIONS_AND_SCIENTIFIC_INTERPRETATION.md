# 03｜这次做了哪些让步，以及会损失什么

| 原要求 | V4 让步 | 为什么允许 | 科学解释变化 |
|---|---|---|---|
| final memory 严格 <=1K/2K native tokens | 同 requested B，记录实际长度并做匹配/调整 | 当前 API 无 visible hard cap；严格门阻塞科学测试 | V4 不能声称严格 fixed-rate causal proof，只能判断 path effect 是否超过 endpoint-length 差异 |
| C1 thinking=low/on | thinking off | V3 reasoning 22K–58K、延迟极大且污染 output cap | 结论限定于 thinking-off compressor policy；GO 后才做 thinking robustness |
| 两个 final budgets | 一个 B | route decision 只需先证明现象存在 | 无法建立完整 rate-response curve |
| 32 exploration + 96 confirmation | 12 + 24 | paired design 下足以判断是否存在值得追的大/条件性效应 | 不是发表级精度，不用于“效应很小”的强等价结论 |
| staged2×2 + staged3 + rewrite + wait | direct + staged2 + rewrite | 最短路径区分 bottleneck vs re-encoding | 暂不回答 schedule geometry 和深度曲线 |
| C2/R2/自然数据全需 | 只要求第二 reader；自然数据可选 | 复用冻结 memory，成本最低 | GO 后仍需第二 compressor/自然数据做发表级泛化 |
| novelty/generalization/evidence audit 全部完成才决策 | pilot 后再完整审计 | 当前近邻风险已知，先确认是否有现象值得审 | GO 只是“值得投资”，不是“已经新颖到可投” |

## 不允许的让步

以下仍是科学底线：
- compressor 不能看到 future query；
- direct/staged/rewrite 必须使用相同 C1 配置；
- 不允许 post-hoc truncation；
- 不允许按结果反复 regeneration 直到长度满意；
- P2 必须是新 histories；
- 长度差异必须公开并进入敏感性分析；
- rewrite control 不能删；
- 最终必须 GO/NO-GO。

## 如何理解 NO-GO

V4 NO-GO 的含义是：

> 在一个可快速、可复现、成本可接受的 DeepSeek compressor policy 与当前任务上，没有看到足以支持“现在切换整个 ICLR 主线”的 path signal。

它不是信息论定理，也不是宣布其他模型/更长 context 永远没有 path dependence。
