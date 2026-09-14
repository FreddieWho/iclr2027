# 01｜当前进展诊断与纠偏

## 1. 当前项目并没有完成一次科学 pilot

截至 2026-09-13：
- V3 visible-budget preflight 仅完成 4/40 个有效响应格；1 个请求中断；35 个未尝试；
- direct vs staged utility 尚未评分；
- R1 正式 calibration 未运行；
- P1/P2 均未运行；
- 当前 `NO_GO/TECHNICAL` 只说明审计/技术条件未满足，不能解释为 memory path hypothesis 阴性。

因此项目真正的科学完成度接近 **0%–10%**，而不是“已经做完一半但结果不好”。工程脚手架、tokenizer、账本、数据合同已经有价值，但它们没有回答主问题。

## 2. 为什么推进不顺

### 2.1 把发表级严格控制提前成了进入实验的硬门

原方案要求：
- 1K/2K native-token final budget；
- 模型自然 stop；
- 不截断；
- thinking 保持；
- prompt 自己精确 obey token limit。

但当前 API 又没有 visible-token hard cap。结果项目实际在测试：

> “DeepSeek 在隐藏 reasoning 共享 completion budget 的情况下，能不能靠 prompt 精确数 native tokens？”

这不是研究假设。

### 2.2 C1 low-thinking 的运行特性与任务不匹配

真实 V3 调用中隐藏 reasoning 可达到 22K–58K tokens，单格延迟约 100 秒到 24 分钟。它既拖慢实验，也使 provider output cap 与 visible memory 控制相互干扰。

compression-path hypothesis 并不依赖 thinking 开启，因此继续坚持它没有科学收益。

### 2.3 过多“未来论文必须项”被放进 route-selection gate

旧规则同时要求：
- 64–96 新确认历史；
- 两个 final budget；
- rewrite + same-K + length；
- 第二 compressor/reader/natural generalization；
- 完整 novelty audit；
- evidence audit。

这些适合 GO 后的正式论文，不适合回答“现在值不值得切题”。

### 2.4 机器化 `decision.json` 过早

当前 decision checker 把“未完成”映射成 NO_GO/TECHNICAL，这在工程审计上没错，但心理上容易让项目看起来像已得出阴性结果。V4 将 route-selection decision 单独输出，不复用旧 decision.json。

## 3. V4 的核心纠正

### 纠正 A：严格 endpoint budget -> 软目标 + 实际长度控制

路线决策时不再要求：

`L_visible <= B exactly`

而要求所有 arm：
- 同 requested B；
- natural stop；
- 无 post-hoc truncation；
- 实际 visible native length 全记录；
- path effect 必须在 length-balanced subset 和 length-adjusted analysis 中仍有信号。

这直接回答更重要的问题：

> path 是否影响 utility，且这种影响是否超过“最后多写了/少写了多少 token”。

若答案阳性，GO 后再解决真正 strict fixed-rate serving。

### 纠正 B：C1 thinking off

V4 采用一个新的 compressor policy：同一 DeepSeek V4.1 Flash，thinking disabled。所有 arm 完全一致。

这会改变“compressor policy”，但不破坏 path 的 paired causal comparison。V4 结论仅适用于该冻结 policy；GO 后再做 thinking-on robustness。

### 纠正 C：一个 final target

先用 B=4096；只有 2-history 联调明显不可用才一次切到 B=8192。route-selection 不需要扫 5 个 rate。

### 纠正 D：12 + 24 histories

本轮目的不是写 final paper，而是做研究投资决策。24 个新确认历史配合 paired design、长度校正和第二 reader，足以把“大效应/条件性效应”与“接近零/主要由长度解释”分开。

### 纠正 E：只保留三条 memory path

- direct
- staged2
- rewrite

这样可以同时问：
1. staged 是否不同于 direct？
2. 若不同，是否只是多做了一次 rewrite？

## 4. V4 不回答什么

即使 GO，V4 也不证明：
- 所有 LLM 都有 path dependence；
- 严格相同 native-token endpoint 已得到 causal proof；
- 1K/2K rate 有特殊规律；
- repeated compression 的所有机制都已定位；
- 已经具备投稿证据。

它只回答一个项目决策：**这个科学方向是否存在足够强、足够独立于长度和简单 rewrite 的信号，值得把主要资源从 Action-Mode 切过来。**
