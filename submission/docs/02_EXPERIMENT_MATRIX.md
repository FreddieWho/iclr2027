# 02｜Pilot 实验矩阵与扩展边界

## 1. 默认尺寸

可控历史默认 L≈32,768 canonical tokens，B∈{1,024,2,048}。K∈{1,2,3}。最大中间输出 8,192 tokens，而非直接从 128K→64K 开始：后者可能被服务商最大输出限制截断，且成本不必要地放大。实际 L 可为 16K/32K；必须在联调阶段依能力选择并记录，不能看到效果后单独给某模型缩短 H。

32 个探索历史；96 个全新确认历史；一次样本扩展上限为确认总数 192。每个历史默认 8 个问题（四类各两题），类型内部均匀。32/96 是操作性起点，不是统计 power 保证。问题共享历史，统计时按 history 聚类。

## 2. 核心路径（B 为最终预算）

| 路径 ID | 序列 | 目的 | 探索 |
|---|---|---|---|
| raw | H，0 次写入 | long-context 参照，不参与预算相等主比较 | 必需 |
| direct | H→B | 一次摘要基线 | 必需 |
| staged2_wide | H→4B→B | 二阶段/较宽中间 memory | 必需 |
| staged2_tight | H→2B→B | 同 K、相同 B、较紧中间 memory | 必需 |
| staged3 | H→4B→2B→B | 渐进收紧 | 必需 |
| rewrite3 | H→B→B→B | 等终点重复改写 | 必需 |
| wait3 | H→4B→4B→B | 同 K、延后收紧 | 必需 |
| no_memory | 空文本 | parametric guessing / 答案先验校准 | 小样本必需 |
| oracle | 只有该问题 gold evidence | reader 能否读懂，答案标签不输入 reader | 小样本必需 |

D 与 R3 的第一次压缩可共享同一个真实 D 输出；S3 与 W3 可共享 H→4B 输出，从而降低噪声和费用。共享产生相关性，必须按 history/replicate 聚类而不是把路径当独立样本。其余候选不能从表现最好的 seed 摘取上游记忆。

raw/no_memory/oracle 不因预算不同重复计算。正式核心对比选至多两个，例如 D vs S3，S2-wide vs S2-tight。对 H3 可选一个明确 difference-in-differences，而不是在两个子组分别看显著性。

## 3. 阶段安排

### P0：只联调，不计效果
6–8 个单独 calibration 历史；测真实 tokenizer、长度利用率、finish_reason、输出 JSON、缓存、usage、服务商是否隐藏 compaction。no-memory 和 oracle 必须拉开，否则先检查任务和 reader。仅作校准；不得把这批数据混进统计。

### P1：探索覆盖，不扩大模型数
C1/R1，neutral prompt，全部核心路径，两预算，seed 1。32 历史。8 个额外技术重复用于估计 compressor/reader 运行波动，不当独立历史。输出完整热图、类型分层、实际长度和调用成本。不能只上报最好路径。

### P2：新历史确认，而非重跑旧种子
从 P1 选择至多两个明确对比/假设；记录比较、符号（允许负向）、类型、预算与 code hash。96 新历史，2 次真实独立压缩重运行（服务不支持 seed 时记录 replicate，不伪称可控 seed）。确认重运行不是命中更好结果重抽。

先使用 C1/R1；R2 读同一批冻结 memory。C2 仅在 32 个确认历史上复现关键路径，避免全笛卡尔积。正式扩展研究是否需全部三模型 family 留到 GO 后。

### P3：自然数据/机制审计
LongMemEval-S cleaned 32 个独立完整历史，按原始 question_type 分层，保持各实际类型的样本数透明；只有 1 个 QA/历史，不宣称它和合成 8-Q 构造具有同等统计分辨率。使用 D、最佳候选及相应 rewrite/schedule 对照，不重跑全套路径。默认是外部效应方向验证，不是发表级完整 benchmark 分数。

不能读入 oracle 文件作为主 H；不能保留 answer sessions 后删 fillers 冒充完整测试。模型输入不足时，采用 H/Q 无关、整 session 保留的公开短历史子集并报告选择偏差，或者换已授权足够窗口的模型。两者都做不到，标为未完成外部验证，不造数据。

从确认池抽 24–48 个失败/成功配对做人工或独立 auditor 证据审计：必须覆盖各路径、信息类型及正确答案样本，避免仅审计失败造成 denominator 偏差。

## 4. 允许的扩展（至多两项，且只一轮）

| 扩展 | 何时值得做 | 上限/不能做什么 |
|---|---|---|
| 预算边界 | 两 B 均地板或天花板；或 interaction 已有趋势 | 新增一个 B；保持 4B<L；不能连扫十个预算 |
| structured prompt | neutral 明显丢时间/来源，怀疑格式所致 | 一个结构化模板；新样本确认；不得按任务逐种调 prompt |
| 更长/更短 H | 长输入能力可能是主混杂 | 新增一个 L，在同 H 各臂上使用；不混合估计 |
| streaming replay | 静态效应与真实长期记忆距离太远，或单调效应值得检验策略后果 | 32 条历史、同 chunks、同最新窗口、相同 final flush；≤3 policies |
| uncertainty/provenance | 更新/关系已有机制线索 | 新模板分布而非换几个同义词；只保留一个新信息类 |
| 轻量 mitigation | 某一失真类型可定位，存在廉价可解释修复 | 一个 timestamp/source-preserving 模板或固定 schedule；不是新架构 |
| 等价/成本边界 | 主差异接近零且区间够窄 | 预设 ±3pp 实践等价带；估计实际节约，不以 p>0.05 代替等价 |

联调 bug 修复不消耗上述两项，但重复修复同一失败三次即停止相关 adapter，报告阻塞。两个新 compressor、三种 prompt 和十个 B 不属于“一次扩展”。

## 5. 不同结果怎么走

总体均值为零但 update 很差：确认 H3，不平均掉机制。所有路径准确率接近零：先 oracle/no-memory 校准和一个较宽 B，不能据此 kill。S3≈R3但 R3≠D：保留“re-encoding drift”路线审查，但必须有新的调度/任务后果，否则不足以超越最近邻。D≈S3 且 CI 很宽：一次增样或 NO_GO_EVIDENCE。D≈S3 且严格等价、有明显成本优势：可以评估 GO_BOUNDARY，不强行寻找灾难性遗忘。
