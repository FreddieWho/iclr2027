# 06｜Agent 施工与资源策略

## 工作安排

| 原子任务 | 依赖 | 可并行 | 必交产物 |
|---|---|---|---|
| T00 环境/API/预算发现 | 无 | 是 | PREFLIGHT.md + model manifest；真实tokenizer验证 |
| T01 文献核查 | 无 | 是 | NOVELTY_AUDIT.md，三篇直接近邻逐项比较 |
| T02 数据编译与gold隔离 | 无 | 是 | histories/queries JSONL、manifest、checksum |
| T03 runner联调 | T00/T02 | 小范围 | 所有steps、usage、输入hash；无mock混入 |
| T04 核心探索 | T03 | 不同history并行 | 全路径×B表、长度/cost、探索报告 |
| T05 选择与新样本确认 | T04 | 两worker可独立评分 | confirmation_selection.json，新样本效果与CI |
| T06 外部/跨reader复现 | T05的冻结memory | 是 | 一项以上独立复现；最好自然与C2各一小组 |
| T07 证据与长度审计 | T05 | 是 | EVIDENCE_AUDIT、LENGTH_AUDIT、反例 |
| T08 一次定向扩展 | 只有明确歧义时 | 有上限 | 扩展理由、全部结果；不能暗中改假设 |
| T09 汇总与裁决 | 上述有效产物 | 否 | decision.json + PILOT_REPORT.md +复跑命令 |
| A-MEMO 摘要核对 | 六原始评审 | 是 | 只核对本包备忘，无Action-Mode新实验 |

## 资源默认值

本包默认15,000次live请求、150M input、15M output tokens作为工程防失控上限，**不是预估一定会花这么多，也不是付费授权**。先用plan计算将要发送的请求；若超出已有预算，把全网格改为阶段性执行：先P1，再只运行选中的P2。不要减少同history配对来凑漂亮数字。

美元上限初始未设置；使用用户已有明确授权可自动填入并记录来源。不凭此包暗示可以花50/100美元。没有价格或授权时，运行mock/CPU准备，不发送live请求；如确需用户提供key/额度，只提出一次具体需求。GPU租赁不在自动授权范围。

静态记忆缓存键包括：history hash、model exact ID、endpoint、prompt hash、generation params、路径前缀、目标长度、replicate。R1/R2共享同压缩输出；same prefix共享只限相同replicate。缓存不能把两个“独立重复”合成一份结果。

## 重试、失败与并发

单次网络请求最多3次尝试；429/5xx退避，不自动改模型/长度/prompt。每次尝试预留费用，网络失败可能已计费，保守保留账本；不把未知费用视为0。对同一个确定的协议错误不得重复跑整套history。

参考runner是单进程顺序执行，便于验证。要并行时按完整history shard划分，多个worker必须共享SQLite费用账本，并在总控分配的out目录内工作；当前脚本重复启动同out目录不支持安全并发，先分片再运行。不得两个agent重做同一history后挑好结果。

## 参考代码与真实施工的衔接

安装 `requirements.txt`（numpy+requests+tiktoken；可替换成本地HF tokenizer；也可接本地HF tokenizer）。先 `python3 -m unittest discover -s tests -v`。补本地模型配置，运行8例真实联调，再扩到探索。

每个自然数据问题需semantic grader，参考runner只给待评分记录，不生成虚假accuracy；模板在prompts/judge.txt，或调用作者脚本。所有adapter更改要添加最小test：history长度、has_answer剥离、cap、finish_reason、queryblind请求、usage/money、重复缓存、缺失分母。

本包无需重新搭一个agent框架、不要求特定CLI/模型品牌，不安装大型训练栈、不训练embedding、不下载所有候选数据。现有环境足够就复用。
