# 03｜数据、模型与下载

## A. 可控历史（优先启动）

`scripts/make_synthetic.py` 不调用 LLM，可生成多项目/多事件文本、独立 random identifiers、时间更新、关系链和规则撤销。它是机制仪器，不是自然对话 benchmark。正式数据必须使用真实固定 tokenizer，demo 模式只测试程序。每条 history 同时产生 public histories 与独立 queries/gold 文件；压缩只读取 public text。

默认四类、每类两题：
- atomic：确切标识符/存储位置，随机值避免模型常识猜中。
- update：一个事实有两次带日期的更新，问最新值；证据包含旧/新及来源。
- relational：项目→负责人→交接对象→标识符的链式关系，不能只查到一个同名词就判保留。
- constraint：一般规则、范围例外和后续撤销；任务是选正确操作，不是背 safety slogan。

所有项目都有同类事件，不把被提问项目写成显眼的“重要事实”。非目标事件充当真实结构干扰，不能靠重复一段 filler 填到 32K。每条问题的 gold_evidence 记录允许 grader 检查，但不能进入 compressor。保持时间顺序；需要位置实验时通过合法 interleaving 实现，不随意打乱“最新”的定义。

同一 history 中 8 个问题可以减少压缩费用，但回答应使用独立调用或固定、无答案互喂的批次；批次设定属于 reader protocol，不可按路径变。默认 runner 逐题读取，费用估算按逐题计。

## B. LongMemEval 外部验证

已核实作者仓库存在 cleaned 发布及下列链接 [EXT-LME-CODE]：

```text
https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
```

下载命令：

```bash
python3 scripts/fetch_longmemeval.py --out data/raw/longmemeval
python3 scripts/import_longmemeval.py --input data/raw/longmemeval/longmemeval_s_cleaned.json \
  --out data/longmemeval --n 32 --seed 20260910
```

原始 dataset checksum、下载日期、resolved revision、schema 和数据许可必须落盘。代码 MIT 许可不自动等于数据许可；若卡片未明确，记录实际条款，不在本包或论文附件重新分发数据。作者联系/登录不得绕过；若下载失败，最多三个不同正规入口/重试，不无限折腾。LongMemEval 原论文及仓库提供多类长期记忆问答及 evidence session/turn 元数据 [EXT-LME, EXT-LME-CODE]。

关键防泄漏：丢弃 compressor 输入中的 `has_answer`、`answer_session_ids`、`answer`、`question`、`question_type`。公开 text 只保留角色、内容、原始日期和无含义事件 ID。原始长历史完整保存；不能为了减少 API 费把 evidence oracle 当自然 memory 输入。

自然问答不普遍适用 exact match。本包 importer 会标记 `scoring=semantic_required`；参考 runner 不把它们错误判零。agent 应复用作者评分或完成 blinded judge adapter，再人工检查至少 20 例与各类边界。主 reader 不同时充当唯一裁判。审计用完整 gold 可以，reader/compactor 不可以。

## C. 模型能力，而不是预先指定品牌

C1：现有授权、非极小弱模型、32K+输入并能输出8K真实文本的写入器。R1：同级或更强 reader。R2：不同 family 优先，读取同 memory。C2：不同 family，仅验证关键效应。不要为“不同名字”把同底层模型换个路由算独立 family。

记录：provider/base URL、精确模型 ID/版本、native tokenizer、canonical tokenizer、context window、最大可见输出、reasoning token 策略、temperature/top_p、seed 是否实际支持、API fingerprint、价格与核验日期、服务是否有隐式检索/记忆/压缩。没有证据的能力写 unknown，先小额探测，不虚构。

`configs/models.example.json` 不含有效模型名或预算授权；其 adapter 为 Chat-Completions-compatible 参考实现，不保证任意厂商兼容。Agent 根据官方文档/本地服务验证 parameter 名称；不要静默丢 seed/temperature 后说实验一致。特殊 reasoning 模型的 hidden output 计入费用，不能把 max_completion_tokens 当可见 memory token 数。

## D. 成本与数据权限

不使用用户私人聊天或原 Action-Mode 数据作为 Memory 数据。付费请求应使用既有授权额度，独立记账。先缓存压缩，再交多个 reader；数据生成、统计和 memo 在 CPU 完成。不得因为本地没有大 GPU 就直接租卡。

估算至少输出 input tokens、visible/hidden output tokens、请求数、按服务商当日价格的上界。未知 hidden token/价格时按保守上限预留；没有费用授权仅完成本地工作并报告 BLOCKED。包中的 token/call 上限不是用户批准的美元消费额。
