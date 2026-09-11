# 10｜参考代码已做什么、agent还需做什么

## 已包含的参考实现

- make_synthetic：四类可控事件、public history/独立gold、随机实体与逐事件来源。
- import_longmemeval：移除has_answer等标签，保持完整sessions，分层抽样，保留gold在query文件。
- build_plan：生成明确target序列；禁止非正预算和中间目标超过history。
- run_pilot：fresh-request压缩、raw/no-memory/oracle校准与逐题reader、每步缓存与落盘、费用预留、token上限、重复前缀缓存、API与mock模式；不自动启动网络。
- analyze：matched question/replicate聚合到history，paired history bootstrap；支持预设类型过滤。
- decide：二元决策与route阈值检查、mock拒绝、必要审计证据存在性验证。

## 必须由agent完成的真实适配

具体服务商响应与原生tokenizer验证；natural-language semantic scoring；多reader/compressor分片运行合并；目标长度合规与敏感性重读；interaction对比（可复用bootstrap函数，但须输出直接ΔΔ）；evidence auditor；外部novelty检索；最终事实审核。它们有明确合同，但没有伪装成已经联调的功能。

默认脚本不完整实现streaming、自适应schedule训练、三方LLM裁判共识和并发GPU部署：这些不是核心pilot开工前提。参考runner顺序执行，agent如需提速应按history分片，不改统计单位。

## 复现实验前须确认

正式tokenizer的词表/版本已存在；model配置不是REPLACE；live明确授权并设美元cap/价格；服务商没有默认agent历史；自然答案不误用exact match；llm失败不转成0分。只要其中一项无法确认，就标出范围，不假称端到端科学验证完成。
