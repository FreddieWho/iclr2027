# Memory Pilot 施工包 + Action-Mode 审计备忘

**日期：2026-09-10｜状态：可启动施工；没有真实 LLM pilot 结果。**

## 交付目标
在同一原始历史、同一最终记忆预算和 query-blind 写入条件下，比较 long context、一次压缩、多阶段压缩及等预算重复改写。完成一轮有上限的探索与独立确认，给出 **GO / NO_GO**，而不是不断补实验的计划。

GO 的意思是“值得把 Memory 升为下一阶段研究主线”，不是“论文已成立或必能中稿”。NO_GO 的意思是“本轮证据不支持升主线”，必须区分效应小、创新不足、证据不足及技术阻塞，不能一律声称假设已被证伪。

**Action-Mode 本轮只整理备忘，不运行新实验、不修改原仓库、不自动取消 AMR 方法目标。**

## 先看这四份

| 文件 | 用途 |
|---|---|
| `MASTER_AGENT_PROMPT.md` | 直接交总控 agent 的启动指令 |
| `docs/02_EXPERIMENT_MATRIX.md` | 核心实验、扩展上限与对照 |
| `docs/05_GO_NO_GO_RULES.md` | 可执行的最终决策规则及误杀保护 |
| `docs/08_ACTION_MODE_AUDIT_MEMO.md` | 六份审计对 A 的修改建议、分歧与采纳意见 |

其余依次是科学定义、数据、测量统计、执行调度、查重、产物合同及代码能力边界。`configs/` 是默认配置，`prompts/` 是原始提示词，`scripts/` 是参考运行器，`tests/` 是离线测试，`sources/` 保存全部六份评审原文及带行号版本。

## 立即启动

```bash
cd ICLR_MemoryPilot_ActionModeMemo_20260910
python3 -m unittest discover -s tests -v
python3 scripts/make_synthetic.py --out work/smoke_data --n 4 --tokens 3000 --tokenizer demo
python3 scripts/build_plan.py --profile configs/smoke.json --out work/smoke_plan.json
python3 scripts/run_pilot.py --histories work/smoke_data/histories.jsonl \
  --queries work/smoke_data/queries.jsonl --plan work/smoke_plan.json \
  --models configs/models.example.json --out work/smoke_run --mock
python3 scripts/analyze.py --scores work/smoke_run/scores.jsonl \
  --contrast direct,staged3 --budget 256 --out work/smoke_contrast.json
```

**上面只验证流水线，不是科研 pilot。** demo tokenizer、mock provider 和内置合成评测样本均被标记，不能生成科研 GO。正式运行先按 `docs/06_AGENT_EXECUTION.md` 完成模型能力/预算发现、真实 tokenizer 校准、数据准备和少量真实 API 联调。无需公开预注册；保留一次简短的探索→确认交接记录即可。

## 本包相对上一轮口头建议的修正

不把“必须出现 crossover/相变”作为门槛；稳定、较大且有决策价值的单调效应也可以 GO。没有找到 path 平均效应，不立即停止，而是检查预先列出的信息类型/预算差异、等调用次数 schedule 对照、证据保留和读出。重复改写解释了退化，不等于项目必死，但不能再冒充独立 bottleneck-schedule 机制。允许一次有理由的补测，禁止无限模型或 prompt 搜索。

## 代码边界

已包含：可控历史生成、LongMemEval 标准化、路径计划、stateless API/mock 参考运行器、逐步记忆落盘、成本预留/缓存、精确答案评分、history-cluster bootstrap、规则决策器和离线测试。

**仍由 agent 完成并以真实输出验证**：具体服务商兼容适配；自然数据语义评分；事实证据审计；严格实际长度敏感性；第二 reader/compressor 复现；必要的 streaming 扩展；最终 novelty 判定。这些均有输入/输出合同，不能用 mock 或手填通过标志替代。

未在此会话调用付费模型、租 GPU、访问私人聊天记录或运行真实 pilot。包内不含 API key、模型权重或第三方数据集正文。
