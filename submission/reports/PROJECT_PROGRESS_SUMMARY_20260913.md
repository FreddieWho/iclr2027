# ICLR Memory Pilot 项目进展汇总

**报告日期：2026-09-13（Asia/Shanghai）**  
**用途：供决定是否续跑当前预检、修订技术方案或停止项目。** 本报告按当前工作区与可追溯 artifacts 汇总；不把技术探针当成科学实验结果。

## 一页结论

- **项目当前机器决策：`NO_GO / TECHNICAL`。** `reports/decision.json` 的理由是必需审计未完成/失败；失败项包括 technical validity、query-blind、budget/length、rewrite controls、independent confirmation、generalization、novelty 和 task relevance。这不是“direct 与 staged 等价”或“科学假设被证伪”。
- **完整正式 P0 尚未运行。** V3 C1 可见预算预检只完成 4/40 格，另有 1 格在 HTTP 等待时被中断、35 格未尝试。R1 未调用，没有正式 memory score。
- **V3 有一个需要正面披露的执行偏离：**冻结计划规定除全局授权、配置或预算阻塞外继续执行；当时没有这些阻塞，我提前停止了矩阵。部分格子已排除首选 `[2048,4096]` 通过的可能，但这不等于完成 40 格率估计。当前状态应读作 **`INCOMPLETE_PROTOCOL_DEVIATION`**。
- **下一步的关键选择：**续跑原矩阵剩余 35 格以完成描述性预算曲线；接受不完整结果并停止；或审批新 prompt/预算候选和预先声明的停止规则，再用新样本重做预检。

## 项目目标与成功门

本项目要在相同原始历史、相同合格的最终 memory 预算、压缩时 query-blind 的条件下，比较 direct 与 staged memory 路径。主要估计量是配对 utility 差；目前没有可用于估计它的有效 scores。

项目级 GO 还需要：至少 64 个新确认历史、至少一个独立泛化方向、长度敏感性与 rewrite/same-K 对照、完成近邻查重及证据/任务相关性审计。P0 技术校准是进入后续实验的前置门，不是科学证据本身。总控执行合同见 `MASTER_AGENT_PROMPT.md`，决策规则见 `docs/05_GO_NO_GO_RULES.md`。

## 各工作流状态

| 工作流 | 当前状态 | 已有证据与边界 |
|---|---|---|
| 本地 runner、tokenizer、数据合同 | 工程准备大体完成 | 有真实 native tokenizer、运行配置和 mock/smoke 路径；smoke 不算 live science。V3 专项测试 6 项通过；V3 代码加入后没有重新跑完整测试套件。 |
| 合成 P0 数据 | 数据准备完成，未进入统计 | 历史报告登记 8 histories / 64 questions；本次 V3 技术预检另使用 8 条约 32K-token 合成历史，且未把 query 暴露给压缩器。 |
| LongMemEval 自然数据 | 准备完成，语义评测未运行 | 已准备 32 条完整历史；没有完成自然数据 grader/semantic evaluation。 |
| P0 C1 预算预检 V3 | **不完整、有协议偏离** | 4 个响应格完成；第 5 格本地中断，35 格未尝试。首选预算对已有失败格，正式预算仍未冻结。 |
| P0 正式校准（raw/no-memory/oracle/direct） | **NOT_RUN** | 没有正式的配对评分、reader oracle accuracy 或 P0 score rows。 |
| R1 reader | **完整校准 NOT_RUN** | 旧合同下有一个 4-token natural-stop 短答控制；它不代表当前 low/12,000 合同的 oracle 校准。 |
| P1 exploration | **NOT_RUN** | 计划中的 32-history 探索矩阵尚未执行。 |
| P2 confirmation | **NOT_RUN** | 至少 96 个新历史的确认实验未执行；没有独立确认数据。 |
| 泛化、length/rewrite、证据审计 | **未完成** | R2/C2/自然泛化和配对 evidence audit 均没有形成完整实测结果；近邻文献审计主要方法/实验核验完成，但附录、代码等覆盖仍不完整。 |

## 当前有效运行配置

V3 文件 `configs/models_opencode_go_p0_visible_budget_v3.json` 是继承式配置；直接文件中的模型字段为空是因为从父配置解析。最近实际运行时的 resolved config 为：

| 角色 | 有效配置 | 当前状态 |
|---|---|---|
| C1 compressor | OpenCode Go `deepseek-v4.1-flash`；`reasoning_effort=low`；thinking enabled；provider `max_tokens=65536`（reasoning 与可见内容共享）；DeepSeek tokenizer revision `dba1be0a40aa45a94ad051997016db3960a90277` | 已用于 V3 预检；没能满足当前可见长度门。 |
| R1 reader | OpenCode Go `glm-5.3-flash`；初始 `low`；provider output 12,000；最终可见答案 ≤512 tokens；oracle accuracy <0.95 或 relational multihop / exception-retraction 系统失败时，才提升至 high 并固定用于所有正式 R1 | 未在本轮调用。模型列表快照中 `glm-3-flash` 不可用、`glm-5.3-flash` 可用；该快照时间为 2026-09-12 17:53:52 UTC，不能当作永久可用性保证。 |

活动 V3 配置仍为 `PREFLIGHT_REQUIRED_NOT_FORMAL`：候选预算 `[1024,1536,2048,3072,4096]`，门槛 90%，样本量 8/预算；`final_memory_budgets_native_tokens=[]`，`preflight_status=PENDING`，没有冻结正式预算。旧 `configs/pilot.json` 仍列 `[1024,2048]`；正式 plan 重建时必须明确同步，不能让旧预算静默覆盖 V3 结果。

## 最新真实模型证据：V3 预检

运行计划为 `configs/visible_budget_preflight_v3_plan_20260913_v3.json`，计划 hash `8d2f143383125d584495f70202f60eb29b41c3d43f981503408f578bffe5a678`。模型目录响应使用已保存且 hash 校验的快照；C1 请求没有重新发模型列表 GET。每个候选的正式预检门要求 8/8 合规，因为 `ceil(0.90 × 8)=8`。

| 候选预算 | 可见 memory tokens | provider output / reasoning tokens | finish | 判定 | 延迟 / 本格 USD-equivalent |
|---:|---:|---:|---|---|---:|
| 2,048 | 4,199 | 26,389 / 22,189 | `stop` | `BUDGET_NONCOMPLIANT` | 100.3 秒 / $0.04159380 |
| 3,072 | 8,123 | 61,560 / 53,436 | `stop` | `BUDGET_NONCOMPLIANT` | 213.8 秒 / $0.08379900 |
| 4,096 | 7,336 | 65,536 / 58,199 | `length` | `TECHNICAL_INVALID` | 1,437.4 秒 / $0.08857020 |
| 1,536 | 2,548 | 30,671 / 28,122 | `stop` | `BUDGET_NONCOMPLIANT` | 115.8 秒 / $0.04673220 |
| 1,024 | 无有效返回 | usage/finish 未知；HTTP 响应等待中由本地中断 | 未知 | 不可作为合规格；不得重试 | ledger 保留上限 $0.08863800 |

以上 4 个已完成响应均来自 `syn-20260913-00000`。2,048 与 4,096 的已观察失败意味着首选对不能达到预定 8/8；1,536、3,072 也各有已观察失败。1,024 不是模型行为证据，而是本地中断导致的无有效 artifact。其余 35 格尚未执行，因此没有完整的候选自然合规率。

**解释边界：** 输出预算只在 prompt 中要求严格上限，没有 native-token constrained decoding 或 post-hoc truncation。当前观测说明这套具体 prompt/model/runtime 合同无法稳定满足已试长度，不说明压缩后的记忆是否保留了科学任务相关信息，也不构成路径效应证据。

### 之前探针的关系

- 更早的高 reasoning / 较小 provider-output 探针多次以 `finish_reason=length` 结束且没有可见 memory；这些是技术故障证据，不是 memory 质量评分。
- 10 次 `low` reasoning length-only 请求曾全部自然 `stop`，但没有启用 formal memory 长度目标，不能当作长度合规证据。
- V5 曾有一个 2,048-token 单次探针自然停止且长度合格，同时 1,024-token 探针超限。它是单点能力探针，不是 8 样本自然合规率；与本次 V3 4/40 结果不合并、不替代 V3 计划。

## 协议偏离、成本和复现

- 计划的 retry/stop 规则原文要求：不重试，provider/API errors 计不合规；除全局授权、配置或预算阻塞外继续执行。本轮停止不是该规则授权的条件。恢复原计划时需保留这一偏离记录，不可把它擦除。
- 当前 run 状态写在 `work/p0_visible_budget_preflight_v3_20260913/run_03/run_manifest.json`；逐格输出在 `calls.jsonl`；费用在同目录 `cost_ledger.sqlite`。机器可读部分结果为 `artifacts/visible_budget_preflight_v3_partial_20260913.json`，完整预检 canonical artifact 未生成。
- 累计账本 actual-plus-uncertain 为 **$1.37211145**：历史结转 $1.02277825、本轮 4 个已结算请求 $0.26069520、被中断请求的保守预留 $0.08863800。该值是 OpenCode Go usage-value 账本估算，不是信用卡账单。
- 原 40 格整批预检的保守上界为 $3.54549；当前 run 估计的 C1 累计最坏上界（含预检及计划中的 P0 C1 请求）为 $5.98658185，低于 DeepSeek $8 模型 cap；项目 cap $18、GLM cap $15，无 auto-topup、无付费 fallback。4 个已返回响应的请求耗时约 100 秒至 24 分钟；不能据此精确预测续跑时长，但剩余矩阵可能需要数小时。
- 复现关键值：C1 tokenizer revision `dba1be0a40aa45a94ad051997016db3960a90277`；`GET /v1/models` 响应 SHA256 `46ffe9c29889eef6fb5f66d087a3c73295f09d8971a500554b09e2cd0d631fb8`；prompt SHA256 `88685aa2c4b3c9677797904c907cc45a0863171ff99ca88b1654fdbd4df46828`。请求/响应原件与所有逐格字段留在 `run_03`；没有记录 API key。

## 下一步可选路径

| 选择 | 做什么 | 好处 | 代价/边界 |
|---|---|---|---|
| **A. 续跑原 V3 矩阵** | 不重试 4 个已完成格；把中断的 1,024 格记为本次无有效结果/不合规；只发剩余 35 个未尝试格，沿用原 model、prompt、candidate order 与账本。 | 补齐候选预算率，遵循原 40 格样本设计；保留同一批次与 hash。 | 早停偏离仍须披露；首选 `[2048,4096]` 已不能通过，续跑只增加其他预算的描述性信息，不能自动选 fallback；耗时可能数小时。正式 P0 仍不会自动开始。 |
| **B. 接受当前不完整结果并停止** | 保留 `INCOMPLETE_PROTOCOL_DEVIATION`，不再调用模型。 | 不新增费用；准确保留当前 evidence boundary。 | 不能说完成 V3 预检，不能给全矩阵合规率，也不能运行正式 P0。 |
| **C. 设计 V4 技术预检** | 先审批新的长度 prompt 和/或候选预算，固定模型/effort/样本及 futility stop；用新预检样本，不重试本轮失败格。 | 目标是找到实际可行的正式预算；规则可避免再次无效跑满矩阵。 | 这是显式协议修订；新预算、成本上界及其对研究问题的影响需先审定。只提高 65,536 provider output 上限不能解决已观察到的可见长度超限。 |
| **D. 终止本项目当前科学验证** | 将现有总体 `NO_GO/TECHNICAL` 作为停止决定，归档技术阻塞。 | 结束继续投入。 | 不产生 direct/staged 的科学结论，不能写成假设被证伪或两路径等价。 |

**建议的决策顺序：** 先决定 A/B/C/D；如果选 A，先实现只续跑未尝试格的 checkpoint-safe 入口，并把中断格纳入记录而不重试。如果选 C，先冻结新的 prompt/budget/sample/stop rule，再发任何请求。无论哪条路，R1 calibration 应等 C1 正式预算门明确后再做；若 C1 门不能打开，R1 结果不会让正式 P0 可评分。

## 交付与工作区状态

- 关键报告：`reports/PILOT_REPORT.md`、`reports/P0_VISIBLE_BUDGET_V3_RESULT_20260913.md`、`reports/PREFLIGHT.md`、`reports/decision.json`。
- `reports/decision.json` 仍是现有 `NO_GO/TECHNICAL` 快照；没有用这份部分预检重新生成 decision input/decision。
- 当前工作区有既有 tracked modifications 和 untracked files。本次没有 commit 或 push；本汇总文件及 P0 部分结果仍需纳入后续审阅的提交 payload。
