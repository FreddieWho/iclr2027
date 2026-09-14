# MASTER AGENT PROMPT — V4 Fast Route Decision

你接手的是一个已经被技术预检拖慢的 ICLR Memory Pilot。你的任务不是修复所有历史协议，也不是把 V3 做完整，而是**最快、尽可能准确地得到足以决定是否切换研究主线的科学证据**。

## 0. 先纠正当前状态

当前 `reports/decision.json = NO_GO/TECHNICAL` 不是科学 NO-GO。正式 direct-vs-staged utility 尚未运行。V3 只完成 4/40 个有效响应格，另有 1 个本地中断；其余未执行。原 hard-budget preflight 把项目阻塞在“模型能否自然严格遵守 1K/2K token 上限”，这不是本轮要检验的核心科学假设。

**不要续跑 V3 剩余 35 格。不要删除或改写 V3 历史。** 按 `docs/05_MIGRATION_FROM_V3.md` 将其冻结为 `SUPERSEDED_FOR_ROUTE_SELECTION`。

## 1. 本轮唯一科学问题

在相同历史 H、相同 compressor 配置、相同 requested final memory target 下：

> direct compression 与 staged compression 是否产生稳定、任务相关、且不能主要由最终可见 memory 长度差异或单纯 repeated rewriting 解释的 utility 差异？

这是 route-selection proxy，不是最终论文里“严格相同 native-token endpoint”的完整证明。

## 2. 允许并要求的协议让步

### 2.1 Compressor

C1 保持模型 `deepseek-v4.1-flash`，但 V4 **关闭 thinking**。理由：V3 low-thinking 已产生 2–6 万 reasoning tokens、100 秒到 24 分钟延迟，并且破坏可见长度控制。path 假设不要求 thinking；只要求各路径使用相同、冻结的 compressor 设置。

默认：
- C1 model: `deepseek-v4.1-flash`
- thinking: disabled
- reasoning_effort: 不发送/记录为 N/A
- provider max_tokens: final calls 12,288；intermediate calls 16,384
- temperature / sampling：沿现有 neutral deterministic-ish 配置；若仓库没有冻结值，用 temperature=0 或 provider 允许的最低值，并全程不变
- query-blind：必须保持
- post-hoc truncation：禁止
- finish_reason=length：technical invalid；只允许按本包规则一次扩大 provider cap，不允许裁剪结果

不要把 V4 与 V3 thinking-on 结果合并。

### 2.2 Final target

默认 requested final target `B*=4096` DeepSeek native tokens；intermediate target `2B*=8192`。

V4 **不要求模型输出必须 <=4096 才能评分**。对所有 natural-stop memory：
- 记录实际 DeepSeek-native visible tokens；
- 记录 compression ratio；
- 进入主 utility 评分；
- 用长度匹配与长度调整分析判断 path 效应是否仍存在。

只有在联调中出现极端不可比（例如大多数 final output <2048 或 >8192，或 finish=length）时，才允许一次调整到 `B*=8192`, intermediate=16384，并冻结。禁止继续扫 1024/1536/2048/3072/...。

### 2.3 样本和路径

本轮只做：
- `raw`
- `direct`: H -> B
- `staged2`: H -> 2B -> B
- `rewrite`: H -> B -> B
- `oracle`, `no_memory` 只做 reader calibration

不要做 staged3、wait3、多预算 factorial、第二 compressor、AMR 或其他旁支。

探索：12 个独立合成历史。  
确认：24 个**新历史**。  
每历史继续使用现有 4 类信息；可沿用 8 questions/history，但统计单位是 history。

### 2.4 Reader

R1 保持 `glm-5.3-flash`。先用现有 low setting；用 oracle/raw/no-memory 做最小校准。

若 oracle accuracy <0.90，允许一次提升到 high 并从头重跑所有正式 R1；否则不得变更。reader 输出尽量用结构化短答案；优先一次回答同一 history 的全部 questions 以减少调用，但如果 parse failure >5%，切回逐题且全部正式条件统一。

确认阶段冻结 memory 后，用 R2=`deepseek-v4.1-flash` 复读相同 memory，作为 reader-family 泛化检查；R2 不重新压缩、不改变 memory。

## 3. 执行阶段

### P0 — 2-history fast technical calibration

只用 2 个不进入统计的 history，生成 direct/staged2/rewrite。

通过条件：
- 至少 5/6 个 final memory natural stop；
- final visible length 的多数位于 [0.5B, 2B]；
- 没有 query leakage；
- R1 oracle >=0.90 或经一次 high 提升后 >=0.90；
- raw 明显优于 no_memory（至少 +0.20 accuracy，若任务先验导致此差不成立则人工审查 8 个问题确认仪器仍有效）。

失败时只允许一次技术修改：
- 若输出过长/finish=length：改 `B=8192, intermediate=16384` 和相应 cap；
- 若 reader 弱：low -> high；
- 其他 adapter bug 可以修一次。

仍失败 -> `NO_GO_TECHNICAL`，停止。

### P1 — 12-history exploration

对 12 个历史生成 direct/staged2/rewrite，raw 不压缩。冻结全部 memories 后由 R1 统一评分。

必须产生：
- per-history utility；
- final visible token length；
- `Δ_SD = U(staged2)-U(direct)`；
- `Δ_RD = U(rewrite)-U(direct)`；
- 四类信息的同类差异；
- paired bootstrap 90% CI（按 history resample）；
- length-balanced subset：`max(L_S,L_D)/min(L_S,L_D) <= 1.25`；
- length-adjusted regression：`ΔU_i ~ 1 + Δlog(length_i)`，报告 equal-length intercept 和不确定性；
- staged-vs-rewrite contrast：`U(staged2)-U(rewrite)`。

P1 不是终局。除非出现明确 futility：
- raw/oracle 仪器失效；或
- 所有三种 memory utility 都接近 no_memory；或
- final lengths 完全不可比且一次允许调整已经用完。

否则进入 P2；不要因为 12 个历史没有显著性而停。

### P2 — 24 new-history confirmation

完全冻结 P1 后的模型、prompt、B、解析、评分和阈值。生成 24 个新历史的 direct/staged2/rewrite；由 R1 评分。

随后不生成新 memory，使用 R2 读取同一批冻结 memory。

只允许在确认结束后计算正式 route decision。不要为了变成 GO 更改 prompt、B 或选择子组。

### P3 — 可选 8-history natural check

仅当 P2 已满足或非常接近 GO 时，如果现有 LongMemEval 32-history 数据和 grader 能**无需新开发复杂语义评分器**直接运行，则取预先固定的前 8 个合格完整历史运行 direct/staged2。它是加分证据，不是 route decision 的硬门。

如果需要超过一次 grader/adapter 修复，跳过并记录，不得因此阻塞最终决定。

## 4. 最终决策

严格使用 `docs/04_ROUTE_DECISION_RULES.md`。最终只允许：
- `GO_MEMORY`
- `NO_GO_MEMORY`
- `NO_GO_TECHNICAL`

不能输出 MAYBE / CONTINUE_EXPLORING。

若 GO：停止继续 Action-Mode 主施工，提出发表级 Memory 下一阶段，但**不要自动开始大 benchmark**。  
若 NO_GO：明确建议回 Action-Mode，并引用之前独立审计备忘中的必要修正。  
若 TECHNICAL：说明未检验科学假设，但因为当前目标是路线选择，本轮仍停止 Memory 投入，除非用户另行授权。

## 5. 明确禁止

- 不补 V3 剩余 35 格；
- 不再要求 1K/2K hard-cap compliance；
- 不做 post-hoc truncation；
- 不把 `max_tokens` 当 final memory length；
- 不为了控制长度重新生成“直到长度满意”为止；
- 不读未来 query 来做 compression；
- 不在 P2 后寻找新的 budget/prompt/model 救结果；
- 不要求 64/96/192 history 才允许 route decision；
- 不把 route-selection pilot 写成发表级 fixed-budget 证明；
- 不把 V3 protocol deviation 隐藏或修史。

## 6. 资源

沿用项目已有授权 cap；本包不提高任何总 cap。优先使用缓存和批量 reader。若预计 V4 会突破现有 DeepSeek/GLM/project cap，在发请求前计算剩余额度并停止请求用户授权；其余本地工作继续。

## 7. 必交付

- `reports/V4_PREFLIGHT.md`
- `configs/v4_frozen_runtime.json`
- `artifacts/v4_exploration_rows.*`
- `artifacts/v4_confirmation_rows.*`
- `reports/V4_FAST_PILOT_REPORT.md`
- `reports/V4_ROUTE_DECISION.json`

`V4_FAST_PILOT_REPORT.md` 第一页必须用通俗语言回答：
1. path effect 有多大？
2. 是否只是因为最后 memory 更长/更短？
3. 是否只是 repeated rewrite？
4. 第二 reader 是否看到同方向？
5. 最终是否值得切换项目主线？
