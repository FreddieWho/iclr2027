# V4 Fast Memory Pilot — Route Decision

## 1. 一页结论

- **Decision:** GO_MEMORY / NO_GO_MEMORY / NO_GO_TECHNICAL
- **最强确认效应:**
- **90% CI / n histories:**
- **长度调整后效应:**
- **rewrite control:**
- **R2 replication:**
- **一句话科学解释:**
- **是否切换 ICLR 主线:**

## 2. 当前实验到底测试了什么

说明：本 V4 使用 same requested final target + natural-stop outputs + observed-length control，不是严格 fixed-token endpoint benchmark。

## 3. P0 技术状态

| item | result |
|---|---|
| C1 config | |
| B / 2B | |
| natural stop rate | |
| visible length range | |
| R1 oracle accuracy | |
| raw vs no-memory | |
| query blind audit | |

## 4. P1 exploration

完整报告 direct / staged2 / rewrite，不挑最好结果。

## 5. P2 confirmation

### 5.1 Primary paired effects

| contrast | mean | 90% CI | history win-rate | n |
|---|---:|---:|---:|---:|
| staged2 - direct | | | | |
| rewrite - direct | | | | |
| staged2 - rewrite | | | | |

### 5.2 Actual memory lengths

| path | median | IQR | min-max |
|---|---:|---:|---:|
| direct | | | |
| staged2 | | | |
| rewrite | | | |

### 5.3 Length controls

- balanced-subset n:
- balanced-subset ΔSD:
- regression equal-length intercept:
- 是否主要由 final length 解释：YES/NO

### 5.4 Information types

四类全部列出，不只列阳性类型。

### 5.5 Reader generalization

R1 与 R2 对同一 frozen memories 的结果。

## 6. 失败案例/成功案例

至少各 3 个，列 H 中的关键证据、三条 memory 的相关片段、最终答案。不要把 anecdote 当统计结果。

## 7. 成本与运行稳定性

调用数、usage-value、finish_reason、technical invalid rows。

## 8. GO/NO-GO 映射

逐条对照 `docs/04_ROUTE_DECISION_RULES.md`，明确哪条满足/不满足。

## 9. 下一步

### 若 GO
只提出发表级下一步：strict fixed-budget serving、第二 compressor、更多 histories、自然数据、完整 novelty audit。不要自动执行。

### 若 NO_GO
明确回 Action-Mode，并列出应采用的 Action-Mode 审计修正：support/coalition matched controls、任务语义拆分、独立确认、避免把 Jacobian 一阶近似当 headline。
