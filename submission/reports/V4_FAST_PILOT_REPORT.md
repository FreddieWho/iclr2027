# ICLR Memory Pilot V4 — P2 科学结果

**记录时间：2026-09-14 UTC。** P2 的 R1 与 R2 评分均已完成。科学结果为 `INCONCLUSIVE_NO_GO_SIGNAL`：点估计略偏向 direct，但不确定区间很宽；既不能确认 staged2 有稳健优势，也不能据此证明路径效应不存在。基于本次证据，不建议切换项目主线到 Memory。

## 一页回答

1. **Path effect 有多大？** R1 的 staged2−direct 为 **−4.55 个百分点**（history 配对 n=11，90% CI [−15.91, +4.55]）；R2 全部完整 history 上为 **−3.13 个百分点**（n=12，90% CI [−16.67, +9.38]）。在 R1/R2 共同完整的 11 个 history 上，两者均为 −4.55 个百分点。方向上是 direct 稍好，但区间跨零且样本有限，不能作有益或有害的确定性结论。
2. **是否只是 memory 长短导致？** 不能确定。8192 是软目标，不是硬上限；52 个自然停止的最终 memory 全部超过它，三条路径的中位长度约 17.4K–18.6K tokens。长度平衡后的点估计仍偏负，但仅 n=11；等长回归截距也偏负、区间宽且跨零。因此长度分析没有消除点估计，却不足以证明长度不是主要解释；本实验不是严格等 native-token endpoint 比较。
3. **是否只是 repeated rewrite？** 总体上 staged2 与 rewrite 的差异很小且不确定：R1 staged2−rewrite **+2.27 pp**，R2 **−1.04 pp**，二者区间都跨零。信息类型的点估计呈现关系多跳偏负、例外/撤回偏正的不同模式，但每类只有 13–14 个 history 且区间宽，尚不能确认任务相关的路径机制。
4. **第二 reader 是否看到同方向？** 是。在共同完整的 11 个 history 上，R1 与 R2 的 staged2−direct 均为 −4.55 pp；R2 全体完整 history 的均值为 −3.13 pp。R2 是同一冻结 memory 的敏感性读取，不是新的压缩实验。
5. **是否值得切换项目主线？** 当前不建议切换到 Memory，因为没有达到预定 GO 证据。这个建议不等于 Memory 无效，也不构成科学 NO-GO：P2 结果不精确，且 technical missingness 较高。

## 技术路线—结果/证据—结论

### 设计和阶段边界

- P2 使用 24 个新 synthetic histories、每个 8 个问题；统计单位是 **history**，不是问题数。比较 direct、staged2、rewrite。
- P2 冻结 prompt 为 `prompts/compress_v4_route_selection_compact_v2.txt`（SHA-256 `2612dc3e…299092d`），requested final target 为 8192 DeepSeek-native tokens；协议明定此值不是硬上限，禁止 post-hoc 截断。
- query-blindness 通过：压缩器未打开 query 文件，未来 query 未进入压缩输入。没有 post-hoc memory truncation。
- 初始 P0 的 `NO_GO_TECHNICAL` 是当时的技术停机状态，不是科学结果。之后 owner amendment `V4_OWNER_CONTINUATION_AFTER_P0_TECHNICAL_FAILURE_20260914_V2` 授权继续；compact-v2 P0 仍只是校准，不计入科学分析。该 P0 的 6 个 final memory 中 4 个自然停止，2 个 direct technical invalid；4 个自然输出均超过 8192。
- P1 的 12-history exploration 使用旧 prompt（SHA-256 `253a1e81…79d3d0bb`），36 个 final memory 中 12 个 technical invalid。P1 与新 prompt 的 P2 不合并；因此 P2 是 owner-amended route-selection evidence，不是对 P1 同一 prompt 的独立复制。
- 按用户后续指示，R2 在 R1 未达 GO 信号后仍执行，读取同一冻结 memory。此项是 reader sensitivity，明确记录为对早先 “primary signal 后才执行 R2” 触发条件的授权覆盖。

### 压缩器与实际长度

C1 为 OpenCode Go `deepseek-v4.1-flash`，thinking disabled，`reasoning_effort` 未发送；final/intermediate provider output budget 分别为 24,576/32,768。P2 有 72 个最终 memory 单元，其中 52 个 natural stop、20 个 technical invalid。technical invalid 按缺失处理，不按 memory 失败计分。

| 路径 | natural stop | technical invalid | natural-stop memory 长度中位数 [IQR] | 范围 |
|---|---:|---:|---:|---:|
| direct | 18/24 | 6/24 | 17,413.5 [15,645, 20,730] | 14,316–22,307 |
| staged2 | 14/24 | 10/24 | 17,767.5 [15,766.75, 18,395.75] | 15,028–19,762 |
| rewrite | 20/24 | 4/24 | 18,643.5 [16,573.25, 21,964] | 14,909–24,541 |

所有 natural-stop memory 都超过 requested 8192；staged2 的 technical invalid 比例也最高。差异性缺失会使 complete-case paired estimate 有选择偏差风险。P2 压缩收到 113 个 provider response：94 个 `stop`、19 个 `length`；另有 1 个 shell-wrapper timeout 请求隔离为 uncertain，没有重发。P2 C1 实际 usage 约 `$3.79955`，另有 `$0.0493242` 隔离预留。

### Reader 结果与元数据

| Reader | Provider/model | Effort；output budget | 完成/有效评分行 | 有效三路径 history | 可见批量响应长度 |
|---|---|---|---:|---:|---:|
| R1 | OpenCode Go / `glm-5.3-flash` | low；12,000 | 768 行；600 valid、160 因 compressor memory invalid、8 schema invalid | 11/24 | 接受响应最大 512 tokens；两次超限尝试为 535、530，按协议各重试一次，没有截断；单题答案最大 27 tokens |
| R2 | OpenCode Go / `deepseek-v4.1-flash` | low（由 R1 calibration 解析）；12,000 | 576 行；416 valid、160 因 compressor memory invalid | 12/24 | 最大 420 tokens；无超限、无重试；单题答案最大 9 tokens |

两位 reader 的 `finish_reason` 均为 `stop`（R1 78 个 response attempts；R2 52 个），返回模型字段均与指定模型一致。provider output usage 包含隐藏 reasoning；provider 未提供单独 reasoning token 数，按 `provider_output_tokens - visible_response_tokens` 记录 native estimate：R1 15,336、R2 156,117。R1/R2 的 provider input/output usage 分别为 1,861,735/39,925 与 990,253/172,300 tokens，cache read 均为 0；reader usage 分别约 `$0.299223` 与 `$0.503836`。

### P2 history-level estimates

效应定义：`Δ_SD = U(staged2) − U(direct)`；所有区间为按 history bootstrap 的 90% CI，20,000 replicates。

| 分析 | R1 (n=11) | R2 (n=12；长度平衡 n=11) |
|---|---|---|
| staged2 − direct | −0.0455 [−0.1591, +0.0455] | −0.0313 [−0.1667, +0.0938] |
| rewrite − direct | −0.0682 [−0.1591, +0.0227] | −0.0208 [−0.1042, +0.0625] |
| staged2 − rewrite | +0.0227 [−0.0909, +0.1364] | −0.0104 [−0.0938, +0.0729] |
| 长度平衡 staged2 − direct | n=11：−0.0455 [−0.1591, +0.0455] | n=11：−0.0455 [−0.1818, +0.0909] |
| 等长回归截距 | −0.0446 [−0.1509, +0.0668] | −0.0345 [−0.1693, +0.1188] |

共同完整的 11 个 history 上，R1 和 R2 的 staged2−direct 均值恰为 −0.0455；这是描述性 matched-subset sensitivity，没有另行计算该子集 bootstrap CI。R1 的路径 win rate 为 0.182，低于稳健 GO 条件。分类型 staged2−direct 的关系多跳点估计在 R1/R2 分别为 −0.231/−0.214；例外/撤回分别为 +0.154/+0.214。例外减关系多跳的交互点估计为 R1 +0.385（n=13）、R2 +0.429（n=14），提示可能存在类型差异；但分类型 90% CI 宽且样本均低于 conditional-GO 的每类至少 18 histories 要求，不能作确认性子组结论。

### 决策和边界

按 `docs/ICLR_MemoryPilot_V4_FastDecision_20260913/docs/04_ROUTE_DECISION_RULES.md`，当前未达到 robust 或 conditional `GO_MEMORY`。同时 NG1–NG5 的强条件也均未被满足，因此不能把结果标成科学 `NO_GO_MEMORY`；旧的 `NO_GO_TECHNICAL` 仅适用于 continuation 前的 P0 停机。原三标签规则没有覆盖本次“真实 P2 已完成、但 GO 与强 NO-GO 均不成立”的状态；按用户最新指示，当前结论记录为 **INCONCLUSIVE_NO_GO_SIGNAL**，而不是伪造负结论。

实际工作建议：**不要仅凭本 pilot 切换 Memory 主线；保留该假设为未决。** 这不是自动启动新的 benchmark 或 P3；P3 触发条件未达到。不得将 600/416 个问题级有效行当作独立样本，也不得宣称等长 endpoint 已得到验证。

## 可复核 artifacts

- 机器可读科学结果与完整统计：`artifacts/v4_p2_compact_v2_science_result_20260914.json`
- P2 每个 history 的 R1/R2 utility、长度和有效评分数：`artifacts/v4_confirmation_rows_compact_v2_20260914.jsonl`（48 history-reader 行；旧 `v4_confirmation_rows.jsonl` 保持为续跑前快照）
- compact-v2 P2 frozen memories：`work/v4_fast_decision_20260914_compact_v2/p2/live/memories.jsonl`；SHA-256 `3d11ba615f468b3603a47201e4a382006c3016ac82fea8bdf4795d9f8590330e`
- P2 compression manifest：`work/v4_fast_decision_20260914_compact_v2/p2/live/compression_manifest.json`；SHA-256 `de812102d31c5cf32bc6853dba8b1848556e7b14ed811701be3b2bcb23c03333`
- R1 manifest/score rows：`work/v4_fast_decision_20260914_compact_v2/p2/r1_low/reader_manifest.json` / `reader_rows.jsonl`
- R2 manifest/score rows：`work/v4_fast_decision_20260914_compact_v2/p2/r2_low/reader_manifest.json` / `reader_rows.jsonl`
- compact-v2 P0 calibration：`artifacts/v4_p0_compact_v2_technical_calibration_result_20260914.json`
- P1 exploratory result (old prompt; not pooled)：`artifacts/v4_p1_exploration_result_owner_continuation_20260914.json`
- P1 12 个 history 的 exploratory utility rows：`artifacts/v4_exploration_rows_owner_continuation_p1_20260914.jsonl`（旧 prompt，明确不与 P2 pooled）
- owner continuation amendment：`artifacts/v4_owner_continuation_amendment_20260914_v2.json`
- 旧 `artifacts/v4_confirmation_rows.jsonl` 与 `artifacts/v4_exploration_rows.jsonl` 保留为 continuation 前的 `NOT_RUN` 状态快照；它们不是当前 phase 结果源。实际 P1/P2 per-history 行见以上 owner-continuation artifacts，问题级收据见对应 reader rows。
