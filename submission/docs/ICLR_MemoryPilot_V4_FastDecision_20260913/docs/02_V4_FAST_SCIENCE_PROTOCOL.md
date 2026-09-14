# 02｜V4 快速科学检验协议

## 1. Scientific estimand

对历史 `H_i`、final requested target `B`：

- `D_i = C(H_i; B)`
- `S_i = C(C(H_i; 2B); B)`
- `R_i = C(C(H_i; B); B)`

reader utility：`U_i(M)`。

核心：

`Δ_SD,i = U_i(S_i) - U_i(D_i)`

rewrite control：

`Δ_RD,i = U_i(R_i) - U_i(D_i)`

机制区分：

`Δ_SR,i = U_i(S_i) - U_i(R_i)`

## 2. 为什么不再要求 exact final token equality

当前 serving API 没有 separate visible-output hard cap。强制模型自然恰好 obey 1K/2K 只会测试 prompt/token-counting 能力。

V4 用三个层次控制终点容量：

1. **same requested target**：D/S/R 最后一步完全相同的 B 与 prompt；
2. **observed length reporting**：每个 final memory 用冻结 DeepSeek tokenizer 重新计数；
3. **equal-length sensitivity**：
   - matched subset：`max(L_a,L_b)/min(L_a,L_b) <= 1.25`；
   - paired regression：`ΔU_i = α + β ΔlogL_i + ε_i`；`α` 是 equal-length proxy；
   - 报告 length difference 与 utility difference 的相关关系。

若 path effect 在这些控制后消失，本轮不得 GO。

## 3. Runtime

### C1
- provider/model: 现有 OpenCode Go / `deepseek-v4.1-flash`
- thinking: disabled
- query-blind: true
- final requested target: 4096 native tokens
- intermediate requested target: 8192
- final provider max_tokens: 12288
- intermediate provider max_tokens: 16384
- natural stop required
- no posthoc truncation

只在 P0 技术失败时允许一次：
- B -> 8192
- intermediate -> 16384
- final max_tokens -> 16384
- intermediate max_tokens -> 24576

### R1
- `glm-5.3-flash`
- initial reasoning: low
- oracle <0.90 时一次升级 high
- 正式 run 只能有一个冻结 reader setting

### R2
- `deepseek-v4.1-flash`
- 只读 P2 已冻结 memories
- 作为 reader generalization，不作为第二 compressor

## 4. Data

### P0 technical
2 histories，不入统计。

### P1 exploration
12 独立 synthetic histories。保留现有四类：
- atomic factual
- temporal update/provenance
- relational multi-hop
- exception/retraction/constraint

若现有每 history 8 Q 已完成，继续使用；不要重造数据增加漂移。

### P2 confirmation
24 个全新 histories。必须与 P1 不重叠；冻结 P1 后才生成/揭示 P2 scoring results。

## 5. Reader scoring

优先使用可程序化 exact/normalized scoring。若答案具有同义表达：
- 先用预先定义的 normalization；
- 不要在看到路径结果后人工给某 arm 改 grader。

一个 history 的 utility 是其 questions 的平均 score。统计 resampling unit = history。

可批量让 reader 回答一个 history 的全部 questions；所有 arm 采用相同方式。若 parse failure >5%，统一切换逐题并重跑正式 reader，不混用两种模式。

## 6. Analyses

### 必须
1. `mean Δ_SD` + paired bootstrap 90% CI；
2. `mean Δ_RD` + CI；
3. `mean Δ_SR` + CI；
4. path win-rate across histories；
5. 四类信息的预设 subgroup effect；
6. direct/staged/rewrite actual visible length distribution；
7. `Δ_SD` on length-balanced subset；
8. equal-length regression intercept；
9. R1/R2 direction agreement on P2 frozen memories。

### 可选但不得阻塞
- LongMemEval 8-history check；
- 更复杂的 semantic evidence audit；
- 第二 budget。

## 7. 一次允许的技术调整

只有 P0 可调整一次：
- B/2B 与 provider cap；或
- R1 low->high；或
- 修复一个明确 adapter/parse bug。

如果需要第二轮协议改动才可运行，终止为 NO_GO_TECHNICAL。本项目此时不值得继续消耗切题窗口。
