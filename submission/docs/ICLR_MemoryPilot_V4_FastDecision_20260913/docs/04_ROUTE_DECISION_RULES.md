# 04｜V4 路线切换 GO / NO-GO

## 0. 这是投资决策阈值，不是论文统计标准

V4 的问题是：**现在是否值得把 Action-Mode 暂停并切到 Memory？**

因此我们优先避免两类错误：
- 因为技术门槛过严而错过一个强现象；
- 因为 12 个样本的偶然波动就切掉已有积累的 Action-Mode。

最终只允许三种结果：`GO_MEMORY` / `NO_GO_MEMORY` / `NO_GO_TECHNICAL`。

## 1. GO_MEMORY — Robust route

P2 的 24 个新 history 满足全部：

1. R1 `|mean Δ_SD| >= 0.05`；
2. paired bootstrap 90% CI 不跨 0，或 CI 仅轻微跨 0 但 path win-rate >= 0.67 且 R2 同方向；
3. 长度控制后仍存在：
   - length-balanced subset n>=12 且同方向、`|mean Δ_SD| >=0.03`；**或**
   - equal-length regression intercept `|α| >=0.03` 且方向一致；
4. 不完全等于 repeated rewrite：至少满足一个：
   - `|mean Δ_SR| >=0.03`；
   - staged 与 rewrite 的信息类型 profile 明显不同，预设 interaction >=0.08；
   - staged 比 direct 更短却 utility 更高（或更长却 utility 更低），直接排除“只是更多 token 更好”的解释；
5. R2 对冻结 memory 的 `Δ_SD` 与 R1 同方向，且幅度至少为 R1 的 40% 或绝对值 >=0.03。

满足即 GO，不要求 crossover/phase transition。

## 2. GO_MEMORY — Conditional route

总体 `|Δ_SD| <0.05` 也可以 GO，但必须：

- 四个**预设**信息类型中有至少一个，在 P2 `|Δ_SD,type| >=0.10`；
- 该类型至少有 18 个可评分 history；
- 90% CI 大部分位于同一侧；
- length adjustment 后仍 >=0.06；
- R2 同方向；
- 不是 rewrite 产生完全同样的 pattern。

此时 claim 必须限定为对应信息结构，不能写 universal path effect。

## 3. NO_GO_MEMORY

满足任一强条件：

### NG1 — small / unstable
- P2 `|mean Δ_SD| <0.03`，且 90% CI 大致落在 [-0.05, +0.05]；
- 没有任何预设 type 达到 Conditional GO。

### NG2 — length explains it
- raw path effect 看似 >=0.05，但 length-balanced 与 adjusted intercept 均 <0.03 或方向反转；
- 即效应主要来自 staged 最终写得更多/更少。

### NG3 — rewrite explains it
- `Δ_SD` 与 `Δ_RD` 接近，`|Δ_SR| <0.02`；
- 类型 profile 也无明显不同；
- 则当前结果更像 generic re-encoding drift，不足以让项目切题。

### NG4 — reader-specific
- R1 有 GO 级信号，但 R2 方向反转或接近 0，且无明确 reader failure 证据。

### NG5 — only P1
- 只有 12-history P1 强，P2 不复现。

## 4. NO_GO_TECHNICAL

P0 一次允许调整后仍发生：
- >1/3 core final calls `finish_reason=length`/API invalid；
- final memory 极端长度导致 arm 不可比；
- oracle reader <0.90 且修复后仍失败；
- query-blind 无法保证；
- 现有授权/cap 无法完成最小 P2。

TECHNICAL 不是科学阴性。但本轮目的就是快速路线决策，因此默认动作仍是**停止 Memory，回 Action-Mode**，除非用户另行决定投入基础设施。

## 5. 不允许的“救结果”

P2 后不得：
- 换 compressor；
- 换 final target；
- 新增 5 个 subgroup；
- 调 prompt 再跑同一 P2；
- 只报告最好 reader；
- 把不显著说成等价；
- 把 8 Q/history 当 8 个独立样本。
