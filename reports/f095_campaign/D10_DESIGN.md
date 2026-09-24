# D10｜设计（决策前沿，纯分析零训练，2026-09-23）

## 目标
把P2从"仿射control拟合某个覆盖点"变成可读的决策边界：同覆盖比质量、同成本比质量、前沿说话，不挑λ讲故事。

## 复用（只读，不重训不重挖）
- `p2_table.npz`（256场景×49动作行，6模型逐行logit，可行性标签，动作成本；manifest sha已记）。
- dev/eval划分沿用v2（排序前86场景∩feasible＝dev，余eval），覆盖/概率均值/阈值选择掩码完全一致。

## 构造（`d10_frontier.py`）
- 策略族：逐场景在maxP0候选行上τ-阈值执行（与v2 `q_at`同族，去掉τ=0.5固定）。**完整可达阶梯**：每模型取dev maxP0的全部互异值（≤86级），不用21-分位网格近似。
- 每（模型，τ）报告：执行率、执行集成功率（own-acted quality）、执行总成本/均成本、候选precision/recall。dev选匹配点，eval公布实际覆盖差（不假设dev匹配能迁移）。
- 同覆盖比较 vs 同成本比较分开做；coverage–cost–risk前沿（risk＝1−own质量）逐模型输出前沿图数据。
- own-acted质量差CI：同场景paired bootstrap（同一次重采样内重算各自比值）；common-acted（双方都执行的固定集）另作不同estimand、各自CI，不借用。
- 强控制：dev选出的clean最佳阈值；dev拟合的严格单调校准器（只动工作点；F3看它能否复制flip增益）。
- test-label优化前沿只作诊断上界，单列，不宣称可部署；任何用eval分布的转导设定单列（本轮不做，只留接口标记）。

## 无泄漏单测（`test_d10_noleak.py`）
1. 选择函数只收dev数组：eval行置换后输出逐位一致。
2. 可达阶梯恰为dev maxP0互异值全集（无网格近似、无四舍五入假ties）。
3. 单调校准器保序（dev分数Spearman=1）。

## 可证伪分支（冻结）
- F1：同覆盖下flip质量－clean质量的paired CI下界>0（任一匹配点）→ 实际决策收益成立。
- F2：同成本下结论同F1 → 收益非"多花钱买来的"。
- F3：单调校准复制flip增益≥一半 → 收窄representation解释（增益多为工作点）。
- 判决：F1∧F2 → 决策收益SUPPORTED；仅F1 → 覆盖-成本权衡 reporting；F3 → 收窄。
