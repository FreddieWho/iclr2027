# 写作与复算的指标口径

四个状态按 `P, A, B, AB` 排列，`c_s` 表示该状态预测正确。统计总体 `E` 由 oracle 定义：A、B 各自保持 P 的标签，而 AB 改变标签。先固定 E，再计算模型正确性；模型筛选子集须另给分母。

| 指标 | 定义 | 必须说明 |
|---|---|---|
| atomic-joint accuracy / S | mean(c_A & c_B) | 不能标成 J3 |
| J3 / J | mean(c_A & c_B & c_AB) | 不要求 P 正确 |
| J4 | mean(c_P & c_A & c_B & c_AB) | 不能复制 J3 的值充当 J4 |
| conditional joint success | sum(c_A & c_B & c_AB) / sum(c_A & c_B) | 是成功率，不叫 CCM |
| CCM / conditional composition miss | sum(c_A & c_B & !c_AB) / sum(c_A & c_B) | 原子联合正确分母为 0 时记 null/undefined，不补 0 |
| baseline-110 子集 H | 基线 A、B 正确且 AB 错误 | 基线固定；跨模型时交代各自基线或共同基线 |
| endpoint repair | H 内修复模型 AB 正确的比例 | 分母固定为 H |
| full repair | H 内修复模型 A、B、AB 全对的比例 | 分母固定为 H |
| migration | H 内修复模型 AB 正确、至少一个原子错误的比例 | endpoint repair = full repair + migration |
| baseline-111 regression | 基线 A、B、AB 全对，修复后不再全对 | 报回归计数与基线 111 分母 |

本轮保存的纠正结果采用这些名称。历史 JSON 的同名字段若采用不同定义，必须按分组审查记录解释，不能按字段名自动混表。`151/176` 是历史封存 holdout 的条件漏检计数；它不是修复前后配对差，也不是 176 个独立 parent。

## 统计单位

- Quartet-level 点估计与 parent-cluster bootstrap 可以同时成立：抽样单位是 parent，每次保留该 parent 的所有 quartet。不能将它自动解释为等权 parent 平均。
- 每个 seed 的配对差和 parent-bootstrap 区间分别报告；seed 是训练重复，不能当作新数据集。三个 seed 均值本身没有替代跨 seed 不确定性的作用。
- 比较不同臂，必须匹配同一 quartet 的次序、标签、parent 和预测来源。完整支持还要求任务、银行、后端、预处理、选择规则一致。
- 多次查看的 dev 银行仍是探索证据；新生成的独立银行也要交代生成与选型顺序。封存汇总的复读不算新确认。

## 解释边界

- 测试样本少、seed 差方向不同：写“在该配方和银行未建立稳定优势”，不能写“不可能有效”。
- 修复 bug 后指标一致：只证明该项数值/代码合同，不证明所有机制成立。
- 有限搜索没找到碰撞：写观察到的最小距离；不能据此给全局最小距离的正下界。
- 换后端后未复现：写跨运行未复现；没有隔离硬件/软件等因素时不能归因为后端。
- 几何前端某配方训练失败：写该配方未学好接口；不能据此排除容量或证明接口不可学习。
