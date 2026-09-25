# Route 2 正式矩阵分析与最多3轮优化规则

## 固定停止原则

核心矩阵完成后先核对12个run是否完整、finite、无OOM、数据hash一致。科学结果不按单个最高J挑选；seed、arm、exposure和parent必须完整保留。

### 核心结果可接受

满足以下条件时，不消耗额外优化预算，直接回收并独立审阅：

1. 12/12 arm-seed run完成，checkpoint和result/prediction文件齐全；
2. 所有指标finite，训练/dev选择日志完整；
3. interaction、representation相对additive的方向在至少2/3 seed上可解释，且没有明显atomic competence崩溃；
4. 任何正负结论都只写在新28-quartet/19-parent银行上，不和旧159银行混合。

**评估合同修正优先于性能优化。** 初始四臂都用clean+single-edit训练，因此`direct`不是clean-only baseline；初始结果中的`full_repair/migration/111_regression`只能标为direct-reference diagnostics，不能当作clean→repair flow。第1轮优化预算固定用于补同seed、direct架构、clean-only singleton训练的baseline，再重算合法flow；不改变四臂正式结果，不扫超参。

“direct已经与interaction相当”不是工程失败，而是结构机制未显示优势；完成合法flow修正后应停止并报告边界，不人为抬高难度。

### 需要优化的情况

仅在核心矩阵出现以下可修复情形时消耗一轮：

- mask/颜色读出覆盖异常或大量zero-mask；
- interaction/representation训练dev BCE或梯度显示未优化；
- 某一arm出现seed性崩溃、NaN、OOM或checkpoint选择异常；
- 结果方向与输入接口契约明显不一致。

每轮只改一个预先记录的因素，并重新使用同一data hash、arm、seed和评价定义。不得用测试J挑因素。

## 三轮优先级

1. **评估合同修正**：补clean-only direct baseline，修正合法flow分母。
2. **读出/输入合同修正**：正式runner必须把native RGB原图送入模型，由模型内部做resize/normalization；禁止把normalized tensor再次当原图生成颜色mask。本轮由审计发现，固定输入hash后整矩阵重跑。
3. **优化/容量因素**：只有前两轮后仍出现明确未优化或seed崩溃，才固定数据与seed做一次head/学习率敏感性检查；不扫大网格。

## 优化预算更新

用户将原“最多3轮”改为：可以持续迭代至04:55。仍遵守提前停止：结果满意、确认失败、没有新的可修复因素或数据/实现合同失效时立即停止；不为了耗尽时间制造新训练。

## 终止

- 核心结果满意：立即回收；
- 同一实现/优化失败连续两轮且没有新的可解释因素：停止，记录失败；
- 到04:55或预算耗尽：停止，保留所有轮次和负面证据；
- 任何结果未经独立review，不写视觉interaction claim；
- 停止后立即通过AI Galaxy退租实例。
