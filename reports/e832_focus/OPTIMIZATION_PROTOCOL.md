# 五路线胜率优化协议

日期：2026-09-25。基准提交：`e832887c938948b23e8783719d493f2b9b8c3a17`。

## 总目标

只回答一个问题：哪些输入表示与跨对象计算条件，使 single-edit 监督更可靠地转化为完整组合正确性，而不是只迁移错误或提高终点准确率。

五条路线：

1. 跨任务确认：source、T1、T2；
2. 视觉结构迁移：同一 ResNet18 encoder 下的 interaction / additive / representation 消融；
3. 强基线：解析、set/relational、容量匹配、有限群轨道/规范形式；
4. 独立稳健性：新 parent、新 renderer、预先指定观测变化；
5. 论文收束：只接最强的 1–2 项新增结论。

## 不可变合同

- 训练只见 clean 与 single-edit；不得见 AB、四元组或测试组合标签。
- 测试指标 `J3 = P(A、B、AB 全部正确 | E)`；E 的构造是两条各自保持标签、合并后翻转标签的编辑。
- 报告 `J4`、A/B/AB 边际、atomic_joint、full repair、migration、baseline-111 回归；不存在的字段写 missing。
- 父场景是重采样单位；seed 是训练变异；群元不是独立重复。
- checkpoint、学习率、特征选择只看 clean+single dev BCE，不看测试 J。
- 不用整数 ID 做数据去重；用原始坐标与 L-inf 距离。
- 不读 `holdout_909`、`confirm_1007`、bridge `holdout_895`、`J03WQQ`、`SoccerTrack-v2`。
- 历史 dev512、159/178 银行只作工程背景，不作新确认。
- 训练、选型、推理成本分开记录；至少记录参数量、训练状态前向次数和单例推理 MAC。
- 不新增付费租机或凭据。路线2先完成实现与CPU冒烟；GPU确认等待已有授权实例可用。

## 路线1：跨任务确认

### 模型

- raw coordinate MLP；
- 任务合理但分段/角色可加的 typed control；
- 角色保持、联合非线性的 repaired interaction；
- 任务合法的有序距离/几何特征；
- 与几何信息匹配的 rich features；
- 合法群排序或有限群平均表示。

每个任务先做函数可表达性与群不变性单元检查。T1、T2 不自动继承 source 的“段间加性不足”。

### 五轮预算

- Round 0：实现、数据生成、模型合同、测试；不产生主张。
- Round 1：source 新银行 3 seeds，确认排序与 interaction 对比。
- Round 2：T1、T2 各 3 seeds，报告输入效应和 interaction 的任务依赖。
- Round 3：只针对 Round 1/2 中结论不稳或反例明显的合法因素优化；不扫无关超参。
- Round 4：独立新 seed/parent 或有限群轨道对照，验证最有竞争的解释。
- Round 5：只对最终 1–2 个对比扩到 10 seeds 并锁定结果；其余保持有界阴性。

成功不是所有任务同号，而是明确区分：输入优势是否迁移、交互符号是否依任务、排序是否只是额外几何量的同义词。

## 路线2：视觉结构迁移

### 共同视觉接口

同一 ResNet18 encoder、同一训练 parent、同一新测试 parent、同一 renderer 与预算。段角色只来自图像可见的红/蓝段；推理不得使用隐藏 A/B 索引、坐标、AB 真值、single-edit 答案或组合 ID。

### 五轮预算

- Round 0：实现数据接口、可见段表示、interaction/additive heads、CPU 冒烟和 GPU bundle。
- Round 1：单 seed 探索，验证流程和感知误差，不选最终模型。
- Round 2：3 seeds × direct / additive / interaction / representation，比较 J3 与 full repair。
- Round 3：按 Round 2 的失败模式只修一个因素：段读出、interaction 形式、优化或匹配预算；不同时改多个。
- Round 4：新 parent + 一个预先指定 observation shift；冻结设计。
- Round 5：最终最强 1–2 个对比扩到 5–10 seeds；生成 GPU 锁定任务包。没有 GPU 时状态写 BLOCKED_GPU，不以 CPU 小样本替代正式结论。

## 路线3：强基线

依次比较：

1. 经典解析几何；
2. raw、six-distance、rich-unsorted、rich-sorted；
3. 标准 DeepSets/Set Transformer 类集合模型；
4. 标准关系交互模型；
5. 参数量匹配 raw；
6. 有限群平均或受限规范形式。

错误对称性的全点求和只作负控，不冒充正确关系模型。若解析或标准基线解决任务，论文如实收窄。

## 路线4：独立稳健性

- 新 parent；
- 新 renderer，但保持语义和标签；
- 开发前固定一种观测变化；
- 不可辨识样本保留在总体并单列，不从分母删除；
- 按 parent 重抽样；
- 不把不同 renderer/任务银行合并为一个分母。

优先复核排序、interaction 和视觉 full repair 三个最强结论。若结论只在一个设置成立，缩窄 claim，不追加任务维持显著性。

## 路线5：论文

路线1–4有结果后才接线。正文只保留：

- 一个主问题；
- 统一测量；
- 最强 1–2 个干预结论；
- 一个视觉边界或迁移验证。

Figure 1 使用真实 quartet：固定规则选 endpoint-only repair、full repair 和新结构正例各一例；图不用于估计效应。所有数字绑定来源文件、分母、seed/parent 单位和允许措辞。
