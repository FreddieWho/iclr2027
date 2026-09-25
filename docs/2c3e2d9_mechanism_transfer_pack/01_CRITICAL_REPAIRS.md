# 01｜结果相关修正与执行合同

不要因本清单一次性重跑全部历史训练。先修共享原语、定位受影响checkpoint与claim，再重跑最小必要对比。旧代码和结果保留，新增v3目录；文件名不得替代科学定义。

## R1｜两个“repaired”不是同一模型

检查：
- `experiments/e832_focus/structure/a_source_compare.py::RepairedRho`：有segment内`psi`，再跨segment非线性，是有效的源任务结构对照。
- `experiments/e832_focus/route1_cross_task/common_runner.py::Typed.forward`：source分支是`sum(phi(points[:2])) + sum(phi(points[2:]))`，等价于四点总和，忽略两条段的分组。即使rho非线性，也不能恢复已经丢掉的配对。

把第一套实现提取为共享模块，不再复制一个简化近似。保留G8、明确不强制S4。用本包相交/不相交的同四点反例测试非法重组会改变可表达关系；单纯G8不变测试不足以检出过大对称群。T1/T2单独确认角色，不自动套源任务证明。

验收：源码/模型类/hash入manifest；输出不只记录`repaired`字符串。训练表达性四例是工程检查，不计入科学结果。仍拟合不佳时报告训练曲线，不用BCE<0.60触发“优化已解决”。

## R2｜T1/T2几何特征

`common_runner.py::rich(T2)`目前对形如[B,2 endpoints,2 coordinates]的张量使用`norm(axis=1)`，得到按x/y聚合的量，不是圆心到两个端点的距离。应使用`axis=-1`。`sort_features(T2)`把段长与第一个其他量排序，也不是端点交换的作用。

T2基本有效输入：`[|A-B|, min(|C-A|,|C-B|), max(|C-A|,|C-B|), R]`；R固定时可不输入，变化时必须输入。额外量各自审计。禁止对不同语义角色的字段直接拼袋排序。

T1独立排序edge bag和query-distance bag有本包精确碰撞；保留旧表示作为有信息损失的对照，不将它称作“相同信息仅换顺序”。代码`sine=area2/(edge1*edge2*edge3)`不是无量纲正弦，应准确命名，或在新的明确合同里改公式，不静默覆盖旧特征。

验收：合法置换、刚体变换、任务角色、非法交换、碰撞反例测试齐备。修T2要重训受影响rich arms；不只在旧checkpoint前更换特征。未受影响的six输入与旧U10不是自动作废。

## R3｜跨任务runner没有保持已有任务能力条件

新runner改变了生成分布、训练父数、类别平衡、MLP结构、候选上限与学习率；它不是旧U10的直接复现。先在同一旧合同重放一个raw/six锚点，再一次只更改一个设计轴。

`mine_E`先遍历按固定顺序生成的keep候选，再保留前12个；多draw不保证新增候选进入这12个。记录实际端点/角色/编辑族/半径覆盖，不只记录draw数。新银行先完整得到候选，再用固定独立RNG分层抽取；训练和测试使用同一生成规则但parent不重叠。旧first12银行继续作为历史分布，不删除不利例。

`gen`中重复判断的最大值/最小值轴也要审计：整场景Linf应先在point和xy两轴求max，再跨参考parent求min，不能把某一个点接近当成整个场景重复。

保存每个选中checkpoint、train/dev曲线、parent来源、特征schema、batch顺序和模型哈希。当前runner不保留权重，不能支持后续冻结模块迁移。

## R4｜视觉mask与上下文隔离

`visual_mechanism.visible_segment_features`把native64二值mask nearest缩到最终7×7，可能把真实可见的整条线变成零mask。先在所有旧/新bank统计：native可见而pool为零率、每色有效网格数、与误差的关联。

替代：面积权重/自适应平均池化，或较早的空间层；同一head做pooling-only对照。缺色和下采样漏色分开统计，禁止oracle补mask。实现保持概率质量，不以empty mask静默平均成零向量。

整图ResNet之后的red/blue池都可能已含另一条线的信息。因此当前additive arm只是“加性读出”，不是“没有跨对象计算”。机制实验需要独立对象输入在首次融合前保持隔离，并保留全局坐标参考。透明/分层renderer可验证隔离；原opaque renderer有遮挡依赖，单独报告，不能宣称其对象token严格独立。

`_image_nchw`先拒绝非NCHW再尝试转换NHWC，逻辑有误；`input_size`参数未传给`prep`，非224实验会失配。这两项不影响已知NCHW/224主跑，但必须在新路线前修好。

## R5｜指标和估计单位

`gpu_run.metrics`的J4仍等于J3，忽略P状态。应J3=all(A,B,AB)，J4=all(P,A,B,AB)，缺失P时null。字段`J3_atomic_joint`重命名，旧alias只作兼容，禁止把它当J3。

所有表同时明确quartet数、eligible parent数和抽取parent数；parent平均与row平均分列。parent-cluster CI；seed不是新数据。小样本19parent结果是试点，不以“不显著”宣布效应不存在。

修复比较保存全部8状态流向、J3、J4、A/B/AB边际、baseline110 full/migration、baseline111回归。训练样本数与每parent曝光分别记账。

## R6｜合同不加门控膨胀

建立一个公共schema与一份语义测试，供所有runner导入。仅对训练/评估隔离、输入可见性、标签/角色语义使用硬错误退出。对弱信号、混合seed和边界效应允许探索，不设“不到10pp永久关闭”。真正的最终1–2项claim才使用一次新独立确认。
