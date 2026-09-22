# 所有新路线共享的最小实现

## 可复用的实际入口
- experiments/discovery_campaign/coord_mlp.py：坐标模型与加载逻辑。
- experiments/discovery_campaign/r02_search.py：合法几何候选生成；r02_deepen.py：新候选库。
- experiments/discovery_campaign/r02_t5r3.py：真数据前向/zone oracle/编辑/重建图。
- experiments/discovery_campaign/r04b_methods.py：现有flipmine矿工与训练对照。
- experiments/discovery_campaign/r04b_t5r3.py：真数据coverage微调。
- docs/iclr2027_discovery_campaign_20260917/core/relations.py：相交oracle（运行时确认位置）。
- artifacts/discovery_campaign/：本地已有场景、checkpoint、翻转样本。
权重/大数组按现有.gitignore可能未推送GitHub。agent在用户原工作环境复用；缺某个二进制不编造结果、不优先重训所有历史模型。

## 新输出布局
experiments/last15h/{shared, n01, ..., n10}/
artifacts/last15h/Nxx/<run_id>/
reports/last15h/Nxx.md
保留旧脚本/结果；新实验可修改任何方法，不必遵守旧“路线层冻结”。付费、外部发送和数据许可不由本包新增授权。

## 路径记录（足够用，不建大框架）
每条路径记录parent_id、source_match（若有）、action_family、t值、状态、oracle_label、预测类别与logits、实际编辑代价、有效性、训练使用状态。基本数组一次生成，让N01/N03/N04/N06复用。
真值与模型用相同采样点；oracle辅助二分精化需记录调用数，不能免费给某方法更密的评价。
单次转折须先扫描定位。多次转折按顺序记录，不用二分误当单调。预测没转折记missing，错转第三类记wrong-target。

## 有效性是新问题的一部分
只检验与本任务实际有关的坐标界限、可见margin、实体对应；不加不必要的生态/运动学大模型。若声称行动安全，则路径中间也必须合法。真值改变时用新真值，不用旧标签制造“对抗失败”。
新指标首先记录原状态正确与否，评价既给整体，也给明确条件子集。多种子均值/散点可正常使用，无“必须10 seeds”规定。

## 计算
默认CPU；从现有profile设线程，6worker每个1–2个BLAS/PyTorch线程起步，防止嵌套并行反而变慢。先做一小批测真实时间，不承诺固定每轮耗时。预训练视觉权重无缓存且下载困难时回退小CNN，不阻断其他线。
首轮常规256父场景、1seed，阳性后扩新父场景和现有额外种子。数字取结论需要的最小充分规模，而不是机械补齐巨大网格。
