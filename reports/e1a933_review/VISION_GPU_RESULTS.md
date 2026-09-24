# O03/N01：GPU21臂独立结果解读

状态：21次GPU训练全部完成并校验回收；逐样本指标与dev选epoch独立复核通过。此处不混入CPU训练结果。

数据：新canonical renderer协议，159个E-quartet来自74个独立测试父场景；dev707图像来自128父场景。每种方法3个训练seed。CI按真实父场景重采样，不将图像/四态当独立重复。旧renderer数字不得与本表当同条件前后比较。

| 方法 | J803 | J805 | J806 | J平均 |
|---|---:|---:|---:|---:|
| pretrained_static | 0.5157 | 0.6038 | 0.4969 | 0.5388 |
| pretrained_flip | 0.6415 | 0.6981 | 0.7107 | 0.6834 |
| random_static | 0.4277 | 0.4843 | 0.4843 | 0.4654 |
| random_flip | 0.7044 | 0.6541 | 0.6478 | 0.6688 |
| ordered | 0.7170 | 0.6415 | 0.7296 | 0.6960 |
| matched | 0.6981 | 0.6730 | 0.7044 | 0.6918 |
| shuffled_matched | 0.5660 | 0.7044 | 0.7421 | 0.6709 |

## O03：同架构single-flip训练带来完整联合修复

两种regimen每epoch都曝光2816张图、88次optimizer step，20epoch共1760步。static循环重复clean图，flip加入single终点图；逐parent额外曝光分布并非完全匹配，因此识别的是同总预算训练regimen效应，不是脱离曝光分布的纯flip内容因果。N01各辅助臂图像及逐parent曝光则完全相同。

pretrained static→flip平均J增加14.465pp，random增加20.335pp；两初始化的三个seed配对差均为正，逐seed parent-bootstrap95%区间也均大于0。新协议支持所测ResNet18在该合成任务上的repair效应，不支持“所有成熟视觉模型普遍失效”，也不能推出预训练无关。

| 初始化/seed | J差pp [parent95%] | baseline110 n | full repair n | migration n | baseline111 n | 111退化 n |
|---|---:|---:|---:|---:|---:|---:|
| pretrained/803 | 12.58 [3.82,21.77] | 43 | 20 | 8 | 82 | 8 |
| pretrained/805 | 9.43 [2.58,17.06] | 34 | 19 | 4 | 96 | 6 |
| pretrained/806 | 21.38 [11.52,31.76] | 44 | 28 | 4 | 79 | 7 |
| random/803 | 27.67 [17.42,38.18] | 51 | 34 | 6 | 68 | 5 |
| random/805 | 16.98 [5.92,27.81] | 34 | 23 | 2 | 77 | 11 |
| random/806 | 16.35 [6.25,27.22] | 34 | 21 | 1 | 77 | 12 |

## N01：辅助目标学会，修复收益未稳定转化

matched相对ordered的平均J差−0.419pp，相对BCE-only +0.839pp，相对shuffled +2.096pp；这些差跨seed不一致。对ordered、BCE的每seed配对J区间都跨0，因此不能把“可观测性匹配”写成已证实改善组合修复的方法。

| 辅助目标/seed | selected epoch | dev matched MSE | physical matched MSE | ordered MSE |
|---|---:|---:|---:|---:|
| ordered/803 | 15 | 0.31272 | 0.03624 | 0.61365 |
| ordered/805 | 2 | 0.44503 | 0.05178 | 0.55778 |
| ordered/806 | 11 | 0.31906 | 0.03701 | 0.59545 |
| matched/803 | 11 | 0.11794 | 0.01402 | 0.96849 |
| matched/805 | 2 | 0.29729 | 0.03276 | 0.95496 |
| matched/806 | 4 | 0.18530 | 0.02093 | 0.76933 |
| shuffled_matched/803 | 1 | 0.71870 | 0.08289 | 1.36491 |
| shuffled_matched/805 | 2 | 0.73813 | 0.08629 | 1.20934 |
| shuffled_matched/806 | 7 | 0.79317 | 0.09195 | 1.23289 |

matched的dev轨道误差在三个seed均低于ordered、shuffle和未训练辅助头；matched−ordered配对MSE差分别−.19479、−.14774、−.13377，其128-parent bootstrap区间均严格低于0。这是几何学习证据，不能仅由分类J均值推断。第20epoch轨道误差进一步到.1137/.1439/.1137；但主任务选中的805模型仅epoch2，误差.2973，说明不能称全部seed精确恢复几何。原始ordered误差较高并不否定matched学习，因为命名冲突仍不可识别。

合理结论：观测匹配改善了关系辅助目标的可学性，但在所测学生/训练配方下尚未稳定改善任务修复。不能推论所有关系蒸馏失败，也不能认定决策head不使用几何已被证明。N03仅作为后续可检验解释启动。

证据：`artifacts/e1a933_review/vision_gpu_interpretation/independent_summary.json`及其`cuda_matrix/`逐样本数组、history、paired_analysis；完整原始回收归档保留在`remote_vision_20260924_ssh30891/collected/`。复核命令：`python experiments/e1a933_review/vision_gpu_interpret.py`。
