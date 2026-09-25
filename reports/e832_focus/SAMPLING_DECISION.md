# C 采样 / 计算前沿：不启动

裁决：**DO-NOT-START** 新训练。

日期：2026-09-25。只读既有 N02。未训练，未租 GPU，未改既有结果文件，未读封存池，未改论文或状态文档。

这不是新的阳性/阴性硬线。启动条件沿用 `docs/e832887_focus_pack/04_C_COMPUTE_FRONTIER_OPTIONAL.md` 与 `docs/e832887_focus_pack/01_MASTER_PROMPT.md`：C 是第三优先；只有 A/B 已有解释缺口，或该效应能独立形成比结构问题更强的主张时才训练。本轮任务合同写明：A/B 仍在跑，默认不启动，除非既有证据单独已经使本 lane 强于结构问题。它没有。

## 已经支持的

支持的是**固定观测信息下的网格/计算配置效应**，不是已确认的采样机制。

- A = 原生 64，B = bilinear(A, 224)。B 不增加原始观测像素。来源：`reports/e1a933_review/N02_SAMPLING_ACCESS_REPORT.md` 第 1 节，观测合同 `config.json` / `receipt.json:config`。
- parent 等权、跨 seed、J、`all` 层，分母 **178 quartet / 85 parent**，bootstrap 4000、按 parent 聚类。来源：`artifacts/e1a933_review/N02_interpretation/contrast_summary.csv` 的 `value_parent`。
  - pretrained，`B_standard_pretrained_flip minus A_standard_pretrained_flip`：+0.340196，CI [+0.258170, +0.423211]。逐 seed parent 等权 +0.379412 / +0.342157 / +0.299020，6 个初始化 seed 里这一对的 3 个都为正。
  - random，`B_standard_random_flip minus A_standard_random_flip`：+0.299673，CI [+0.223529, +0.380392]。逐 seed +0.251961 / +0.331373 / +0.315686，3/3 为正。
- 原生 224（C）相对 B 的额外信息小得多。同一分母、同一加权：
  - pretrained C−B：+0.003922，CI [−0.029412, +0.036601]，含 0。
  - random C−B：+0.040196，CI [+0.011111, +0.075817]。
  - 低通 D 相对 A：pretrained +0.003922，CI [−0.067320, +0.073203]；random −0.094118，CI [−0.145760, −0.043791]。
- 因此可以写：同一 64 格信息、换成大输入网格，J 大约提高 0.30–0.34（上表两个跨 seed 点估计）。不能写：这已经证明是信息可访问性或 aliasing。

机制未解决的直接证据是 lowstride 交互，不是缺少 48 臂。`INT_lowstride_on_BminusA_*`，同一 178/85 分母：

- pretrained：−0.133007，CI [−0.213072, −0.058170]。逐 seed parent 等权 −0.168627 / −0.128431 / −0.101961；seed 806 的 CI [−0.221569, +0.024510] 含 0。
- random：+0.019935，CI [−0.041838, +0.081046]。seed 803 为 +0.109804，CI [+0.029412, +0.190196]，符号与预测相反且区间不含 0。

专报据此把机制标为 UNRESOLVED，且不给 aliasing 归因。本裁决沿用该判定，不升级。

## 12.25× MAC 混杂

`structure_matrix.json` 四格参数都是 11,177,025。conv+linear MACs/图：

- standard_64 = 148,046,336
- standard_224 = 1,813,561,856
- 比值 = 1,813,561,856 / 148,046,336 = **12.2499610933**

这就是报告里的 12.25×。复算笔记：`experiments/e832_focus/sampling/mac_ratio_audit.md`。B 的 standard stem 相对 A 不是“只改网格、算力不变”。同一文件里 lowstride_64 / standard_64 = 563,282,432 / 148,046,336 = 3.8047711765，所以 lowstride 干预自己也带算力变化。

standard_64 的 layer1 是 16×16，standard_224 的 layer1 是 56×56；(56×56)/(16×16) = 12.25 整。在这个 ResNet18 上，网格变大和 MAC 变大是同一改动的两侧。N02 的训练合同是 20 epoch（`sampling_results/config.json` 与 `receipt.json` 的 `epochs`），不是 MAC 匹配。`04_C_COMPUTE_FRONTIER_OPTIONAL.md` 已写明：不能靠匹配了 epoch、没匹配 MAC 的对照回答“更多计算普遍能买到，还是分到空间网格的计算特别有效”。N02 正是这种对照，所以它回答不了 C 的问题，也不因此授权再开一轮去回答。

## pretrained 与 random 的差别

两件事不要并成一句“有/没有预训练效应”。

1. **B−A 的 J 差距两边都大。** 上节 +0.340196 与 +0.299673，分母都是 178/85。这不是 pretrained 专属的尺度效应。random 下“更多算力自动提高 J”也不成立：`A_lowstride_random_flip minus A_standard_random_flip` 的跨 seed parent 等权只有 +0.036928，CI [−0.011111, +0.087255]，含 0。pretrained 下同一 A 臂对比是 +0.211438，CI [+0.148039, +0.279428]。
2. **去掉早期池化是否削弱 B 相对 A，取决于初始化。** pretrained 交互 −0.133007，区间上界小于 0；random 交互 +0.019935，区间含 0，且有一个 seed 显著反向。两个跨 seed 点估计之比 −0.133007 / 0.340196 = −0.390971，只是 pretrained 下交互占 B−A 的描述性比例，不是决策阈值。random 下这个比例不能用来声称机制成立。

因此现有证据只允许写：早期池化/网格配置与初始化有条件交互。不允许写成通用采样机制，也不允许写成“预训练已被排除”。

static 对照不改变这个边界。同一 178/85、parent 等权，B−A 的 J：pretrained static +0.390523，CI [+0.310784, +0.471593]；random static +0.365033，CI [+0.283660, +0.442484]。尺度差距不依赖 flip 监督，但这仍是未匹配 MAC 的网格/计算配置效应。

## small-edit 格子

`contrast_summary.csv` 每个 `small_edit` 行都是 **5 quartet / 3 parent**。`other_edit` 是 173 quartet / 84 parent。`all` 是 178/85。parent 不是划分：seed 803、上述 B−A 对比里，small_edit parent 是 135（1 行）、205（2 行）、471（2 行）；135 与 205 也出现在 other_edit。见审计笔记。

冻结预测的“小编辑更明显”在这个格子上不可判定，点估计也不支持：

- pretrained 交互，small_edit：−0.055556，CI [−0.333333, +0.333333]；other_edit：−0.130622，CI [−0.205365, −0.054886]。小编辑点估计弱于 other_edit。
- random 交互，small_edit：+0.055556，CI [−0.166667, +0.333333]；other_edit：+0.021495，CI [−0.039352, +0.084656]。方向与“负交互”相反。

不能引用“尺度效应主要在细小编辑”。要检验它必须换新 parent 分层；那不在本轮授权内，也不是现在启动 C 的理由。

## A/B 现在有没有本 lane 能填的解释缺口

没有。

- 2026-09-25 读 `reports/e832_focus/` 时，只有 `OWNERSHIP.md` 和空的 `handoff/`。没有 `STRUCTURE_DECISION.md`，没有 `VISUAL_DECISION.md`。A、B 还没有交出缺口。
- A 问的是：在表达能力足够、预算可比的模型里，合法不变性是否让 single-edit 监督变成完整联合正确，以及收益是否还要额外几何量（`02_A_RELATIONAL_STRUCTURE.md`）。N02 的 ResNet18 网格/MAC 矩阵不回答这个问题。
- B 问的是：感知已经够用时，关系的联合计算是否影响完整修复（`03_B_VISUAL_TRANSFER.md`）。总控写的是：只有 C 能够解释视觉结论时，才继续分离网格/算力（`01_MASTER_PROMPT.md`）。B 还没有视觉结论需要 C 来解释。
- N02 已有的大 J 差距是混杂的配置效应，机制 UNRESOLVED。它比尚未完成的结构问题更弱，不能当成更强的独立主张，也不能因为 48 臂已经跑完就扩大。

## 不启动之后可以保留的说法

可以保留，且必须带分母与 MAC 限定：固定 64 信息、把输入网格放到 224，并因此把 conv/linear MAC 提高到约 12.25×，在 178 quartet / 85 parent 上把 parent 等权 J 提高约 0.30–0.34；原生 224 的额外贡献小；早期池化的作用依赖初始化；小编辑层只有 5/3，不能支撑机制。旧 12-checkpoint 零重训诊断用的是另一套 renderer 和 bank，数字不能与本版相减（专报第 9 节）。

不能写：采样机制已确认；aliasing 已建立；算力已被排除；小编辑是主作用层；C 应替换论文中心。

## 本轮不做的比较

`04` 里的 A/B/C/D/E 前沿（低 stride、用更多训练步匹配 B 的训练 MAC、必要时改 width/depth 靠近推理 MAC）是以后若 A/B 真有缺口时的最小比较，不是本裁决授权的训练。本文件不设新的效应量门槛。
