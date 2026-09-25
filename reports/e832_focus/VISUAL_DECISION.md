# 视觉四臂：终点收益是不是完整修复

这是对已经完成的 ResNet18 GPU 结果的同条件比较，不是新训练，也不是新架构。比较只使用 canonical224 的 159 个 E-quartet、74 个测试 parent、cuda FP32、同一渲染器和同一测试 parent。来源是 `artifacts/e1a933_review/vision_gpu_interpretation/cuda_matrix/` 下四个臂的 `result.json`、`predictions.npz`，以及 `paired_analysis.json` 和 `independent_summary.json`。重排脚本是 `experiments/e832_focus/vision/build_canonical_fourarm_table.py`。它没有改旧脚本，没有读封存池，也没有租机。

## 两套银行不能相减

159 quartet 来自 `artifacts/e1a933_review/vision_canonical224_v2/data.npz`，sha256 `63151a7670d2cadcab34d7f52b88f488e27ea0fb04ff4fcdd421d064efca0de6`。74 个 parent 的 quartet 数是 1、2、3、4 的分别有 31、15、14、14 个 parent。这组数字在 `artifacts/e832_focus/vision/numbers_for_decision.json`。

178 quartet 的 N02 银行是另一份数据，sha256 `5f773e0b8fc906ce287e0aa1b1161b4aa5a036d0c48d2a485187264e3ec75032`，身份计数来自 `artifacts/e1a933_review/N02_interpretation/arm_parent_values.csv` 里 `A_standard_pretrained_flip`、seed 803、stratum `all`：178 行、85 个 parent。本表没有复制它的任何准确率。两套银行的差不是提升。

## 字段怎么命名

论文里的 J3 是 E 上 A、B、AB 三个状态同时正确。这 159 行的标签全部满足 E：两个原子状态保持基态标签，AB 翻转基态标签。因此文件里的字段 `J` 在这张表上就是 J3，分母是 159。J4 在 `result.json`、`per_run` 和 `paired_analysis.json` 里都不存在，写 missing。基态边际准确率 P 不是 J4，没有拿来填空，也没有从 logits 现推 J4。

原子准确率用已有字段 `atomic_joint`，即两个原子状态都对，分母 159。AB 准确率用已有字段 `AB`，分母 159。完整修复、迁移和 baseline-111 回归只存在于 static 对 flip 的配对里，所以单个臂的这三列是 missing。

论文的 CCM 是条件组合失误率。这些文件里名叫 CCM 的字段是 `joint_count / atomic_count`，那是条件联合正确率，不是失误率。本表的 paper CCM 全部写 missing。没有把该字段改名成 CCM，也没有用 1 减它来填失误率。同一命名冲突也出现在 `artifacts/e1a933_review/vision_n03_analytic/result.json`：那里的 `CCM` 是 114/128 的条件正确率，不是论文 CCM。

逐 parent 行原来没有存成表。下面的逐 parent 文件是同一批 `predictions.npz` 的拆分，seed 合计已锁回官方计数。这不是从 178 银行重算。

## 四臂，按 quartet 池化

下表每个 seed 的分母都是 159。J3、原子、AB 的计数来自同一 `predictions.npz`，比例与对应 `result.json` 一致。seed 均值只取 `independent_summary.json` 里已经存好的 `J_mean`，它是三个 seed 比例的不加权平均，每个比例的分母仍是 159，不是 477。原子和 AB 没有官方 seed 均值，所以那两格是 missing。完整修复、迁移、111 回归在单臂上是 missing。

| 初始化 | 训练 | seed | J3 | 原子准确率 | AB 准确率 | J4 | 完整修复 | 迁移 | 111 回归 |
|---|---|---:|---:|---:|---:|---|---|---|---|
| pretrained | static | 803 | 82/159 | 125/159 | 116/159 | missing | missing | missing | missing |
| pretrained | static | 805 | 96/159 | 130/159 | 123/159 | missing | missing | missing | missing |
| pretrained | static | 806 | 79/159 | 123/159 | 112/159 | missing | missing | missing | missing |
| pretrained | static | 三 seed 均值 | 0.5387840670859538 | missing | missing | missing | missing | missing | missing |
| pretrained | flip | 803 | 102/159 | 118/159 | 142/159 | missing | missing | missing | missing |
| pretrained | flip | 805 | 111/159 | 123/159 | 144/159 | missing | missing | missing | missing |
| pretrained | flip | 806 | 113/159 | 125/159 | 145/159 | missing | missing | missing | missing |
| pretrained | flip | 三 seed 均值 | 0.6834381551362684 | missing | missing | missing | missing | missing | missing |
| random | static | 803 | 68/159 | 119/159 | 106/159 | missing | missing | missing | missing |
| random | static | 805 | 77/159 | 111/159 | 121/159 | missing | missing | missing | missing |
| random | static | 806 | 77/159 | 111/159 | 121/159 | missing | missing | missing | missing |
| random | static | 三 seed 均值 | 0.4654088050314466 | missing | missing | missing | missing | missing | missing |
| random | flip | 803 | 112/159 | 125/159 | 146/159 | missing | missing | missing | missing |
| random | flip | 805 | 104/159 | 119/159 | 144/159 | missing | missing | missing | missing |
| random | flip | 806 | 103/159 | 118/159 | 142/159 | missing | missing | missing | missing |
| random | flip | 三 seed 均值 | 0.6687631027253668 | missing | missing | missing | missing | missing | missing |

机器可读副本是 `artifacts/e832_focus/vision/arm_pooled.csv`。逐 parent 副本是 `artifacts/e832_focus/vision/arm_per_parent.csv`，888 行，每个 parent 的 J3 分母是该 parent 的 quartet 数，不是 159。

AB 在六个 static 到 flip 的配对里都上升。原子准确率不是同样上升：pretrained 的 803 从 125/159 降到 118/159，805 从 130/159 降到 123/159。所以 J3 上升不能读成原子也一起变好。

## 终点收益里，多数是完整修复，不是全部

修复只对同一 seed 的 static 和 flip 计算。baseline-110 是 static 的 A、B 对而 AB 错，分母随 seed 变化，不是 159。完整修复是这个子集里 flip 变成 A、B、AB 全对。迁移是这个子集里 flip 的 AB 变对、但两个原子不再都对。二者不重叠。终点收益是二者之和。baseline-111 是 static 已经 A、B、AB 全对的 quartet；回归是其中 flip 不再全对的个数。计数来自 `paired_analysis.json`，并与同一 `predictions.npz` 重新计数一致。区间是该文件里的 parent bootstrap 95%，不是新的检验。

| 初始化 | seed | 完整修复 | 迁移 | 终点收益中的完整修复 | 110 上没有终点收益 | 111 回归 | J3 差 [parent 95%] |
|---|---:|---:|---:|---:|---:|---:|---|
| pretrained | 803 | 20/43 | 8/43 | 20/28 | 15/43 | 8/82 | 0.12578616352201258 [0.03821051358542288, 0.21769944341372904] |
| pretrained | 805 | 19/34 | 4/34 | 19/23 | 11/34 | 6/96 | 0.09433962264150944 [0.02579823299773988, 0.17060435132957283] |
| pretrained | 806 | 28/44 | 4/44 | 28/32 | 12/44 | 7/79 | 0.2138364779874214 [0.11515151515151516, 0.31756756756756754] |
| random | 803 | 34/51 | 6/51 | 34/40 | 11/51 | 5/68 | 0.27672955974842767 [0.17418653576437587, 0.38181818181818183] |
| random | 805 | 23/34 | 2/34 | 23/25 | 9/34 | 11/77 | 0.16981132075471697 [0.05921052631578947, 0.2781074885379521] |
| random | 806 | 21/34 | 1/34 | 21/22 | 12/34 | 12/77 | 0.16352201257861634 [0.0625, 0.2721518987341772] |

六个 seed 都是完整修复多于迁移。终点收益里的完整修复比例是 20/28、19/23、28/32、34/40、23/25、21/22。这是多数，不是全部。pretrained 803 的 20/43 还说明：多数终点收益成为完整修复，不等于 baseline-110 的多数都变成了完整修复。六个 seed 的 111 回归都不是 0。

`paired_analysis.json` 里两个初始化的 `mean_J_difference` 分别是 0.14465408805031446 和 0.20335429769392033。它们是三个 seed 差的不加权平均，每个差的分母是 159，不是把六个 seed 加成一个更大的样本。六个 parent 区间都大于 0。这个区间只重抽样测试 parent，不能抵消下面的训练曝光问题。

逐 parent 的修复计数在 `artifacts/e832_focus/vision/contrast_per_parent.csv`，444 行。某个 parent 的 baseline-110 或 baseline-111 为 0 时，比例写 missing，计数仍保留。pretrained static 到 flip、seed 803 的逐 parent 合计锁回 20、8、43、8、82，与上表一致。

## 总曝光对齐，不是逐 parent 曝光对齐

两种 regimen 每个 epoch 都是 2816 个图像槽、88 次 optimizer step，20 个 epoch 共 1760 步。这是总预算对齐。数据文件里 clean 图像 2048 张，single-flip 图像 768 张，训练 parent 512 个。static 把 clean 索引重复到 2816，因此看不到那 768 张 single-flip 图像；flip 每个 epoch 每张训练图像出现一次。按 `experiments/e1a933_review/vision_cuda_protocol.py` 的这个索引规则，512 个训练 parent 里有 441 个的 static 槽数不等于 flip 槽数，单个 parent 的绝对差最大是 4 个槽。来源是 `artifacts/e832_focus/vision/exposure_caveat.json`，数据是上面的 `data.npz`。

因此这里识别的是同总预算的训练 regimen 效应，不是脱离曝光分布的纯 flip 内容因果。N01 的辅助臂之间图像和逐 parent 曝光相同，但那不是这张四臂表，本文件不重新解释辅助损失。

开发集是同一数据文件里的 707 张图像、128 个 parent，只用于说明数据身份，不进入上表分母。parent 的整数编号在每个 split 内重新从零开始，不能拿来和 N02 的 parent 编号对接。

## 结论

在这个已经测完的 ResNet18 上，static 到 flip 产生的终点收益，多数变成了完整修复。pretrained 和 random 的六个 seed 都是这样。它没有把全部终点收益变成完整修复，也没有把 baseline-110 全部修好，更没有保住全部 baseline-111。

这没有建立的东西同样明确。它不是所有胜任视觉系统的普遍规律，不能和别的银行里迁移占优的结果拼成一条定律。它不是纯终点内容因果，因为总曝光对齐不是逐 parent 曝光对齐。它不是 J4，也不是论文 CCM。它不是相对 178 quartet N02 银行的提升。它不是新 parent 上的新确认；159 这个银行已经被这条视觉线用过，只能当工程证据。它也不是神经方法优于经典像素解析：同一数据上的经典红蓝分割加 PCA 解析，J 是 `vision_n03_analytic/result.json` 里的 0.7169811320754716，即已报告的 114/159，高于本表最好的 pretrained flip seed 113/159。本文件不做新的优劣检验，只拒绝“ResNet 已经胜过该解析器”这句。

交互读出和加性消融没有跑。因此这张表不能说明完整修复来自跨段非线性，也不能宣布一个新架构。

## 以后的结构消融，现在是 NOT RUN

协议写在 `experiments/e832_focus/vision/PROTOCOL_A_ALIGNED_ABLATION.md`，状态文件是 `artifacts/e832_focus/vision/ablation_protocol_status.json`。状态是 NOT RUN。没有训练，没有租 GPU，没有新架构。

如果以后跑，需要三件已经规定的东西，而不是再起一个名字。第一，编码器仍是这条已经测完的 ResNet18，渲染器不变。第二，加上一个交互读出：两个段的表示要被同一个非线性同时看到。第三，加性消融使用同一编码器和同一段表示，只去掉跨段非线性。段的角色必须是图里可见的红段和蓝段，不能在推理时送入隐藏的 A/B 编号、坐标、AB 真值、单编辑答案或组合编号。真几何只允许作为标明的训练特权监督，不能当推理输入。

那一次运行之前，不能把本文件写成“关系结构已经在视觉上得到验证”。如果直接的 ResNet18 已经同样好，可写的是可靠性边界，不是新架构优胜。
