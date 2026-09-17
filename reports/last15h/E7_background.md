# E7 背景综合（4 researcher 并轨，2026-09-18）

问题（L-007）：转折位置精度需要新机制。v1（双侧标签）敢转不准、v2（方向 margin）零增益、N02（梯度方向诊断）无判别、N07（覆盖）≈随机——死因各不相同，但都绕开了同一个东西：**标注点之间的插值函数形状**。

## 四轨 top pick（各轨详见 subagent 原文，已归档 artifacts 引用略）

| 轨 | top pick | 一句话机制 | 与死亡名单的区别 |
|---|---|---|---|
| A 导数 | bracket 内斜率下限 anti-flatness（+A-4 gap/斜率耦合修 v2） | MISSING=路径上分数太平无过零点；bracket 内 |ds/dt| 下限＋外罚平 | 约束斜率模长；v1 管值、v2 管端点 gap、cos 管方向——三者都留斜率自由 |
| B mixup | flip-mix：mixup 采样＋oracle 硬标签（删线性先验） | 随机对中点软标签在 τ* 偏心时恒错（manifold intrusion）；oracle 标签釜底抽薪 | 监督整段台阶形状；v1 只钉两点（可被错位 sharp edge 蒙混） |
| C 蒸馏 | 切向导数匹配（Sobolev/Jacobian 沿路径切线） | 抄 teacher 斜率场；无穷多边界分开同样标注点，斜率定位置 | 抄形状；v2 无 teacher 参照、v1 无斜率 |
| D 跨领域 | SDF 回归头（签距离；备选 DSNT heatmap） | CE 在分对后梯度饱和，位移不付代价；SDF 处处付代价，零点钉死 | 连续几何量；v1 是 flank 0/1（梯度仍会饱和） |

收敛句（独立提出）：四轨都说同一件事——**用 oracle 接地的信息约束标注点之间的插值函数**（斜率/斜率场/距离值/稠密标签）。v3 不是四选一，是正交组合。

## 引文亲验（2026-09-18，独立检索）

- ✅ Srinivas & Fleuret 2018（ICML，PMLR v80）：Jacobian 匹配≡加噪蒸馏——C-1 承重墙成立。
- ✅ DSNT（Nibali et al. 2018）：可微、无参数、空间泛化——D-2 机制成立。
- ✅ RKD（Park et al. CVPR 2019）：distance/angle-wise 关系转移——C-2 成立。
- ⚠️ Finlay 1910.06922：实为 **GAN margin 框架**（"Gradient penalty from a maximum margin perspective"），不是分类器梯度惩罚鲁棒性——A-1 的"2–3× margin"引用线**错配**，已剔除；A-1 的隐函数论证是数学不受影响，机制先例改引 Ross RRR＋Drucker-LeCun double-backprop。
- ✅ CoD（researcher web_read 验摘要）：CFE 映射 teacher 边界——C-3 传承成立。
- 其余（Sobolev/mixup/AdaMixUp/RegMixup/LocalMixup/SPKD/Hinton-KD/BAN/CURE/Novak/Sokolic/BSN/BMN/HED/Tversky/Audebert-SDF/Kervadec）为标准文献，取摘要级匹配，全文公式未逐验——写稿引用前需再对一次公式。

## v3 提案（待开火）：正交组合，一次 falsification 网

基线：flipmine（积分 0.229）＋v1（参考死人）。同 paths/种子，积分第一指标，5 种子（过线者进 E6 上 10 种子）。

| 臂 | 内容 | 预注册 kill 线 |
|---|---|---|
| B-1 flip-mix | 路径稠密 oracle 标签（uniform K=8；τ* 集中变体） | \|terr\|/积分不胜＝死；只赢 recall＝v1 翻版＝死 |
| A-1 anti-flatness | bracket 内斜率下限＋外平坦 | 斜率起来了积分不动＝死（证瓶颈在表示不在监督） |
| D-1 SDF 头 | 签距离回归 λ×clip-δ sweep | 零点散＝死 |
| C-1 tangent-match | teacher 切向导数拷贝 | 不胜 matched-noise-only＝降格加噪＝死 |
| A-4 coupled-margin | gap/斜率耦合（v2 补丁） | 同 v2 表现＝死 |
| 对照组 | vanilla mixup（预期 \|terr\| null）、global 梯度惩罚（预期中性偏害）、BAN 一轮（预期 null） | 反向超预期则重写理论 |

通过线（= E6 提名）：任一臂 5 种子同向且均值效应 ≥12pp（L-006 门 1/门 2），对照臂同跑。
预算：每臂 5 种子 × 小 MLP CPU 分钟级；全网约数 CPU 小时。先在坐标 falsify，胜者再 port T5R3（E2 模式）。
红线：J03WQQ、holdout_303 不碰；不新增数据/费用。

## 待用户拍板
开火 v3 falsification 网？或砍某臂/加臂。开火即 L-007 状态→执行中。
