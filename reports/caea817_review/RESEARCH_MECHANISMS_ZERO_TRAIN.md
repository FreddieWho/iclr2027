# Research: 小坐标MLP沿连续路径转折位置精度（|terr|/turn_miss）的剩余候选机制

## Summary
E7 v3（12臂60训练全败）已证伪"监督侧"（斜率下限/稠密标签/距离回归/切向蒸馏/自蒸馏），L007零训练诊断已证伪"输入几何"与"隐空间轨迹几何"，唯一可预测量 min_abs_logit 与目标同构、不算解释。剩余空间中，能用**冻结模型 + shared/paths.py 扫描契约零训练验证**的机制还有至少5个，其中信息增益/成本比最高的是：(1)跨种子分歧（方差分解）、(2)边界–路径掠射角×过渡带宽交互项、(3)路径上分段线性kink密度（谱偏置的可观测版本）。三者都与已死机制正交（详见§4）。

方法说明：本简报只给"若X成立，应观测到Y"的可证伪形式，不做因果断言。`web_read` 原文抓取在本轮全部失败（arXiv/OpenReview 返回 fetch failed），以下引用以检索到的官方摘要页与可核实元信息为准，全文级公式核验未做——写稿引用前需再对一次原文。

## Findings

### 候选机制总表（全部零训练可算）

**1. 边界–路径掠射角（grazing incidence）**
- **机理一句话：** 同一条决策边界，路径以小角度掠过时，同等法向扰动会被放大为 1/sinθ 倍的切向位置误差，即 |terr| ≈ w/|cos(grad,e)|，w 为过渡带半宽。
- **Claim:** 路径切向与边界法向夹角是|terr|的一阶放大器；E7只约束了斜率模长、N02只看了无条件方向，两者都没测"角度×带宽"交互项。 **Sources:** [Fawzi et al. CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Fawzi_Empirical_Study_of_CVPR_2018_paper.pdf)（边界曲率不对称、分类器沿某些方向曲、沿正交方向平坦）. **Support:** interpretation（把Fawzi的曲率不对称结论转写为单路径口径；原文是图像分类器经验研究，非坐标MLP）. **Confidence:** medium.
- **可观测签名（冻结模型+现有契约）：** 在 t_theta 处取 logit 对输入的梯度 g=∇_x s（一次 autograd，不训练），算 a=|g·ê|/‖g‖（ê=e/‖e‖）；带宽 w 取 |logit| 从 -k 到 +k 的 Δt（k=1或2，纯扫描量）；检验 |terr| 与 w/a 的相关（Spearman），以及 early/late 是否与符号 (g·ê) 一致。
- **最小验证成本：** 零训练；每路径 ~2–5 次 forward + 1 次 backward（200路径×3种子，CPU分钟级）。
- **支持/反对 prior art：** 支持：Fawzi et al. 2018（边界曲率高度各向异性，掠射方向最脆弱）；Balestriero & Baraniuk NeurIPS 2019（MASO/ReLU网络决策边界分段线性公式与曲率度量，给分段线性边界+路径夹角分析提供数学件）——[NeurIPS页面](https://proceedings.neurips.cc/paper/2019/hash/0801b20e08c3242125d512808cd74302-Abstract.html)。反对方：N02梯度方向诊断无判别（本地证据；但N02测的是无条件方向，本机制测的是交互项，见§4正交性）。
- **证伪线：** w/a 与 |terr| 的 Spearman < 0.2 且 early/late 与符号无关 → 本机制死。

**2. 过渡带宽本身（transition width）作为诊断量（非训练目标）**
- **机理一句话：** 翻转点定位精度下限由过渡带宽决定，带越宽、零点越易被噪声搬移。
- **Claim:** 即使"拉高斜率"作为训练目标已死（E7 A-1），带宽作为被动诊断量仍可能预测|terr|（目标≠诊断）。 **Sources:** 本地 E7 v3（斜率拉升后积分反更差）为反例背景；无外部强引用，列为弱候选。 **Support:** researcher inference. **Confidence:** low.
- **签名：** 同上 w；检验 w–|terr| 单变量相关。
- **成本：** 零训练，纯重扫描（更密 t 网格即可）。
- **证伪线：** 相关 < 0.2 → 死。与机制1的区别：若 w 单变量死而 w/a 活，则精度瓶颈是角度不是带宽——这本身就是信息。

**3. 分段线性分辨率 / kink密度（谱偏置的可观测版本）**
- **机理一句话：** ReLU坐标MLP沿一维路径是分段线性函数，线性区断点（kink）密度决定了它能安置翻转点的分辨率；谱偏置使网络倾向用尽量少的kink拟合训练点，标注点之间的翻转位置于是落在"最近的可用kink"而非真值处。
- **Claim:** |terr| 应与 t_star 邻域内 kink 稀疏程度（到最近kink距离 / 单位t内激活模式切换次数）正相关。 **Sources:** [Tancik et al. NeurIPS 2020 "Fourier Features…"](https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html)（低维输入MLP谱偏置：低频先学，用于NTK视角解释+Fourier映射补救）；[Ramasinghe et al. NeurIPS 2022 "On the Frequency-Bias of Coordinate-MLPs"](https://openreview.net/forum?id=oR5WIUtsXmx)（坐标MLP频率偏置行为与常规MLP不同、与激活区结构相关）；[Yüce et al. arXiv 2301.05816 "Understanding Spectral Bias of Coordinate-Based MLPs via Training Dynamics"](https://arxiv.org/html/2301.05816v4)（positional encoding与ReLU死亡/激活区关系）。 **Support:** interpretation（上述文献研究的是信号拟合/收敛速度，本简报将其转写为"翻转位置分辨率"口径，原文无|terr|概念）。 **Confidence:** medium.
- **可观测签名：** 冻结模型密扫描（n_scan=129或256，用 scan_linear 同契约）+ 记录每点激活模式（forward hook，无梯度）；量：(a) t_star ± δ 内 kink 计数；(b) t_star 到最近 kink 的距离；(c) t_theta 是否恰落在 kink 上（命中率 vs 随机）。检验 (a)(b) 与 |terr| 的相关。
- **最小验证成本：** 零训练；每路径 ~129 forward（200路径×3种子 ≈ 8万次小MLP forward，CPU可接受）；需加 hook 取激活模式（推理侧改动，不碰训练）。
- **证伪线：** kink密度/距离与|terr|无相关，且 t_theta 落 kink 命中率≈随机 → 本机制死。注意与已死机制正交：E7动的是损失函数，L007量的是隐空间轨迹"形状"，本机制数的是分段线性"断点 budget"，三者互不蕴含。

**4. 跨种子分歧（disagreement）→ 方差分解**
- **机理一句话：** 若高|terr|路径恰是多种子间 t_theta 分歧最大的路径，则误差主成分是训练噪声/欠定（variance），不是系统性表示偏（bias）；反之分歧小而误差大，则是系统性偏。
- **Claim:** 跨种子 t_theta 标准差（或 t_star 处投票熵）预测 |terr|；这是 L007 报告末尾点的"噪声分解"方向的最小第一步。 **Sources:** [Jiang et al. "Assessing Generalization of SGD via Disagreement" (ICLR 2021, OpenReview)](https://openreview.net/forum?id=WvOGCEAQhxl)（同架构同数据多种子训练的分歧可估计测试误差；引用~193）。 **Support:** interpretation（原文结论是数据集级误差估计，转写为路径级|terr|预测是研究者外推，需实证）。 **Confidence:** medium.
- **可观测签名：** 现有冻结 checkpoints（r04b flipmine 3种子；E7 flipmine 5种子）同 paths 同契约重跑 locate_model_turns；量：每路径 t_theta 跨种子 std、分歧熵；检验与 |terr|（取种子均值或配对）相关；顺带做 bias–variance 分解：mean(t_theta)−t_star（bias²）vs var(t_theta)。
- **最小验证成本：** 零训练（checkpoints 已存在）；成本 = 重扫描次数（种子数×路径数 forward）。
- **证伪线：** 分歧与|terr|无相关（Spearman<0.2）→ 误差不是种子方差主导，关闭"噪声地板"解释，逼向系统性偏。无论哪边结果都有信息（双向判据）。

**5. 离路(off-path)边界褶皱探测（小正交扰动）**
- **机理一句话：** t_star 附近若在垂直于 e 的方向上极近处另有边界褶皱，沿路一维扫描看到的翻转位置会被褶皱"劫持"。
- **Claim:** 在 t_star 处使预测翻转的最小扰动范数（不限方向）显著小于沿路到 t_theta 的距离 ⇒ 一维|terr|只是高维边界错位的投影。 **Sources:** [Fawzi et al. CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Fawzi_Empirical_Study_of_CVPR_2018_paper.pdf)（曲率刻画最脆弱方向）；Moosavi-Dezfooli 博士论文 "Geometry of Adversarial Robustness of Deep Networks"（曲率↔通用扰动鲁棒性的几何关系）——[EPFL记录](https://infoscience.epfl.ch/server/api/core/bitstreams/339dbb20-c75c-4b3f-976a-ff112943c54d/content)。 **Support:** interpretation. **Confidence:** low-medium.
- **可观测签名：** 冻结模型，在 x+t_star·e 处做小预算 PGD/梯度法求最小翻转扰动 δ*（不训练，只推理+梯度）；比较 ‖δ*‖ 与 ‖e‖·|terr|；检验两者比例与 turn_miss 的关系。
- **最小验证成本：** 零训练；每路径数十次 forward+backward。成本高于机制1–4。
- **证伪线：** ‖δ*‖ 普遍 ≫ 沿路误差距离 → 一维误差非褶皱劫持，本机制死。

### 次级（需新训练才能验证，不作首选，仅列）
- (S1) 隐式偏置 max-margin 方向（Lyu & Li ICLR 2020, [OpenReview](https://openreview.net/forum?id=SJeLIgBKPS)：齐次网络GD方向收敛到max-margin KKT点）——可解释"边界为何系统性地钉在训练点决定的max-margin处而非oracle真值处"，但干净验证需控制训练点几何重训 → 次级。
- (S2) 参数sharpness/flatness（Keskar et al. 2016；Dinh et al. 2017 "Sharp Minima Can Generalize"指出sharpness可重参数化；Andriushchenko et al. ICML 2022 "Towards Understanding Sharpness-Aware Minimization"指出m-sharpness并非总能区分泛化）——E7的global-grad惩罚臂（sharpness相邻干预）已死；且sharpness需Hessian/扰动、与|terr|链路过长 → 次级。
- (S3) NTK vs feature learning regime、grokking、mode connectivity、neural collapse——全部需要训练轨迹或改架构（如Fourier特征/SIREN对照），违反零训练优先 → 次级。其中"加Fourier特征重训看|terr|是否改善"是谱偏置假设的训练侧验证，若机制3诊断为阳性，它是自然的下一步（但那是新训练，不在本次诊断内）。
- (S4) 校准/ECE（Guo et al. 2017）——与已死的BAN（温度软化有害）及同构量 min_abs_logit 高度重叠，不正交 → 不推荐。

### 明确排除（不再碰）
- E7已死：bracket内斜率下限、vanilla-mixup、flipmix稠密标签、SDF距离回归（同logit口径）、tangent匹配、coupled-margin、global-grad、BAN、noise-only对照。
- L007已死：oracle输入几何（边际深度/斜率/代价/位置）、隐空间轨迹几何（速度集中/夹角/路径长度）、输出斜率/曲率单变量（除同构邻接量）。
- 本简报机制1–5均不重复以上：1测交互项（非单变量），2是被动诊断（非训练目标），3数断点（非轨迹形状），4用跨种子统计（非单模型量），5探离路方向（非沿路量）。

## Contradictions
- Tancik 2020 vs Ramasinghe 2022：前者称坐标MLP有标准谱偏置（Fourier映射可解），后者称坐标MLP频率偏置行为与常规理解不同（含反直觉的高频偏好条件）。对本设定的影响：机制3的"谱偏置→kink稀疏"方向性预测不能直接抄文献，必须以本地kink计数为准——文献只提供"断点budget值得数"的合法性，不提供方向。
- Sharpness–泛化相关性本身有争议（Dinh 2017重参数化反例；Andriushchenko 2022 m-sharpness不总能区分）→ sharpness列为次级而非首选的直接原因。
- Jiang分歧预测误差：后续有"Nagel et al. 2202.01851 A Note on…"指出其适用边界（regular ensemble vs deep ensemble校准差异）→ 机制4若做，必须同时报告校准侧记账，避免单边解读。
- web_read 全失败：以上引用未做全文公式核验，§2中"interpretation"标签均为研究者转写，非原文直接断言。

## Missing evidence
- N02梯度方向诊断的具体口径（是否测过 t_theta 处局部 g·ê、是否做过 w/a 交互项）——未读原文，只凭任务描述"无判别"认定；若N02实际已测交互项，机制1优先级下调。
- 现有checkpoints是否都保留完好（r04b flipmine 3种子 / E7 flipmine 5种子）——机制4可行性依赖此。
- 激活hook在当前代码结构下的接入成本——机制3假设"推理侧hook易加"，未验证。
- Balestriero & Baraniuk 2019的分段线性边界公式能否直接给出"kink密度↔翻转分辨率"定量预测——只核实了摘要级，未推公式。

## Sources
- Kept: Fawzi et al. CVPR 2018 Empirical Study of Topology and Geometry (https://openaccess.thecvf.com/content_cvpr_2018/papers/Fawzi_Empirical_Study_of_CVPR_2018_paper.pdf) — 边界曲率不对称/掠射脆弱性的直接先例，支撑机制1、5。
- Kept: Tancik et al. NeurIPS 2020 Fourier Features (https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html) — 低维坐标MLP谱偏置+NTK视角，支撑机制3合法性。
- Kept: Ramasinghe et al. NeurIPS 2022 On the Frequency-Bias of Coordinate-MLPs (https://openreview.net/forum?id=oR5WIUtsXmx) — 坐标MLP频率偏置特殊性，支撑机制3并给出矛盾侧。
- Kept: Lyu & Li ICLR 2020 Gradient Descent Maximizes Margin (https://openreview.net/forum?id=SJeLIgBKPS) — 隐式偏置次级机制的理论源头。
- Kept: Jiang et al. ICLR 2021 Assessing Generalization via Disagreement (https://openreview.net/forum?id=WvOGCEAQhxl) — 机制4的直接先例。
- Kept: Balestriero & Baraniuk NeurIPS 2019 Geometry of Deep Networks (https://proceedings.neurips.cc/paper/2019/hash/0801b20e08c3242125d512808cd74302-Abstract.html) — ReLU网边界公式+曲率度量，支撑机制1、3的数学件。
- Kept: Yüce et al. arXiv 2301.05816 Spectral Bias of Coordinate MLPs via Training Dynamics (https://arxiv.org/html/2301.05816v4) — 激活区/kink视角，支撑机制3。
- Rejected/deprioritized: SAM/sharpness系（Andriushchenko ICML 2022等）— 争议大且链路长，降为S2；校准/ECE系 — 与死机制重叠，降为S4；grokking/mode-connectivity/NTK-regime系 — 需新训练，降为S3。

## Next steps
1. 先做机制4（分歧分解）：零成本复用checkpoints，且结果双向都有判决力（variance主导→开噪声分解；bias主导→逼向表示偏）。
2. 再做机制1（w/a交互项）：一次autograd+密扫描，若阳性则直接给出"掠射路径是误差放大器"的可发表诊断。
3. 再做机制3（kink计数）：若阳性，自然导出下一步训练实验（Fourier特征对照）；若阴性，关闭谱偏置解释。
4. 三个诊断的预注册判据建议统一为：Spearman≥0.3且三种子同向 = 阳性；<0.2 = 阴性；之间 = 不确定（加种子/加路径）。

## Supervisor coordination
无阻塞，无需协调。本地文件未做任何改动。
