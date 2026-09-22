# Research: 小坐标MLP转折位置精度的训练侧非幼稚解释（文献与机理评估）

**问题：** 二分类坐标MLP（64-64-32-feat，BCE）沿单转折路径 gamma(t)=x+t·e，oracle 在 t_star 翻转、模型在 t_theta 翻转；|terr|=|t_theta−t_star|，turn_miss=无同向翻转。基线 flipmine：miss≈0.4、|terr|≈0.16。
**任务边界：** 只做文献与机理评估，不写代码、不训练、不改仓库。另一条训练线正在做多种子方差分解+ensemble（进行中），本简报提案不得与之重复。
**硬约束（项目纪律）：** CPU only，无GPU租赁；训练只用 train_101，评价只用 eval_202/dev池；禁碰封存池（holdout_909/confirm_1007/holdout_895 等）。
**已死清单（禁重复，已读原文确认）：** 监督目标侧——斜率下限、平坦性、mixup、稠密标签、SDF距离回归、tangent、coupled-margin、global-grad、BAN自蒸馏（E7 v3，12臂60训练全灭，见 `reports/last15h/E7_v3_verdict.md`）；零训练诊断侧——oracle输入几何、隐空间轨迹几何、输出斜率/曲率（除同构邻接量）、掠射角×带宽交互、kink密度（见 `reports/caea817_review/L007_DIAGNOSTIC_REPORT.md`、`L007_FOLLOWUP_REPORT.md`）。kink阴性→谱偏置重训动机已连带 park；max-margin隐式偏置 park 待重训；离路褶皱 park（成本高）。

## Summary

最锋利的训练侧信息不在"端点方差是噪声还是系统性"（正在跑的线），而在三个正交问题：(1) 转折精度在训练**时间轴**上何时锁定（零新训练，用 checkpoints 回答）；(2) **哪些训练点**因果决定边界位置（TracIn 筛选 + 小规模精确 LOO 验证，CPU 可行）；(3) 边界是被**训练点几何**钉住而非 oracle 真值（沿路径方向增删/移动训练点的重训对照）。另有两个低成本辅助探针：集成多样性来源消融（init vs shuffle）与 mode connectivity 精度-损失正交性检验。Sharpness/Hessian 与 Fourier/SIREN 经诚实评估后建议 park（理由见文末）。

## Findings

### P1（排序第1，信息增益/成本最高）：检查点轨迹锁定分析——转折精度是早期锁定还是全程漂移
1. **Claim:** 若跨种子的最终 |terr| 排序在训练早期就已建立并保持稳定，则精度由关键学习期锁定，后续训练只是"雕刻"；若排序全程漂移/交叉，则精度是晚期动力学产物，早期干预无意义。该检验零新训练，只用已有/计划中多种子 checkpoints 的 eval_202 前向评估。 **Sources:** Critical Learning Periods in Deep Networks (https://arxiv.org/abs/1711.08856); One Period to Rule Them All / Layer Rotation 方法 (https://arxiv.org/html/2506.15954v1); Grokking as lazy-to-rich transition (https://arxiv.org/html/2310.06110v3)。 **Support:** interpretation（将"早期缺陷不可逆"类比迁移到 |terr| 排序稳定性；原文献结论是分类泛化，不是对转折位置精度）。 **Confidence:** medium。
   - 机理一句话：转折附近边界的函数形式在训练早期（特征学习窗口）定型，晚期主要是 margin 膨胀而非位置移动。
   - 可观测签名：对每个种子 s 与检查点 epoch e，计算 |terr|(s,e) 与 miss(s,e)（eval_202 aimed 200 路径，沿用 E7 0→1 匹配规则）；算 (a) 跨种子 Spearman( |terr|(·,e), |terr|(·,final) ) 随 e 曲线，(b) 锁定 epoch 分布（首次进入 final±容差带且不再离开），(c) NTK 距离或表征几何距离（若已有）作对照，检验"排序锁定早于 loss 收敛"是否成立。对照：正在跑的种子方差分解只看端点分布，本探针看**排序动力学**，信息正交。
   - 最小验证成本：零新训练；评估量 = n_seed(已有，≥8) × n_ckpt(~10–20) × eval_202 前向，CPU 小时级。
   - 与已死机制正交点：已死斜率/SDF/稠密标签全是"终态监督形状"假设；本探针问"终态何时决定"，与其全部正交；kink/掠射角是冻结模型几何，本探针是时间维度。
   - 证伪：若排序相关系数直到最后 10% epochs 才上升，或锁定 epoch 分布与 loss 收敛 epoch 重合且跨种子不一致，则"早期锁定"死，关闭"早期干预"类后续。
   - 若阳性→下一步：只在锁定窗口内做干预实验（如窗口内数据扰动，另立 proposal）；若阴性→关闭"关键期"解释，转向晚期 margin 动力学（P3）。
   - 文献适用边界（诚实）：Achille 等的"缺陷不可逆"针对分类泛化与数据增强/正则时机（Golatkar 等亦证早期正则最敏感），未核全文（检索摘要+综述级证据）；grokking 的 lazy→rich 解释在小宽度 MLP 上是否成立有争议（2024 NeurIPS 有"表征几何变化比 NTK 距离更贴近泛化跳变"的报告），故本提案只做**观测性锁定检验**，不做"NTK 距离解释精度"的因果断言。

### P2（排序第2）：TracIn 筛选 + 小规模精确 LOO——哪些训练点因果决定转折位置
1. **Claim:** 小训练集（train_101，n=101）+ 小 MLP 上，TracIn（一阶 checkpoint 梯度点积，无需 Hessian）可筛出候选"支持向量式"训练点；对 top-k 做精确 leave-one-out 重训，若删掉它们使 |terr| 系统性移动（而随机删 k 个不移动），就是"数据几何决定论"的硬证据。 **Sources:** TracIn 原论文 NeurIPS 2020 (https://arxiv.org/abs/2002.08484)（未核全文；依据 Google Research 博客方法总结 https://research.google/blog/tracin-a-simple-method-to-estimate-training-data-influence 与 AAAI 2022 综述公式页 https://ojs.aaai.org/index.php/AAAI/article/view/20791/20550）；Influence Functions 脆弱性警告 Basu 等 (https://openreview.net/forum?id=xHKVVHGDOEk)（未核全文；依据摘要级多源一致转述）；dattri 库方法枚举 (https://github.com/TRAIS-Lab/dattri)。 **Support:** direct evidence（TracIn 公式本身）+ interpretation（迁移到 |terr| 指标）。 **Confidence:** medium（方法可行性 high；"top-k 删除必移动 |terr|"为待验证假说，low→medium）。
   - 机理一句话：转折附近的决策边界位置由少数训练点（margin 最小/梯度对齐最强的点）钉住，其余点是观众。
   - 可观测签名：(a) TracIn 影响分：I(z→z_path) = Σ_c η_c ⟨∇ℓ(z;θ_c), ∇ℓ(z_path;θ_c)⟩，其中 z_path 为 eval 路径转折邻域点的代理损失（用已有 checkpoints 梯度点积，CPU 可算）；(b) 精确 LOO：仅对 TracIn top-k（k=5–10）逐个删点重训（小种子数），对照组为随机 k 点删除；观测 Δ|terr| 与 Δmiss。先验判据示例（需预注册）：top-k 删除的 |Δ|terr|| 中位数 ≥ 随机删除中位数的 2× 且符号一致跨种子，则数据决定论阳性。
   - 最小验证成本：TracIn 筛选零新训练（梯度点积，CPU 小时级）；LOO 重训 = k(~5–10) × 3种子 × 单训成本（小MLP+101样本，CPU 可承受，是全候选中最便宜的因果干预）。
   - 与已死机制正交点：已死全是"改监督函数形状"（斜率/SDF/mixup/蒸馏）；本探针**不改损失**，只改训练集成员，问题从"监督形状"转为"数据归因"，完全正交。零训练几何（输入/隐空间/kink）是观测相关，本探针是删除干预。
   - 证伪：若 top-k 删除与随机删除的 Δ|terr| 分布无差异（permutation test 不显著），则"少数点钉住"死，关闭数据归因线，转向优化偏置线（P3）。
   - 若阳性→下一步：对致因点做几何分析（它们相对路径的位置/标签/法向），并与 P3 的"沿路径方向增删点"对照合并；若阴性→关闭 LOO 线（TracIn 分数仅作描述性统计，不再重训）。
   - 诚实警告：经典 influence function（Koh & Liang 的 H^{-1} 形式）在深度非凸下脆弱（Basu 等，未核全文但多源一致），**禁用 Hessian 逆路线**，只用 TracIn（一阶）做筛选、LOO 做确证；TracIn 对 checkpoint 频率敏感，需固定频率并报告敏感性。

### P3（排序第3）：训练点几何控制重训——max-margin 钉位假说的直接对照
1. **Claim:** GD 的隐式偏置是把边界钉在**训练点决定的 max-margin 位置**而非 oracle 真值处；沿路径方向 e 增删/移动训练点应系统性拖动 t_theta，而远离路径的同等数量增删不应。这是用重训对照验证隐式偏置的最锋利形式。 **Sources:** Lyu & Li, Gradient Descent Maximizes the Margin of Homogeneous Neural Networks, ICLR 2020 (https://ar5iv.labs.arxiv.org/html/1906.05890)（未核全文；依据 ICLR OpenReview 页 https://openreview.net/forum?id=SJeLIgBKPS 摘要与后续统一分析 Tsilivis 等 https://arxiv.org/html/2410.22069v4（已核全文：late-stage、指数尾损失、KKT 点、算法依赖 margin）；UBC 课程笔记对 LL20/Soudry 等的定位 https://www.cs.ubc.ca/~dsuth/532D/25w1/notes/13-margins.pdf。 **Support:** direct evidence（理论设定）+ interpretation（本场景适用性评估为研究者推断）。 **Confidence:** medium（理论转述 high；本场景可验证性 medium）。
   - 机理一句话：BCE+GD 在可分后进入 margin 最大化相，边界位置由训练点包络决定，oracle 真值 t_star 不在目标函数里。
   - 可观测签名：三组对照（每组 3–5 种子）：(A) 沿路径方向在 t_star 附近**插入** 1–3 个 oracle 标签训练点（train_101 内增广，不碰封存池）；(B) 沿路径方向**移走/翻转**最近的 1–2 个训练点；(C) 远离路径的等量增删（sham 对照）。观测 t_theta 位移方向与量级：若 (A)(B) 系统性拖动 t_theta 向干预方向、(C) 不动，则钉位假说阳性。附带量：训练点 margin 分布 vs t_theta 的相关性。
   - 最小验证成本：3组 × 3–5种子 × 单训成本；CPU 中等（比 P2 LOO 贵约 2–3×，但仍在 101 样本小MLP 可承受范围）。
   - 与已死机制正交点：已死 SDF/稠密标签/tangent 是"给模型更多 oracle 形状信息"；本探针**不给新形状信息**，只动训练点位置/存在性，检验的是"优化器听谁的"（训练点包络 vs oracle 真值），正交。kink 阴性（表示容量非瓶颈）反而为本假设让路：容量够，位置仍错→偏置/数据问题。
   - 证伪：若 (A)(B) 与 (C) 的 t_theta 位移分布无差异，则"训练点几何钉位"死，关闭 max-margin 线，转向优化器动力学（P4/P5）。
   - 若阳性→下一步：二分搜索"拖动 1σ|terr| 所需的最小点移动距离"，量化钉位刚度；若阴性→关闭隐式偏置解释。
   - 适用边界（必须写入简报）：Lyu&Li 定理要求 (i) 同质网络（ReLU MLP 近似满足，bias 项破坏严格同质性）；(ii) logistic/指数尾损失（BCE/logistic 满足）；(iii) 可分后晚期相（late stage after perfect fit）；(iv) 梯度流/无穷小步长（实际有限 LR + Adam/SGD + early stopping 偏离理论）；(v) 结论是收敛到 KKT 点**方向**（非全局 max-margin，且是参数 margin 非输入空间几何 margin）。Tsilivis 等（已核全文）进一步强调 margin 是算法依赖的（GD vs sign/Adam 的几何不同）。因此本提案**不援引定理作预测**，只借其"边界由训练点包络决定"的定性预言设计对照；任何"理论预言 t_theta 位置"的说法都是越界。

### P4（排序第4）：集成多样性来源消融——init vs 数据顺序（寄生在跑中的 ensemble 线上）
1. **Claim:** 若 ensemble 降低 miss/|terr|，diversity 主要来自随机初始化（不同 basin）而非数据 shuffling 顺序；2×2 消融可归因，且成本寄生在已跑的多种子复训上。 **Sources:** Fort, Hu & Lakshminarayanan, Deep Ensembles: A Loss Landscape Perspective (https://ar5iv.labs.arxiv.org/html/1912.02757)（未核全文；依据摘要级多源一致转述：non-bootstrap、仅随机初始化的 ensemble 亦强，BNN 单 basin 对照）；diversity-accuracy 平面与对称性 ensemble 后续工作 https://arxiv.org/html/2303.02484v2。 **Support:** interpretation。 **Confidence:** medium。
   - 机理一句话：ensemble 增益 = 函数空间多 basin 采样；init 决定 basin，shuffle 只决定 basin 内位置。
   - 可观测签名：2×2：(同init,异shuffle) vs (异init,同shuffle)，每格 4–5 成员；量 = 成员间 t_theta 两两分歧（std/ pairwise |Δt_theta|）+ ensemble 后的 miss/|terr| 增益。对照就是正在跑的"全随机 ensemble"。
   - 最小验证成本：约 2× 当前 ensemble 线的成员数（或复用已跑种子补一格）；CPU 中等偏低。
   - 与已死机制正交点：已死全是单模型监督/几何；本探针问"多种子分歧的结构"，与"种子方差分解"（量级分解）正交（本探针是**来源归因**）。
   - 证伪：若两格的分歧与增益无差异，则"init 主导"死，关闭多样性归因，ensemble 只作方差缩减器报告。
   - 若阳性→下一步：追问不同 basin 的边界有何结构差异（连 P5）；若阴性→ ensemble 按纯 bagging 效应处理，不再深挖。

### P5（排序第5）：Mode connectivity 精度-损失正交性检验（纯评估，CPU 便宜）
1. **Claim:** 若不同种子解之间存在低损失连接路径（linear + permutation 对齐后 barrier≈0）但沿路径 |terr| 大幅变化，则"转折精度与训练损失正交"——这是对已死损失 shaping 全族的一次性总关闭；若 barrier 高且 |terr| 在各 basin 内各自稳定，则精度是 basin 属性。 **Sources:** Entezari 等 permutation 对齐后线性连通性 https://arxiv.org/pdf/2110.06296.pdf（未核全文）；Mechanistic Mode Connectivity（同损失不同机制仍可连通）https://proceedings.mlr.press/v202/lubana23a/lubana23a.pdf（未核全文）；LMC 综述页 https://www.emergentmind.com/topics/linear-mode-connectivity。 **Support:** interpretation。 **Confidence:** medium（方法成熟度 high；本小MLP上 barrier 大小未知，low→medium 待测）。
   - 机理一句话：损失地貌的连通性不蕴含函数位置精度的一致性；精度可能是损失零空间里的自由度。
   - 可观测签名：取 3–5 对种子（已有复训权重）：(a) naive 线性插值 θ(α)=(1−α)θ_a+αθ_b，α∈[0,1]（~9–17 点），每点评估 train loss + eval |terr|/miss；(b) 若 naive barrier 高，加 permutation 对齐（Entezari 法）后再测。对照量：barrier 高度 vs |terr| 沿程极差。判据示例：barrier < 5% 相对隆起但 |terr| 极差 > 1σ_seed，则正交性阳性。
   - 最小验证成本：零新训练；评估 = 对数 × 插值点数 × eval 前向，CPU 小时级。注意小 MLP permutation 对齐实现成本高于大网相对值，但仍远低于重训。
   - 与已死机制正交点：已死假设"损失信号携带位置信息"（斜率/SDF/稠密/BAN 全是改损失）；本探针**直接检验该前提**（损失不变，精度变否），是元检验。
   - 证伪：若 |terr| 沿低损失路径恒定（精度随损失走），则正交性死，说明精度仍是损失地貌属性，重启损失设计有理；若正交阳性，则永久关闭"再调损失形状"族（与 E7 v3 的 12 臂阴性汇合为双重关闭）。
   - 若阳性→下一步：报告"精度是损失零空间自由度"，后续只做数据/优化偏置（P2/P3）；若阴性→重新打开损失设计，但需新理由。

## Contradictions
- Influence function 可用性矛盾：Koh & Liang（ICML 2017 talk 页 https://icml.cc/virtual/2017/talk/1107）称非凸非光滑下近似仍可提供有用信息；Basu 等称深度下总体脆弱（未核全文），Epifano 等（2023，https://www.sciencedirect.com/science/article/abs/pii/S0893608023001648）称"不如先前认为的脆弱"。本简报取保守立场：禁用 H^{-1} 路线，只用 TracIn+LOO。如实记录分歧，未裁决。
- Sharpness 与泛化关系矛盾：SAM 等称 flat 有益；Dinh 等（转述自 https://www.inference.vc/sharp-vs-flat-minima-are-still-a-mystery-to-me 与 https://tuananhbui89.github.io/blog/2024/sharpness）指出重参数化可任意改变 sharpness 而不改变函数，故 sharpness alone 不能解释泛化。本简报据此 park sharpness（见下）。
- Grokking/lazy→rich 解释力矛盾：Kumar 等称 lazy→rich 转变控制 grokking；2024 NeurIPS 论文（https://neurips.cc/virtual/2024/102692）称 NTK 距离可与泛化跳变解耦、表征几何更贴近。本简报 P1 只用排序稳定性观测，不站队。

## Missing evidence
- Lyu & Li 原文未核全文：定理的精确假设（bias 处理、同质度、离散 GD 步长条件）未逐字核对；以上适用边界基于摘要级+ Tsilivis 统一分析（已核全文）+ UBC 笔记，转述可信但非逐字验证。
- TracIn/Fort/Entezari/Lubana/Achille/GHD 谱偏置原文均未核全文（web_read 对 arXiv PDF 连续失败，jina/firecrawl 均不可用或被限流；仅 Tsilivis html 抓取成功）：本简报中凡涉及这些文献的精确公式与量化结论处已标注"未核全文"，只依赖其方法级公开摘要（公式来自 Google 博客与 AAAI 综述页）。
- 本地关键事实（train_101 大小 n=101、单训 CPU 成本、已有 checkpoints 数量/epoch 网格、eval_202 路径数 200）来自任务书与历史报告引用，未独立 re-verify；各提案"最小验证成本"为量级估计，需执行方按实测单训时间换算。
- 未验证 dattri/TRAK 在本仓库 PyTorch 版本可用性；P2 默认用手写 TracIn 点积（依赖已有 checkpoints），不引入外部库。

## Sources
- Kept: Tsilivis et al., Flavors of Margin (https://arxiv.org/html/2410.22069v4) — 唯一已核全文；用于 Lyu&Li 适用边界（late-stage/KKT/算法依赖 margin）。为什么重要：防止 max-margin 被庸俗化为"理论保证边界在真值处"。
- Kept: Lyu & Li ICLR 2020 页 (https://openreview.net/forum?id=SJeLIgBKPS) + ar5iv (https://ar5iv.labs.arxiv.org/html/1906.05890) — 钉位假说的理论源头（未核全文）。
- Kept: TracIn 方法页 (https://research.google/blog/tracin-a-simple-method-to-estimate-training-data-influence) + AAAI 公式页 (https://ojs.aaai.org/index.php/AAAI/article/view/20791/20550) — P2 公式与"checkpoint 频率敏感"警告。
- Kept: Basu et al. 脆弱性 (https://openreview.net/forum?id=xHKVVHGDOEk) — 禁用 H^{-1} 的依据（未核全文）。
- Kept: Fort et al. ensembles (https://ar5iv.labs.arxiv.org/html/1912.02757) — P4 来源归因依据（未核全文）。
- Kept: Entezari et al. LMC (https://arxiv.org/pdf/2110.06296.pdf)；Lubana et al. Mechanistic MC (https://proceedings.mlr.press/v202/lubana23a/lubana23a.pdf) — P5 方法依据（未核全文）。
- Kept: Achille et al. critical periods (https://arxiv.org/abs/1711.08856)；Layer Rotation 识别法 (https://arxiv.org/html/2506.15954v1)；Kumar et al. lazy→rich grokking (https://arxiv.org/html/2310.06110v3) — P1 依据（未核全文）。
- Kept: Tancik et al. Fourier Features (https://arxiv.org/pdf/2006.10739)（未核全文）；Sitzmann et al. SIREN (https://proceedings.neurips.cc/paper/2020/file/53c04118df112c13a8c34b38343b9c10-Paper.pdf)（未核全文）；Coordinate-MLP 频率偏置 NeurIPS 2022 (https://proceedings.neurips.cc/paper_files/paper/2022/file/0525fa17a8dbea687359116d01732e12-Paper-Conference.pdf)（未核全文）— 仅用于 park 论证。
- Rejected/deprioritized: 通用 sharpness 博客与 SAM 医疗评测 — 方法虽热但 Dinh 重参数化批评未被回答，且 Hessian 链路长；本任务明确要求诚实评估性价比→ park，不占 5 个名额。YouTube/二手讲解（Fort 解说、TracIn MLBBQ）— 仅作发现辅助，未作证据引用。

## 诚实 park 项（不占 5 名额，但任务点名要求表态）
- **Sharpness/Hessian（park）：** 性价比为负。成本：Hessian 谱（Lanczos/幂迭代）或 SAM 式扰动评估链路长，且 Dinh 等重参数化批评下"sharp 值"本身在 ReLU 缩放对称下可被任意操纵而不改变函数（从而与 t_theta 无关）；E7 已证"斜率拉上去精度反更差"，sharpness 是斜率在参数空间的 cousin，重蹈覆辙概率高。重开条件：P5 证伪正交性（精度确是损失地貌属性）之后。
- **Fourier 特征/SIREN 对照（park，维持 L007 判断）：** kink 密度双阴性（|ρ|<0.3，t_theta 落 kink 率仅 0.15–0.20）已抽掉"断点 budget 不足"前提；此时加 Fourier/SIREN 是"无动机重训"，且它引入的新归纳偏置（高频振荡）与 miss（无翻转）方向关系不明。重开条件：P1 显示晚期高频拟合与锁定相关，或有人先给出"高频能力→位置精度"的可证伪链条。注意 Tancik 的 Fourier 特征针对的是"拟合高频信号值"，Sitzmann 的 SIREN 针对连续信号表示，**两者都不是为"分类边界位置精度"设计的**，直接迁移属范畴误用——如实记录。

## Next steps
1. 执行顺序建议（按增益/成本）：P1（零新训练，先做）→ P2 筛选部分（零新训练）→ P5（零新训练）→ P2 LOO（小重训）→ P3（中重训）→ P4（寄生 ensemble 线）。
2. 所有重训提案执行前需预注册：可观测签名中的判据数值（如 2×、barrier 5%、排序相关阈值）必须先冻结，再看数。
3. 若 P1+P2+P3 全阴 + 正在跑的方差分解显示噪声主导：诚实结论是"转折位置精度在当前设定下无训练侧系统解释"，与 E7/L007 的排除网汇合为论文的边界声明，不再开新训练线。

## Supervisor coordination
无需协调；本任务为独立文献评估，无阻塞。如实声明：web_read 对 arXiv PDF 多次失败（jina/firecrawl/默认 reader 均失败或被限流），仅 Tsilivis html 抓取成功；其余文献依赖摘要级+二级来源，已逐项标注"未核全文"。
