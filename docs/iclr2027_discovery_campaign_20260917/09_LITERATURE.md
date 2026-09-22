# 09｜近邻研究与新颖性边界

本表是有针对性的原始来源检索，不是穷尽性新颖性证明。已读范围标在每项；不把摘要核验写成全文复现。文献用于淘汰“换名词”的方案，不设文献数量门。

## [S1] The Lie Derivative for Measuring Learned Equivariance（2023）
核验范围：已读摘要/元数据。
微分测量等变性已有先例；不能把Jacobian诊断单独当创新。
来源：`https://arxiv.org/abs/2210.02984`

## [S2] Improving Robustness of 3D Point Cloud Recognition from a Fourier Perspective（2024）
核验范围：已读正文3.3/3.4及截图第1/5页。
图傅里叶loss梯度敏感性及频域对抗训练FAT；R01/R04需越过此近邻。
来源：`https://proceedings.neurips.cc/paper_files/paper/2024/file/7e0af0d1bc0ec2a90fc294be2e00447e-Paper-Conference.pdf`

## [S3] Excessive Invariance Causes Adversarial Vulnerability（2018/2019）
核验范围：已读摘要/元数据。
任务相关变化被忽略已有先例；R02必须增加受控协同与新后果。
来源：`https://arxiv.org/abs/1811.00401`

## [S4] Exploiting Excessive Invariance caused by Norm-Bounded Adversarial Robustness（2019）
核验范围：已读摘要/元数据。
降低敏感性可能增加不当不变性；R04须同时检验两类表现。
来源：`https://arxiv.org/abs/1903.10484`

## [S5] Model metamers reveal divergent invariances between biological and artificial neural networks（2023）
核验范围：已读期刊页面摘要与研究描述。
metamer已有成熟传统；不能把activation matching重新命名成发现。
来源：`https://www.nature.com/articles/s41593-023-01442-0`

## [S6] PointCloud Saliency Maps（2019）
核验范围：已读摘要。
梯度定位关键点/片段；单点重要性必须是强对照而非刻意弱化。
来源：`https://arxiv.org/abs/1812.01687`

## [S7] Universal adversarial perturbations（2016/2017）
核验范围：已读摘要/元数据。
共享扰动跨输入已有先例；R03不能以universality自身宣称新。
来源：`https://arxiv.org/abs/1610.08401`

## [S8] Rethinking Transferable Adversarial Attacks on Point Clouds from a Compact Subspace Perspective（2026）
核验范围：已读摘要与方法4.1–4.3；预印本。
CoSA使用prototype与低秩空间做迁移；R03需明确同边际合同和角色迁移差异。未核验代码发布。
来源：`https://arxiv.org/html/2601.23102v1`

## [S9] Neural Relational Inference for Interacting Systems（2018）
核验范围：已读摘要及官方repo说明。
相互作用与动态预测已有框架；R06的新问题是同边际依赖风险，非首次做多体预测。
来源：`https://arxiv.org/abs/1802.04687`

## [S10] Distributionally Robust Portfolio Optimization under Marginal and Copula Ambiguity（2024）
核验范围：已读期刊摘要；非本任务方法复现。
边际与依赖不确定性已有正式框架；只作理论邻域，不借用其应用结论。
来源：`https://link.springer.com/article/10.1007/s10957-024-02550-y`

## [S11] Structured ambiguity sets for distributionally robust optimization（2023）
核验范围：已读摘要与问题定义。
利用独立结构的DRO及已知边际邻域；不能宣称联合分布鲁棒优化是新算法。
来源：`https://arxiv.org/html/2310.20657v1`

## [S12] A Kernel Classification Framework for Metric Learning（2013）
核验范围：已读摘要/元数据。
涵盖LMNN/ITML与成对/三元组监督的度量学习；R05必须对比普通metric learning。
来源：`https://arxiv.org/abs/1309.5823`

## [S13] DINOv2 official README/model entry（checked 2026-09-17）
核验范围：已通过GitHub读取README及S/14权重链接。
可用冻结视觉模型入口；没有在本次下载权重/测试设备。
来源：`https://github.com/facebookresearch/dinov2`

## [S14] NRI official implementation README（checked 2026-09-17）
核验范围：已通过GitHub读取README。
有spring/charged数据生成入口；老依赖不能假装现代即装即用。
来源：`https://github.com/ethanfetaya/NRI`

## 开工前最值得记住的区别

R01不是“首次发现相关扰动危险”，而是在逐实体边际和实际预算固定时，测量表示/任务对联合依赖的失效及迁移。
R02不是“首次做feature collision”，而是用可解释协同编辑和oracle检验关系失察，并检验标准稳定性测试的盲点。
R03不是“首次低秩UAP”；R04不是“首次DRO”；R05不是“首次metric learning”；R06不是“首次多体预测”。每条的价值需要实测新问题和现有方法的不足，不能由术语推出来。

若worker找到更直接前作，修改贡献位置/对照后继续小试；若只剩重复，关闭该支。不要为了“不阻塞”装作没有前作。也不要求worker读完一个领域才开始计算。
