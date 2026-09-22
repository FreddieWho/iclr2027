# GPT6astra-high 独立评审 Prompt（转折位置精度 · 一轮意见）

> 使用说明：把本文件全文粘贴给 GPT6astra-high（高思考档）。若它能读到本仓库
> （根目录 `/home/huyudi/012_conference/iclr2027`），则按下文文件路径核查代码；
> 若读不到，正文已内含全部关键数字、判据与可疑点，可直接出意见。
> 要求：只评审、不写代码；引用凡未核对全文的一律标注"未核全文"。

---

你是独立评审员，对"小坐标MLP沿连续路径的转折位置精度"问题做一轮高强度审计，
并提出新的尝试方案。先读完全部背景再下结论，不要在信息不全时断言。

## 1. 问题定义

二分类坐标MLP（CoordMLP：Linear(8→64)→ReLU→Linear(64→64)→ReLU→Linear(64→32)→Linear(32→1)，
BCE训练，train_101上512场景，300 epoch，Adam 1e-2，全批量）沿单转折路径
gamma(t)=x+t*e（x为父场景，e为编辑）：oracle在t_star翻转（Bisection 12轮精化），
模型在t_theta翻转（logit过零点Bisection精化）；|terr|=|t_theta-t_star|，
turn_miss=模型无同向（0→1）翻转。评价池恒为eval_202上200条aimed单转折路径
（seed 202），只保留恰好1个oracle转折的路径。
基线flipmine（clean监督+翻转监督，lam=1.0）：miss约0.4、|terr|约0.16。
目标：找到决定翻转位置精度的机制。注意：miss路径的|terr|无定义，各分析只用非miss行——
审计时必须评估这是否引入选择偏差（见§4靶点）。

## 2. 已产生的阴性结果（逐条审计对象）

| # | 线 | 投入 | 结果数 | 判决 |
|---|---|---|---|---|
| 1 | E7监督侧（e7_v3.py） | 12臂60训练 | 积分基线0.229–0.242，各臂全差/持平 | 斜率非binding；监督侧死 |
| 2 | L007零训练诊断（l007_diagnostic.py） | 冻结模型重算 | ORACLE家族R²≤0；REPR在输出之外零增量；min_abs_logit Spearman 0.78–0.85 | 输入/隐几何死；邻接量承认同构 |
| 3 | 掠射角×带宽（l007_grazing.py） | 零训练 | w/a与\|terr\|相关+0.12/-0.06/-0.10 | 死 |
| 4 | kink密度（l007_kink.py） | 零训练 | 距离+0.06–0.17，计数-0.22–-0.29（线0.3） | 死；S3 Fourier连带park |
| 5 | 种子方差（l007_seedtrain.py/l007_seedvariance.py） | 8新种子复训 | 跨种子Spearman 0.41；ensemble +8.3%（门槛15%） | 噪声主导 |
| 6 | P1锁定动力学（l007_wave_train.py/l007_lockin.py） | 8种子×13检查点 | 中位锁定287.5 vs 收敛300；跨种子一致性全程~0.4 | drift/late，关键期关闭 |
| 7 | P2 TracIn+LOO（l007_tracin.py/l007_loo.py） | 筛选+36次复训 | top-k/随机比值0.85/0.84/0.17 | 数据决定论死 |
| 8 | P3几何钉位（l007_pin.py） | 12次复训 | A仅2/4、C 0/4且sham亦动±6–11% | 钉位死 |
| 9 | P5 connectivity（l007_connectivity.py） | 11权重5对 | 仅r04b11-r04b23对齐（barrier 0.9%），其余作废 | inconclusive |

共同契约（experiments/last15h/shared/paths.py）：scan_linear（17点均匀扫描）、
locate_oracle_turns / locate_model_turns（Bisection 12轮）、E7匹配规则
（只取0→1同向翻转中离t_star最近者）。汇总报告：
reports/last15h/E7_v3_verdict.md、reports/caea817_review/L007_DIAGNOSTIC_REPORT.md、
L007_FOLLOWUP_REPORT.md、L007_TRAIN_REPORT.md、L007_SEEDVARIANCE_REPORT.md。

## 3. 已知bug史（审计时复查同类残留）

- B1：l007_pin.py第一版最近邻距离矩阵reshape压成标量，B/C删集相同（已修＋"近/远不交"断言）。
- B2：l007_connectivity.py贪心匹配perm索引错位（已修＋测试）；注意对齐仍用贪心而非精确Hungarian。
- B3：l007_diagnostic.py的spearman guard曾经过严（已修n<3）。
- B4：P5判据设计缺陷（高barrier=检验作废而非阴性）已修正为inconclusive；但要复查"1个弱阳性实例"（r04b11-r04b23，|terr|极差0.053≈3.5σ_seed）是否只是噪声 cherry-pick。
- B5：min_abs_logit与目标同构已承认；hyper_dist=|logit|/|w|是同一邻接量的重缩放（已承认）。

## 4. 指定审计靶点（必须逐条给结论）

- T1：miss路径被排除在|terr|分析外，是否构成选择偏差？miss率本身（0.39–0.49）是否应作为第一指标而非|terr|？
- T2：LOO的k=6、种子3个是否足够？top-k删除搬动千分之几，能否排除"效应存在但k太小"？
- T3：TracIn代理损失（转折邻域±0.15的oracle标注态）是否合理？checkpoint每50epoch是否太疏？
- T4：P3的sham（删最远20场景）与干预组（删最近/插入邻域）是否可比？远处场景删除本身是否改变数据分布？
- T5：贪心匹配 vs 精确Hungarian——当前"仅1对可连"结论对匹配算法有多敏感？
- T6：种子方差0.41的残差相关 + ensemble +8.3%，"噪声主导"判决是否过早？0.41是否值得一个解释？
- T7：P1的锁定容差（max(0.03, 20%终值)）与"永不离开"定义是否把锁定epoch系统性推晚？
- T8：r04b种子 vs 2001+种子跨era几乎不可连（配方逐字相同）——是 basin 真差异，还是对齐算法天花板？

## 5. 输出要求（结构化中文）

- A. 正确性审计：9条逐条给"实现正确/有bug（文件+行级位置）/口径存疑"三态结论。
- B. 修正分级：必须修（结论可能反转）/建议修（加强稳健性）/可忽略，每条给具体修法；对T1–T8逐条表态。
- C. 新尝试方案：与已死清单（监督目标9项、几何5项、数据LOO、钉位、种子方差）正交的新探针，零训练优先；训练侧报最小成本（种子数×epoch量级，CPU可行性）。每个附可证伪判据、与已死机制的正交点、信息增益排序。不许提已死方向的换皮版本。
- D. 引用凡未核对全文的一律标注"未核全文"。

## 6. 硬约束

CPU only、无GPU租赁；训练只用train_101、评价只用eval_202/dev池；
禁碰封存池（holdout_909/confirm_1007/holdout_895等）；判据先行、不追线、
不为统一故事设计实验；禁因果断言，只给可证伪形式。
