# R01–R10 修复与复算

2026-09-24 执行快照（8 条 lane 全部收尾后更新）。基准 e1a933e6b32dd07c6895c7925cbccd9004da7f54；旧产物只读。

**状态：本轮 R/O/N 施工项全部收尾，无 RUNNING/NOT_RUN 残留。** 主要生产修复、复算与训练均已完成并回收校验；
下面逐条写明"可保留范围"，不把工程通过当作科学结论。仍属边界而非欠项的内容集中在文末。

| 项目 | 实际状态与证据 | 可保留及未决范围 |
|---|---|---|
| R01 | 完成四臂逐parent重算；U10_CLAIM_CORRECTION.md | T1 I为正、T2 I为负；两任务distance+flip优势保留，不称普遍正交互 |
| R02 | 完成role-shared归一化及18 typed重训；完整保存恢复流水线与D02群平均指标已核查 | typed仍低于distance只限该配方；source scalar无此bug，不重训 |
| R03 | **完成**；96模型×3bank血缘矩阵，96/96训练来源由checkpoint内嵌mu/sd指纹唯一定位（63 N/27 4N/6 16N）；392412个增广状态对三个quartet银行**0 exact/0 near**（最近Linf 0.112/0.0131/0.0324），检测器经400/400+400/400+注入1条标定；R03_LINEAGE_REPORT.md + AUGMENTATION_STATE_AUDIT.json + TRAIN_SOURCE_RESOLUTION.json | 不称"全历史数据独立"（父场景重叠是另一已知事实）；非几何来源（足球/U10/视觉线）不在同一输入空间、未逐状态审计；审计只覆盖"状态相等" |
| R04 | 完成起点置信修复、72行前后敏感性及逐样本分母；R04_CONFIDENCE_REPORT.md | 未追加训练时顺序干预；原P1B不被局部bug直接撤回 |
| R05 | 完成生产匹配目标/RNG修复、512真实parent渲染审计与反例；AUX_OBSERVABILITY_AUDIT.md；21臂GPU矩阵已回收校验并入稿 | 旧renderer492/512整轨道相同，新canonical512/512；匹配目标只提高辅助可学性，不产生稳定额外J增益 |
| R06 | 完成冻结BN、指标修复与旧factorial重算；VISION_FACTORIAL_CORRECTED.md；21臂GPU矩阵已回收校验并入稿 | 224是上采样；保留全部seed，初始化均值差29.80pp；static→flip为regimen效应，逐parent曝光分布仍不同 |
| R07 | 完成30臂同起点公平续训；CONTINUATION_FAIR_COMPARISON.md | protection对plain无稳定跨seed J优势；未见原子回归11–38%，不能关闭U05 |
| R08 | **完成**；真实E、角色/时间/cap修复、三场有限搜索，以及**自然E发生率**（主格8/1449=0.55%，事件级8/162=4.94%，帧对10/7117；搜索可构率670/1449=46.2%→富集~84×；条件率自然10/6485=0.15% vs 搜索5066/216924=2.34%）；窗口依赖0.2s→2.0s为0.55%→16.98%；R08_NATURAL_E_RATE.md + NATURAL_E_RATE.json | 只测240事件样本（162审计事件），非三场全部传球，其余4场未用（J03WQQ封存）；联合格是两个真实单人位移的交叉、非真实观测帧；0.2s下被选中接球者0/155，不足以做策略偏好检验；标签是几何oracle不是传球成功率 |
| R09 | **完成**；PAVA、全部candidate断点、3编码×3seed策略重算，以及**全256场景无oracle筛选部署估计**（139/256可行、117/256不可行=45.7%；dev目标覆盖0.4884→eval_full实测0.1455–0.3239；b0.5下net-benefit差0/9对排除0）；DECISION_FRONTIER_V2.md + FRONTIER_FULLSCENE.json + 27648行逐场景CSV | 只能称"dev匹配策略的外推"，不称同覆盖已实现；dev场景集本身仍按可行性筛出；同一批256场景再分析、无独立新场景确认；净收益是说明性效用、无业务效用验证 |
| R10 | **完成**；8条lane全部接线进附录（N02 48臂/N01-P1/O04前瞻+D10回声/R09全场景/O02匹配预算/O01 warm-start/R03血缘/R08自然E），claim ledger新增49行，状态文档同步，构建回执重生成；PAPER_PATCH_AUDIT.md | 新结果只按可支持范围入稿；未把 N02 机制、O04 主判定、N03 增益写成阳性 |

## 对审计包本身的修正（本轮新增两条）

D09整数parent属于d09_fresh局部namespace。逐坐标验证发现：对N模型236个quartet全部新；4N有58个训练parent quartet；16N全部236已见。历史剔除18个得到218并不是有效去重，只保留为错误ID筛选的敏感性结果。dev512的237个quartet坐标均未匹配N/4N/16N，但仍是已暴露旧bank，不称首读确认。旧D03E与D09还共享8个真实E父（D03的74个quartet、D09的14个quartet），不能称两bank完全独立；未来builder已修正索引偏移，旧银行保留。

**新增 1（须披露的阳性）**：`d09fresh665` **单编辑**银行有 761/126295（0.603%）条目与 16N 池模型的增广训练输入逐位相同、163/126295（0.129%）与 4N 池相同（涉及 512 个抽样父场景中的 127/30 个，命中条目全部是标签翻转条目）。成因是银行父场景取自 16N 池且与 flip 挖掘共用同一确定性候选网格。**不影响 quartet 矩阵（0/708）**，但使用该单编辑子集的统计必须附排除敏感性结果。

**新增 2（实现细节，影响血缘口径）**：银行 `Sx` 是**起点状态**、`Se` 才是编辑（`build_bank.py:82`）。任何只用 `Sx` 的状态比对实际在比父场景，这是本轮发现的口径陷阱。

## 必要检查

R02/R04/D02 生产回归、R07–R09 相关检查、readout 数值梯度/solver 检查通过。本轮另新增两组：

- `experiments/e1a933_review/test_vision_contract.py`（10 项）+ `vision_contract_mutation_check.py`：把 R05/R06 合同检查移植为调用生产函数的回归测试，并证明 4/4 mutation 守卫在恢复旧行为时**确实失败**。
- `experiments/e1a933_review/test_r08_natural_e.py`（6 项）、`test_n01_p1_diagnostic.py`（4 项）、`test_fair_football_frontier.py` 新增 6 项生产调用回归（共 19 项通过，9 个原始反例保留）。

**修正一处此前的过度陈述**：审计所称"R05/R06 十一项合同检查通过"中，`normalization_commutes_<p>` 四项**不可能失败**（点置换只行置换轨道，沿群轴排序后任何逐维仿射归一化都相同；实测 pooled 与有 bug 的 per-index 方案最坏差均 0.000e+00）。已删除并替换为两项可失败检查；其余检查经逐条核对确实具备判别力。没有跑仓库全量测试；这些工程检查不替代训练或科学审核。

## 仍属边界、不属欠项的范围

- **N02 机制 UNRESOLVED**：算力未匹配（B 的 MACs 是 A 的 12.25×）；冻结方向预测仅 pretrained 命中、random 未命中；`small_edit` 分层 5 quartet/3 parent 无辨识力。
- **O01 raw 未被证明优化到容量上限**（13/72 warm full-dose raw 臂训练误差 >5%，未做参数匹配）。
- **O02 只测 w=40 一种宽度、只做单向匹配**；难度/手性任务按施工卡属可选。
- **O04 typed encoder 类在新任务上的同口径结论未做**（发布检查点无独立特征头）。
- **投稿行政**：Figure 1、AI-use 定稿、双盲扫描、匿名打包属另行投稿收尾。

本轮对PLAN的影响：H1在新source pool保留距离表示优势，且该优势在一个被充分搜索的raw基线上仍成立（O01），但正交互仍依任务（T2为负、4N/scratch/f100两个seed CI含0）；H2视觉线给出regimen级效应但采样机制未决（N02）；H3自然E发生率已测（稀有但非零），不构成足球不可行；H4的S与分段归属明确依赖readout（O04）。没有把未完成问题改写成支持或反驳。
