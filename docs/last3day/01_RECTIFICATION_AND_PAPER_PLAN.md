# caea817 审计整改与论文迁移施工单

审计基准：`FreddieWho/iclr2027@caea817a3bc1e09a0ebe33fa1e917cdd085727a7`  
日期：2026-09-22  
用途：直接交给总控 agent 执行。本文处理正确性、可复现性和主稿；新探索见 `02_EXPLORATION_AND_UPGRADE_PLAN.md`。

## 0. 执行原则与审计边界

本审计核对了该提交的论文全部输入章节、当前权威报告、核心脚本和部分实际 JSON，并对已有汇总数字作了独立算术复核；未独立重训模型、完整运行测试或编译论文。因此，“报告支持”“代码支持”“本轮算术推导”必须分开。

你最接近现场，可以调整顺序、实现和小规模诊断。改变一个不适用的实现不需要机械等待批准；但应写明理由和结果可见性，不能把事后分析包装成原先冻结的验证。不要再重做一套目录治理体系。

优先级：纠正具体事实 → 保护已有有效结果 → 让主稿覆盖真正贡献 → 投稿交付。探索不得阻塞前三项。单个任务受阻时，完成其余独立任务。

允许：读取既有公开归档结果、完善生产分析入口、生成论文图表、直接修改 `paper/`。遵守现有封存权限；新模型评价或新选样需要使用封存原始输入时，不得绕过权限。已有结果重算与新假设验证应标为不同事件，不能重新“封存”后声称从未见过。

## 1. 本轮应保住、并重新组织的科学主线

不是回到 AMR/谱几何，也不是把文章改成抽象的“正确性有很多维度”。建议主线为：

> 在精确知道真实标签如何变化的受控编辑中，原子状态判断正确不足以保证联合结果正确；起点置信度也不等于变化后的可靠性。进一步，终点变对不等于完整组合修复。已有关系特征与 flip 数据的组合提供了一个有边界的正向修复实例。

对应四个角色：

| 内容 | 角色 | 不应写成 |
|---|---|---|
| 原子保持、联合翻转的 quartet 与连续路径 | 主问题与主要证据 | 新发现一切组合推理都会失败 |
| P1 当前置信与后续风险分离 | 描述性主发现，坐标域限定 | 置信造成盲点、神经网络有记忆惰性 |
| P3 终点修复/完整修复/错误迁移 | 主分析与修复评价 | AB 变对就是完整恢复能力 |
| relfeat + flipmine | 有边界的正向结果；必须进入主线 | 新通用架构、所有问题已解决 |
| 足球、像素 | 不同观察方式下的有关行为验证 | 已验证 P1/P3/relflip 全部跨域 |
| P2 行动接口分析 | 支持性结果或 Discussion | 所有收益都只是覆盖，或已排除所有阈值解释 |
| 旧失败分支 | 附录中的具体范围边界 | 排除了所有其他机制 |

## R1 — P1G：修复“冻结文件传了，但阈值没有冻结”【必须】

### 已确认的问题

`experiments/next_novelty/p1g_repair.py` 读取 `freeze.get(f"s{seed}", confirm_quantile)`；实际 `artifacts/next_novelty/p1_freeze.json` 使用 `s11_clean` 等键，值是含 `t25/t50/t75` 的字典。复算说明给 P1G 传入的正是这个文件。

归档 `artifacts/next_novelty/p1g_confirm/P1G.json` 已显示：

| seed | 开发集冻结 t25 | P1G 确认结果实际 thr |
|---|---:|---:|
| 11 | 20.7555 | 18.2866 |
| 23 | 22.1445 | 18.7195 |
| 47 | 20.7557 | 21.5690 |

这是局部分析 bug。不能因此作废 P1B 主曲线：`p1b_confirm/P1B.json` 的对应冻结阈值读取是正确的。

### 施工

1. 建立一个生产级阈值加载函数，明确 schema；P1G 读 `freeze[f"s{seed}_clean"]["t25"]`，或使用确实在开发阶段固定的专用高风险阈值表。不能在确认模式静默退回当前数据分位数。
2. 缺失键、模型 SHA 不对应、bank 标识错配时直接报错。开发模式可计算阈值，但输出必须含模式字段。
3. 用已归档逐样本输出优先重新计算该受影响分析；原结果留作历史，生成修正版，不覆盖成“当时就是这个结果”。无法在当前权限下重算时，删除“冻结高风险确认”表述并列明缺口。
4. 高风险主群体由 clean 决定后，计算全体 baseline-fixed 的终点、起点、联合正确性变化。当前脚本最后的 `paired_delta` 仍只在 repair 起点正确的 `both` 集上计算，必须重命名为 common-start-correct secondary，不能替代主比较。
5. 给出全部主群体分母、共同起点正确分母、repair 新增起点错误数；parent 聚类配对区间。

验收：生产函数直接接受真实 schema；三个实际阈值逐位等于指定冻结值；缺键单元测试会失败；主比较不再通过 repair 起点正确筛掉回归样本。

## R2 — P1：分离有效主曲线与尚未成立的机制解释【必须】

### 2.1 统计实现

当前 `p1b_analysis.py::grouped_cv` 的 OOF 部分按 parent 划分、训练折标准化，是有效改进；但其 `refit_boot_ci` 在三个模型循环内都用全部 X 列拟合，并在同一个 bootstrap 样本上评价。因此不是 conf-only/no-conf/full 各自的 OOF 区间，也不是 held-out AUC 差的区间。

修法任选一种，但命名必须准确：
- 最小方案：保留正确的 OOF 预测，用完全相同 parent draws 对模型间 OOF AUC 差做配对重采样，标明区间条件于该 CV 拟合；若评估训练不确定性，再重复分组划分或重拟合完整流程。
- 完整方案：每次在训练 parent 重采样并拟合，在独立/OOB parent 评价；每个比较使用同一切分、同一 bootstrap draw，真正按 `cols` 取特征。

保留训练集拟合 AUC 只能标 `in_sample_diagnostic`，不得和 held-out 列混用。不要为了得到“无混杂”而换预测器反复挑模型；一个预先说明的线性基线与一个低复杂度非线性敏感性分析足够。

### 2.2 输入与群体

- 路径实际起点是 x+A 或 x+B；核验 `m_start/m_end` 来自该起终点，不是原始 x 的 m0。没有值不得用 0 伪装精确测量。
- 单编辑与 quartet 路径来自不同构造；使用相同 parent 池与阈值，不等于每个起点都同时具有配对 preserve/flip。主文应准确说明这一点。
- `p1_unified.py` 取每 parent 枚举中最前 40 个合格 single edits，而 `candidates_for_scene()` 按 node/family 顺序生成。输出编辑族、节点和位移分布；增加一个开发集确定的分层截取/等 parent 权重敏感性分析。不要将枚举靠前的子集描述为随机总体。
- `matched_check()` 目前是粗分箱高低置信比较，未真正输出承诺的 SMD；改称 stratified analysis，或补实际平衡表和区间。大样本数不等于共同支持或匹配成功。
- 数据中重复起点不能因出现在多个 quartet 就伪装成独立状态。报告 row、distinct start、quartet、parent 四种数量；主要 CI 按 parent。

### 2.3 概念与条件

主稿删除“within currently correct states, static error decreases”：该条件下起点错误率按定义为零。正确写法是：当前错误在未按正确性筛选的起点总体上计算；flip 更新错误额外条件于起点正确。

同一图上可以展示不同 estimand，但图注逐个写出条件与分母。十二个 dev/confirm 结果不是十二个独立模型，而是六个训练模型在两个池上的评价。

`top25` 是开发集参考分位阈值，不保证确认集每个子群实际只保留 25%；输出各子群实际覆盖率。

保留两套完整指标：
- endpoint reliability：preserve/flip 都不筛起点正确；
- conditional update reliability：preserve/flip 都筛起点正确。

R(tau,rho) 仅是明确条件下、给定重加权的混合风险。rho 必须定义为该保留总体内的变化组成或标准化权重；不能直接当未经选择的部署变化发生率。全为 1 的 bootstrap [1,1] 不是总体误差率被无不确定性地证明为 100%。

### 2.4 端点 Δf 的循环性

`|f(x)|/|f(x+e)-f(x)|` 使用了结果端的分数。二分类、起点正确且真实翻转时，终点错等价于 `f(x) * f(x+e) > 0`（边界相等时按判别规则处理）。因此高 AUC 不构成独立机制验证，更不是变化前预警效果。

保留为后验描述即可；删除“它控制后 confidence 系数仍正，所以存在独立机制/独立预测价值”。需要真正变化前的风险对照时，只允许起点分数、已知编辑 e、起点局部响应等当时可用量，且先明确使用成本；不强制另造风险模型。

验收：主曲线、OOF 点估计、区间、回归调整、后验诊断分别标注；错误区间不再流入 ledger；P1 主结论范围为构造的坐标变化风险，而非 causal confidence。

## R3 — 将关系表示的正向结果提升为主线，并补复算入口【必须】

已归档、同一 dev bank 的四组 J：

| seed | raw | raw+flip | relational | relational+flip |
|---|---:|---:|---:|---:|
| 11 | .0549 | .0675 | .1308 | .3840 |
| 23 | .0506 | .0591 | .1688 | .4219 |
| 47 | .0549 | .0759 | .2110 | .3966 |

`artifacts/next_novelty/relflip/RELFLIP_CONFIRM.json` 确实已存在：509 quartets，relflip J=.4735/.4479/.4361，R_full=.4954/.4709/.4485，M=.2508/.2905/.3000。不能说确认结果不存在。

但 `relfeat_eval.py` 当前硬编码 dev bank，`REPRODUCE_FINAL.md` 未给出生成上述确认 J 与 migration 的完整命令。补可参数化入口 `--bank --model-map --out`，保存模型 SHA、输入 bank SHA、对应原始预测/聚合来源。若结果由临时脚本生成，先恢复准确实现再统一，不要求另训模型。

明确 R_full、M 对哪个 baseline-110 集计算；不能跨 bank 比较 raw-flip 的 dev 与 relflip 的 confirm，制造放大的“反转”。四臂应先在同一 bank 配对，确认集缺的行明确缺失，不能从别的表拼进来。

保留手工特征边界：六个有序点对距离＋四个质心半径；节点配对身份是输入的一部分；这是任务感知的确定性表示，不是新基础模型、也不是 oracle-free 无先验。输入没有标签不等于没有结构先验。

论文中将“Representation lead”改为正式受控表示干预结果。J<.5 仍有明显残余错误，使用“substantially improves / shifts toward full repair”，不要“solves”。不因早期 relational pooling 失败而继续概括“relational variants all fail”。

验收：Abstract/Introduction/主表或主图至少有一处明确出现这项正向结果；确认数有实际脚本链；旧失败变体与新关系输入分开命名。

## R4 — P3 与训练结论的边界【必须，优先复用】

1. 完整 8×8 状态转移矩阵归档，不只留 top12。状态位对应 A/B/AB 的正确性，不是预测类别。
2. 在基线 H={A对,B对,AB错} 上：R_endpoint=R_full+M；另报 M/R_endpoint，它才是“端点修复中有多少迁移”的比例。三 seed 当前约 79%–94%，不能混用概率分母。
3. J、R_full、M 及差值使用同一 parent 重采样；给出 paired ΔJ 区间，不用两条单独 CI 重叠与否替代配对检验。
4. “J unchanged”收窄为“小幅变化/尚未证明有大的联合收益”，除非有明确等效界限。当前 raw flip 并非所有指标都零收益。
5. 已修公平 keepbal 仅说明该固定 BCE 采样/权重方案没有稳定优势，不是否定全部正例保护或回归约束。不要再为句子扩大阴性范围而重训十种方案。
6. 如果观察到多数 110→001，检查这是否伴随共同阈值/分数排序问题；先做升级文档中的无训练排序诊断，而不是重启“二阶 interaction 架构”。

## R5 — P2：修正对照与结论，不再把拟合失败解释成不可能【必须】

### 已知实现问题
- `p2_operating.py::dev_stats()` 的行动覆盖只取 dev，但 `P0.mean()` 取全表，包括 eval 候选分数。它不是标签泄漏，却不符合“仅 dev 拟合”；应对同一个 dev 行掩码求平均，并重算受影响对照。
- alpha,b 按匹配 repair 的覆盖与平均概率选取，不是按任务效用或全部可达到性能前沿优化。某个拟合失败只能排除这个拟合配方，不能排除所有 operating-point 解释。
- 阈值来自各自分位数已改进，但先把概率 round(4) 会人为制造饱和并列。用原始可行性 logit 排序；并列真实存在时完整保留。
- 等覆盖 q 差是两个策略各自在所选对象上的质量差，而当前 `quality_diff_ci` 只对两边共同执行对象计算。拆成 `policy_quality_difference` 与 `common_act_difference` 两个 estimand，分别配对计算，不能共用一个区间。
- dev 同覆盖不保证 eval 同覆盖，报告实际 gap；行动成本更高也不能称无条件质量支配。

### 最小施工

A. 修数据掩码、生产脚本与 JSON schema，报告 dev/eval/feasible 的准确计数。
B. 保留已有宽网格作为有限对照，明确 s11 贴边。更有辨识力的选择是：对硬分类，正仿射变换只改变一个共同阈值，可直接在开发集枚举排序分数间的所有可达阈值，无需盲目扩 b 网格。测试集仅评估已定策略。
C. 区分“与 repair 的分数形状近似”与“开发集上最优的阈值基线”。不能将前者的失败充当前者之外全部方案的失败。
D. 若要写能力改变，优先复用升级文档的 quartet 分数排序不变量；它能排除单调校准对同一组分数的解释，但不能单独归因到 encoder。
E. P2作为支持性短节/Discussion，除非得到清晰公平对照，不能挤掉关系表示的正向主结果。

验收：代码能生成每个引用数字；固定 dev 策略与 eval 报告对应；相同 coverage 和相同成本的含义不混；不存在“拟合没成功⇒所有校准都不可能”的句子。

## R6 — 主稿必须直接修改：问题定义、结构与准确措辞【必须】

### 6.1 最小形式化放正文，不要藏在运行日志

记 x_A=x+e_A、x_B=x+e_B、x_AB=x+e_A+e_B。定义测试总体：

`E = {y(x_A)=y(x_B)=y(x), y(x_AB)≠y(x)}`。

给出 CCM 在 E 且 A/B 预测正确上的条件错误；给出 J=P(A/B/AB同时正确|E)。定义 H=基线 A/B对、AB错，给 R_full、M、R_endpoint 恒等式。另报基线 x 是否正确作为审计列，不能悄悄改变 151/176 的既有定义。

明确本研究多数是 feed-forward classifiers 对各状态独立重新计算；“updating”是行为检验，不暗示内部记忆、递归状态或世界模型。当前反应与结果端分类错误的关系要说清楚。

简表交代：坐标输入8维/MLP64-32、关系输入10维的构造、CNN图像大小及训练规模、足球模型和真实标签是什么、训练量/采样/优化。有关参数须从真实配置提取，本文提到的64/32仅对应已读修复脚本，不应无审计推广到全部模型。

### 6.2 推荐结构（可合并，名称不拘泥）

1. Introduction：受控编辑问题＋原子正确条件失效；P1；迁移与有边界的正向修复。最多3项核心贡献。
2. Controlled changes and evaluation：quartet、路径、模型、数据、指标、确认资源用途。
3. Failure and confidence under change：85.8%、路径例子、P1主曲线；主要展示确认池，dev放附录。
4. Endpoint repair, migration, and relational input：raw flip 的局限＋四臂同表＋relflip；这是新主图所在。
5. External observations and decision interfaces：足球/像素有关行为；P2紧凑支持结果。篇幅不足时P2后移。
6. Related work；Discussion/Limitations；短Conclusion。
7. 不计页数声明与References；Appendix承接复现、失败分支和完整统计。

当前 `06_external_confirmation.tex` 实际是失败方法目录，不是外部确认。移到附录，主文只保留最能排除简单替代解释的2–3行，不展示“我们试过很多”的工作量。

### 6.3 逐项必须改掉

| 当前问题 | 位置例子 | 准确方向 |
|---|---|---|
| 原子保持/联合翻转定义不完整 | §2 | 明确 E 的方程与采样 |
| 当前正确条件下静态错误仍下降 | Abstract、Intro、§3 | 当前误差未按正确筛选；更新误差有额外条件 |
| `feasible-set size 3.6×` | §4 | 实际为拒动率约缩小3.6倍；非拒动覆盖 .57→.88 |
| 温度改选动作被当成“看到了输入变化” | §6旧文件 | 温度实验只支持行动规则对分数尺度敏感 |
| `thresholds ... do not rescue; probabilities not cutoff` | §6旧文件 | T>0、0.5阈值不变是代数性质，不能作经验排除 |
| 所有 relational variants 失败 | §6旧文件 | 指定旧变体与指标；新关系输入确有正向证据 |
| `misses propagate`后又说无因果证据 | Intro、§3 | 各结果共现/有关；不把WP4否定的因果链写回来 |
| 所有pixel styles unseen | §5旧文件 | 分清原样式重复与真正改变的样式 |
| pixel低atomic仍称完整组合修复 | §5旧文件 | 报AB/emergent endpoint改善；完整一致性未测 |
| 无P1像素验证写成“像素不支持P1” | Abstract、External | 足球此口径未复现；像素未测试，二者不同 |
| 108/128全部是严格零转折 | §3 | 查107纯零与108总漏的原始定义；未核清前只用“漏转折” |
| `never averaged`、撤回旧稿等反复出现 | 多节 | 方法一次说明分母，审计史移附录/内部文档 |

外部数据必须标“真实坐标上构造的受控改变”还是“实际比赛的自然事件”。不得用E2合成路径代替L002真实事件的阴性结论。T5R6属于旧任务，按既定边界保留历史储备，不当本论文主结论的外部确认；rebuttal也一样。

### 6.4 摘要/贡献/图表应包含什么

摘要保留一个主失败数字、P1范围、迁移反转、关系输入+flip的有边界正结果。不要把三个分母说明和所有组件百分比塞满摘要。不得在对照统计没完成时使用“synergy”“causal mechanism”。

贡献建议：受控变化与静态置信的可靠性边界；配对分析区分终点修复和完整修复；关系表示×训练覆盖的受控正向例子。足球/像素为外部观察证据，不必独立挤出一项贡献。

主表必须有同一 dev bank 四臂 J/A-B通过/AB正确/条件miss，以及确认池可追溯的对应行；未提供的确认行留缺口，不能跨池填。

## R7 — 图表和真正可编译的主稿【必须】

1. `main.tex` 从 paper 目录编译时，章节里的 `../figures/fig2...` 指到不存在的根目录 figures；生成器输出在 `paper/figures/`。统一为可验证的根路径，例如 `\graphicspath{{figures/}}`＋文件名，或直接 `figures/...`。
2. 当前fig2仅seed11 dev、fig3仅raw-flip migration，正向关系图被打印而未绘出。把主图换成确认池三种子可比曲线/小面板，加四臂联合正确率与修复流向。图里展示 count、pool、conditioning，不以复杂命名代替说明。
3. Figure1应是实际受控quartet/路径示例，说明oracle标签与模型预测的差异。未生成前不得写已完成。不要选只支持口号却不属于统计总体的示例。
4. `main.tex` 只有占位article/geometry；换成官方样式后编译、查缺图/undefined refs/overfull/主文页数。官方当前要求：主文不超过9页，2026-09-25 23:59 AoE提交；AI-use statement必需且不计主文页数。以执行时官方页面为准。[L5,L6]
5. 当前未输入Discussion、Conclusion、Appendix；创建并接入，而不是仅给reports写“已迁移”。技术细节与失败目录放参考文献后的附录。
6. 不继续追求40+引用配额。以每个关键claim的最近工作和准确支持为验收标准。
7. AI使用声明应反映本项目实际用在问题探索、代码、分析、审计、写作等环节的范围，由作者核准；不能默认写“仅语言润色”。匿名补充包移除姓名、邮箱、仓库所有者直链、机器路径与日志身份字段，不包含`.git`历史。
8. 标题与投稿系统已提交摘要核对；不自动改成另一问题。当前同问题内强化正向结果不等于必须换标题。不要推定行政提交已成功或遗漏。

## R8 — 文献定位：先修最近邻，不按数量堆引用【必须】

- Jacobsen等 `Excessive Invariance Causes Adversarial Vulnerability` 已直接研究task-relevant变化后模型不变。现稿将其描述为只研究fixed-decision robustness是不准确的，必须改。
- `The Hard Positive Truth about Vision-Language Compositionality`是P3的直接近邻：困难负例训练会伤害困难正例。需要比较其成组样本/完整正确性评价与我们的同quartet错误转移，不可只用“他们两个集合、我们同例”未经核实地制造区别。
- `Selective Classification Can Magnify Disparities Across Groups`比只引校准论文更接近P1：置信拒识可改善总体而损害群体。我们的潜在区别是已知标签变换下的受控配对/条件风险，而非首次发现置信筛选有坏处。
- 关系标量表示与不变量已有深厚工作；`Scalars are universal...`可为关系输入的已知基础。真正要争取的是本任务中联合修复的证据，不是发明距离特征。
- “task-oriented prediction无canonical对应”不能作为研究结论；Donti等2017 `Task-based End-to-end Model Learning in Stochastic Optimization`已有正式NeurIPS版本。可简短引用来界定P2，不需要扩成优化论文。
- Press与Composition Collapse覆盖条件组合失败的先例，保留不争首创的立场。核实元数据与核实我们的区分是否成立，是两项不同工作。

## R9 — 复算、资源历史与结构整理【必须】

1. 修 `REPRODUCE_FINAL.md` 中 `--seed {11,23,47}` 等用法。普通Bash会展开成多个值交给单参数，改为明确for循环；对output和seed一一对应。
2. 每条正文主结果提供实际支持的命令、所需输入/权重/版本和输出路径。关系确认、migration、图形与表格导出的入口都应有，不能只靠本地手工代码。
3. 给confirm1007建立claim级使用表：P1首分析、后来关系特征评价、脚本纠错重算分别记录。训练parent独立不自动等于所有后续假设都“从未见过”；读取归档结果也不自动等于污染，二者别走极端。
4. 若已封存资源暂不可读，利用既有逐样本输出完成重算；新数据访问权限不足时列为局部缺口，不阻断稿件其他修改。
5. 结构迁移的旧SHA保留。新增当前release manifest与旧→新路径映射，不把新SHA覆盖历史manifest后说旧实验已被预先锁定。历史设计完整性测试与当前可执行依赖校验分开。
6. `PROJECT_MAP.md` 对旧 `FOOTBALL_CONF.json` 标成VOID/历史错误产物；对修正自然转折结果标negative，不再叫主线复现。
7. 单元测试调用生产函数：真实冻结schema、错误键必失败、同parent分组不重叠、cols控制正确、同数据模型共享编辑、R_endpoint=R_full+M、排序诊断的单调不变性。不能只对手写toy数组通过就宣称科学分析通过。
8. 已报告166项通过不应替代本轮重跑结果；输出实际执行测试、未运行测试及原因。编译成功与训练复现实验是不同验收项。

## R10 — 推荐分工与最终验收

可并行：统计worker修R1/R2；证据worker修R3/R4/R5；写作worker先按R6/R8重组；发布worker处理R7/R9。写作可先使用保守措辞，不能等待探索全部结束才开工。

每项输出一张短表：问题 → 变更 → 实际结果 → 受影响claim → 保留/降级/删除。所有更新落入一个当前ledger，旧版本顶部标history，不再创建多个“唯一真相”。

最终交付建议：
- `reports/caea817_review/RECTIFICATION_LOG.md`
- `reports/caea817_review/CLAIM_SOURCE_MAP.csv`
- `reports/caea817_review/CONFIRMATION_USAGE.md`
- `reports/caea817_review/REPRODUCE.md`
- 更新后的主文、附录、图表及可编译匿名PDF/补充包（确实生成后才能标完成）。

结论不要再只有“closure完成”。明确回答：P1哪个估计量成立；P1G纠错影响多大；raw flip的迁移事实；relflip正向证据能支持多强方法说法；P2仍有哪些解释无法排除；稿件主线是否包含真正最强结果；还有哪些投稿阻塞。

## 来源索引（均针对本审计commit，非未知最新分支）

[R1] `experiments/next_novelty/p1g_repair.py`; `artifacts/next_novelty/p1_freeze.json`; `artifacts/next_novelty/p1g_confirm/P1G.json`。  
[R2] `experiments/next_novelty/p1b_analysis.py`; `p1_unified.py`; `experiments/discovery_campaign/r02_search.py`; `artifacts/next_novelty/p1b_confirm/P1B.json`。  
[R3] `artifacts/next_novelty/relfeat/RELFEAT.json`; `relflip_eval/RELFLIP.json`; `relflip/RELFLIP_CONFIRM.json`; `p3/P3_FINAL.json`。  
[R4] `experiments/ccm_audit/relfeat_eval.py`; `relflip_train.py`; `reports/final_closure/RELFEAT_DECISION.md`; `P3_CLOSURE_REPORT.md`。  
[R5] `experiments/repair_decomposition/p2_operating.py`; `p2_affine_atoms.py`; `reports/final_closure/MASTER_CLAIM_LEDGER.md`。  
[R6] `paper/main.tex`; `paper/sections/00_abstract.tex`至`08_reproducibility.tex`（含04b）；`scripts/figures/fig_p1_p3.py`。  
[R7] `PROJECT_MAP.md`; `paper/SUBMISSION_CHECKLIST.md`; `reports/final_closure/REPRODUCE_FINAL.md`; `REBUTTAL_RESERVE_T5R6.md`。  
[L1] Jacobsen et al., Excessive Invariance Causes Adversarial Vulnerability, ICLR 2019, arXiv:1811.00401。  
[L2] Kamath et al., The Hard Positive Truth about Vision-Language Compositionality, ECCV 2024, arXiv:2409.17958。  
[L3] Jones et al., Selective Classification Can Magnify Disparities Across Groups, ICLR 2021, arXiv:2010.14134。  
[L4] Villar et al., Scalars are universal: Equivariant machine learning, structured like classical physics, NeurIPS 2021, arXiv:2106.06610；Donti et al., Task-based End-to-end Model Learning in Stochastic Optimization, NeurIPS 2017, arXiv:1703.04529。  
[L5] ICLR 2027 Author Guidelines（官方，2026-09-22核验）。  
[L6] ICLR 2027 AI Policy for Authors（官方，2026-09-22核验）。
