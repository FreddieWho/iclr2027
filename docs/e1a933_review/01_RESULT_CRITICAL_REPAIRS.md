# 第一施工单：可能影响结果的错误修复

基准：`e1a933e6b32dd07c6895c7925cbccd9004da7f54`。本单授权修代码、复算受影响结果、有限公平重训、同步论文。不是为旧结论寻找更好数字。遵守 `04_AGENT_MASTER.md` 的预算与数据纪律。

## R01｜先修主claim：跨任务优势不等于跨任务正交互

**确定问题。** `U10_SOURCE_PREDICTION.md` 的P-A测 `six_flip − raw_flip`，不是I。归档T1 I=+.3936/+.4393/+.4127，T2 I=−.1891/−.1719/−.2503。

**操作。** 读取两任务四臂逐quartet预测；按任务、种子计算J、四个均值、I和parent配对bootstrap。每次抽同一组parent，同时重算四个臂，不按quartet iid。将P-A记为“相同flip协议下输入特征优势命中”，P-B为子集量比较，P-C待R02修复。单独保留T1正I和T2负I；检验是否受天花板、任务难度、flip方向、类别混合影响。误差相对下降/赔率尺度只作标明的次级分析，不能替代原J尺度主结果。

**交付。** `U10_CLAIM_CORRECTION.md`、四臂原始计数与I区间、`claim_diff.csv`；摘要不得写“正交互在两个新任务全复现”。P-A/B已有大差距不因typed问题自动撤回。

**验收。** 恒等式I两种算法一致；T2归档值重算为负；保留失败/饱和任务，不只汇总命中个数。

## R02｜恢复U10端到端对称性；分清原模型与群平均模型

**确定问题。** `u10_targets.py::fit_stats` 的typed分支返回每个输入索引自己的均值/标准差，却叫`scalar8`。`u10_models.py`只保证裸网络不变，不能修复进入网络前的不对称。`u10_orbit_eval.py`未断言完整流水线logit相等；J差小于.01也不等于严格不变。

**操作。** T1三个可交换顶点使用共享二维统计，query可以有独立二维统计；T2 A/B共享统计，center独立。或所有点共用一个标量统计，说明物理单位。归一化仅fit训练集。新增 `F(Px)=F(x)` 全流水线测试，输入必须是原始坐标、包含真实featurize和保存/恢复checkpoint。测试各合法群元素及非对称统计的故意反例。修正后重新训练typed T1/T2 clean+flip（f25若用于主表再补）；不得只换评估统计处理旧权重。main source任务的scalar归一化是另一实现，先验证再决定是否重训，不连坐。

D02同时修指标：原模型的`all8_raw`与群平均模型`all8_groupavg`分别输出。后者应与groupavg identity J在浮点容差内一致。8×前向开销计入，原模型all8低不能用作平均模型仍不稳的证据。

**交付。** `SYMMETRY_PIPELINE_AUDIT.json`（最大logit差、预测差）、旧/新typed逐种子表与训练曲线、原始/平均模型轨道指标。

**验收。** 正确流水线容差通过，错误索引标准化被测试抓住；重新检验“该typed结构不如距离输入”，不预设修正后仍阴性。

## R03｜建立每个模型自己的训练/评估血缘矩阵

**确定问题。** `d03_build_bank.py`从`D01/scenes_16N`的512之后取样，所以D03只对N模型是新parent，对4N可能部分重叠，对16N必然是训练parent。`D09.md`已有n236中18quartet重叠并剔除的历史。dev512与train101的父场景关系也已有审计记录。不要用“所有臂统一”代替独立性。

**操作。** 给每个checkpoint建立 `train_source_hash + parent_id + float64/原始序列化坐标hash` 血缘；不要单用没有namespace的整数parent。按模型输出旧parent新编辑、从未见parent、exact state重复、near duplicate四种口径。对N结果保留正确的new-parent子集；任何4N/16N新泛化claim在完全未进入N/4N/16N及其augmentation的独立新pool复核。新生成数据用新seed且进行跨bank哈希检查，统计字段不预设“fresh”。标签新颖不等于parent未见。

D09 dual仅报告符合该模型训练史的两套样本，不声称多个seed=多个独立数据集。复算旧银行不再称一次性首读；模型冻结后一次确认即可，不增加无关门控。

**交付。** `MODEL_BANK_OVERLAP.csv`、各bank纳入流图、版本化sample IDs、清洗前后敏感性表。新数据只为受影响的泛化结论服务，不重跑全部旧工作。

## R04｜D03置信度使用了终点，改回预测时可得的起点

**确定问题。** `d03_eval.py`用 `abs(lgS[sel]) >= thr` 定义high confidence，`lgS`是已编辑终点的logit。已有`lg0`和parent映射应被使用。

**操作。** 改为 `abs(lg0[p0map[parent]])`，在同一个起点parent上定义保留集，再测preserve/flip。明确同时报无条件endpoint风险和统一start-correct条件下的update风险；分母不可混用。阈值在已声明开发/训练参考集固定；报告实际覆盖率而不把top25标签当实际25%。side字段 `risk_hiconf_*`全部复算，原P1B不受该局部bug直接影响。

候选顺序结论限于此候选集、此评估筛选方式。端点评估三顺序差很小，不证明训练时cap-first对模型无影响；检验候选集合重合率和训练集顺序偏差后决定是否值得小对照。

**验收。** 反例：两样本start logit=[10,1]、end=[.1,10]，阈值5必须选第一个，不得选第二个；序列重排后同parent结论不变。

## R05｜U06辅助目标与可见输入不匹配：先审计，再改目标

**确定问题。** `n08_visual.render`不标识同色线段内部端点身份；`u06_relation_distill.rel10`却要求按指定端点索引回归距离/半径。同一图片可对应不同目标。附带fixture实际生成完全相同的两张图、rel10距离约.159；这是合同反例，不是原始训练bank的损失估计。普通MSE可学条件均值，但不能完美恢复不可见命名；这不自动证明U06阴性的唯一原因。

**先审计。** 对训练场景及其合法不可见重标号，用同noise seed渲染，测像素相同/最大差，计算标签恒定率与target变化。图片区分红/蓝两线，因此不可见群优先H=交换红线两端×交换蓝线两端（4个），不要直接用会交换可见颜色的8群。目标均值/方差报告于原始物理单位与训练归一化单位。

**修复。** 先做同一目标信息、允许一致端点置换匹配的loss，或可见商空间上的target；具体可按N01实施。整个几何结构使用同一个合法置换，不能逐条距离独立排序。交换前/后标准化需共同统计或先反标准化到物理单位再匹配，防止indexed mean/std再次破坏等价性。保留未匹配MSE、matched MSE、打乱且匹配、无aux的对照；通用坐标aux同样需要不可见端点匹配才能公平。

**另外必须修。** Cshuf在seed设定之前调用randperm，改成专用固定Generator；batch shuffle、模型init、aux init、render seed各自独立。extra aux层不能改变数据批次次序。训练日志必须给aux dev误差、orbit匹配后误差、BCE与aux梯度量级；不能仅凭λ=.5宣称loss-scale matching。

**交付。** `AUX_OBSERVABILITY_AUDIT.md`和新目标协议；只否定旧单aux-head固定配方，不把未配对失败当关系迁移普遍不可能。

## R06｜D04/视觉线：冻结BN、区分尺度与信息、保留初始化效应

**确定问题。** head-only分支冻结参数后仍`net.train()`，BatchNorm统计会更新；64→224是同一64图的双线性上采样，不是native高分辨率。224的pretrained-random仍有+72/+7.5/+9.8pp，不支持“非预训练”。

**操作。** 分开实现纯frozen probe（backbone.eval，缓冲区和参数哈希均不变）与BN-adaptation probe（明确允许统计更新）；不回填旧probe为完全冻结。固定sample列表、图像字节、batch序列，配对不同初始化/分辨率。报告P,A/B,AB/J/E-CCM：现字段atomic_acc实际上是A/B同时正确率，需更名/增补各自准确率。

用全部三个seed计算init×input-scale效应与不确定性。塌缩是结果，先保留；要归因需要匹配起点的随机初始化/训练轨迹实验。延长或优化random训练可作为公平强基线，用static/single-dev选，不用E-J选。增加ResNet static-only同架构对照后，才有“flip修复标准视觉模型”的效应估计。

**交付。** `VISION_FACTORIAL_CORRECTED.md`、BN buffer test、每个run train样本hash与逐parent预测；任何“同分辨率”与“同信息量”分开标。

## R07｜D08公平对打与塌缩归因

**确定问题。** pct从已训练clean继续300ep，普通flip从随机初始化训练300ep；`balanced`是flip池内endpoint类别平衡，不是preserve/flip平衡；旧正确回归在训练集而非未见原子集上评价。温度保持argmax是恒等事实，不能排除一般工作点解释。

**操作。** 从每seed同一个clean checkpoint复制：plain continuation、训练旧正确保护、hard-label replay（额外前向数匹配）、真实preserve+flip平衡。固定base系数、总新增样本、增广均值权重和训练步数。scratch arm另列作为训练路径对照，不能与续训arm混为loss因果比较。比较训练轨迹救援要有同checkpoint无保护续训；记录初始参数hash和每个batch顺序。新测试parent上的old-correct regression、A/B/AB联合状态迁移是主结果；训练集回归只是拟合诊断。

U07“联跑塌缩=RNG流伪影”先追溯真实run源代码、初始state、batch流。当前`d01_capacity.py`构建每arm都reset seed，aug=none的full-batch路线不能仅凭结果变化就断言RNG原因。修复后的统一流水线重新评估受影响格，不只替换被救好的格。

**交付。** `CONTINUATION_FAIR_COMPARISON.md`。若保护没有超出plain续训，撤回loss归因，但保留优化可恢复这一事实。U05是否值得做由未见原子回归决定，不由训练回归<5pp决定。

## R08｜D06重新测真正的E可构性

**确定问题。** `d06_roles.py`的paired仅为“至少一个keep且至少一个flip”。它没有执行y(x+eA)=y(x+eB)=y(x)、y(x+eA+eB)≠y(x)；两个命题互不蕴含。故3–7%不是E可构率。`opp[:6]`不是最近阻挡者；doc的same-section未落实；nearest timestamp可能跨半场或双侧时间并列。

**操作。** 首先保留旧统计并更名`supported_single_keep_and_flip`。按match+period+timestamp+person ID取唯一一帧，审计球员/裁判/球类别、缺测、recipient/team一致性。位移cap来自同球员连续时间差的速度/加速度分布，按明确时间窗变换，不以未验证slot相邻帧p99代替。

构建两个独立作用变量（如接球者位置与一个真正约束通道的防守者）和完整A/B/AB。fixed endpoints且“所有防守者均不阻塞”的AND oracle下，只分别移动不同防守者可能使目标组合在逻辑上根本不可构；先给oracle布尔分解，再选有交互的变量。对可行候选用model-blind解析边界/二分加随机对照，不靠无限增大位移。

观察真实选择的传球与所有潜在接球候选的几何代理必须分开。后者有可计算几何真值，不需伪造未发生传球的成功标签。报告自然发生率、候选抽样富集率、E条件率三个分母。它低仍可作为稀有压力测试，不用“必须够大”门槛自动叫不可行。

**交付。** `FOOTBALL_E_FEASIBILITY_V2.md`、逐事件坐标/角色/时间审计与三状态标签测试；D07/U09重新独立判断，不被该探针连带关闭。

## R09｜D10完整策略前沿与正确保序校准

**确定问题。** `reachable_ladder`只枚举scene maxP0。最低成本动作在覆盖不变时也会切换，故这不是完整动作策略前沿。`fit_isotonic`以合并块的x均值为插值点，在训练x上都不等于PAVA最小二乘解；重复x未正确聚合。

**操作。** 用原logit避免sigmoid饱和；枚举所有会改变候选/选择动作的分数断点、两侧和全拒动/全执行边界。可先完整枚举dev所有candidate scores再按结果去重；不要只枚举max。用正式IsotonicRegression或验证过的PAVA，测试重复x、持平段、域外剪裁。保序变换后的平坦分数采用明确无标签tie rule。

在dev选定目标预算策略，再在eval报真实coverage、总成本、成功率/每场景净收益；若eval覆盖不同，只称“dev匹配策略的外推”，不称同覆盖已实现。另可做明确oracle标签参与的纯诊断前沿；不得和部署估计混合。置信区间按同scene抽样，重算两策略各自完整ratio；数十个相关阈值不是数十次独立重复。

**验收反例。** 同一scene候选：P=.4/cost1/可行，P=.9/cost2/不可行；max-only梯子漏掉覆盖仍1但成功且便宜的策略。保序数据x=[0,1,2],y=[1,0,1]在训练点输出应[.5,.5,1]，不能[.5,.667,1]。样例脚本已附。

## R10｜统一收敛/机制口径与论文迁移

**问题。** 有固定配方失败就宣称容量/结构/优化被排除；有单次救援就归因RNG/损失；S的相似值被称encoder固有属性。均超出实验可识别范围。

**操作。** 每条“because/排除/因此”补一个竞争解释栏。O01/O04完成前只写“所测配方下”；S记作S(f_head∘h)，不是h单独属性。完成计数与效果分开；负面不等价于信息不够或结构无用。

论文主文优先：可信跨任务输入优势、真实完整修复、标准视觉模型的同任务结果。摘要不再重复所有旧数字；U10的T2负I作为正交互边界明确保留。D04有static-only前不写repair因果；U06修目标前不称原则迁移阴性；D06探针不进入足球不可行结论。修改自动表生成器而非手改结果数。所有图表标明n_parent、n_quartet、训练seed与模型种类的差别。

**统一验收。** 必须将本单至少R02/R04/R05/R06/R08/R09的测试移植为调用真实生产函数的回归测试，故意恢复旧错误应使测试失败。输出`AFFECTED_CLAIMS.csv`逐项标PASS/REVISED/UNRESOLVED，不以“全测试通过”替代科学审核。
