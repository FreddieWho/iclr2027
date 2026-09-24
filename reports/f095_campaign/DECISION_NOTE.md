# DECISION_NOTE 模板（总控§1：局部设计变更记录）

每条路线凡改动包内原问题/设计/统计/预算/顺序，写一段，存于
`reports/f095_campaign/<ROUTE>/DECISION_NOTE.md`。无需预注册审批。

## 格式

- 原问题：
- 证据（文件/数字/命令）：
- 修改点：
- 是否看过目标结果（何时、哪个）：
- 会怎样改变 claim：

- 交付：`reports/f095_campaign/U03.md`（三段表＋预测式 next）；`reports/f095_campaign/D02.md`（群平均对照表）；`test_symmetry.py` 7 项通过。

### D02-群平均结果分支（已执行，2026-09-23）

- 原问题：强制合法重标号不变后，失效与 repair 优势还剩多少？
- 证据：`artifacts/f095_campaign/D02/D02_SUMMARY.json`——rel+flip 群平均 J 三种子 +5–12pp（.384→.506/.422→.515/.397→.452），raw 仍 ~0.04–0.06；身份元 12/12 复现 U1（实现验证通过）；raw 臂 all8=0。
- 修改点：无（按合同执行前两项；增广训练版后补）。
- 是否看过目标结果：执行前看过 C1c 的 24/24 同向与 U1 四臂 J；群平均增益是执行后新见。
- 会怎样改变 claim：走"去命名依赖不充分"分支——命名伪影为真调制但非主因，支持继续 U01/U02；群平均增益另记（8× 推理成本，需同预算 ensemble 基线并列）。

### U01-typed 负结果（已执行，2026-09-23）

- 原问题：typed-pair 精确不变结构能否把手工距离成功变为学习机制？
- 证据：`U01_TYPED_W64_EVAL.json`——J .008–.025 全面低于同条件 raw（.051–.076），S 仅 .23–.37（raw .54–.57），J*≤.05；三种子同向；clean 臂已收敛。
- 修改点：无（按合同执行；未加层/调参救解释）。
- 是否看过目标结果：执行前看过 U1 四臂 J 与 D02 群平均增益；typed 负结果为执行后新见。
- 会怎样改变 claim：走"命名不变性不足"分支；U02 sixdist/concat 对照升为关键裁决（已开火）。

### D06-不可行判决（已执行，2026-09-23）

- 原问题：真实传球几何能否构造配对 E-bank？
- 证据：721 真实传球（全对上人）配对可构 3.05%；失败≈成功；pre-pass 3.3%（更开放）；窄通道 7.2% 仍稀疏。
- 会怎样改变 claim：D06 原设计 INFEASIBLE；足球线收窄（E2 锚＋contested 测量＋D07 自然审计）；不另换数据源（outcome 硬约束）。详见 `reports/f095_campaign/D06.md`。

### D01-4N 数据格（已执行，2026-09-23）

- 证据：`D01_W256_4N_EVAL.json`＋`D01_REL_4N_EVAL.json`——raw-clean J .11/.14/.08（N 时 .03–.05），S .69–.72；rel-clean .28/.33/.30，rel-flip .48/.51/.50（N 时 .38–.44），三种子全收敛。
- 会怎样改变 claim：数据 4× 双臂都抬，但鸿沟持续（4N raw-flip .15 ≪ 4N rel-flip .50；4N raw-clean .11 < N rel-clean .17）。宽度 confound 保守（raw 侧 12× 参数仍输）。U02 第四个问题转向 CONTRADICTED。

### D01-w512（已执行，2026-09-23）

- 证据：`D01_W512_EVAL.json`——300161 params（44×）下 clean J .059/.038/.030、flip s11/s23 .055/.051；s47-flip 第三次塌缩（恒 1.3067，与 w256 同值——同 seed 同挖掘的 init-basin 效应，与 L-006 噪声地板主题一致，该格未决）。
- 会怎样改变 claim：容量前沿 6.8k→84.6k→300k 全平（J .03–.11），"模型太小"解释关闭；4N/16N 数据格已开火（两 worker），待收。

### D01-w256 首格（已执行，2026-09-23）

- 原问题：12× 参数（84609）能否消除失效？
- 证据：`D01_W256_EVAL.json`——clean J .025/.046/.038（w64 为 .055/.051/.055），atomic 相近；flip s11/s23 J .080/.106（w64 .068/.059）；训练 loss→0.0000（记忆训练集）仍无改善。s47-flip 塌缩（loss 恒 1.3067，J=S=0）→该格未决，不计入结论。
- 会怎样改变 claim：容量不足解释被削弱，机制实验起点可信；待 w512＋数据格补全前沿。

### U02-竞争解释裁决（已执行，2026-09-23）

- 原问题：收益来自对称性、几何特征，还是训练实现？
- 证据：`U02_W64_EVAL.json`——sixdist J .12–.44≈旧 relflip；concat≈raw；typed 最差；s47-sixdist-flip 欠收敛 caveat。
- 会怎样改变 claim：机制中心从"任务对称性"转向"几何非线性组织×flip 监督互补"；U07 响应面列为后继，需授权。

### U06-图像学生迁移 NEGATIVE（已执行）

- 证据：`U06/s{803,805,806}[shuf]/result.json`——C≈A（−.068/−.010/+.006），Cshuf≈C；R_full .14–.21；B一致更差。
- 会怎样改变 claim：主claim限坐标域；视觉线止于D04适配结论；不追大backbone。详见 `U06.md`。

### W2-附录批量更新（已执行）

- 内容：`gen_app_tables.py` 加 D02-aug 行、tab:u07、D09/LAM/U06 段；重编译零错误零未定义零overfull（U07表改footnotesize+3pt）；主文仍9页；tests 212全过。

### D02-aug 增广训练（已执行，2026-09-23）

- 原问题：训练时随机重标号能否学到不变性/抬 J？
- 证据：`D02_AUG_EVAL.json`——g8-flip ident J .24–.36（基线 .06–.08），6/6 收敛（含 s47 治愈）；但 disagreement .46–.76、all8≈0；repeat 对照无增益。
- 会怎样改变 claim：命名/任务形式承认解释重要部分（覆盖多样性机制），几何仍居首；详见 `D02_AUG.md`。lam 伸缩（0.25/0.5/2.0×3 种子）已开火。

### W1-稿件迁移（已执行，2026-09-23）

- 内容：`fig_u03_panel.py`（U03 stacked 图，来源断言 S≡H/J≡J）→ `paper/figures/figU03_decomp.pdf`；`gen_app_tables.py`→`paper/sections/app_f095.tex`（D01/D02/U01-U02/D06 表＋ verdict 句）；`04_mechanism.tex` 加机制句（六距离充分/typed 失败/concat 持平）；main.tex 附录区 \input。
- 证据：TinyTeX 全链（pdflatex→bibtex→pdflatex×2）零错误零未定义零 overfull；主文 9/9 页（8→9 系新机制句，SUBMISSION_CHECKLIST 已更新）；tests 212 全过。
- 偏离：U03 panel 进附录而非主文图（页数约束；Fig4 为冻结脚本生成不动）。主文仅一句＋引用。

## 已登记（2026-09-23，开工准备）

### D01-嵌套银行（已执行）

- 原问题："原 N、4N、16N 嵌套父场景"未指定与 train_101 的关系。
- 证据：`d01_make_data.py`＋`data_manifest.json`——N sha 与 train_101 一致（N≡锚点）；4N＝N＋1536 fresh（seed401），16N＝4N＋6144 fresh（seed1601），严格前缀嵌套；r04b 挖掘复现 K=1534 一致。
- 会怎样改变 claim：宽度效应在锚点数据上可比；数据效应无历史锚点，只作新矩阵内部比较。

### U01-标量标准化（已执行）

- 原问题：r04b 的逐坐标 mu8/sd8 本身不对置换不变，typed 模型的"精确不变"需输入侧配合。
- 证据：`test_u01_models.py` 精确不变测试通过（容差内）；ckpt 以 `featurize=scalar8`＋8 维相同元组存标量统计，复用现有 loader。
- 会怎样改变 claim：typed 的不变性为真精确保证；与 raw/relfeat 的标准化口径差异如实记录（比较时并列，不混为"同预处理"）。

### D06-占位角色试探（已执行）

- 原问题：T5R3 views 无球，固定占位角色（p=home0, r=home1）500 帧 open_frac=1.0——任意角色不构成任务。
- 证据：`D06_PROBE.json`（slot-team 连续性 1.0；步长中位 0.035/p99 0.149，为位移上限输入；margin 中位 0.58）。
- 修改点：角色语义必须来自 ball/events（IDSSE canonical 有 ball 行），另起设计步骤；位移上限取自然短窗分布。
- 会怎样改变 claim：未改变（仅探针）；D06 主任务仍待角色定义后才算开工。

### S5-1 四臂表 AB 列（已执行）

- 原问题：`paper/sections/04_mechanism.tex` 四臂表 `AB endpoint err.` 混列。
- 证据：`artifacts/next_novelty/u1_factorial/U1_SUMMARY.json`（s11 raw AB_correct 0.519→err 0.481；relfeat 0.4515→err 0.5485；raw+flip 0.6456→err 0.3544）＋逐实例复算一致；生成器 `experiments/f095_campaign/gen_fourarm_table.py` 输出 `raw .481/.523/.481`、`relational .548/.532/.506`（raw+flip 原值即 error，不变）。
- 修改点：raw 行 `.519/.477/.519`→`.481/.523/.481`；relational 行 `.452/.468/.494`→`.548/.532/.506`。schema 统一为 `_err`，由逐实例生成、禁手填。
- 是否看过目标结果：看过 U1 归档 J/AB 汇总（f095dfa 已有）；本次只纠正列含义，未改任何模型结果。
- 会怎样改变 claim：只修正表意错误；J/atomic 主结论不受影响。

### S5-2 对称性措辞（已执行）

- 原问题："距离特征携带 raw 没有的置换对称性"过强（两者输入都随索引置换；通用 MLP 都不保证输出不变）。
- 证据：总控§5(2)；C1/C1c 实测的是稳定性差距（raw J 摆动 75–133% 均值 vs 关系+flip 15–29%）。
- 修改点：`04_mechanism.tex` 改为描述性稳定性差距＋"两者都置换、MLP 都不精确不变，原因交对称性对照"。`05_repair.tex` 与 discussion (viii) 原已收敛（称描述/联合修复证据），不动。
- 会怎样改变 claim：收窄机制解释，为 D02/U01/U02 留出对照空间。

### S5-3 八标号/S4（已核查，无需改 paper）

- 核查：正文仅称 "eight relabelings"（abstract/04_mechanism/discussion(xi)），未主张 S4、未提议无类型四点 DeepSets。约束已编入 `experiments/f095_campaign/README.md`（合法 8 元群、typed-pair，U01 禁无类型四点集合作主基线）。

### S5-4 residual 多数措辞＋pending 标记（已执行）

- 原问题："residual majority still wrong"与 0.32–0.62 不一致（s23 残留 0.32 为少数）；附录 "marked pending until the re-run lands" 已过期（P1G_CORRECTED.json 已落盘，见 RECTIFICATION_LOG R1）。
- 修改点：`03_phenomenon.tex` 改为 "substantial residual"；`appendix.tex` 改为 "re-run landed; see R1"。
- 确认集使用：CONFIRMATION_USAGE.md 已登记 5 类使用并声明无首读资格，与 discussion (x) 一致；P2 边界/表格分母在附录 estimand map 已标注描述性对比。未发现新的分母不一致。

### U03-state0（已澄清：无需补，2026-09-23）

- 原问题：初判"复用U2归档逐quartet分数"不够做三段分解，需补 state-0 前向。
- 证据：总控§3的 J 定义为 E 上 A/B/AB 同时正确；E 本身 yA=yB=y0、yAB≠y0，每 quartet 天然含正负标签。`u03_decompose.py` 用 [n,3]（A/B/AB logits＋yS/yS/yAB）跑出 S 与 U2 的 H 逐格一致（s11：0.5654/0.7300/0.8734/0.8734）、J 与 J_t0 一致——独立代码交叉验证通过。
- 修改点：state-0 补测取消；U03 口径明确为 3-state（A/B/AB）诊断证书。
- 会怎样改变 claim：不改变；"诊断上界非部署成绩"定位不变。

### U04层间互补（已执行，2026-09-23）
- 原问题：U03"表示降第一段、flip降第三段"是行为描述，需因果定位到encoder/readout。
- 证据：`artifacts/f095_campaign/U04/U04_SUMMARY.json`——P1：relflip-encoder静态readout S与原整网四位一致（3/3）；P2：flip-readout增益仅relfeat-encoder为正（+.160/+.068/+.059）；P3：MLP相对线性增益全格<+.05（最大+.021，12/12）；raw-encoder上flip-readout呈纯迁移（R_full .05–.06、M .39–.48）。静态重训复现原整网J（init噪声内）；主干sha前后一致。
- 修改点：无（按U04_DESIGN.md冻结合同执行；P2判PARTIAL而非强行SUPPORTED）。
- 是否看过目标结果：执行前看过U1/U03汇总（f095dfa已有）；重训数是执行后新见。
- 会怎样改变 claim：层间互补成立且收窄——S是encoder属性，J增益需好表示×flip监督同时成立。主文机制段可引用，附录进表。

### D08公平对打（已执行，2026-09-23）
- 原问题：相对最接近的训练思想，flipmine多解决什么。
- 证据：`artifacts/f095_campaign/D08/D08_SUMMARY.json`——17臂（7新训：raw fliprand×2、rel fliprand×3、balanced×6、pct×6；余复用记hash）＋temp-calib 24格。Q1：pct 3/3落在relflip−0.05内；Q2：回归差距最大1.2pp（门5pp）；Q3：temp 24格J不动。
- 修改点：balanced用337+337下采样（unique样本少一截，如实记录未补量）；lam_pres=1.0单点零搜索；pct为continuation-estimand。
- 是否看过目标结果：执行前看过U1/U03汇总；对照数是执行后新见。
- 会怎样改变 claim：flipmine特殊性不成立→方法贡献让位诊断；U05不开（门条件未满足，有证据）。

### U10跨任务预测（已执行，2026-09-23）
- 原问题：几何组织×flip原则能否离开线段题。
- 证据：54训（T1/T2 × raw/six/typedT × clean/flip/f25 × 3种子）＋双E-bank（2724/3443 E）。P-A 6/6（差+.36–+.56）、P-B 6/6、P-C 6/6，全中。分解同构源任务；T2-raw J .61证非难任务假象。
- 修改点：砍concat（源任务两次≈raw，记录理由）；typed多~40%参数（输了结论更强）；E-bank每parent cap 12（记录）；T2半径R0=0.25任务常数。
- 是否看过目标结果：看过U02/U07源汇总（冻结预测依据）；目标任务数是执行后新见（预测文件时间戳先于运行）。
- 会怎样改变 claim：主claim升级为跨任务可预测成立；U02/U07的"线段题"边界摘除。

### D03顺序审计（已执行，2026-09-23）
- 原问题：强反转/迁移/表示优势是否由候选顺序制造。
- 证据：`D03_EVAL.json`——三顺序flip误差差≤0.01（12格）、preserve≤0.005；E-bank（无序，1838 quartets）四臂格局与dev512同序；order-free复现。
- 修改点：共享oracle缓存使三顺序预算严格相等（原early-break不可比，记录为adaptation）；E配对均匀子采样≤3000（记录）；D09父集排除（512个，manifest记；附带发现D09银行含7个train_101内E-parent＋31个S-parent，已开D09重算审计，见下条）。
- 是否看过目标结果：看过U1/U03汇总；顺序差值是执行后新见。
- 会怎样改变 claim：顺序解释排除；**不加训**（step-4门条件未满足）。

### D09重叠审计（已执行，2026-09-23）
- 原问题：D03排除 audit 发现 d09fresh665 银行含7个train_101内E-parent（＋31个S-parent），"未见父场景"口径不纯。
- 证据：`D09_AUDIT.json`——剔除18个重叠quartets后（n=218），Δflip +.257/+.367/+.216、Δclean +.101/+.069/+.184，Q1/Q2仍3/3。
- 修改点：D09.md＋附录段加限定句；原236数保留但注明18个来自重叠父场景。
- 会怎样改变 claim：不改变 verdict（CONFIRMED），收窄口径到218严格未见。审计链条：D03的银行清单反哺了D09的质量控制。

### 七项改进+大预算升级批次（已批准执行，2026-09-23）
- 原问题：已执行路线是否存在可优化/可转正的空间；用户批准全部七项并解除时间约束（"只要结果更好"）。
- 设计（执行前记录）：
  1. U03分解在D03-E无序银行复现（零成本，产物已在）。
  2. D09升级为双新鲜银行确认（D03-E即第二银行，措辞合并）。
  3. D04全析因：224-random×3＋224-headonly×3＋96/128×(pretrained,random)×3。原224只有pretrained臂，confound（ImageNet分辨率匹配）只有散文承认；random-224是关键对照。预期两分支：random-224≈random-64→confound坐实；远大于→分辨率本身贡献，改写224解释。
  4. U06-224全臂：B/C/Cshuf×3（A-224=D04 pretrained-224已存在，J .720/.737/.709）。唯一可翻转判决的候选：C>A→scoped-positive（"迁移需足够视觉可读性"）；≈A→阴性跨分辨率稳健。需--res参数（已加，训练与评估两侧同改）。
  5. U07剂量面：盘点发现six_4N_f25已存在（six_4N_s*/flip_fx即f25臂，PART已评）；真正缺口仅raw_4N full-flips×3。raw4q_s47=0.0塌缩格保留为数据，不refill。
  6. D10前沿扩到rel12/sixdist编码器（评估-only，脚本参数化）。
  7. U10 orbit平均（T1 S3六元/T2二元）＋sixdist-dev512群平均（评估-only）。
  8. bonus：sixdist×pct损失×3（s47塌缩归因：损失设计 vs 种子；协议镜像D08 pct臂，续训自同种子six-clean）＋U04 --bank d03E复现。
- 是否看过目标结果：D04-224 random/headonly、U06-224、raw_4N full、six-pct、U04-d03E、D10-rel、U10-orbit 全部未跑，结果未知。
- 会怎样改变 claim：见各分支预期；全部阴性也有价值（confound坐实/阴性稳健化）。
