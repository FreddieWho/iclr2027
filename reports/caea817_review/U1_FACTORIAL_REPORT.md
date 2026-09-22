# U1 — 关系表示 × flip 监督因子重算报告（dev512，exploratory）

重算脚本：`experiments/ccm_audit/u1_factorial_eval.py`；
逐 quartet 派生表：`artifacts/next_novelty/u1_factorial/U1_PERQUARTET.csv`；
汇总：`artifacts/next_novelty/u1_factorial/U1_SUMMARY.json`。
四臂全部为冻结 checkpoint，同 dev bank（sha `21786139…`）、同 quartet 顺序、同 parent-cluster 自助法序列（rng 20260922，2000 次）。

## U1.1 四臂可比性核验

- 训练合同：r04b 三种子 manifest（epochs 300、lr 1e-2、lam 1.0、mine_seed 5、n_train 512）；
  relflip manifest（epochs 300、lam 1.0、n_flips 1534，seed 各异）。
  relflip 与 r04b 用同一 `mine_flips(..., seed=5)`、同一 `train_101/scenes.npz`（512 scenes）。
- 同一 flip 监督：train_101 上重算 `mine_flips(seed=5)` 得 1534 条，与 r04b s11/s23/s47 三份
  `mined.npz` 逐条 scene/edit/new 全一致；flip 列表 sha256 `abf78112…`；relflip 用同一函数与种子，
  故四臂共享同一 flip 监督预算。
- 泄漏检查（坐标哈希）：训练 512 场景、1534 mined 终点与 dev 298 起点/237 AB 终点三集合
  两两交集全为 0——测试 parent 与目标组合监督均无泄漏。
- 参数量（同 trunk h64f32）：raw 两臂 6849，关系两臂 6977（首层 8→10 维差 128 参数，
  无需架构匹配工程；解释力不足时已声明）。
- 归一化：raw 臂用训练集坐标均值方差，关系臂用关系特征均值方差（各臂 checkpoint 自带统计，
  评价时原样复用，无跨臂混用）。

## U1.2 主要分析（同 bank、同 parent 序列）

J（完整联合正确率）三种子：

| seed | raw | raw+flip | relfeat | relflip | I (pp) | I parent-cluster 95% CI |
|---|---|---|---|---|---|---|
| 11 | 0.0549 | 0.0675 | 0.1308 | 0.3840 | +24.1 | [0.134, 0.346] |
| 23 | 0.0506 | 0.0591 | 0.1688 | 0.4219 | +24.5 | [0.121, 0.377] |
| 47 | 0.0549 | 0.0759 | 0.2110 | 0.3966 | +16.5 | [0.066, 0.264] |

配对 J 差（CI 全为同 parent 序列）：
- raw+flip − raw：+0.008–0.021，三 CI 全跨零（flip 监督单独几乎不提高 J）。
- relfeat − raw：+0.076/+0.118/+0.156，三 CI 全不含零。
- relflip − relfeat：+0.253/+0.253/+0.186，三 CI 全不含零。
- relflip − raw+flip：+0.317/+0.363/+0.321，三 CI 全不含零。

互补形状（原子 vs AB 分解）：
- relfeat-only：原子通过率 0.68–0.71（raw_clean 0.52–0.56），但 AB 端点 0.45–0.49
  （raw_clean 0.48–0.52，基本不动）——J 提高主要伴随原子能力提高。
- raw+flip：AB 端点 0.57–0.68，但原子通过率掉到 0.37–0.46（原子回归率 0.41–0.43）。
- relflip：原子通过率 0.59–0.67 且 AB 端点 0.75–0.79——两端同时保住。
- 以 raw_clean H 集（base110，n=110/120/110）为基准：relflip R_full 0.36–0.44、
  R_endpoint 0.75–0.78、M 0.32–0.42，恒等式 R_endpoint = R_full + M 三种子全成立（残差 <1e-9）。

## U1.3 证据等级

- 本交互对比为 dev 上事后提出的新对比，标记 **exploratory**；J 尺度上的经验正交互，
  不称独立机制或普适协同定律（J 为联合事件，加性交互受非线性和上限效应影响）。
- 既有确认池：relflip J 另一 parent 池仍高（CONFIRM），但确认摘要缺 raw+flip 整行，
  故**不宣称确认了完整 2×2 交互**。
- 更强阳性句（“关系表示决定了这些编辑样本能否同时保住原子正确性和修好联合结果”）
  的支持条件满足：同输入、同监督预算、同 parent 序列的配对结果（relflip−relfeat、
  I 的 CI 全不含零）；但仍限所测表示/任务，不推广到 foundation models，且保留
  exploratory 标记。

## U1.4 论文落点

- 最小阳性成立：主图呈现四臂，写“关系输入与翻转样本结合显著提高完整联合正确性”。
- 更强阳性句可用（带 exploratory 标记与范围限定）。
- 主文四臂图数据源：本报告 U1_SUMMARY.json（替代审计期手算的 audit 数字；
  fig4 生成脚本 `scripts/figures/fig_fourarm.py` 下轮应切到此源）。
