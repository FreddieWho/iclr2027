# O01 equal-budget optimization and retained-label factorial

本文件保留 108 scratch 候选的既有记录（第 1 节），并新增 e1a933 施工包要求的
warm-start 因子闭环、收敛稳健配置与 oracle 三账本（第 2–7 节）。

---

## 1. 既有记录（2026-09-24 第一轮，原文保留不改）

中文要点：第一轮完成 108 个 scratch 候选（N/4N × raw/sixdist × 保留比例 0/25/100% ×
seed 11/23/47 × Adam 学习率 .01/.003/.001，width64/feat32，300 步全批）；全部候选保留、不删 seed；
每个格子只用独立的 static/single 开发 BCE 选学习率，不用 E-J 选；只用训练集统计做标准化；
初始化逐候选重置；receipt 含初始化/子集/数据/checkpoint 哈希与完整 loss 曲线。
新评价池为 1229 quartet / 250 eligible parent（512 个抽取 parent 中其余未产生合法 E），
另有 256 个不重叠 dev parent；测试标签从不参与选择。
表中 6 个 size×seed 的选中比较都保留很大的“距离+flip”优势，且 distance25 超过同一池上的 raw100；
raw100 平均 J 从约 .145（N）升到 .413（4N），distance100 从 .635 升到 .853。
这是**挖完全池之后**的保留标签效率，不是省下 75% oracle 查询。

Completed 108 scratch candidates: N/4N × raw/six-distance × 0/25/100% retained flips × seeds11/23/47 × Adam learning rates .01/.003/.001, same width64/feature32 and 300 full-batch gradient steps. All candidates retained, no seed exclusions. Each cell chooses LR only by independent static+single dev BCE, not E-J. Train-only normalization, fixed subset RNG934, initialization reset per candidate; receipt contains initialization/subset/source/checkpoint hashes and full loss curves.

This is finite matched training-budget evidence, not globally optimized model equivalence. Param count differs slightly with input dimension. Source data and fixed candidate search differ from older recipes; compare the complete new factorial internally.

New evaluation: 1229 quartets / 250 eligible parents out of 512 drawn evaluation parents (others yielded no valid E under fixed sampling). 256 disjoint dev parents; test labels never used for selection.

|Size|seed|raw0|raw25|raw100|six0|six25|six100|
|---|---|---|---|---|---|---|---|
|N|11|0.081367|0.139138|0.144020|0.268511|0.682669|0.642799|
|N|23|0.073230|0.113100|0.171684|0.276648|0.616762|0.646867|
|N|47|0.071603|0.131001|0.117982|0.285598|0.589910|0.614321|
|4N|11|0.151343|0.268511|0.475183|0.542718|0.768104|0.888527|
|4N|23|0.126932|0.262002|0.333605|0.545972|0.814483|0.851098|
|4N|47|0.141579|0.268511|0.430431|0.484133|0.809601|0.818552|

All six size×seed selected comparisons retain a large distance+flip advantage, and distance25 exceeds raw100 on this same fixed pool. Original raw100 mean J rises from about .145 (N) to .413 (4N); distance100 from .635 to .853. This is retained-label efficiency after mining the full pool, NOT 75% fewer oracle queries. Oracle candidate calls, retained labels and training forward counts are separately stored.

Full four-arm parent-bootstrap contrasts, same-pool R_full/M, CCM denominators and 111 degradation: `artifacts/e1a933_review/data_o01/O01_PAIRED.json` and `O01_FACTORIAL.csv`. All 108 optimizer recipes evaluated AFTER LR selection for same-recipe sensitivity in `artifacts/e1a933_review/data_source_eval/FROZEN_AND_ALL_RECIPES.csv`; these additional test scores did not alter model selection.

---

## 2. 本轮补全：设计与执行（新增）

审计缺口：`scratch/warm-start 配方因子未在同一新 pool 完整闭环；R07 不能替代全部剂量/规模格；额外稳健优化配方未执行`
（`reports/e1a933_review/PACKAGE_COMPLETION_MATRIX.csv`，O01 = PARTIAL_REQUIRED）。

脚本：`experiments/e1a933_review/o01_continuation.py`（确定性、可复跑，`--stage preserve|train|summarize|check`）。
产物目录：`artifacts/e1a933_review/data_o01_continuation/`。旧产物（`data_o01/`、`fair/`）只读，未改动。
线程数固定 2（`torch.set_num_threads(2)`，与 O01 campaign 相同），因此本轮的 scratch 臂可与 O01 逐位对齐。
脚本 sha256（本轮最终版，训练代码自 train/summarize/check 三阶段起未再改动；
第 7 节的守卫分析是在 check 之后追加到 `stage_summarize` 的纯分析代码）：`7f783af323d2a32d780b749174c5f9feedfd60bb518918c700fb2435fd9844fc`。

**clean 起点定义**：每个 (size, arch, seed) 取 O01 中 frac=0（只用基础数据、无 flip 监督）且按
static/single 开发 BCE 选出的候选作为 clean checkpoint，例如 `N_raw_s11_f0_lr0.001`
（`artifacts/e1a933_review/data_o01/N_raw_s11_f0_lr0.001/model.pt`）。
它与 O01 scratch 臂共用同一训练银行、同一网络与同一特征标准化，因此 scratch 与 warm-start
的差别只是训练路径，不是数据或编码器。起点文件哈希、起点参数哈希逐臂记入 `receipt.json`
（`clean_checkpoint_sha256` / `clean_state_sha256`）。本轮没有使用 `fair/` 的 clean_ckpt
（那是 train_101 银行上的旧配方，R07 用它做的是另一个 bank 的公平续训，不能替代本 lane）。

**warm-start 配方因子**（全部同起点、同 300 步、每步前向数相同）：

| 方法 | 每步损失 | 说明 |
|---|---|---|
| `plain` | BCE(base) + BCE(flips_dose) | 普通续训 |
| `protection` | + 1.0 × BCE(clean 起点 logits 的 soft teacher on 起点正确的 base 点) | PCT 式保护 |
| `hard_replay` | + 1.0 × BCE(同一些点的 hard label) | 硬标签 replay |
| `preserve_flip_balanced` | BCE(base) + BCE(半量 flips + 半量真实 preserve) | 真实 preserve+flip 平衡 |

所有 warm 臂每步都做 base / augmentation / replay 三次前向（plain 与 balanced 的 replay 系数为 0），
因此同一剂量下四方法的额外前向数完全相等（`forwards_per_step` 记入 receipt）。
`preserve` 池：对每个已挖 flip 生成一个同标签、模长受该 flip 范数限制的真实扰动，
N 用 1,534 条（2,801 次 oracle 尝试）、4N 用 6,390 条（11,955 次尝试），见 `preserve_{N,4N}.npz`。

**剂量格与规模**：frac = 0/25/100（25% 用固定 RNG934 置换的前 1/4，N=383 条、4N=1,597 条），
输入 raw / sixdist，训练规模 N（512 场景）/ 4N（2048 场景）只作为同一规模因素的两个水平，
不是两个独立测试集；所有评价都在同一个 1229 quartet / 250 parent 未见 parent 池上。

**运行规模**：648 次真实训练 = 432 warm 因子臂（2 size × 2 arch × 3 seed × 3 dose × 4 method × 3 LR）
+ 216 收敛稳健臂（2 start × 2 arch × 2 size × 3 seed × 3 dose × 3 config）。全部保留，无删除。
选择只用 static/single 开发 BCE（`dev_bce`，分母 = 256 clean + 784 single 状态），
评价池 J 从不参与选 LR / 选步数 / 选 seed / 选方法。

## 3. 同起点同步数的合同断言（209/209 通过）

证据：`artifacts/e1a933_review/data_o01_continuation/CHECK_REPORT.json`（`all_passed: true`，209 项断言）。

- `single_vs_batch::*`（3 臂）：同一方法单跑与批量跑给出**同一个**初始 state 哈希、**同一个**
  batch 顺序哈希、同一个末状态哈希（含 4N 平衡臂与 hard_replay 臂）。
- `same_start::*`（12 格）：同一 (size,arch,seed) 的全部 warm 臂只有 1 个不同的
  `init_state_sha256` 与 1 个不同的 `clean_state_sha256`。
- `same_batch_list::*`（144 格）：同一 (size,arch,seed,dose,method) 的 3 个 LR 臂 batch 列表哈希完全相同
  （哈希范围 = base/flips/preserves/replay 四个索引列表，优化器与步数作为元数据单列）。
- `clean_rederived::*`（6 格）：从随机初始化重跑 clean 配方（同 seed、同 LR、300 步基础数据）
  与 O01 存档 clean checkpoint **逐位相同**。
- `o01_scratch_reproduced::*`：本 harness 在相同设置下**逐位复现** O01 scratch 臂
  （`N_raw_s11_f100_lr0.001`、`4N_sixdist_s23_f25_lr0.003` 的末状态哈希与 O01 存档一致）。
- `balanced_f0_eq_plain_f0::*`（18 格）：剂量 0 时平衡臂与 plain 臂逐位相同（增广为空的预期结果）。

**执行过程披露**：第一遍 648 臂在 `bank()` 里把坐标先 `astype(np.float32)` 再算标准化统计，
与 O01 的 float64 输入路径相差约 1e-7（相对量级），导致 `clean_rederived` 与
`o01_scratch_reproduced` 断言失败。代码修正为保留原始 dtype 后全部 648 臂重跑。
第一遍的逐臂目录已删除，其 `ARMS.csv` 与日志原计划留档到 `superseded_first_pass/`，
但在清理目录时被同一条 `rm -rf */` 一并删除，**没有保留下来**；因此本报告只以重跑后的结果为准。
可核对的痕迹：第一遍 check 输出中 `N_raw_s11_f100_lr0.001` 的 dev_bce 为 1.2733，
修正后为 1.2749（与 O01 存档一致），差异 1.2e-3，方向性结论不受影响。
本报告不把第一遍当作可用证据。

## 4. 主结果：同一起点、同一 300 步下的剂量—表示响应（新增）

评价池 = 1229 个 quartet（250 个 eligible parent，来自 512 个抽取的 evaluation parent）；
J 的分母恒为 1229（先按 parent 聚类再配对 bootstrap，2000 次重采样，seed 924）。
每个格子是 3 个 seed 的均值；逐 seed 值在 `O01_CONTINUATION_FACTORIAL.csv`（`selected_by_dev_bce=1` 行）。

| 规模 | 输入 | 起点 | f0 | f25 | f100 |
|---|---|---|---|---|---|
| N | raw | warm（dev 选中） | .079 | .116 | .133 |
| N | raw | scratch（O01 选中） | .075 | .128 | .145 |
| N | raw | warm + 稳健配置 | .075 | .119 | .138 |
| N | sixdist | warm | .281 | .579 | .592 |
| N | sixdist | scratch | .277 | .630 | .635 |
| N | sixdist | warm + 稳健配置 | .280 | .583 | .636 |
| 4N | raw | warm | .144 | .239 | .288 |
| 4N | raw | scratch | .140 | .266 | .413 |
| 4N | raw | warm + 稳健配置 | .140 | .218 | .249 |
| 4N | sixdist | warm | .538 | .776 | .808 |
| 4N | sixdist | scratch | .524 | .797 | .853 |
| 4N | sixdist | warm + 稳健配置 | .525 | .752 | .798 |

- 剂量响应：six 从 f0 到 f25 的涨幅占总涨幅的 88–96%（N: .281→.579→.592；4N: .538→.776→.808），
  即**在同一个已挖全池的条件下**，再用 25% 的保留标签几乎拿到全部收益；raw 的涨幅小一个量级。
- 起点响应（warm − scratch，同一 LR、同一剂量、同一评价池）：
  f0 差 ≈ 0（−0.001…+0.020，CI 全部含 0）；six 在 f25/f100 多为负
  （N f100 −0.081/−0.069/−0.024；4N f100 −0.056/−0.052/−0.033，部分 CI 不含 0）；
  raw 在 4N f100 明显为负（−0.14…−0.20，CI 不含 0）。
  **warm-start 没有带来收益**：从 clean 出发再补 flip 训练不如一开始就把 flip 放进训练；
  这与 R07“PCT 从 clean 续训”的旧因果解释被撤回是同一方向的证据。
- 逐样本迁移（主臂，warm plain，起点状态分布见 `clean_start__transition_8x8`）：
  1229 个 quartet 中起点为 110（两原子对、AB 错）的数量在 N/raw=647、N/six=606、4N/raw=663、
  4N/six=403；`000→111` 在所有臂都是 0 个（起点没有 000 的 quartet，分母事实）；
  4N/six f100 的 `110→111` 为 284–307、`111→111` 为 544–618，而 N/raw f100 的 `110→111` 只有 86–101。

## 5. 明确回答：六距离相对一个被充分优化的 raw 基线是否仍保留优势

**是，且优势很大。** 三种设置、两种规模、三个 seed 的方向完全一致，没有一个 CI 跨 0。

（a）**等配方比较**（固定学习率、同一起点、同一步数，6 格 × 3 LR × 2 剂量 = 36 个配对对比）：
six − raw 的 ΔJ 在 f25 为 +0.388…+0.539，在 f100 为 +0.434…+0.561，CI 全部不含 0
（`O01_CONTINUATION_PAIRED.json` 的 `equalLR*_six_minus_raw_f*`）。
f0（未做任何 flip 训练）也有 +0.177…+0.433，即优势并非来自 flip 阶段。

（b）**dev 选中配方比较**（收敛稳健三候选 SGD momentum .01/.003 与 Adam 5e-4 × 600 步，
只按开发 BCE 选；与第 4 节的 Adam 三档 dev 选中臂同属“最优配方”口径，不是等配方）：
warm 起点 f100 为 +0.491…+0.570，scratch 起点 f100 为 +0.475…+0.574，CI 全部不含 0
（`robust_*_six_minus_raw_f100`）。

（c）描述性范围（**不用于任何选择**）：在全部 648 个配置里，raw 在 f100 的 J 最高为
0.168（N，scratch）/ 0.345（4N，scratch），six 在 f100 的 J 最低为 0.562（N，warm）/ 0.737（4N，warm）。
两个分布不重叠，因此“raw 只是没调好”无法解释该差距。

（d）保留标签效率：six@25% − raw@100% 在所有设置下为正且 CI 不含 0：
warm 选中 +0.409…+0.506；等配方 +0.378…+0.542；scratch +0.293…+0.539；
稳健配置 warm +0.422…+0.512、scratch（守卫后）+0.374…+0.454。

（e）交互项 I（J 尺度）(six_d − six_0) − (raw_d − raw_0)：warm f25 +0.088…+0.324、
warm f100 +0.077…+0.288、scratch f25 +0.108…+0.356，CI 全部不含 0；
scratch f100 为 +0.022…+0.312，其中 4N 的两个 seed CI 含 0（−0.057…+0.104、−0.032…+0.123）。
即：**正交互在本轮 24 个 cell×dose×起点组合里 22 个得到 CI 支持，4N/scratch/f100 两个 seed 是例外**。

边界：raw 臂普遍存在训练集欠拟合（warm f100 有 13/72 臂训练误差 >5%），
因此“raw 已被优化到容量上限”不成立；能说的是：在本次覆盖的优化器族
（Adam .01/.003/.001 + SGD momentum .01/.003 + Adam 5e-4 × 600 步）与两个起点下，
raw 的最好观测结果仍远低于 six 的最差观测结果。

## 6. 配方因子：保护 / hard replay / 平衡（新增）

同一剂量、同一起点、同一步数，与 `plain` 的配对差（`warm_{arch}_{method}_minus_plain_f*`）：

- **J 上没有稳定收益**：|ΔJ| ≤ 0.053；raw 上 −0.027…+0.028（多数 CI 含 0），
  six 上 −0.053…+0.024。不能声称保护损失提升联合正确率。
- **未见原子回归下降**（起点正确的原子在训练后变错的比例，分母 = 起点正确的 quartet 数，
  见 `matched_noflip__atomic_regression` 与 `atomic_regressed_n`）：

| 格子 | plain | protection | hard_replay | preserve+flip 平衡 |
|---|---|---|---|---|
| N raw f100 | .241 | .197 | .198 | .150 |
| 4N raw f100 | .209 | .167 | .180 | .141 |
| N sixdist f100 | .153 | .135 | .132 | .081 |
| 4N sixdist f100 | .042 | .042 | .038 | .021 |

  ΔReg 的配对 bootstrap：raw 上 −0.04…−0.12（3/3 seed CI<0），N/six 上 −0.03…−0.12（2–3/3 CI<0），
  4N/six 上 −0.003…−0.043（0–2/3 CI<0，因为该格原子回归本来已只有 4%）。
  平衡臂的 J 与 plain 基本相当（最差一格 N/six/f25 为 −0.053，其余 |ΔJ| ≤ 0.029）但只用了一半的 flip 标签，属于“同标签量下更省遗忘”的候选解释，
  本轮不足以判定机制。
- 剂量 0 的 protection / hard_replay 臂是纯蒸馏正则（没有 flip 数据），其 dev_bce 保持 clean-only 水平
  （如 `warm_N_raw_s11_f0_protection_lr0.01` dev_bce=2.885，起点为 2.649），
  这些臂只作剂量对照，**不用于论证保护损失有效或无效**。

## 7. 收敛稳健配置与失败配置完整披露（新增）

候选（只按 static/single dev BCE 选，绝不用评价 J）：
`sgd_m09_lr0.01`、`sgd_m09_lr0.003`（SGD momentum 0.9，300 步）、
`adam_lr0.0005_steps600`（低 LR 长跑）。

- **失败配置（保留并计数）**：`sgd_m09_lr0.003` 从 scratch 出发在 raw 上系统性崩溃
  （N/4N × 3 seed 共 12 臂 J<0.05，最低 0.000，训练误差 36–38%）；全 648 臂中 J<0.05 共 30 臂，
  全部来自该配置。warm 起点下没有一臂 J<0.05。
- **dev 选择准则的已知弱点**：在 scratch/raw/dose-0 角，dev BCE 会选中一个近常数解
  （dev_bce≈0.69 恰是常数预测器的 BCE 水平）。72 个稳健格中有 5 格因此选中 J≈0.02 的退化臂。
  已按“不删结果、不改用测试 J 选择”的原则处理：`ROBUST_GUARDED_SELECTION.json` 给出
  守卫规则（排除训练误差 >20% 的候选，仍只用 dev BCE 选）与 12 格选择变化；
  守卫后 six − raw 的结论不变（scratch f100 +0.475…+0.509，six25 − raw100 +0.374…+0.454）。
- 观察到的有效候选：`adam_lr0.0005_steps600` 在多数 scratch 格最好，
  `sgd_m09_lr0.01` 在 warm/six 格最好；**不存在单一稳健配置在所有格胜出**，
  因此主表用 dev 选中臂，同时给出等配方比较作为不受选择影响的对照。

## 8. oracle 成本三账本（新增，`ORACLE_LEDGER.json`）

| 账本 | N | 4N | 来源 |
|---|---|---|---|
| 实际查询候选数（flip 挖掘） | 147,514 | 587,320 | `data_o01/{N,4N}_flips.npz` 的 `oracle_calls` |
| 保留 flip 标签数 | 1,534（246 parent） | 6,390（1,020 parent） | 同上 |
| 开发集单编辑挖掘 | 73,622 次查询 → 784 条保留标签（256 parent） | — | `data_o01/dev.npz` |
| 评价池挖掘 | 1,656,780 次候选/配对查询 → 1,229 quartet（250/512 parent） | — | `data_o01/bank_manifest.json` |
| preserve 挖掘（本轮新增） | 2,801 次尝试 → 1,534 条 | 11,955 → 6,390 | `data_o01_continuation/preserve_*.npz` |
| 合计 oracle 查询 | **2,479,992** | | 上表求和 |

每轮前向/梯度（本轮实测，取 warm/raw/seed11 的 plain 臂）：

| 剂量 | 基础场景 | 保留 flip 标签 | replay 点 | 每步前向 | 300 步总前向 | 总反向 |
|---|---|---|---|---|---|---|
| N f0 | 512 | 0 | 512 | 1,024 | 307,200 | 300 |
| N f25 | 512 | 383 | 512 | 1,407 | 422,100 | 300 |
| N f100 | 512 | 1,534 | 512 | 2,558 | 767,400 | 300 |
| 4N f0 | 2,048 | 0 | 2,048 | 4,096 | 1,228,800 | 300 |
| 4N f25 | 2,048 | 1,597 | 2,048 | 5,693 | 1,707,900 | 300 |
| 4N f100 | 2,048 | 6,390 | 2,048 | 10,486 | 3,145,800 | 300 |

**口径（必须照此表述）**：25% 子集是在**挖完全池之后**从保留标签里抽取的，
因此挖掘阶段的 oracle 查询成本对所有剂量完全相同（N 147,514 / 4N 587,320 次），
**只能主张“保留标签/监督侧的数据效率”，不能主张“省下 75% oracle 查询”**。
若要在部署上省查询，需要另外做“只挖 25% 场景”的实验，本轮没有做。
平衡臂用一半 flip + 一半真实 preserve，标签总量与 plain 相同（4N f100 为 3,195 + 3,195）。

## 9. 未完成与边界（新增）

1. **scratch × {protection, hard_replay, preserve+flip} 未运行**：scratch 没有 flip 之前的阶段，
   “起点正确”的 replay 掩码没有自然定义（R07 也只把 scratch 当作训练路径对照）。
   因此 start×method 交互不能从 scratch 侧估计，本轮只能给出 warm 侧的 4 方法比较
   与 plain 方法下的 scratch/warm 对比。
2. **warm-start 的 clean 起点是 300 步的 static-only 模型**，不是长期预训练的表示。
   本 lane 没有用它做“预训练表示 + flip 微调”的强主张；O01 里也没有更长的 clean 存档，
   若要检验“更长 clean 起点是否改变结论”，需要一个更长训练的 clean checkpoint（本 lane 不擅自替换）。
3. **raw 侧未被证明优化到上限**：raw 臂普遍欠拟合（warm f100 13/72 臂训练误差 >5%），
   优化器族也仅限于 Adam 三档 + SGD momentum 两档 + Adam 低 LR 长跑；
   参数个数也随输入维度略有差异（raw in_dim=8 vs sixdist in_dim=6），未做参数匹配（属 O02 的范围）。
4. **评价池不是 sealed 确认集**：1229 quartet / 250 parent 是本轮新 parent 池，
   但同批实验已多次使用它做内部比较，不能再称“未见首读”。
5. **preserve 池是模型无关的同标签扰动**（模长受对应 flip 范数限制），不是自然保持事件，
   不能解释成真实数据中的保持分布。
6. **交互项 I 在 4N/scratch/f100 的两个 seed 上 CI 含 0**，不得写成“所有格都正交互”。
7. 本轮没有做阈值/温度校准、没有做跨任务迁移（T1/T2 属 O02），也没有重跑 R07 的 30 臂。

## 10. 复现命令

```bash
cd /home/huyudi/012_conference/iclr2027
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python experiments/e1a933_review/o01_continuation.py --stage preserve
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python experiments/e1a933_review/o01_continuation.py --stage train      # 648 臂，约 12 分钟
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python experiments/e1a933_review/o01_continuation.py --stage summarize
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python experiments/e1a933_review/o01_continuation.py --stage check      # 209 项断言
```

证据索引：`ARMS.csv`（648 行全部运行）、`O01_CONTINUATION_FACTORIAL.csv`（含每臂对三种参考的
配对指标与 8×8 迁移矩阵）、`O01_CONTINUATION_PAIRED.json`（全部配对 CI）、
`ORACLE_LEDGER.json`、`COLLAPSE_SUMMARY.json`、`ROBUST_GUARDED_SELECTION.json`、
`CHECK_REPORT.json`（209 项断言）、`TRAIN_RECEIPT.json`（含 source commit `e1a933e6b32dd07c6895c7925cbccd9004da7f54` 与真实 argv），
`check_single_runs/`（check 阶段单独重跑的 5 个臂，用于单跑/批量一致性对照），
逐臂目录含 `receipt.json` / `batch_order.{json,npz}` /
`evaluation.npz`（逐样本分数）/ `loss.json` / `model.pt`。
