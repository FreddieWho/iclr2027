# N01：执行协议与当前边界

状态更新：GPU完整21臂已完成并独立核验；N01改善辅助目标可学性，但没有稳定的分类/完整repair收益。完整数值和分母见`VISION_GPU_RESULTS.md`。下文保留执行前协议；N03已按真实辅助学习证据启动，见`N03_GATE_AND_PROTOCOL.md`。

正式矩阵：pretrained_static、pretrained_flip、random_static、random_flip、ordered、matched、shuffled_matched；seed803/805/806，总21次训练。前12次对应O03，后9次与共享的pretrained_flip组成N01的4臂3seed矩阵。同一个ImageNet ResNet18、同native64上采样224、Adam3e−4、batch32、每臂20epoch。模型选择仅基于dev的clean/single终点BCE；从不以测试J选择。

新数据在物理父场景层面独立生成后才生成编辑和图像：train512、dev128、test512父场景。训练2816图像=2048clean+768single flip，与旧训练2048clean+至多920flip不同；dev707图像；test2803单状态图。固定测试E事件159quartets/74父场景，最多每父场景4个，不能当159独立父场景。训练单编辑由预先固定16次随机采样、radius .1/.2/.35/.5循环、oracle margin≥.02和至少20改变像素产生，每父场景最多2个。未使用训练组合标签。

渲染协议纠正了旧renderer在20/512场景上的方向舍入身份泄漏；所有新臂采用同色端点坐标排序后的相同图像。train/dev/test原父场景canonical哈希无交集，生成图像跨split精确字节无交集。完整manifest记录图像/坐标哈希，runner记录压缩bank SHA256和源代码工作树哈希。

ordered/matched/shuffled三个臂具有同一辅助头和同一初始主干/分类器；shuffle使用专用固定Generator。三个aux臂使用同一个train-only标定规则：初始化有序目标的主干梯度范数，将aux初始梯度量级设为BCE的0.25倍（clip1e−4至10）；每seed所有aux臂共享由同一原始目标计算的lambda，不以dev/J调参。每epoch记录实际BCE/aux梯度量级、dev有序/匹配误差和训练曲线，训练中比例允许漂移，不能称全程严格scale matching。BCE-only也分配同样辅助头但不给梯度。

每臂交付checkpoint哈希、test单状态/四态logit和parent id、dev逐样本有序/匹配误差、epoch曲线；矩阵完成后自动生成同样本配对J、baseline110 full repair、migration、111退化和按真实parent bootstrap区间，三个seed分别报告。

N01方法收益判决：NOT_SUPPORTED_BY_THIS_RECIPE。C未同时稳定超过ordered、shuffle和BCE baseline；不能支持所测学生范围内匹配辅助监督带来完整repair的方法主张。N03已由辅助学习与分类结果共同触发并启动；其结论仍待新结果。旧A模型、旧产物只读。

独立噪声无关查重：train/dev/test/quartet之间相应red/blue颜色mask精确交集均为0，防止背景noise掩盖同图重复；train到dev/test/quartet的canonical坐标最近L∞距离分别.1492/.1210/.1584，.01邻域计数均0。坐标近邻仅为诊断，不能据此证明一切感知近似图都不存在。证据`vision_canonical224_v2/split_visual_duplicate_audit.json`；复现`python experiments/e1a933_review/vision_split_audit.py`。

---

## N01-P1 补验：起点置信（lg0）下的 preserve/flip 诊断（lane `n01p1`）

审计缺口（`reports/e1a933_review/PACKAGE_COMPLETION_AUDIT.md` N01 行、`PACKAGE_COMPLETION_MATRIX.csv`）：施工卡要求 N01「最后看 J、full repair、migration 与 P1」，但本专报此前只有 J 与 aux 学习证据，没有独立的 P1 证据。本节补齐该诊断。

**结论摘要：P1 诊断可做、已做，零重训、无 GPU；在 P1 维度上没有观察到 matched 相对 ordered / BCE-only / shuffled-matched 的一致额外差异。N01 的方法判定不变（`NOT_SUPPORTED_BY_THIS_RECIPE`）。**

### 1. 数据口径与可追路径

- 只读输入：`artifacts/e1a933_review/vision_gpu_interpretation/cuda_matrix/{arm}_s{seed}/predictions.npz`（159 quartet × 4 状态 logits、labels、parent），逐文件 sha256 见同目录 `ARTIFACT_MANIFEST.json`；四臂 = `pretrained_flip`（BCE-only）、`ordered`、`matched`、`shuffled_matched`，seed = 803/805/806，共 12 个 run。
- **起点置信来源**：quartet 第 0 列就是未编辑父场景的渲染（`experiments/e1a933_review/vision_protocol.py:generate` 中 `states=[x, x+ea, x+eb, x+ea+eb]`），因此 `lg0 = logits[:,0]`，不需要也不允许用终点 logit 冒充起点。同一 parent 的多个 quartet 起点图像与起点 logit 完全相同（脚本内断言 `spread==0`，12/12 通过）。
- **keep/flip 定义**：quartet 标签模式恒为 `(y0, y0, y0, 1-y0)`（`mine_quartets`：两个原子编辑保持标签、组合编辑翻转），故 keep = 第 1/2 列（318 状态），flip = 第 3 列（159 状态）。
- **阈值**：逐模型、固定在已声明的开发参考集上——dev 单状态 bank（707 张，其中未编辑 clean 512 张 / 128 parent）的 `|logit|` 的 **2/3 分位**（沿用 D03/R04 的 top-tertile 约定）。dev logits 由回收包内 hash 校验过的 checkpoint 做一次 CPU 前向得到（`model.pt` sha256 同时等于 `ARTIFACT_MANIFEST.json` 与各 `result.json:checkpoint_sha256`，12/12 通过）；**无训练、无 GPU、无测试集泄漏**。敏感性另报 dev 中位数与上四分位阈值。
- **覆盖率按实测值报告**，不写「top 25%」。
- 回归检查：脚本用与 `analyze()` 相同的公式重算配对 ΔJ、baseline110、full repair、migration、111 退化，并断言与已归档 `cuda_matrix/paired_analysis.json` **逐值相等**（15 个 pair×seed 行全部通过）。

### 2. 分母（三者分开，不混用）

| 记号 | 含义 | 说明 |
|---|---|---|
| D1 | 全部编辑状态（keep 318 / flip 159） | 不做任何保留筛选 |
| D2 | 仅起点高置信（retained） | 不要求 start-correct（=R04 的「无条件 endpoint 风险」） |
| D3 | retained ∧ start-correct | 统一 start-correct 条件下的 update 风险 |

### 3. 逐臂逐 seed 结果（D1/D2/D3）

| 臂 | seed | J | P(start) | 覆盖率(quartet) | 覆盖率(parent) | keep D1 | keep D2 | keep D3 | flip D1 | flip D2 | flip D3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BCE-only | 803 | 0.642 | 0.994 | 0.497 | 0.486 | 0.157 | 0.165 | 0.165 | 0.107 | 0.127 | 0.127 |
| BCE-only | 805 | 0.698 | 1.000 | 0.421 | 0.459 | 0.129 | 0.164 | 0.164 | 0.094 | 0.090 | 0.090 |
| BCE-only | 806 | 0.711 | 1.000 | 0.528 | 0.514 | 0.119 | 0.131 | 0.131 | 0.088 | 0.083 | 0.083 |
| ordered rel10 | 803 | 0.717 | 0.981 | 0.384 | 0.392 | 0.116 | 0.139 | 0.139 | 0.082 | 0.049 | 0.049 |
| ordered rel10 | 805 | 0.642 | 1.000 | 0.434 | 0.432 | 0.167 | 0.138 | 0.138 | 0.069 | 0.072 | 0.072 |
| ordered rel10 | 806 | 0.730 | 1.000 | 0.453 | 0.446 | 0.107 | 0.118 | 0.118 | 0.075 | 0.069 | 0.069 |
| matched rel10 | 803 | 0.698 | 1.000 | 0.516 | 0.514 | 0.132 | 0.134 | 0.134 | 0.075 | 0.061 | 0.061 |
| matched rel10 | 805 | 0.673 | 1.000 | 0.503 | 0.500 | 0.132 | 0.100 | 0.100 | 0.107 | 0.125 | 0.125 |
| matched rel10 | 806 | 0.704 | 1.000 | 0.220 | 0.216 | 0.123 | 0.171 | 0.171 | 0.082 | 0.171 | 0.171 |
| shuffled-matched | 803 | 0.566 | 0.994 | 0.509 | 0.527 | 0.151 | 0.123 | 0.123 | 0.189 | 0.259 | 0.259 |
| shuffled-matched | 805 | 0.704 | 0.987 | 0.409 | 0.432 | 0.123 | 0.146 | 0.146 | 0.101 | 0.077 | 0.077 |
| shuffled-matched | 806 | 0.742 | 1.000 | 0.472 | 0.446 | 0.101 | 0.127 | 0.127 | 0.075 | 0.080 | 0.080 |

分母（D2/D3）与阈值：

| 臂 | seed | n(keep,D2) | n(flip,D2) | n(keep,D3) | n(flip,D3) | 阈值(dev 2/3 分位) |
|---|---|---|---|---|---|---|
| BCE-only | 803 | 158 | 79 | 158 | 79 | 7.80 |
| BCE-only | 805 | 134 | 67 | 134 | 67 | 6.72 |
| BCE-only | 806 | 168 | 84 | 168 | 84 | 10.40 |
| ordered rel10 | 803 | 122 | 61 | 122 | 61 | 9.57 |
| ordered rel10 | 805 | 138 | 69 | 138 | 69 | 7.18 |
| ordered rel10 | 806 | 144 | 72 | 144 | 72 | 9.75 |
| matched rel10 | 803 | 164 | 82 | 164 | 82 | 10.08 |
| matched rel10 | 805 | 160 | 80 | 160 | 80 | 7.02 |
| matched rel10 | 806 | 70 | 35 | 70 | 35 | 8.42 |
| shuffled-matched | 803 | 162 | 81 | 162 | 81 | 7.64 |
| shuffled-matched | 805 | 130 | 65 | 130 | 65 | 6.73 |
| shuffled-matched | 806 | 150 | 75 | 150 | 75 | 9.59 |

要点：

1. **覆盖率不是 1/3**：实测 0.220–0.528（quartet 级），跨臂与跨 seed 差异很大；matched_s806 只有 35/159 quartet、16/74 parent（低辨识力，单独标注）。这正是 R04 要求「报实际覆盖率」的原因。
2. **D2 与 D3 在 12/12 行完全相同**：起点准确率 P(start)=0.981–1.000，而所有起点判错的 quartet 都落在保留集之外（|lg0| 低于阈值），因此本矩阵中 D3 相对 D2 没有剔除任何保留样本（分母 n 逐行相等）。两套分母仍然都报出，且不得把 D2 的数当作 D3 用：这一巧合只在「高起点置信 ⇒ 起点判对」成立时才出现，不能推广。
3. **高起点置信并不降低编辑风险**：keep 上 D2 ≥ D1 出现在 9/12 行，flip 上 D2 > D1 出现在 6/12 行（例如 matched_s806 keep 0.123→0.171、flip 0.082→0.171；shuffled-matched_s803 flip 0.189→0.259）。起点置信在这里是「静态识别置信」，不能当作「编辑后仍正确的保证」。
4. **端点自置信口径（诊断，非本 lane 主口径）**：若按 R04 所修 bug 的形状、用状态自身 `|logit| ≥ thr` 定义保留集，覆盖率为 0.17–0.36，错误率降到 keep 0.000–0.044、flip 0.000–0.012。也就是说，用终点置信会把 P1 风险报成近似 0，而用起点置信得到的是 keep 0.100–0.171、flip 0.049–0.259。本 lane 主口径一律用起点置信。

### 4. 与 J、full repair、migration 同表对照（配对，共同保留子集）

配对比较的 P1 统计量在**两臂保留集交集**上计算（同一起点判据对两臂是同一个样本集合），避免「各臂覆盖率不同→比的是不同子集」。parent bootstrap 2000 次，95% 区间按 74 个 parent 聚类重抽。

| 对比 | seed | ΔJ(全 159) [CI] | ΔJ(共同保留) | 共同保留 n(q/p) | Δkeep错误(D2) [CI] | Δflip错误(D2) [CI] | full repair / baseline110 | migration | 111 退化 |
|---|---|---|---|---|---|---|---|---|---|
| matched − ordered | 803 | −0.019 [−0.072,+0.031] | +0.000 | 48/23 | +0.000 [+0.000,+0.000] | +0.000 [−0.061,+0.062] | 3/12 | 0 | 8 |
| matched − ordered | 805 | +0.031 [−0.034,+0.103] | +0.000 | 54/24 | −0.037 [−0.091,+0.000] | +0.074 [+0.000,+0.167] | 1/10 | 0 | 7 |
| matched − ordered | 806 | −0.025 [−0.093,+0.032] | +0.000 | 25/11 | +0.000 [+0.000,+0.000] | +0.000 [+0.000,+0.000] | 3/12 | 0 | 9 |
| matched − BCE-only | 803 | +0.057 [−0.012,+0.139] | +0.098 | 61/28 | −0.025 [−0.079,+0.000] | −0.049 [−0.120,+0.019] | 6/16 | 0 | 6 |
| matched − BCE-only | 805 | −0.025 [−0.073,+0.017] | +0.000 | 61/30 | −0.016 [−0.062,+0.017] | +0.033 [−0.042,+0.117] | 2/12 | 0 | 7 |
| matched − BCE-only | 806 | −0.006 [−0.068,+0.049] | −0.031 | 32/14 | +0.000 [+0.000,+0.000] | +0.031 [+0.000,+0.125] | 2/12 | 2 | 8 |
| matched − shuffled | 803 | +0.132 [+0.037,+0.233] | +0.068 | 59/29 | +0.017 [+0.000,+0.044] | −0.102 [−0.232,+0.000] | 18/29 | 1 | 8 |
| matched − shuffled | 805 | −0.031 [−0.078,+0.006] | +0.000 | 56/25 | −0.027 [−0.075,+0.000] | +0.036 [+0.000,+0.120] | 0/12 | 0 | 6 |
| matched − shuffled | 806 | −0.038 [−0.096,+0.011] | −0.033 | 30/14 | +0.000 [+0.000,+0.000] | +0.033 [+0.000,+0.133] | 1/12 | 1 | 8 |
| ordered − BCE-only | 803 | +0.075 [+0.018,+0.149] | +0.075 | 40/20 | −0.037 [−0.120,+0.000] | +0.000 [+0.000,+0.000] | 7/16 | 0 | 4 |
| ordered − BCE-only | 805 | −0.057 [−0.123,+0.000] | −0.020 | 49/22 | +0.020 [+0.000,+0.054] | −0.020 [−0.067,+0.000] | 4/12 | 0 | 14 |
| ordered − BCE-only | 806 | +0.019 [−0.026,+0.074] | −0.018 | 55/26 | +0.000 [+0.000,+0.000] | +0.018 [−0.047,+0.088] | 3/12 | 1 | 4 |
| shuffled − BCE-only | 803 | −0.075 [−0.173,+0.007] | +0.019 | 54/26 | −0.046 [−0.114,+0.000] | +0.074 [−0.024,+0.210] | 3/16 | 0 | 20 |
| shuffled − BCE-only | 805 | +0.006 [−0.031,+0.050] | +0.000 | 61/28 | +0.008 [+0.000,+0.026] | +0.000 [−0.053,+0.044] | 2/12 | 0 | 4 |
| shuffled − BCE-only | 806 | +0.031 [−0.013,+0.081] | +0.032 | 63/27 | +0.000 [+0.000,+0.000] | −0.032 [−0.092,+0.030] | 3/12 | 1 | 3 |

seed 均值（描述性，不把 3 个 seed 当 3 个独立任务）：

| 臂 | J | 覆盖率 | keep D1 | keep D2 | flip D1 | flip D2 | J(保留子集) |
|---|---|---|---|---|---|---|---|
| BCE-only | 0.683 | 0.482 | 0.135 | 0.153 | 0.096 | 0.100 | 0.616 |
| ordered rel10 | 0.696 | 0.424 | 0.130 | 0.132 | 0.076 | 0.064 | 0.693 |
| matched rel10 | 0.692 | 0.413 | 0.129 | 0.135 | 0.088 | 0.119 | 0.632 |
| shuffled-matched | 0.671 | 0.463 | 0.125 | 0.132 | 0.122 | 0.139 | 0.624 |

**matched 相对 ordered / BCE-only 在 P1 维度上是否有额外差异：没有一致证据。**

- matched − ordered：ΔJ 全样本 seed 均值 −0.004（3/3 seed 的 parent bootstrap 区间含 0）；P1 上 Δkeep错误 均值 −0.012、Δflip错误 均值 +0.025，**0/3** seed 的区间排除 0，且符号在 seed 间翻转。full repair 3/12、1/10、3/12，migration 0/0/0。
- matched − BCE-only：ΔJ 全样本 seed 均值 +0.008（0/3 区间排除 0）；Δkeep错误 均值 −0.014、Δflip错误 均值 +0.005（均 0/3）。matched 在 P1 上不比 BCE-only 稳定更好。
- matched − shuffled-matched：ΔJ 均值 +0.021，只有 seed803 的区间排除 0（+0.132 [+0.037,+0.233]），另两 seed 为负；P1 上 Δflip错误 均值 −0.011（0/3）。
- 因此 P1 分解没有给出 J 之外的额外信号：matched 的辅助目标可学性改善（`independent_summary.json:auxiliary_parent_paired`）并未转化为起点置信条件下的编辑风险下降。

### 5. 起点置信测量的稳健性

同一 parent 的 4 张未编辑 clean 单状态渲染的平均 logit 与 quartet 起点 logit 的相关 r = 0.992–0.9999，符号一致率 1.000（shuffled-matched_s805 为 0.986），说明起点置信不是某一次渲染的偶然产物。

### 6. 未完成与边界

- 本 lane 只覆盖 N01 的四臂（BCE-only / ordered / matched / shuffled-matched）；未对 `pretrained_static`/`random_*` 做 P1（P1 关心的是编辑风险，static 臂没有编辑监督，不构成同一诊断）。
- dev 阈值需要一次 checkpoint 前向（CPU，12 臂 × 707 张 ≈ 2 分钟）；这是本 lane 唯一的新计算，checkpoint 来自回收包且已 hash 校验。若只允许读取已归档的逐样本文件，则 dev 起点 logits 不在其中，阈值只能退化为测试集自参考——本 lane 明确选择了前者，并已在 `PROVENANCE.json` 记录 commit/hash/命令。
- 阈值分位（top-tertile）沿用 D03/R04 约定，没有预注册；敏感性结果（dev 中位数 / 上四分位）已写入 `n01_p1_per_arm_seed.csv` 的 `cov_dev_median`、`cov_dev_top_quartile`、`keep_err_retained_dev_*`、`flip_err_retained_dev_*` 列，覆盖率随阈值变化（三种分位下实测 0.182–0.635），keep/flip 保留错误率区间（keep 0.100–0.207、flip 0.018–0.295）方向不变。
- matched_s806 的保留集只有 35 quartet / 16 parent，其 P1 数字（keep/flip 0.171）单独看不可外推；seed 间不做池化。
- P1 只是静态置信的使用边界诊断，不替代表示/修复主张；本 lane 不改变 N01 的 `NOT_SUPPORTED_BY_THIS_RECIPE` 判定，也不构成方法结论。
- 未做：起点置信的概率校准（只用 |logit| 分位）、按编辑半径/方向分层的 P1、训练时顺序干预（属 R04 边界，非本 lane）。

### 7. 产物与复现

产物目录 `artifacts/e1a933_review/vision_n01_p1/`：

- `n01_p1_per_arm_seed.csv`（逐臂逐 seed：J、P、覆盖率、D1/D2/D3、阈值、敏感性）
- `n01_p1_denominator_table.csv`（长表：臂×seed×keep/flip×D1/D2/D3，含 n 与错误率）
- `n01_p1_paired_contrasts.csv`（配对 ΔJ / Δkeep / Δflip + parent bootstrap CI + full repair / migration / 111 退化）
- `n01_p1_table_combined.csv`（第 3、4 节合一的单表：逐臂水平 + 配对修复/迁移）
- `n01_p1_start_confidence_stability.csv`、`n01_p1_summary.json`、`PROVENANCE.json`、`dev_logits/*.npz`（12 个臂的 dev logits + 阈值）

复现（CPU，≤6 线程，确定性；独立重跑两次逐字节相同）：

```bash
cd /home/huyudi/012_conference/iclr2027
OMP_NUM_THREADS=6 python experiments/e1a933_review/n01_p1_diagnostic.py \
  --out artifacts/e1a933_review/vision_n01_p1 --force
python -m pytest experiments/e1a933_review/test_n01_p1_diagnostic.py -q   # 4 passed
```

`experiments/e1a933_review/test_n01_p1_diagnostic.py` 覆盖：R04 反例（起点 [10,1]/终点 [.1,10]、阈值 5 必须选第一个）在 vision 保留规则上成立、行序无关；D1/D2/D3 分母不被合并（构造一个「保留但起点判错」的 quartet，D3 必须把它排除）；覆盖率按实测值报告（0.25 / 0.50 而非名义比例）。

脚本在运行时校验：bundle `data.npz` 哈希、回收包 `model.pt` 哈希（同时对齐 `ARTIFACT_MANIFEST.json` 与各 `result.json`）、quartet 标签几何、同 parent 起点 logit 一致性，以及重算的配对统计与归档 `paired_analysis.json` 逐值相等。
