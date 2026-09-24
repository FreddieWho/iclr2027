# N02 采样机制专报（48 臂完整版）

状态：`MATRIX_COMPLETE`，48/48 臂训练与全部逐样本预测、配对统计、结构预测均已回收并**独立重算**通过。本文件是 N02 的**唯一现行版本**；2026-09-24 15:26 的 12-checkpoint 零重训诊断降级为第 9 节历史子节（其 `NOT_RUN` 项已由本版回收，见 9.1）。

- 结果根目录（只读输入）：`artifacts/e1a933_review/gpu_finish_20260924/sampling/extracted/sampling_results/`
- 本轮独立复算脚本：`experiments/e1a933_review/sampling_report_gpu.py`（只读、无 GPU、无 torch、确定性）
- 本轮产物：`artifacts/e1a933_review/N02_interpretation/`（下表每个数字都能索引到其中一个文件）
- 判读过程与自我反驳：`reports/e1a933_review/N02_INTERPRETATION.md`
- 论文文字：`reports/e1a933_review/handoff/n02_PAPER_TEXT.md`

## 0. 一页结论

| 问题 | 检验 | 观测（parent 等权，85 个 fresh test parent） | 判定 |
|---|---|---|---|
| (a) 纯上采样（A→B，不增加观测信息）是否就够 | `B_standard_{init}_flip − A_standard_{init}_flip`，J | pretrained **+0.340** [+0.258,+0.423]；random **+0.300** [+0.224,+0.380] | 方向明确、种子一致；但 B 臂计算量是 A 的 **12.25×**，不能归因"信息可访问性" |
| (b) 真实原生高分辨率信息是否额外重要 | `C_standard_{init}_flip − B_standard_{init}_flip`，J | pretrained **+0.004** [−0.029,+0.037]（不可判定）；random **+0.040** [+0.011,+0.076] | 仅 random 下有小的额外价值（≈尺度总效应的 13%） |
| (c) lowstride 是否消掉 B 相对 A 的优势 | 交互 `(B−A)_lowstride − (B−A)_standard`，J | pretrained **−0.133** [−0.213,−0.058]；random **+0.020** [−0.042,+0.081] | 冻结预测在 pretrained 命中、random 未命中 → **机制 UNRESOLVED** |
| (d) 尺度效应作用在哪一层 | static-only 对照 + 分解 | static 臂 B−A=**+0.391**(pre)/**+0.365**(rand)，与 flip 臂同级；分解落在 atom/AB 状态 | 效应不依赖 flip 监督；主要在**被编辑状态与其合取**，非基础识别、非仅阈值 |

一句话：**同一像素信息、只换输入网格（并顺带把 conv/linear MACs 提高一个数量级）就能大幅提高"部分正确→完整正确"的转化；真实原生分辨率的额外贡献小得多；去掉 maxpool 只在预训练初始化下削弱该优势，"小编辑更明显"的半句预测未被支持，机制保持未解决。**

## 1. 数据与合同（全部为冻结输入）

- 几何/数据：`data.npz` sha256 `5f773e0b…e75032`；test quartet **178 行 / 85 parent**（每 parent ≤4），其中 `small_edit` **5 行 / 3 parent**（定义：`min(max|Δ端点|(编辑a), max|Δ端点|(编辑b)) × 32 ≤ 2`，即以 native64 像素计的较小组成编辑位移 ≤2；掩码存在 `data.npz:quartet_small_edit`，生成于 `sampling_gpu_prepare.py`）。
- fresh parent 证据（`data_manifest.json`）：train/dev/test parent 两两最近 L∞ ≥0.1028 且 1e-6 内匹配 0；全部 train 状态 vs test 状态最近 L∞ 0.0707、匹配 0；对 19 个旧暴露 bank 的 parent 匹配数合计 **0**。
- 观测合同（`config.json`、`receipt.json:config`）：A=原生 64 连续胶囊渲染；B=bilinear(A,224)；C=原生 224；D=area(C,64)。低通 D 与 A 只差光栅/降采样方式。
- 16 个实验格 × 3 个模型初始化 seed（803/805/806）= 48 臂；`seeds` 是**模型初始化，不是独立任务**。
- 训练与选型（`receipt.json:config`）：20 epoch、Adam 3e-4、batch 32、FP32、TF32/AMP 关闭；只按 singleton-dev BCE 选 checkpoint，无 AB train/dev，无目标 J 选型，未删除任何 seed（`selection` 字段）。static 臂把 clean 索引重复到与 flip 臂相同的每 epoch 曝光数。
- 源码与校验：`receipt.json:source_sha256`（`sampling_gpu_run.py` `1d78cc3d…`、`sampling_gpu_protocol.py` `aa2f7765…`）、`bundle_manifest_sha256` `badd5300…`、`data_sha256` `5f773e0b…`；源 commit `e1a933e6b32dd07c6895c7925cbccd9004da7f54`（`data_manifest.json:source_commit`）。本轮用 `--verify-hashes` 重算了 `ARTIFACT_MANIFEST.json` 的全部 **250 个文件**，0 处不符（`N02_interpretation/verification.json:artifact_manifest`）。
- 设备：RTX 3080 Ti、torch 2.5.1+cu124（`receipt.json`）。

## 2. 因子矩阵（J、原子、CCM；分母 178 quartet / 85 parent）

指标命名与冻结协议 `sampling_gpu_protocol.metrics()` 一致：`P_state0/A_state1/B_state2/AB_state3` 为四个状态准确率，`atomic_joint`=两个单编辑都对，`J`=atomic_joint 且合取编辑对，`CCM`=`J/atomic_joint`。下表为三 seed 均值；逐 seed、逐 parent 值见 `per_arm_metrics.csv`、`arm_parent_values.csv`。

| obs | stem | init | 监督 | J 803 | J 805 | J 806 | J 均值 | atomic_joint 均值 | CCM 均值（分母均值） | 选中 epoch 均值 | 训练秒/臂 均值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | standard | pretrained | flip | 0.506 | 0.500 | 0.483 | **0.496** | 0.682 | 0.728 (121.3) | 12.0 | 42 |
| A | standard | random | flip | 0.618 | 0.573 | 0.551 | **0.581** | 0.751 | 0.773 (133.7) | 16.7 | 44 |
| A | lowstride | pretrained | flip | 0.697 | 0.713 | 0.669 | **0.693** | 0.822 | 0.843 (146.3) | 15.0 | 44 |
| A | lowstride | random | flip | 0.584 | 0.624 | 0.584 | **0.597** | 0.779 | 0.767 (138.7) | 16.7 | 44 |
| B | standard | pretrained | flip | 0.876 | 0.854 | 0.764 | **0.831** | 0.936 | 0.888 (166.7) | 12.0 | 68 |
| B | standard | random | flip | 0.837 | 0.888 | 0.871 | **0.865** | 0.927 | 0.933 (165.0) | 16.3 | 68 |
| B | lowstride | pretrained | flip | 0.910 | 0.938 | 0.871 | **0.906** | 0.949 | 0.955 (169.0) | 10.7 | 204 |
| B | lowstride | random | flip | 0.921 | 0.899 | 0.916 | **0.912** | 0.955 | 0.955 (170.0) | 17.0 | 204 |
| C | standard | pretrained | flip | 0.888 | 0.820 | 0.831 | **0.846** | 0.936 | 0.904 (166.7) | 8.3 | 93 |
| C | standard | random | flip | 0.888 | 0.899 | 0.893 | **0.893** | 0.936 | 0.954 (166.7) | 11.7 | 88 |
| D | standard | pretrained | flip | 0.522 | 0.461 | 0.461 | **0.481** | 0.665 | 0.724 (118.3) | 7.0 | 57 |
| D | standard | random | flip | 0.421 | 0.472 | 0.472 | **0.455** | 0.678 | 0.689 (120.7) | 15.7 | 58 |
| A | standard | pretrained | static | 0.264 | 0.247 | 0.225 | **0.245** | 0.586 | 0.420 (104.3) | 5.7 | 44 |
| A | standard | random | static | 0.258 | 0.140 | 0.146 | **0.182** | 0.639 | 0.283 (113.7) | 8.0 | 44 |
| B | standard | pretrained | static | 0.685 | 0.764 | 0.607 | **0.685** | 0.861 | 0.794 (153.3) | 7.0 | 68 |
| B | standard | random | static | 0.612 | 0.562 | 0.556 | **0.577** | 0.745 | 0.774 (132.7) | 9.3 | 68 |

来源：`N02_interpretation/per_arm_metrics.csv`（48 行，含 checkpoint sha256、dev BCE、单状态准确率、AUC、训练秒数、曝光数）。

计算预算（`structure_matrix.json`，conv+linear MACs/图；preflight 单步秒数）：

| 配置 | 参数 | MACs/图 | 相对 standard64 | preflight step s |
|---|---|---|---|---|
| standard 64 | 11,177,025 | 148,046,336 | 1.00× | 0.0120 |
| lowstride 64 | 11,177,025 | 563,282,432 | 3.80× | 0.0171 |
| standard 224 | 11,177,025 | 1,813,561,856 | 12.25× | 0.0349 |
| lowstride 224 | 11,177,025 | 6,900,204,032 | 46.61× | 0.1054 |

参数完全相同，**计算量完全不同**：B 相对 A 是 12.25×，lowstride 相对 standard 是 3.80×。这是本报告所有"网格/低stride"结论的必要限定。

## 3. 独立复算与校验（不沿用汇总数字）

`sampling_report_gpu.py` 直接读 48 个 `predictions.npz` / `test_single_predictions.npz` 重算：

1. 逐臂 9 个指标与冻结 `result.json` 逐字段比对：48×9 全部 ≤1e-12（脚本内 assert；`verification.json:recomputation_matches_frozen_result_json`）。
2. 与冻结 `paired_analysis.json` 逐项比对 **324** 个数值（18 对比 × 3 seed × 3 分层 × 2 口径），另与 `structural_prediction.json` 比对 **18** 个条目（6 条目 × 3 分层 × 2 口径）：最大绝对差 **4.97e-7**（仅来自本报告 6 位小数记录）。
3. `ARTIFACT_MANIFEST.json` 250 文件逐一重算 sha256：**0 处不符**。
4. 确定性：连续两次运行，除 `run_manifest.json`（记录墙钟耗时）外的全部输出字节一致。
5. 统计口径：J 等"成功率"类对比用 **parent 等权**（先在 parent 内平均，再对 85 个 parent 等权），CI 为 4000 次按 parent 聚类的 bootstrap（`BOOT_SEED=962010`，每个 (对比,分层,seed) 用独立派生种子）；CCM 为行池化比值并按 parent 重采样。行等权值同时保留在 `contrast_summary.csv:value_row`。

配对统计全表：`contrast_summary.csv`（对比 × 指标 × 分层 × seed，含 CI）；逐 parent 配对差：`contrast_parent_deltas.csv`（约 1.1 万行）；110→111/迁移/退化计数：`paired_transition_counts.csv`。

逐样本分数不从汇总回填：每臂每个 quartet 的 4 个 logits 存在 `<arm>_s<seed>/predictions.npz`，测试单状态的 2805 个 logits 存在 `<arm>_s<seed>/test_single_predictions.npz`，两者都在上面第 3 项的 250 文件清单里逐一核过 sha256；本报告只在其上做聚合。

## 4. 四个正交问题的独立判定

### (a) B vs A：不增加观测信息、只放大输入网格

| init | ΔJ | per-seed | 分解（B−A） |
|---|---|---|---|
| pretrained | **+0.340** [+0.258,+0.423] | +0.379 / +0.342 / +0.299 | atomic_joint +0.243；AB_state3 +0.124；P_state0 +0.016 |
| random | **+0.300** [+0.224,+0.380] | +0.252 / +0.331 / +0.316 | atomic_joint +0.171；AB_state3 +0.146；P_state0 +0.012 |

行等权 0.496→0.831（pretrained 三 seed 均值）级别的差距不可能来自新增观测像素——B 与 A 是同一 64 格数组的双线性放大。但同一改动把 MACs 提高 12.25×、把层1特征图从 16×16 提到 56×56。因此本报告只主张：**在固定观测信息下，"网格/早期采样配置"能造成 0.30–0.34 的 J 差距**；不主张这已被证明为"信息可访问性"而非算力/容量效应。

### (b) C vs B：真实原生高分辨率信息是否额外重要

| init | ΔJ(C−B) | per-seed | atomic_joint | AB_state3 |
|---|---|---|---|---|
| pretrained | +0.004 [−0.029,+0.037]（不可判定） | +0.011 / −0.034 / +0.067 | −0.003 | +0.003 |
| random | **+0.040** [+0.011,+0.076] | +0.051 / +0.011 / +0.022 | +0.014 [−0.004,+0.031] | +0.035 [+0.007,+0.074] |

C−A 与 B−A 几乎相同（pretrained +0.344 vs +0.340；random +0.340 vs +0.300）。即**上采样已拿到几乎全部收益**，原生 224 的额外信息只在随机初始化下贡献约 +0.04（≈总尺度效应的 13%）。低通对照支持同一读法：D（native224 的 area 降采样回 64）与 A 无差别（pretrained +0.004 [−0.067,+0.073]；random −0.094 [−0.146,−0.044]，即 D 略差），而 D−C 为 −0.340/−0.434。

### (c) lowstride 是否消掉 B 相对 A 的优势

冻结预测原文（`config.json:frozen_prediction`，受 manifest 校验）："removing maxpool reduces (J_B-J_A) relative to standard; predicted strongest for min constituent endpoint displacement <=2 native64 pixels; report interaction even if opposite, no threshold search"。

| init | 交互（J 尺度） | per-seed（parent 等权） | 分量 | 判定 |
|---|---|---|---|---|
| pretrained | **−0.133** [−0.213,−0.058] | −0.169 [−0.279,−0.061] / −0.128 [−0.237,−0.021] / −0.102 [−0.222,+0.025] | A 臂自身 +0.211 [+0.148,+0.279]，B 臂自身 +0.078 [+0.043,+0.116] | **hit** |
| random | **+0.020** [−0.042,+0.081] | +0.110 [+0.029,+0.190] / −0.072 [−0.163,+0.015] / +0.022 [−0.081,+0.124] | A 臂 +0.037（不可判定），B 臂 +0.057 | **未命中**（seed803 显著反向） |

消掉的比例：pretrained 下 −0.133/0.340 = 39%；random 下为 0 甚至反向。按 `N02_GPU_EXECUTION_PLAN.md` 预先写下的规则（"若不消除，不要继续把 aliasing 当解释，报告未解决"）：**机制判定 UNRESOLVED**。另外 lowstride 同时改变局部池化/感受野并带来 3.80× MACs，任何单一"aliasing"归因都不成立。

### (d) static-only 对照：尺度效应作用在哪一层

| 量 | pretrained flip | pretrained static | random flip | random static |
|---|---|---|---|---|
| B−A，J | +0.340 | **+0.391** | +0.300 | **+0.365** |
| B−A，atomic_joint | +0.243 | +0.235 | +0.171 | +0.096 |
| B−A，P_state0 | +0.016 | +0.098 | +0.012 | +0.094 |
| B−A，AB_state3 | +0.124 | +0.183 | +0.146 | +0.299 |
| flip−static，A | +0.226 [+0.154,+0.302] | — | +0.348 [+0.272,+0.425] | — |
| flip−static，B | +0.175 [+0.124,+0.231] | — | +0.282 [+0.218,+0.349] | — |
| 交互 `(B−A)_static − (B−A)_flip` | +0.050 [−0.044,+0.141] | | +0.065 [−0.048,+0.176] | |

读法：
1. 尺度效应**不依赖** flip 监督——static-only 臂里 B−A 同样大（甚至更大），交互 CI 含 0，故"尺度改变的是监督利用方式"这一说法不被支持。
2. 在 flip 监督下，gap 集中在**被编辑状态与其合取**（atomic_joint、AB_state3 各约 +0.12–0.24），基础静态识别几乎不动（P_state0 +0.016/+0.012，AUC 已 ~0.999–1.000）；static 监督下 P_state0 也出现 +0.098/+0.094 的差距（即交互 +0.082 [+0.039,+0.129]），因为 flip 臂的基础识别已接近天花板。
3. **不是"只改阈值"**：状态 3 的阈值无关 AUC 从 0.755→0.954（pretrained）、0.735→0.938（random）同步上升，单状态测试准确率也从 0.950→1.000（pretrained flip 单片）上升；CCM 从 0.728→0.888（pretrained）、0.773→0.933（random）。
4. 完整 repair 的计数（`paired_transition_counts.csv`，178 行分母）：A 臂"原子对但合取错"的 110 基线为 36/32/31（pretrained 三 seed），B 臂把其中 29/21/23 修成 111；111 退化 4/2/7，迁移 1/2/2。

## 5. 结构预测对照（逐格 hit / miss / 无法判定）

判定规则（写在脚本里，`prediction_vs_result.csv:verdict_rule`）：预测符号为负；`observed<0 且 CI 上界<0` 记 hit；`observed>0 且 CI 下界>0` 记 refuted；其余记 undecidable；`small_edit` 分层因 5 行/3 parent 一律记 `undecidable_small_denominator`（符号同时照报）。

| init | seed | 分层 | 观测（J 尺度，parent 等权） | 判定 |
|---|---|---|---|---|
| pretrained | 803 / 805 / 806 | all | −0.169 / −0.128 / −0.102 | hit / hit / undecidable |
| pretrained | cross-seed | all | **−0.133 [−0.213,−0.058]** | **hit** |
| pretrained | cross-seed | other_edit（173 行/84 parent） | −0.131 [−0.205,−0.055] | hit |
| pretrained | cross-seed | small_edit（5 行/3 parent） | −0.056 [−0.333,+0.333] | 不可判定（且点估计**弱于** other_edit） |
| random | 803 / 805 / 806 | all | +0.110 / −0.072 / +0.022 | refuted / undecidable / undecidable |
| random | cross-seed | all | +0.020 [−0.042,+0.081] | 未命中 |
| random | cross-seed | small_edit | +0.056 [−0.167,+0.333] | 不可判定（点估计**强于** other_edit，方向相反） |

结论：**预测的第一半（低stride 削弱 B−A）在预训练初始化下命中，在随机初始化下未命中；第二半（小编辑更强）在两个初始化下都未被支持。** 命中的幅度为 B−A 的 39%（pretrained）。

### 5.1 这份预测"事前"到什么程度（按证据说话）

可证明为事前的部分：

- 预测文本、分层定义（≤2 native64 像素）、报告规则（反向也照报、不搜索阈值）都写在 GPU bundle 的 `config.json:frozen_prediction` 里；该 bundle 的成员时间戳为 **2026-09-24 17:13**，`data_manifest.json:created_utc` 为 `2026-09-24T09:13:56Z`；bundle 的 `manifest.json` 在运行开始时被逐文件校验（`receipt.json:bundle_manifest_sha256=badd5300…`）。
- 训练从 **17:32:54**（`receipt.started_utc`=09:32:54Z）开始，首个臂目录 `A_standard_pretrained_flip_s803` 时间戳 **17:34**，最后一臂与埋点结束 **18:36**。也就是说预测文字、分层掩码（`data.npz:quartet_small_edit`，随 bundle 一起冻结）在训练前约 20 分钟固定。
- 分层掩码是**数据侧**冻结量（随 `data.npz` 一起哈希），不是看结果后划的层。

**无法**证明为事前的部分（必须写明）：

- `structural_prediction.json` 本身是**训练之后**的产物：运行入口在 `assert len(receipt['completed'])==48` 之后才调用 `protocol.analyze(out, …)`，该函数同时写出 `paired_analysis.json` 与 `structural_prediction.json`，本地文件时间戳 18:36 = 运行结束时刻。
- 事前只固定了**符号方向、分层阈值、报告规则**，没有固定任何**数值幅度**（没有事前写"预期 −0.13"这类量）。
- 唯一的裸时间戳旁证 `reports/e1a933_review/N02_GPU_EXECUTION_PLAN.md`（17:17，"预先固定结构预测"一节）**未纳入版本控制**（`git ls-files reports/e1a933_review/` 为空），只能算本地 mtime，不能当独立存证。

因此本报告对该预测的定位是：**"事前固定的符号+分层预测，在新 85 个 fresh parent 上逐格对照"成立；"事前固定了数值/机制"不成立。**

## 6. 塌缩 seed 与退化检查（不剔除任何格子）

- 新 48 臂：**没有任何臂出现塌缩**。判据：任一状态的全部 178 行预测是否被压成同一类（`posrate_state*` ∈{0,1}）——48 臂全部为空集（`verification.json:degeneracy.degenerate_state_predictions=[]`）。最低 J 的臂是 `A_standard_random_static` seed805（J=0.140），仍是有信息的工作点，保留。
- 历史塌缩格（保留、不剔除）：旧 12-checkpoint 诊断中 **seed803、检查点 train224_random、输入 64 与 224 两种条件 J 均为 0.000**，四状态准确率恒定（0.3125 / 0.3125 / 0.3125 / 0.6875），即常量输出塌缩。它在旧几何格 = "训练分辨率 224 × random 初始化"；路径：`artifacts/e1a933_review/N02_sampling_summary/observation_matrix.csv` 第 26–27 行、`artifacts/e1a933_review/N02_sampling_seed803_n64/result.json`。
- 注意区分：新合同下对应格 `B_standard_random_flip`（224 上采样输入 × random 初始化）**不塌缩**（J=0.837/0.888/0.871）。两者不同的 renderer、训练配方与 bank 都不相同，不能直接对比，也不能据此说旧塌缩"被修好"。

## 7. 机制判读与已检查的替代解释

- **计算量不匹配**：B/A = 12.25×、lowstride/standard = 3.80× MACs（第 2 节表）。因此"上采样=白拿收益"是错误的读法；应写成"在信息不变、算力/网格改变的条件下出现大幅 J 差距"。
- **低通对照**：D≈A（差 +0.004/−0.094），D−C≈−(C−A)，说明收益与"原生像素信息量"关系很小，而与大网格/早期特征图尺寸关系大。
- **阈值 vs 判别力**：AUC 与单状态准确率同步上升（第 4(d) 节），排除"只改阈值"。
- **初始化是必要条件**：lowstride 交互在 pretrained 下 −0.133、random 下 +0.020，说明"早期池化/网格"这一解释依赖权重先验；不能写成通用采样机制。
- **不做的因果宣称**：本矩阵仍是单骨干（ResNet18）、单渲染合同、单任务（关系 quartet），每个格子只有 3 个初始化 seed（不是 3 个独立任务）；85 个 parent 是 fresh 的，但只覆盖同一几何分布。

## 8. 机制主图（`N02_interpretation/n02_mechanism_figure.pdf`）

只画一张，由脚本从真实结果生成（`draw_figure`）。是否出图有一个写死在脚本里的判据：**pretrained 的跨 seed 交互 CI 上界<0，或三个 seed 全为负且至少两个 seed 的 CI 排除 0**。实际满足（−0.133 [−0.213,−0.058]，3/3 负、2/3 显著），故出图。

- 左：pretrained、三 seed（柱=均值、点=各 seed，均为 **85 parent 等权**）的 A/B × standard/lowstride 的 J：0.463 / 0.803 / 0.675 / 0.882，标注 ΔJ=+0.340（standard）与 +0.207（lowstride，= 0.340 − 0.133）。
- 右：冻结预测的交互量（J 尺度）逐 seed 与跨 seed 均值，pretrained（蓝）/random（橙），含 85-parent bootstrap CI 与 0 线。
- 图**不支持**"aliasing 机制已建立"：random 面板 CI 跨 0 且 seed803 显著为正，图中如实呈现。

若判据不满足，脚本不写 PDF 并在 `run_manifest.json:figure` 里写明原因（`drawn:false` + 具体数值）。

## 9. 历史子节：旧 12-checkpoint 零重训前向诊断（2026-09-24 15:26，保留）

**本节是历史记录，不代表当前状态。** 当时的产物：`artifacts/e1a933_review/N02_sampling_seed{803,805,806}_n64/`、`artifacts/e1a933_review/N02_sampling_summary/`（只读保留，未修改）。当时的合同与本版**不同**：旧整数方形笔刷 renderer、旧 bank（547 行、546 行资格、抽 64 quartet / 58 parent）、只在既有 checkpoint 上做输入尺寸/相位切换，**零重训**。

同一 checkpoint 输入切换（相位00，64 quartet / 58 parent，parent 等权）：

| seed | checkpoint | J@64 | J@224 | ΔJ row | ΔJ parent | parent bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---|
| 803 | train64_pretrained | 0.2188 | 0.0000 | −0.2188 | −0.2155 | [−0.3190, −0.1207] |
| 803 | train64_random | 0.0938 | 0.0000 | −0.0938 | −0.0862 | [−0.1554, −0.0259] |
| 803 | train224_pretrained | 0.0156 | 0.7031 | +0.6875 | +0.6638 | [+0.5431, +0.7759] |
| 803 | train224_random | 0.0000 | 0.0000 | +0.0000 | +0.0000 | [+0.0000, +0.0000] |
| 805 | train64_pretrained | 0.2969 | 0.0000 | −0.2969 | −0.3017 | [−0.4224, −0.1897] |
| 805 | train64_random | 0.1406 | 0.0000 | −0.1406 | −0.1293 | [−0.2155, −0.0517] |
| 805 | train224_pretrained | 0.0625 | 0.6719 | +0.6094 | +0.5862 | [+0.4655, +0.7069] |
| 805 | train224_random | 0.0469 | 0.6250 | +0.5781 | +0.5690 | [+0.4483, +0.6983] |
| 806 | train64_pretrained | 0.2031 | 0.0000 | −0.2031 | −0.2069 | [−0.3103, −0.1034] |
| 806 | train64_random | 0.0781 | 0.0000 | −0.0781 | −0.0862 | [−0.1552, −0.0172] |
| 806 | train224_pretrained | 0.0312 | 0.6875 | +0.6562 | +0.6379 | [+0.5172, +0.7586] |
| 806 | train224_random | 0.0156 | 0.5625 | +0.5469 | +0.5345 | [+0.4138, +0.6638] |

来源：`artifacts/e1a933_review/N02_sampling_summary/observation_matrix.csv`（每行 256 图前向）。当时结论（仍成立）：**固定 checkpoint 只换输入尺寸 → 64 训练模型的 J 全为 0，说明"把图放大"不能替代"在大网格上训练"**；尺度效应主体属于训练/优化配置。当时保留 seed803 塌缩，未剔除。相位平移扫描只证明对具体光栅平移敏感，不能单独确定 aliasing。

### 9.1 当时标为 NOT_RUN / 未决、现已回收的项

| 当时的 NOT_RUN 表述 | 现已回收为 |
|---|---|
| "原生 224（C）与低通 64（D）：NOT_RUN" | 第 2 节 C/D 各 6 臂 × 3 seed；第 4(b) 节 |
| "低stride / 抗混叠结构：NOT_RUN" | `A/B × lowstride` 共 12 臂；第 4(c) 节 |
| "四格完整重训、pretrained/random 公平开发：NOT_RUN" | 48 臂完整矩阵；第 2 节 |
| "新 parent 命中的预先固定结构预测：无" | 第 5 节（85 个 fresh parent，逐格 hit/miss/不可判定） |
| "机制未决（无低stride 干预）" | 仍为 UNRESOLVED，但现在是**有条件的**未决：pretrained 命中、random 未命中 |

仍属本 lane 之外、建议父 agent 统一清理的旧状态文字：`reports/e1a933_review/N02_GPU_EXECUTION_PLAN.md` 首段"48臂训练尚未执行"；`reports/e1a933_review/PAPER_PATCH_AUDIT.md:37`"低stride/native224/完整重训与新parent前瞻验证NOT_RUN"。

## 10. 未完成与边界

1. **细小编辑层的辨识力不足**：`small_edit` 只有 **5 quartet / 3 parent**（占 178 行的 2.8%）。预测中"小编辑更明显"这半句在这里**无法判定**，点估计还与预测方向相反。任何"尺度效应主要作用于细小编辑"的外推都不成立；要检验它必须重新挖一个以 ≤2px 编辑为主的分层（新数据，不在本轮范围）。
2. **每次比较的计算量都不匹配**：B/A 12.25×、lowstride/standard 3.80×。要做"信息 vs 算力"的干净分离，需要参数/算力匹配的对照（例如把 A 的 backbone 加宽到同 MACs），本轮没有做。
3. **只测了 ResNet18 + 一个渲染合同 + 一个任务**：不能外推到其它骨干或自然图像；`docs/e1a933_review/03_NEW_HIGH_VALUE_ROUTES.md` 也明确"不得声称首次发现 CNN 对输入尺度敏感"。
4. **机制未解决**：`aliasing` 只有在 pretrained init 下被"低stride 削弱优势"部分支持，random 下不成立，且 lowstride 同时改变感受野与算力。**本报告不给 aliasing 归因。**
5. **预测只有方向与分层是事前的**（5.1），数值幅度事后才知道；因此不能把"命中"表述成"事前数值预测成功"。
6. **3 个 seed 是 3 个初始化，不是 3 个独立任务**；bootstrap 只覆盖 parent 抽样不确定性，不含种子/配方不确定性。
7. 旧诊断（第 9 节）与本版使用**不同 renderer 与不同 bank**，两版数字不可拼接为一个因果差值。
8. 本 lane 只负责 N02；不代表 `docs/e1a933_review/` 施工包整体完成。

## 11. 复现

```bash
# 独立复算 + 生成全部 CSV/JSON + 机制图（只读输入，无 GPU，无 torch）
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
  python experiments/e1a933_review/sampling_report_gpu.py \
  --results artifacts/e1a933_review/gpu_finish_20260924/sampling/extracted/sampling_results \
  --out artifacts/e1a933_review/N02_interpretation --bootstrap 4000 --verify-hashes
```

输出：`per_arm_metrics.csv`、`arm_parent_values.csv`、`contrast_summary.csv`、`contrast_parent_deltas.csv`、`paired_transition_counts.csv`、`prediction_vs_result.csv`、`questions.json`、`verification.json`、`run_manifest.json`、`n02_mechanism_figure.pdf`。运行参数、源 commit（`e1a933e`）、48 个 checkpoint hash、`data.npz` hash、线程设置都写在 `run_manifest.json`。原地重跑会覆盖同目录产物（脚本判据与输入完全确定，输出字节一致，唯一例外是 `run_manifest.json` 里的墙钟耗时）。

未新增外部生信数据，`infra/bioinf-data-index/` 无需更新。
