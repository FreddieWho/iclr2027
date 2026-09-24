# O04 前瞻干预验证：先声明"本干预影响哪一段"，再在新任务上验证

状态：**EXECUTED**（新任务前瞻预测 + 执行 + hit/miss 判定完成）；D10 下游回声 **已完成**（r09 的 `FRONTIER_FULLSCENE.json` 已存在，见 §6；判定：分解能解释形状与段位归属，不能解释份额/幅度，且对 117/256 不可行份额是结构性无法处理）。
基准 commit：`e1a933e6b32dd07c6895c7925cbccd9004da7f54`（本轮改动未提交，版本控制由父 agent 处理）。
旧产物只读：`artifacts/f095_campaign/`、`experiments/f095_campaign/`、`artifacts/e1a933_review/readout_refit_20260924/` 均未改动；全部新产物写入新目录。

---

## 1. 一句话结论

在**新任务**（U10 T1 点在三角形内 / T2 线段与圆盘）的**新冻结模型**上，事前声明的预测"head 级变化监督干预 I_flip 主要影响 J 段"**没有在严格判据下全中**：
**primary verdict = MISS**（P1 在 8 个单元中 7 个成立，唯一失败的单元是 T1/raw_clean；P2 在 8/8 成立）。
但 P2（"获益不是 oracle 阈值上界移动"）**8/8 全中**，且 miss 发生在一个三段效应都很小的单元里。
本 lane 因此得到的是一个**带边界的负面结果**：S/J/J\* 分解可以前瞻预测"干预不会动 oracle 上界"，但**不能**在每个单元上预测"干预主要动 J 而不是 S"。

---

## 2. 施工范围与"新"的确切含义

- **新任务**：U10 T1（4 点，点在三角形内）与 T2（3 点，线段与圆盘相交）。评估 quartet 是 E-quartet（同 parent 的两次保标签单编辑 + 一次翻标签双编辑），分母：
  - T1：**2724 quartet / 322 parent**（`artifacts/p123_upgrade/bank/bank_u10T1.npz`，manifest `E=2724, Eparents=322`）
  - T2：**3443 quartet / 369 parent**（`bank_u10T2.npz`，`E=3443, Eparents=369`）
- **新模型**：U10 T1/T2 的**冻结主干**，4 个 encoder 类 × 3 seed：
  `raw_clean, raw_flip, six_clean, six_flip`（raw = 原始坐标，six = 6/3 维成对距离；clean/flip = 主干训练时是否用了单编辑状态）。
  主干检查点来自 `artifacts/f095_campaign/U10/{T1,T2}_{raw,six}_s{11,23,47}/{clean,flip}/model.pt`，**本 lane 未训练任何主干**，只冻结取特征后重拟合 head。
- **"新"的边界（必须说清）**：这里的"新"是**相对 O04 readout 再分析**而言——O04 旧产物用的是源任务（线段交叉，`bank_dev512`，237 quartet/106 parent）与它的 4 个 arm。T1/T2 的主干与银行是 f095 campaign 早已产出的既有材料，**因此这不是"新鲜银行独立确认"**，也不能用来加强主 claim；它检验的是"分解能不能前瞻指导干预"这一诊断性问题。
- **未见性检查**：T1/T2 的 train 与 eval 场景由不同随机种子生成（10101 / 20202、30303 / 40404），实测 train↔eval 场景最小逐坐标最大差 = 0.1437（无重复）；T1 的 4 点场景与源任务 train_101/bank_dev512 的 parent 场景最小最大差 = 0.182–0.188（无重复）。T2 是 3 点任务，与 4 点源任务不存在 parent 重叠概念。→ **不触发施工卡里的"与旧 bank 有 parent 重叠"停止条件**。

---

## 3. 事前预测（可证明早于结果）

- 文件：`artifacts/e1a933_review/readout_forward_20260924/PREREGISTRATION.json`
- 封印：`artifacts/e1a933_review/readout_forward_20260924/PREREGISTRATION.seal.json`
- 写出时间：**2026-09-24T11:47:13+00:00**（本机 mtime 2026-09-24 19:47:13 +0800）
- 结果写出时间：**2026-09-24T11:48:38+00:00**（mtime 19:48:38 +0800）→ 预测早于结果 **85 秒**
- payload hash：`52199f0bcdaf75c6c468731e1ce732c78a6ceb7c0a1c1c6b89d350a4b1e4f990`
- 封印文件 hash（PREREGISTRATION.json 字节）：`ce22ae4b62fdbeb3…`（完整值见 seal 文件）
- 脚本 hash：写在 `PREREGISTRATION.json:script_sha256`；`--stage run` 会校验脚本未变，脚本一旦被改就拒绝运行（防止事后改预测）
- 顺序证据（可复算）：`python experiments/e1a933_review/readout_forward.py --stage verify --output artifacts/e1a933_review/readout_forward_20260924` 输出
  `{"integrity":"ok","ordering_ok":true,"prereg_written_utc":"…11:47:13…","results_written_utc":"…11:48:38…","prereg_mtime":1790250433.26,"results_mtime":1790250518.53}`
- 预先声明"结果尚不存在"：`PREREGISTRATION.json:absence_assertions` 记录写出时 `results.json / solver_logs / scores` 均不存在（当时目录本身都不存在）。
- 披露：预测由**源任务旧 bank 的 O04 再分析**（`reports/e1a933_review/O04_READOUT_REPORT.md`、`artifacts/e1a933_review/readout_refit_20260924/summary.json`）外推而来，旧 bank 的 seed 均值已原样写进 `PREREGISTRATION.json:prior_evidence_source_bank`。写出前**没有**读取任何 T1/T2 评估数值（`U10_T1_EVAL.json` 等只读过顶层 key 名，未读数值）。

### 3.1 预测内容（摘要，原文见 `prediction_text`）

干预 **I_flip**：冻结主干与冻结特征标准化，只改 head 监督——`static` 只用 clean 单状态；`flip` 在同一 clean 子集之外加入单编辑状态（两组各 50% 权重）。head 家族、初始化、L2/宽度候选、dev 选择预算两种 regime 完全一致。
记 ΔX = X(flip) − X(static)，X 为评估 quartet 上的 **parent-equal 配对均值**。

| 编号 | 预测 | 判据（全部单元成立才算 SUPPORTED） | 单元数 |
|---|---|---|---|
| P1 | I_flip 是 **J 段**干预 | `|ΔJ| > |ΔS|` 且 `|ΔJ| > |ΔJ*|` | 8（2 任务 × 4 encoder 类） |
| P2 | 获益不是 oracle 上界移动 | `|ΔJ*| < |ΔJ|` | 8 |
| P3 | 不提高关系型表示的 S | six 单元 3-seed 均值 `ΔS ≤ 0` | 4 |
| P4 | 关系型 encoder 的 J 变好 | six 单元 3-seed 均值 `ΔJ > 0` | 4 |
| P5 | MLP head 下分段归属随 encoder 类改变 | six 单元 `|ΔJ|` 最大、raw 单元 `|ΔS|` 最大 | 8 |

primary verdict 预先定义为 **P1 ∧ P2**。判据不含任何自由数值阈值（只用"最大/符号"），因此不存在事后调阈值空间。

---

## 4. 方法（全部产物可追）

产物根目录：`artifacts/e1a933_review/readout_forward_20260924/`
脚本：`experiments/e1a933_review/readout_forward.py`（确定性；`--stage prereg|run|verify`）

- **特征缓存**：每个主干对 clean 512 场景、2728/2468 个单编辑状态、评估 3×quartet 状态统一前向，取 `feat_head` 输出（32 维）缓存到 `features/{task}_{class}_s{seed}.npz`（含 `mu/sd/whitening`）。主干冻结哈希（`net.*`+`feat_head.*`）逐主干记在 `results.json:backbone_hashes` 与 `rows[...]["backbone_sha256"]`。
- **特征标准化**：只用 **head-fit parent 的 clean 状态**拟合 mean/std（T1/T2 各 409 parent），冻结到 dev 与 eval；`whitened` 变体用同一批 fit 协方差（`W` 也存盘）。
- **head 拟合**：`logistic` = L2-logistic，L-BFGS-B 凸解，L2 ∈ {1e-4, 1e-3, 1e-2} 按 head-dev BCE 选；`mlp` = width ∈ {16, 64}，300 Adam 轮、每 10 轮按 head-dev BCE 选（与 static regime 同预算、同初始化种子）。
- **static / flip 交叉**：同一 head 初始化（logistic 零向量；MLP 固定 torch seed）、同一 clean 数据集，只加不减地加 flip 状态。
- **split**：新任务 train 512 场景按 `rng=4404` 确定性 80/20 分 parent → fit 409 / dev 103；单编辑状态跟随其 parent（T1：fit 2150 / dev 578；T2：fit 1862 / dev 606）。**eval 只用 head-dev 之外的 T1/T2 评估银行**。
- **阈值三分离**：`J` = 固定阈值 0；`J_head_dev_threshold` = 仅用 head-dev 单状态加权准确率选的阈值；`J*` = **用评估真值选的 oracle 公共阈值下的联合正确率，是诊断上界，绝不是可部署准确率**。
- **S 的记法**：一律写成 **S(f_head∘h)**，即"最终标量 score 的局部可分性"，是 encoder 与具体 head 的联合性质，不是主干固有属性（本 lane 直接给了证据：同一主干换 head/换特征空间，S 会变，见 §5.5）。
- **配对统计**：`paired(...)` 对同一 parent 内先取均值再对 parent 平均，parent-cluster bootstrap 2000 次（`rng=904`）给 95% 区间；只报告同一批 parent 的差，不做 seed/parent 分母膨胀。

---

## 5. 结果

### 5.1 pipeline 有效性核验（对照 f095 归档）

把本次"冻结主干 + 重拟合 head（static、标准化、logistic）"与 f095 归档的原始联合训练 head（`artifacts/f095_campaign/U10/U10_{T1,T2}_EVAL.json`，同样是 `threshold_certificate` 口径）对比，3-seed 均值：

| 单元 | 归档 S / J / J\* | 本次 S / J / J\* |
|---|---|---|
| T1 raw_clean | 0.6985 / 0.0797 / 0.1027 | 0.6988 / 0.0793 / 0.1030 |
| T1 raw_flip | 0.8371 / 0.2172 / 0.2497 | 0.8366 / 0.2313 / 0.2498 |
| T1 six_clean | 0.9164 / 0.1970 / 0.2928 | 0.9167 / 0.1946 / 0.2932 |
| T1 six_flip | 0.9842 / 0.7497 / 0.7916 | 0.9842 / 0.7483 / 0.7905 |
| T2 raw_clean | 0.8337 / 0.1475 / 0.1664 | 0.8338 / 0.1461 / 0.1673 |
| T2 six_flip | 1.0000 / 0.9982 / 0.9996 | 1.0000 / 0.9993 / 0.9995 |

差异 ≤ 2.5pp，说明重拟合 head 的复算是原 pipeline 的忠实近似（归档数字来自既有产物，非本 lane 重算）。

### 5.2 主结果表（标准化 + convex logistic，3-seed 均值）

分母：T1 2724 quartet / 322 parent；T2 3443 / 369。S/J/J\* 为 quartet 等权（`threshold_certificate` 默认权重），J_dev 为 head-dev 阈值在评估集上的联合正确率。

| 单元 | S static→flip | J\* static→flip | J static→flip | J_dev static→flip |
|---|---|---|---|---|
| T1 raw_clean | 0.6988 → 0.7170 | 0.1030 → 0.1156 | 0.0793 → 0.0669 | 0.0510 → 0.0859 |
| T1 raw_flip | 0.8366 → 0.8374 | 0.2498 → 0.2500 | 0.2313 → 0.2177 | 0.2414 → 0.2301 |
| T1 six_clean | 0.9167 → 0.9214 | 0.2932 → 0.3229 | 0.1946 → 0.2841 | 0.0920 → 0.3184 |
| T1 six_flip | 0.9842 → 0.9842 | 0.7905 → 0.7920 | 0.7483 → 0.7544 | 0.4574 → 0.7796 |
| T2 raw_clean | 0.8338 → 0.8475 | 0.1673 → 0.1669 | 0.1461 → 0.1506 | 0.0994 → 0.1515 |
| T2 raw_flip | 0.9925 → 0.9924 | 0.6794 → 0.6790 | 0.6632 → 0.6298 | 0.6578 → 0.6729 |
| T2 six_clean | 0.9976 → 0.9976 | 0.7582 → 0.7776 | 0.6952 → 0.7441 | 0.5081 → 0.7627 |
| T2 six_flip | 1.0000 → 1.0000 | 0.9995 → 0.9995 | 0.9993 → 0.9978 | 0.8921 → 0.9946 |

**开发拟合 / 评价分离的阈值结果（次级观察，不是预注册的三段之一）**：`J_dev` 只用 **head-dev 单状态**选阈值，再拿到评估 quartet 上算联合正确率。它与 `J*`（oracle 上界）的差距在部分单元很大：T1/six_flip 的 static 头 `J_dev`=0.4574 而 `J*`=0.7905（差 33pp），flip 头把 `J_dev` 提到 0.7796；T2/six_flip 由 0.8921 提到 0.9946（`J*`=0.9995）。即在主干已见过变化监督的关系型单元上，head 级 flip 干预几乎不动 S 也不动 J\*，却大幅改善了开发集选出的工作点。这条只作为观察量报告，不参与 P1–P5 判定，也不据此新增一个"第四段"。

### 5.3 逐段配对差（parent-equal，3 seed 均值；逐 seed 数值与区间在 `results.json:paired`）

| 单元 | ΔS | ΔJ | ΔJ\* | argmax\|Δ\| |
|---|---|---|---|---|
| T1 raw_clean | **+0.0188** | −0.0146 | +0.0121 | **S** ← P1 唯一失败单元 |
| T1 raw_flip | +0.0004 | −0.0141 | −0.0003 | J |
| T1 six_clean | +0.0071 | **+0.0638** | +0.0283 | J |
| T1 six_flip | 0.0000 | −0.0115 | +0.0011 | J |
| T2 raw_clean | +0.0113 | −0.0187 | −0.0015 | J |
| T2 raw_flip | −0.0001 | −0.0341 | −0.0003 | J |
| T2 six_clean | 0.0000 | **+0.0337** | +0.0119 | J |
| T2 six_flip | 0.0000 | −0.0016 | 0.0000 | J |

T1 raw_clean 逐 seed ΔS = +0.0339 / +0.0115 / +0.0110（parent bootstrap 95% 区间分别 [0.0192,0.0494] / [−0.0052,0.0268] / [−0.0055,0.0281]）——只有 1/3 seed 的区间不含 0，且该单元三段效应全部 < 2pp，是全场基线最弱的单元（J ≈ 0.07–0.08）。

### 5.4 预测判定（不改判据、不删单元）

| 预测 | 成立单元 | 判定 |
|---|---|---|
| **P1**（主要影响 J） | 7/8（失败：T1/raw_clean） | **NOT_SUPPORTED** |
| **P2**（\ΔJ\* 不是获益来源） | 8/8 | **SUPPORTED** |
| P3（six 单元 ΔS ≤ 0） | 3/4（失败：T1/six_clean，ΔS=+0.0071） | NOT_SUPPORTED |
| P4（six 单元 ΔJ > 0） | 2/4（失败：两个 six_flip 单元，ΔJ = −0.0115 / −0.0016） | NOT_SUPPORTED |
| P5（MLP 下归属随 encoder 类改变） | 5/8（six 部分 4/4 成立；raw 部分仅 1/4） | NOT_SUPPORTED |

**primary verdict = MISS**（P1 未全中）。

### 5.5 次生 pipeline（预先声明为 secondary，不进 primary 判定）

- **MLP head 家族**（同预算、同初始化规则）：P4 4/4 成立（six 单元 ΔJ = +0.0791 / +0.1715 / +0.0623 / +0.0033），但 P1 7/8、P2 7/8、P5 5/8。P5 的 six 部分（预测归属 J）4/4 成立，raw 部分（预测归属 S）只 1/4 成立——只有 T1/raw_clean 这个最弱单元在 MLP 下确实归属 S。
- **whitened 特征空间 + logistic**：与 primary **明显分歧**——P1 仅 4/8，且 T1/raw_clean、T1/raw_flip、T2/raw_flip 三个单元的 argmax 变成 **J\***；同时干预效应本身被放大（T1/six_flip ΔJ = +0.3484、T2/six_clean ΔJ = +0.1731）。
  → 直接证据支持"S 与分段归属都是 **S(f_head∘h)** 的性质"：**换特征空间参数化就换结论**。因此本 lane 的分段归属结论必须写明 pipeline 条件，不能当成 encoder 的固有属性。

### 5.6 solver 收敛证据（`solver_logs/*.json`，24 主干 × 2 空间 × 2 regime × 3 L2 = 288 个凸解）

- 288/288 `success=True`，最大最终梯度无穷范数 **5.39e-07**，迭代数中位数 28.5（范围 11–286）——凸对照（L2-logistic + L-BFGS-B）可靠收敛，不是只跑了 Adam 300 轮。
- MLP：96 个候选（24 × 2 regime × 2 width），head-dev 选择落在第 300 轮的有 **77/96**——说明 MLP 的 dev 选择经常顶到预算上限，MLP 结论只能限定在这两个宽度与这个预算内。

### 5.7 2×2 交叉：encoder 级 vs head 级干预（J，标准化 logistic，3-seed 均值）

| 任务/参数化 | 主干 clean + head static | 主干 clean + head flip | 主干 flip + head static | 主干 flip + head flip |
|---|---|---|---|---|
| T1 raw | 0.0793 | 0.0669 | 0.2313 | 0.2177 |
| T1 six | 0.1946 | 0.2841 | 0.7483 | 0.7544 |
| T2 raw | 0.1461 | 0.1506 | 0.6632 | 0.6298 |
| T2 six | 0.6952 | 0.7441 | 0.9993 | 0.9978 |

- **主干级变化监督是主导干预**（T1 raw 0.079→0.231，T1 six 0.195→0.748），head 级变化监督只在"主干是 clean 训练"的关系型单元上有实质增益（T1 six_clean +0.064、T2 six_clean +0.034），在主干已见过 flip 的单元上**没有增益**（six_flip 两个单元 ΔJ ≈ 0 或略负）。
- 在 raw 主干上，head 级 flip 干预的 ΔJ **4/4 为负**（−0.0141 ~ −0.0341）——与源任务旧 bank 上 raw_clean 的负 ΔJ 一致。
- 这是本 lane 最有信息量的正面发现：**两级变化监督是替代关系而非叠加关系**，且"干预落在哪一段"依赖 encoder 类别与 encoder 已有训练制度。

---

## 6. D10 下游回声（R09 全场景结果 × S/J/J\* 分解）—— 已完成

**判定（先说结论）**：分解对 R09 的结果**能解释形状与段位归属，不能解释份额，也不能预测幅度**；对 117/256 的 oracle 不可行份额是**结构性无法处理**（不是"暂时没算"）。本节的回声**不改变** §5.4 的 primary verdict（P1 7/8，MISS）。

### 6.1 输入（只读，r09 lane 产物，本 lane 未重算任何 D10 数字）

- `artifacts/e1a933_review/frontier/FRONTIER_FULLSCENE.json`（sha256 `818dfc64a42cfa0923c1737670bc21e36f52e2da688e767cd1e04b0ce1f543b7`；内部 `table_sha256`=`e7ee314b…`、`script_sha256`=`1cd535b5…`、`source_commit`=`e1a933e6…`）
- `artifacts/e1a933_review/frontier/FRONTIER_FULLSCENE_PERSCENE.csv`（sha256 `583f571fb937c96afc2939ec581a72879dfac043ff9bedcf87aa8fb172561ecf`，27,648 行逐场景记录）
- `reports/e1a933_review/DECISION_FRONTIER_V2.md` 的 R09-A / R09-B / R09-C / R09-D 节（本 lane **未修改**该文件）

### 6.2 被回声的 R09 数字（分母写清）

| 量 | 数值 | 分母 | 来源 |
|---|---|---|---|
| oracle 可行 / 不可行场景 | 139 / 117 | 256 场景（每场景 49 候选） | `FRONTIER_FULLSCENE.json:scene_universe` |
| oracle 不可行占比 | 45.7%（部署宇宙内 117/213 = 54.9%） | 256 / 213 | 同上 |
| dev 目标覆盖 → eval_full 实测覆盖 | 0.4884 → 0.1455–0.3239 | dev 43 场景；eval_full 213 场景；18 个 clean/flipmine 策略（b0.5） | R09-A 结论 1 |
| 同覆盖 oracle 点 − 部署点净收益 | ≤0.0093（多数 \|差\| ≤0.001） | eval_full 213 场景 | R09-B 诊断读数 1 |
| 取消 dev 端可行性筛选后的覆盖 | 0.3412–0.6059 | dev 86 场景；eval_rest 170 场景（**不可与 R09-A 直接比较**） | R09-C |
| b0.25 时不可行场景被动作数 | rel12/sixdist 0/117；raw 4–11/117（clean） | 117 个不可行场景 | R09-A 结论 2 |

（本 lane 抽验了 JSON 与上表一致，例如 `raw_s11` clean、b0.5：dev 覆盖 0.4884、τ=3.759、eval 覆盖 0.3005、动作 64、不可行场景动作 18/117、每场景净收益 0.0446。）

### 6.3 分解**能**说什么：覆盖塌缩在段位语言里是"工作点/覆盖段"问题，不是 S 段问题

1. **R09-B 的"同覆盖差距 ≤0.0093"与我的 P2 是同形的。** R09-B 说明：在**同一个工作点（同覆盖）**上，dev 选出的阈值已经基本达到 oracle 上界，oracle 多出来的收益几乎全部来自把覆盖从 0.15–0.20 推到 0.24–0.41，而不是阈值微调。我的 P2（`|ΔJ*| < |ΔJ|` 在 **8/8** 单元成立）说的是同一件事的另一面：变化监督干预动的是 J（固定阈值下的联合正确率），不是 J\*（oracle 阈值上界）。两边都指向"可达到的上界不是杠杆，工作点/覆盖才是"。
2. **R09-C 把归属进一步钉在"目标错位"上，与"段位归属随条件改变"一致。** 只取消 dev 端可行性筛选，覆盖转移就从 0.1455–0.3239 变成 0.3412–0.6059。这说明塌缩主要来自 dev/eval 分布错位（目标错位），而不是部署端缺少 oracle——也就是**分布与工作点**这一段的性质，而不是 score 排序能力的性质。这与我 §5.5 的观察同向："干预影响哪一段"随条件（特征空间、评测分布构造）改变，不是某个主干的内在属性。
3. **我的 `J_dev` 观察是跨任务的同形实例。** T1/six_flip 的 static 读出层：dev 选出的阈值给出 `J_dev`=0.4574，而同一 score 的 `J*`=0.7905（2724 quartet / 322 parent），flip 读出层把 `J_dev` 提到 0.7796 而 S、J\* 基本不动。也就是说"开发集选出的阈值在新分布上远低于可达到的上界"这个失败段，在 quartet 任务上独立存在。

### 6.4 分解**不能**说什么（含一条结构性理由）

1. **不能解释 45.7% 的 oracle 不可行份额——这是结构性不能，不是数据不足。** S/J/J\* 的定义域要求每个 case **至少有一个正状态和一个负状态**：生产函数 `docs/f095dfa_review_pack/checks/threshold_certificate.py` 对"全负状态"的输入直接抛 `ValueError: Each case must have at least one positive and one negative state`（本 lane 已实测复现）。oracle 不可行场景的 49 个候选**全部** `feas=False`，没有正状态，因此这些场景**根本不在分解的定义域内**。117/256 是场景宇宙（候选可行性契约）的性质，与任何 score 无关。→ 分解对"不可行份额"没有话语权。
2. **不得反推"概率/score 本身有问题"。** R09-B 的读数 1 正好给出反证：同覆盖下 dev 选出的阈值与 oracle 前沿的净收益差 ≤0.0093，说明在该工作点上 score 的排序已经接近可达上界。把"覆盖塌缩"读成"概率坏掉"会与 R09-B 直接矛盾。本 lane 的 P2（8/8）同样不支持"概率本身有问题"这类全称结论。
3. **不能在 D10 上"验证"分解，也不能预测它的幅度。** 两者对象不同：D10 是**同一场景内 49 个候选**的排序 + 动作/成本决策，我的分解是**单个状态**的标量 score 在 A/B/AB quartet 上的一致性；场景宇宙（256 个足球场景 vs T1/T2 的合成几何场景）与 parent 不共享，没有任何一个 D10 场景进入过我的分析。因此 0.1455–0.3239、0.4884、45.7% 这些数字**不能**被写成"分解预测到了"。
4. **不得把 R09-C 的 0.3412–0.6059 当作"分解修好了塌缩"。** 那是 r09 的敏感性分析（dev 端去掉可行性筛选），与读出层/分解无关。

### 6.5 一条未被本 lane 分析、但结构上值得注意的边界

在 b0.5 上，raw 编码动作了 18–25/117 个不可行场景，而 rel12/sixdist 动作了 0/117（R09-A 结论 2）。用分解语言说，这是"score 在可行/不可行候选之间不可分"的 S 型差异，看起来像编码器类的差异；但**本 lane 没有做这个分析**（不做新计算），而且可行性是**整场景共享**的属性（49 个候选同真值），任何逐候选阈值都无法把"不可行场景"单独摘出来，只能靠"拒动"——也就是回到覆盖轴。因此这条只作为边界记录，不作为结论，也不进稿。

### 6.6 本节对主判定的影响

**不改变。** primary verdict 仍是 **MISS**（P1 7/8，P2 8/8）。理由：P1/P2 判的是"head 级变化监督干预在 T1/T2 上动哪一段"，R09 是另一批数据、另一个对象上的决策前沿分析，两者没有共享场景/parent，R09 的结论既不能把 P1 的 7/8 补成 8/8，也不能推翻 P2。R09 只提供了**同形旁证**（上界不是杠杆、工作点/覆盖才是），并把"分解的定义域不覆盖无正状态的场景"这一边界显式化。

**matrix 状态**：O04 的"D10 下游回声未做"一项**至此闭合**（其余 O04 缺口此前已闭合）。本 lane 完成 ≠ 整个施工包完成。

---

## 7. 未完成与边界（诚实清单）

1. **primary 预测是 MISS，不是 hit**。失败单元只有 1/8（T1/raw_clean），且该单元三段效应都 < 2pp、ΔS 区间 3 seed 里只有 1 个不含 0；但判据是事前定的"全中"，**不做事后放宽**。因此可以说"分解能预测干预不会动 oracle 上界（P2 8/8）"，**不能说**"分解能预测干预主要动 J"。
2. **分段归属依赖 pipeline**：whitened 空间下 P1 只有 4/8 且三个单元 argmax 变成 J\*。任何"该干预影响 X 段"的句子都必须带 head 家族 + 特征空间条件。
3. **不是新鲜银行独立确认**：T1/T2 主干与评估银行都是 f095 campaign 既有产物；本 lane 只训练 head。不得用它加强项目主 claim，也不得写成"未见首读"。
4. **MLP 结论受限**：只有 width ∈ {16,64} 与 300 轮 Adam；77/96 个候选的 dev 最优落在最后一轮，说明预算可能不足，**不能**据此说"所有非线性 head 都如何"。
5. **typed 类被排除**：U10 的 `typed` 主干（任务适配结构）在发布的检查点里没有独立特征头（只有 `rho` 标量输出），纳入需要重新定义"冻结表示的边界"，属于新的设计决定，本 lane 未做。因此"任务适配表示"这条线在新任务上没有同口径的 encoder 类。
6. **单任务对**：只有 T1/T2 两个新任务、每个 3 个 seed；seed 共享同一任务与银行，**不是独立任务数**，区间为 parent bootstrap 描述性区间，未做多重比较校正。
7. **J\* 是诊断上界**：全部 J\* 数字都用评估真值选阈值，**不是部署准确率**；配对 J\* 区间固定样本选出的 oracle 阈值，只是描述性。
8. **S 不是 encoder 固有属性**：本 lane 的 S 一律写作 S(f_head∘h)；同一主干换 head 家族或换特征空间 S 会变（§5.5）。
9. **没有做的事**：context-conditioned 校正、更宽的非线性搜索、跨任务"未见组合"训练合同下的复现——均未做，也不在本 lane 的完成声明内。D10 回声已在 §6 完成，但它是**对 r09 结果的解读**（同形旁证 + 结构性边界），不是本 lane 在 D10 上的新计算。
10. **本 lane 完成 ≠ 施工包完成**。

---

## 8. 复现

```bash
cd /home/huyudi/012_conference/iclr2027
# 1) 事前预测（会拒绝覆盖已存在的 PREREGISTRATION.json）
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python \
  experiments/e1a933_review/readout_forward.py --stage prereg \
  --output artifacts/e1a933_review/readout_forward_FRESH_DIR
# 2) 执行（先校验预测封印与脚本 hash，再计算；拒绝覆盖已有 results.json）
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python \
  experiments/e1a933_review/readout_forward.py --stage run \
  --output artifacts/e1a933_review/readout_forward_FRESH_DIR
# 3) 校验预测完整性与"预测早于结果"
python experiments/e1a933_review/readout_forward.py --stage verify \
  --output artifacts/e1a933_review/readout_forward_FRESH_DIR
```

实际执行目录：`artifacts/e1a933_review/readout_forward_20260924/`（完整 stdout：同级 `readout_forward_20260924.stdout.log`）。
确定性核验：同一脚本在 `artifacts/e1a933_review/readout_forward_repro_20260924/` 重跑一次，`rows / paired / predictions / sample_hashes` **逐位相同**（最大行差 0.0），两目录的 `prereg_payload_sha256` 相同（预测文本未改；第二次运行仅用于确定性验证，主运行结果当时已存在）。
资源：单次运行墙钟 82 s / 92 s，CPU 峰值 ≤ 6 线程（`OMP/OPENBLAS/MKL=6`，`torch.set_num_threads(6)`）。

---

## 9. 对主 claim / PLAN 的影响

- 本项目主 claim 是"任务适配的表示决定变化监督能否转化成完整联合正确性"。本 lane 在**新任务**上给出一条**边界**：head 级变化监督只有在主干未见过变化监督时才对关系型表示有增益（T1 six_clean ΔJ=+0.064、T2 six_clean ΔJ=+0.034），主干已见过 flip 后 head 级干预不再增益——即"变化监督的收益在表示层与读出层之间是替代的，不是可加的"。这条观察**不改变** PLAN 的假设，属于对它的条件化补充。
- O04 的"分解是前瞻诊断"这一说法必须降级为：**"可以前瞻预测干预不会动 oracle 上界（P2 8/8）；不能保证预测干预主要动 J（P1 7/8）"**。审计方"不得继续使用的解释"里没有本 lane 新增的越界语句。
