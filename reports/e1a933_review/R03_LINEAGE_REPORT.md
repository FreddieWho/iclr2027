# R03 模型-银行血缘矩阵与增广状态审计

基准 commit `e1a933e6b32dd07c6895c7925cbccd9004da7f54`（HEAD 未变，本轮改动未提交）。
本 lane 范围：`docs/e1a933_review/repairs/R03.md`（等价于 `01_RESULT_CRITICAL_REPAIRS.md` 的 R03 章节）。
状态：**本 lane 范围内的两处缺口已补齐**——①历史增广的逐状态 exact/near 穷尽核对；②15 行"历史推断"训练来源的 manifest/hash 证据。
本轮新增（旧产物只读，未覆写任何 `artifacts/f095_campaign/`、`experiments/f095_campaign/` 结果）：

| 文件 | sha256 | 说明 |
|---|---|---|
| `artifacts/e1a933_review/data_lineage/AUGMENTATION_STATE_AUDIT.json` | `e722c156d7e7d817…` | 增广状态审计全文（口径、重建忠实性、9 个增广块 × 3 bank、检测力对照、逐命中明细） |
| `artifacts/e1a933_review/data_lineage/TRAIN_SOURCE_RESOLUTION.json` | `2a5fdacfce5c3563…` | 96 个 checkpoint 的训练来源与证据链 |
| `artifacts/e1a933_review/data_lineage/MODEL_BANK_OVERLAP.csv` | `ad7bb61e179f99de…` | 仅 `augmentation_exact_state_audit`、`train_source_resolution` 两列被填充；其余 13 列逐字节未变（脚本内断言，并已用 `--mode matrix` 全量重算复核） |
| `experiments/e1a933_review/data_lineage.py` | — | 新增 `--mode augmentation`（可复跑、确定性；两次运行产物 sha256 完全相同） |
| `experiments/e1a933_review/test_data_lineage_augmentation.py` | — | 10 条契约测试（阈值语义、bank 状态语义、mined 重建、命中清单） |

---

## 0. 既有结论（前一轮已完成，本轮未改动，仅译录以保上下文）

- 96 个 checkpoint × 3 个银行 = 288 行矩阵；父坐标由 `Qx_B − Qe_A` 重建并与原始 float64 源场景匹配（最大重建 Linf = 5.86e-8，来自银行 float32 序列化）。每模型四种口径（旧 parent 新编辑 / 从未见 parent / exact 状态重复 / near 重复）已逐行给出；`≤1e-6` 用于吸收 float32 序列化，`1e-6 < d ≤ 1e-4` 为 near 带。

|Bank|train size|seen-parent quartets|unseen-parent quartets|
|---|---|---|---|
|dev512|512|0|237|
|dev512|2048|0|237|
|dev512|8192|0|237|
|d03E|512|0|1838|
|d03E|2048|391|1447|
|d03E|8192|1838|0|
|d09fresh665|512|0|236|
|d09fresh665|2048|58|178|
|d09fresh665|8192|236|0|

- **对审计包自身的重要更正**：`d09_fresh` 的位置**逐位等于** `X16[512:]`，其整数 parent id 是**局部命名空间**。旧的 `parent<512` 排除规则误删了 18 个与 N 训练无重叠的 quartet；完整 N 评估保留全部 236 个，历史 n218 是"错误 ID 过滤敏感性子集"，不是经验证的去重。生产用 `d09_audit.py` 现按真实源坐标判定，前/后 J 见 `data_d09/D09_AUDIT.json`。D03 构建脚本的排除命名空间也已更正（供将来生成用）；既有银行保持不变。
- **dev512 就是 `eval_202`**；其 237 个 quartet 父场景在 N/4N/16N 坐标清单中全部未见。它仍是**已暴露的历史银行**，因此那里的结果属再分析。不声称"多模型种子 = 多个独立数据集"。
- **新 seed241933 池**：768 个原始 float64 父场景，在生成编辑**之前**切分为 256 dev / 512 eval；已对完整 16N 血缘做最近坐标排除。新 E 池 1229 个 quartet；保留父数与全部样本坐标存于 `data_o01/eval.npz`。冻结的旧 D01/U02 checkpoint 与全部新 O01 配方都在其上评估：`data_source_eval/FROZEN_AND_ALL_RECIPES.csv`。
- **跨银行更正**：旧 D03 排除规则用了同一错误命名空间。D03E 与 D09 实际共享 8 个 E 父场景，分别贡献 74 与 14 个 quartet。它们是两个银行，不是完全独立的父数据集（`D03_D09_CROSSBANK.json`）。
- 新池对 11 个历史场景池**无父坐标匹配 ≤1e-4**，对这里用到的 N/4N flip 训练池**无状态匹配 ≤1e-4**（`NEW_POOL_CROSSBANK.json`；注意该文件不是本脚本产出，其口径以文件内字段为准）。

## 1. 口径定义（exact / near）

坐标一律指**未归一化、扁平 [4,2] 的原始坐标**，距离用 **Linf**。

- **exact（同一点）**：`Linf ≤ 1e-6`。
  理由：同一个点在两条序列化路径上的最大差 = 池坐标 float64→float32 取整（实测 2.98e-8）+ 编辑向量 float32 取整（实测 0.0）+ 银行 float32 存储 / 父坐标重建误差（实测 5.86e-8），合计 < 2e-7。1e-6 是其 5 倍。
  注意：字节级 float64 hash 相等在"银行存 float32、训练侧算 float64"的边界上一般不可能成立，因此同时报告 `n_byte_identical`（两侧无损提升到 float64 后的字节相等条数，实测只占命中数的一部分，见 §3）。
- **near（近似重复）**：`1e-6 < Linf ≤ 1e-4`。
- **阈值保守性论证（本轮实测，不用任何"slot 相邻帧 p99"）**：`r02_search.candidates_for_scene` 的确定性候选网格（4 半径 × 8 方向 × 单端点/成对/全局）中，**任意两个不同候选编辑的最小 Linf 间隔 = 0.015**（在两个场景各 234 个网格候选上实测）。因此 1e-4 比"两个真正不同的设计样本"的最小间隔小 150 倍，1e-6 小 1.5e4 倍：这两个带只能捕捉"同一点的表示误差"，不可能把两个不同的设计样本误判成重复。
- **状态语义（本轮发现的一个陷阱，已写进代码与测试）**：银行 quartet 的 `Qx` 是**编辑后**状态；但单编辑银行的 `Sx` 是**起点/父**状态、`Se` 才是编辑（`experiments/p123_upgrade/build_bank.py:82`），被评估的原子状态是 `Sx+Se`。实测：`d09fresh665` 的 126295 条 `Sx` **全部**与 16N 池场景逐位相同（Linf=0），而 `Sx+Se` 与 16N 场景的最小 Linf = 0.0095。**任何直接用 `Sx` 做的"状态"比对，比的其实是父场景**（`AUGMENTATION_STATE_AUDIT.json → singles_start_state_diagnostic`）。

## 2. 增广状态审计（96 checkpoint × 3 bank）

**重建方式与忠实性**（`reconstruction_faithfulness`、`mined_count_crosscheck`）：

- 增广族只有四类：`clean`（无增广）、mined-flip（`mine_flips(seed=5)`，N=1534 / 4N=6390 / 16N=26120）、D02 `g8` 的 G8 角色置换轨道（用全轨道作 300 轮抽样的保守超集）、`repeat`（重复 batch，不产生新点）。
- N 池的 mined 集重建结果与落盘的 `artifacts/discovery_campaign/r04b_s11/mined.npz` **逐字节相同**（1534 条 edit、scene id、new label 全部相等，margin 最大差 8.3e-8 来自 float32 存储）。
- 4N/16N 的 mined 集**没有落盘**，只能重建；重建条数与所有相关 `run_manifest.json` 记录的 `n_mined_flips` 完全一致（6390、26120）。

**审计结果**：9 个增广块（3 池 × {mined_flip, g8_orbit_clean, g8_orbit_mined}），其中 5 个被至少一个 checkpoint 实际使用，共 **50412** 个增广状态；全部 9 块共 392412 个状态也逐一对三银行做了核对。96 个 checkpoint 中 **59 个**至少用一个增广块（其余 37 个是 clean-only，CSV 中记为 `NO_AUGMENTATION`，其输入状态由原有 `exact_static_state_duplicates` / `serialized_static_state_matches` 两列覆盖）。

| 增广块 | 状态数 | 被使用 | dev512 quartet / singles | d03E quartet | d09fresh665 quartet / singles |
|---|---|---|---|---|---|
| `N:mined_flip` | 1534 | 32 ckpt | 0 / 0（min 0.201 / 0.111） | 0（min 0.101） | 0 / 0（min 0.136 / 0.109） |
| `N:g8_orbit_clean` | 4096 | 6 ckpt | 0 / 0（min 0.132 / 0.052） | 0（min 0.092） | 0 / 0（min 0.106 / 0.089） |
| `N:g8_orbit_mined` | 12272 | 3 ckpt | 0 / 0（min 0.112 / 0.047） | 0（min 0.083） | 0 / 0（min 0.088 / 0.060） |
| `4N:mined_flip` | 6390 | 18 ckpt | 0 / 0（min 0.151 / 0.101） | 0（min 0.039） | **163** / 0（min 0.053 / **0**） |
| `16N:mined_flip` | 26120 | 3 ckpt | 0 / 0（min 0.122 / 0.065） | 0（min **0.0131**） | **761** / 0（min 0.032 / **0**） |
| `4N/16N:g8_orbit_*` | 16384 / 65536 / 51120 / 208960 | 0（该池从未用 `--aug g8`） | 0 / 0 | 0 | 0 / 0 |

- **三个 quartet 银行（即 `MODEL_BANK_OVERLAP.csv` 的分母）在全部 96 个 checkpoint 上均为 0 exact、0 near**。最小 Linf：dev512 = 0.1117、d03E = 0.0131、d09fresh665 = 0.0324（均在"不同设计样本"的量级上）。
- **唯一的阳性在 `d09fresh665` 的单编辑银行**，见 §3。
- **检测力对照（positive/negative controls，`detector_power`）**：把银行状态自己当查询 → 400/400 判 exact；整体 +5e-5 → 400/400 判 near；整体 +5e-4 → 0 条被标记；把一条 16N mined 状态人为植入 d03E 目标集 → 恰好 1 条 exact 且字节相同。即"0 命中"是**有检测力前提下的 0**，不是检测器失效。

## 3. 新发现：d09fresh665 单编辑银行对 4N/16N 增广训练输入存在重复

- **16N 池**：26120 条 mined-flip 训练状态中，**761 条**（2.9%）与 `bank_d09fresh665.npz` 的单编辑状态**逐位相同**（Linf = 0），对应 **761 条**银行条目（0.603% of 126295）、**127 个**父场景（24.8% of 银行抽样的 512 个父）。
- **4N 池**：6390 条中 **163 条**（2.6%）命中，对应 163 条银行条目（0.129%）、30 个父场景（5.9%）。
- 逐条明细核对：命中条目**全部是 `flip=1` 条目**（`y1≠y0`），且银行父 id + 512 == 训练池父 id 全部一致；家族分布 single 550 / pair 211（16N）。也就是说这些条目不仅是"同一点"，而且是**同一父场景、同一编辑、同一翻转标签**——即模型在训练中确实以该标签见过这个输入。
- 命中的产生机制：`d09fresh665` 的父场景取自 `d09_fresh = X16[512:]`（正是 16N 模型的训练场景），而银行与 flip 挖掘**共用同一个确定性候选网格**，所以只要银行抽到的 (父, 网格编辑) 恰好也是该父的一条 mined flip，状态就重合。
- **独立复算**：不经审计代码，用 float32 状态字节做一次直接 join，得到完全相同的 761 / 163 条，父场景一一对应（见 §5 命令）。
- **影响范围（必须写清）**：
  - 不影响三个 quartet 矩阵（0/711、0/5514、0/708）。
  - 只影响**使用 d09fresh665 单编辑银行**的统计，且只影响 **4N/16N 训练的那 33 个 checkpoint**（27 个 4N + 6 个 16N）；N 池模型对同一银行是 0 命中。
  - 影响面为 0.13% / 0.60% 的条目。`AUGMENTATION_STATE_AUDIT.json → sets["{4N,16N}:mined_flip"].banks.d09fresh665.exact_match_detail.details` 逐条给出 `bank_id`（`S<eid>`），可直接用于排除敏感性复算（该复算属其他 lane 的执行范围，本 lane 不代跑）。
- 按合同，这里**不**把它升格为"方法"或"结论反转"，只作为需要披露并在使用单编辑银行时做排除检验的血缘事实。

## 4. 15 行"历史推断"训练来源：逐行解决

原矩阵中 `train_source_resolution == "source-family documented N history; no embedded data hash"` 的 15 个 checkpoint（对应 45 行 = 15 × 3 bank）**全部解决**，证据为三条独立腿：

1. **运行 receipt**（`.pi/tasks/**/*.json` 的真实命令）——权威 receipt 已逐个读过命令与 `.output` 日志（`saved <run dir>` / `TRAIN_DONE`）；
2. **脚本字面量**——训练脚本中加载训练数据的源码行；
3. **checkpoint 内嵌指纹**——`model.pt` 里的 `mu/sd` 是训练时从该池现算并写入的；用同一代码路径重算，**位级完全相同**，且在 3 个候选池（+`eval_202`）中唯一。

| checkpoint | 解析出的训练来源 | 权威 receipt | 脚本字面量 | 指纹 |
|---|---|---|---|---|
| `discovery_campaign/r04b_s11/{clean,flipmine,fliprand,supmine,auxmargin,relfeat}` | `scenes/train_101/scenes.npz` | `.pi/tasks/session-2971751-2971751/bae255145.json`（`--train $S/train_101`） | `r04b_methods.py:79 d = np.load(a.train / "scenes.npz")` | raw8/rel12 唯一命中 N |
| `discovery_campaign/r04b_s23/{clean,flipmine}` | 同上 | `.pi/tasks/session-2971751-2971751/b0d835971.json`（`for sd in 23 47 … --output $R/r04b_s$sd`） | 同上 | raw8 |
| `discovery_campaign/r04b_s47/{clean,flipmine}` | 同上 | 同上 | 同上 | raw8 |
| `discovery_campaign/r04b_s23_relfeat/relfeat` | 同上 | `.pi/tasks/session-2571586-2571586/b28952361.json` | 同上 | rel12 |
| `discovery_campaign/r04b_s47_relfeat/relfeat` | 同上 | `.pi/tasks/session-2571586-2571586/b83386ed0.json` | 同上 | rel12 |
| `next_novelty/relflip/s11` | 同上 | `.pi/tasks/session-2571586-2571586/bb8c3f344.json`（`relflip_train.py --seed 11`） | `relflip_train.py:32 d = np.load(ART/"scenes"/"train_101"/"scenes.npz")` | rel12 |
| `next_novelty/relflip/s23` / `s47` | 同上 | `…/bed44b19c.json` / `…/b447e5cec.json` | 同上 | rel12 |

**关键等式**：`artifacts/discovery_campaign/scenes/train_101/scenes.npz` 与 `artifacts/f095_campaign/D01/scenes_N/scenes.npz` **逐字节相同**（sha256 `620ea1dc1e443dc6c238317463dc64ab3a9ae05fb592743f3a7d67d7ce17262c`），所以 15 行的训练来源就是 N 池，与 D01/w512 等 N 池 run 共用同一份数据（同一份 augmentation：1534 条 mined flip）。

**不可解决项：无。** 但有一条证据强度限制：receipt 位于本地 `.pi/tasks/`（**未纳入版本控制**）。若该目录缺失，这 15 行只剩"脚本字面量 + checkpoint 指纹"两条腿（仍足以定位到 `train_101`，但少了"实际执行的命令"这一环）。

## 5. 全矩阵训练来源升级

- **96/96** 个 checkpoint 的 `mu/sd` 指纹都在 {N, 4N, 16N} 中**唯一**命中：**63 N / 27 4N / 6 16N**；4N/16N 与各自 `run_manifest.json` 的 `train_dir`+`data_sha256` 完全一致（243 行）。
- 因此 `MODEL_BANK_OVERLAP.csv` 的 `train_source_resolution` 列现在全部是 `resolved: …; ckpt mu/sd bit-exact`，不再有"父系推断"字样。

## 6. 复现命令

```bash
cd /home/huyudi/012_conference/iclr2027
# 矩阵（重建 13 个原有列；输出到全新目录，不覆写现场）
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 python3 experiments/e1a933_review/data_lineage.py \
  --mode matrix --out-dir /tmp/r03_matrix
# 增广审计（就地填充 MODEL_BANK_OVERLAP.csv 的两列，并写两个新 JSON；确定性）
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 python3 experiments/e1a933_review/data_lineage.py \
  --mode augmentation --out-dir artifacts/e1a933_review/data_lineage
# 契约测试
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 python3 -m pytest \
  experiments/e1a933_review/test_data_lineage_augmentation.py -q
```

独立复算 §3（不经审计代码，float32 状态字节直接 join）：

```bash
python3 - <<'EOF'
import numpy as np, json, sys, hashlib
sys.path.insert(0,'experiments/discovery_campaign')
from r04b_methods import mine_flips
b=np.load('artifacts/p123_upgrade/bank/bank_d09fresh665.npz',allow_pickle=True)
Sx,Se=b['Sx'],b['Se']; sm=json.loads(str(b['Smeta']))
bank=(Sx+Se).reshape(len(Sx),-1)
X16=np.load('artifacts/f095_campaign/D01/scenes_16N/scenes.npz')['positions'].astype(np.float32)
y16=np.load('artifacts/f095_campaign/D01/scenes_16N/scenes.npz')['labels'].astype(int)
m=mine_flips(X16.astype(float),y16,seed=5)
ii=np.array([x['scene'] for x in m]); ee=np.array([x['edit'] for x in m],dtype=np.float32)
tr=(X16[ii]+ee).reshape(len(ii),-1)
key=lambda A:[hashlib.sha256(np.ascontiguousarray(r).tobytes()).digest() for r in A]
t={k:i for i,k in enumerate(key(tr))}
hits=[(i,t[k]) for i,k in enumerate(key(bank)) if k in t]
print(len(hits),'bank entries hit;',len({sm[i]['parent'] for i,_ in hits}),'distinct bank parents;',
      'all flips:',all(sm[i]['flip']==1 for i,_ in hits),
      'parents agree:',all(sm[i]['parent']+512==int(ii[j]) for i,j in hits))
EOF
# -> 761 bank entries hit; 127 distinct bank parents; all flips: True; parents agree: True
```

---

## 7. 未完成与边界

1. **本 lane 完成 ≠ 施工包完成**。R03 只是 `PACKAGE_COMPLETION_MATRIX.csv` 的一行；其他 lane（R09、O01、O02、O04、R10、N02 等）的状态不由本报告背书。
2. **不得声称"全历史数据独立"**。本报告能支持的只是：*针对这三个评估银行、在 [4,2] 原始坐标的 Linf 口径下、对几何 quartet 任务的全部可重建历史增广族，除了 §3 列出的 761/163 条 d09 单编辑条目外，没有 exact 或 near 状态重复*。这既不等于"独立数据集"，也不等于"标签/父场景都新"（父场景重叠已由 `old_parent_new_edit_quartets` 单独给出：d03E/d09fresh665 对 16N 模型是 1838/236 个 quartet 全为已见父）。
3. **未审计的历史训练来源**（列明，不是"查不到"）：
   - 非几何 quartet 任务：T5R3 足球（`r04b_t5r3_*`、`r04c_t5r3_*`，输入是 phase/zone 特征，`base_ckpt` 来自 `phase3`）、U10 T1/T2（不同任务池，已在 `SOURCE_NAMESPACE_ASSERTIONS.json` 用父场景最近距离 0.144/0.067 单独断言）、视觉线（D04/U06 的图像输入）、足球/`p3d`/`l007` 系列。这些来源的训练输入与三个银行的坐标不在同一空间，不构成本审计的对象。
   - 几何任务上**已纳入**的额外来源：`discovery_campaign/r04_*` 的 DRO 对抗扰动（`balanced_sign_patterns((8,),seed=7)` × eps=0.1 × 2 臂 × 35 模式 = 35840 状态）与独立符号 `unif` 臂（`rng(seed+1000)`，35840 状态）、`r04c_*` 的 `unifmatched`（`rng(seed+31337)`，每种子 1534 状态）。结果同样 **0 exact / 0 near**（`extra_geometry_sources`）。
4. **审计口径本身的边界**：只看**未归一化原始坐标**的 Linf。不覆盖：(a) 归一化/特征空间里的重复（例如两个不同坐标在 rel12 特征下完全相同）；(b) 其它距离度量；(c) 除"状态相等"之外的间接泄漏（如同一父场景、同一标签的其它编辑）。这些不是本 lane 的合同项，但不应被"0 命中"读成"没有任何血缘联系"。
5. **4N/16N mined 集是重建而非落盘证据**。忠实性由两点支撑：N 池逐字节复现 + 4N/16N 条数与 manifest 完全一致。残余风险：若挖掘代码或 numpy 版本自训练后改变，重建细节可能有偏差（种子固定、条数完全一致，风险很小但非零）。
6. **`.pi/tasks` 未入版本控制**，§4 的 receipt 腿在 clone 后不存在。
7. **本报告未做任何新训练、未改任何旧结果**；§3 的排除敏感性复算未执行（属其他 lane）。
