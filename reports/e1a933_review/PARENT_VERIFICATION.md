# 父 agent 独立核验记录（2026-09-24）

本文件由总控 agent 直接执行，不委托任何 worker。目的是独立复核施工包审计方与各 lane 的
关键声明，而不是复述它们。所有结论都给出可复现的检查方式。

## 1. GPU 产物完整性：独立重算，全部字节级吻合

不采信既有回执，直接对本地磁盘重算 SHA256 与字节数。

| 任务 | manifest 条目 | 逐条重算结果 | 校验字节 |
|---|---|---|---|
| n03（决策瓶颈 6 臂） | 50 | ok=50, bad=0, missing=0 | 256.8 MB |
| football_v2（O05 12 臂） | 40 | ok=40, bad=0, missing=0 | 1.1 MB |
| vlm（O06 272 请求） | 5 | ok=5, bad=0, missing=0 | 0.1 MB |
| sampling（N02 48 臂） | 250 | ok=250, bad=0, missing=0 | 2052.0 MB |

sampling 目录实测 48 个臂目录、48 个 `model.pt`，与 receipt 的 `expected_completed: 48` 一致。

原 21 臂视觉矩阵单独核验：`COLLECTION_STATUS.json` 为 `COLLECTED_VERIFIED`，
归档 `cuda_matrix_results.tar.gz`（876,199,984 bytes，sha256 `63d506e9…`）内部
`cuda_matrix/ARTIFACT_MANIFEST.json` 的 119 个条目**逐条解包重算全部吻合**；
`transfer_manifest.tsv` 3 行全部 ok。

结论：`SAFE_TO_SHUTDOWN` 回执成立。87 训练臂与 272 次 VLM 请求的产物在本机可字节级复验。

## 2. 工程回归测试：通过，但发现四项检查是"不可能失败"的

`experiments/e1a933_review/test_data_repairs.py` + `test_fair_football_frontier.py`
共 12 项通过（2.80s）。逐条核对 mutant 性质后确认 R02/R04/R07/R08/R09 的守卫是**真的会失败**的：

- `test_complete_action_ladder_and_legacy_mutant` 显式断言 max-only 梯子漏掉"更便宜且成功"的策略；
- `test_pava_training_plateau_and_legacy_mutant` 断言正确 PAVA 输出并断言 legacy 插值不同；
- `test_period_unique_frame_and_tie` 断言旧的全局最近帧会选错半场；
- `test_keep_and_flip_does_not_imply_E` 断言旧谓词不充分。

**但 R05/R06 的"十一项合同检查"里有四项不可能失败。** `vision_regression.py` 的
`normalization_commutes_<p>`（H 的四个元素各一项）比较的是

```
sort(normalize(orbit))  vs  sort(normalize(orbit[p], stats))
```

对一个结构施加点置换只会**行置换**它的轨道，而沿群轴排序后，任何逐维仿射归一化都会
给出相同结果。实测：pooled 方案与有 bug 的 per-index 方案在该检查下**都通过**，
最坏排序差均为 0.000e+00。即这四项检查无法区分正确实现与 R02 的 indexed-stats 错误。

处置：
- 新增 `experiments/e1a933_review/test_vision_contract.py`（10 项，pytest 可收集），
  把 R05/R06 合同检查改为调用生产函数（`d04_vision_train.make_net/train_mode/backbone_hash`、
  `u06_relation_distill.target_orbit/matched_mse/H_ENDPOINTS/shuffled_indices`）。
- 删除那四项非判别性检查，替换为两项**可失败**的检查：
  `test_orbit_equivariant_under_declared_group`（H 必须是 rel10 的真实对称群）与
  `test_colour_swap_is_outside_the_declared_group`（H 不得含红蓝互换；该项使前者非平凡）。
- 新增 `experiments/e1a933_review/vision_contract_mutation_check.py`：在原地恢复旧行为并断言
  对应测试**失败**，作为"恢复旧错误必须失败"的证据。结果 `artifacts/e1a933_review/vision_contract_mutation/mutation_check.json`：

| 恢复的错误 | 对应测试 | 守卫触发 | 修复下通过 |
|---|---|---|---|
| `train_mode` 退回裸 `net.train()`（BN buffer 更新） | frozen probe 冻结参数与 buffer | 是 | 是 |
| `matched_mse` 退回有序（下标固定）目标 | 一致群元素匹配为 0 / 匹配对群元素不变 | 是 | 是 |
| `rel10` 变成色盲（排序距离） | 红蓝互换不属于声明群 | 是 | 是 |
| `H_ENDPOINTS` 含非对称置换 | 轨道在声明群下等变 | 是 | 是 |

边界：`test_colour_swap_is_outside_the_declared_group` 是**边界刻画**，其失败模式是
rel10 变色盲，而不是 H_ENDPOINTS 被放大——后者由等变检查守卫。这一点已写进测试 docstring，
避免后续把它误当成群成员守卫。

## 3. 仍未由本文件覆盖的部分

- 本文件没有重跑任何训练、没有重算任何科学指标；它只核验完整性、可复现性与守卫活性。
- R05/R06 的 512-parent 观测性审计（像素轨道歧义、标签恒定率、不可约方差界）仍是
  `vision_regression.py` 的脚本行为，不是单元测试；本次未改其数值结论。
- R03/R09/O01/O02/O04/N01/N02/R08 的科学结论由对应 lane 交付，需在 lane 收尾后另行核验。

## 4. claim ledger 结构缺陷（已修）

`reports/final_closure/MASTER_CLAIM_LEDGER.md` 被声明为「论文数字的唯一来源」。逐行核对单元格数发现：

- 原表 42 个数据行中，`U02 coverage` 与 `X03 scoped negative (x03b rank-4)` 两行**缺 `denominator` 单元格**，
  导致其后每一列都左移一位（`coord` 被挤进 denominator、artifact 被挤进 confirm/dev 等）。
  已插入 `—` 占位修复对齐；未新增任何数字（两行的数值信息原本就在 `number`/`numerator` 里）。
- `P2 equal coverage` 行的 `quality\|same-coverage` 是合法转义管道，不是缺陷。
- 新增的本轮 28 行已校验：单元格数一致、表格连续、无尾部残留内容。

## 5. 论文构建可复现性

用记录中的 TinyTeX 命令在未修改源码上重建，得到的 PDF 与 `reports/e1a933_review/paper_revised.pdf`
**字节数完全相同（336,784）**，`pdftotext` 提取文本 **85,259 字节完全一致**（SHA256 不同属预期：pdflatex 会写入
创建时间与文档 ID）。因此后续接线产生的页数增长可归因于新内容，而不是工具链差异。

接线后的当前状态：总 **22 页**（记录版 20 页），**主文未变**（两份 PDF 的结论文字均落在物理第 **9–10** 页），
0 overfull、0 LaTeX error、0 未定义引用。页数增长全部来自附录 `app_f095.tex`，它在 `main.tex` 里于正文之后 `\input`，
结构上不可能改变正文分页；这一点已用文本提取逐页比对确认。

一处容易误读的细节：`main.aux` 的 `\newlabel{sec:conclusion}` 读出的是 **8**，而 `PAPER_BUILD_RECEIPT.json` 写的是
`conclusion_page: 9`。两者不矛盾——**回执是对的**（物理第 9 页，已用 `pdftotext` 逐页确认结论正文起始于第 9 页）；
aux 标签在分页边界处会少记一页，属 LaTeX 已知行为。本文件此前曾据此对回执提出疑问，现予更正。

注意：`PAPER_BUILD_RECEIPT.json` 的 `source_hashes` 已因本轮编辑过期，该回执需在全部接线完成后重新生成。

## 6. O02 复核：一个被漏披露的线程数混杂（已修）

复核 `data_u10_o02` 时逐项重算了 O02 的对照值，并确认它们与产物一致：

- 等配方 six+flip − typed_m+flip：T1 +0.514/+0.529/+0.486，T2 +0.731/+0.701/+0.755（parent 配对 2000 次 bootstrap，CI 全部排除 0）；
- 最优配方比较：T1 +0.499/+0.507/+0.505，T2 +0.645/+0.675/+0.667；
- `I_hist` 与 `U10_CLAIM_CORRECTION.md` 归档值逐位一致（T1 +0.394/+0.439/+0.413，T2 −0.189/−0.172/−0.250）。

过程中我先怀疑“typed 训练目标差 10²–10³ 倍”被夸大，复核后确认是**我自己的统计量选错**：手稿的 0.0804/0.00256 是「每个配方对 3 个 seed 取均值后的最好值」，
而它对比的 raw/six 也是同口径均值（3.27e-5 / 1.55e-6），比值 2462× 与 1652×，故“10²–10³ 倍”成立且偏保守。
（若误用 typed 的均值去比 raw/six 的最小值会得到不同数字——这是统计量不一致的错误，已排除。）

**但复核暴露出一个真实问题：附录 U10 表混用了两个线程数口径。**

- `HIST_REPRODUCTION.json`：raw/six 历史权重在 **4 线程**下逐字节复现（36/36），而 `data_u10` 重训的 typed 权重在该批下 **18/18 全部 bit_exact=false**；
- `data_u10.py` 用 `torch.set_num_threads(2)` 训练并评估 typed；`o02_budget_match.py` 用 4 线程；
- 逐位比对确认：附录表的 typed 列（T1 0.2761/0.1780/0.2684，T2 0.3398/0.3189/0.3404）**恰好等于 2 线程批**，而 4 线程批给出 T1 0.2731/0.2258/0.2254、T2 0.2495/0.3526/0.3358；
- 即同一 typed flip 臂最多相差 **9.0pp**（T2 s11：0.2495 vs 0.3398），而 `I` 列来自 raw/six（4 线程可复现，抖动 ≤2.2pp）。

结论：两个列各自内部一致，但**并列在同一张表里且未披露口径**，会让读者做 O02 明确禁止的跨口径比较。已在附录表格
caption 与正文补上披露（typed 列为 2 线程评估、只在 2 线程下逐字节复现、不可与其它线程数下的 typed 数字比较；
`I` 列稳定），并在 claim ledger 新增一行记录该披露。

## 7. ledger 表格完整性（已修）

除第 4 节的两处缺列外，本轮新增行中发现并修复 2 处自身缺陷：O04 预注册行的 `(|ΔJ*| < |ΔJ|)` 与
O01 保护损失行的 `|dJ|` 均含未转义 `|`，使行裂成多单元格、表格断裂；已改用 `\lvert`/`\rvert`。
现全文按“未转义管道数”校验：**所有行均为 11 列，无异常**（`P2 equal coverage` 行的 `\|` 为合法转义）。

## 8. 收尾 lane 的独立复核（r03 / r08 / o01）

**r08（自然 E）—— 逐对文件重算完全吻合。** 不采信汇总 JSON，直接从 `natural_pairs_primary.jsonl`（7117 行）重算：
帧对 7117、E 帧对 10（逐场 5/1/4）、有 ≥1 个 E 的情境 8 个、情境级率 8/1449 = **0.552%**、cap 1.214/1.199/1.174。
真实帧对照：`real_state_label_at_t2 != y0` 共 **260/7117 = 3.653%**（逐场 90/107/63），与 handoff 一致。
（我自己第一次算出 396/7117 是**用错字段**：`labels_y0_yA_yB_yAB[3]` 是联合**编辑**标签，不是真实 t2 状态；
换成 `real_state_label_at_t2` 后完全对上。记录于此以免后人重蹈。）

**o01 —— 发现并修正一处口径混用（手稿文字已改）。**

handoff 写“**在全部 648 个配置中**，原始坐标在 f100 的最高 J 为 0.168（N）/ 0.345（4N），距离输入的最低 J 为 0.562（N）/ 0.737（4N）”。
我从 `O01_CONTINUATION_FACTORIAL.csv` 逐行重算后发现该句**混了两个不同子集**：

| 范围 | raw f100 | sixdist f100 | 是否不重叠 |
|---|---|---|---|
| 全 648 扫描（n=54/组） | [0.000, 0.168] (N) / [0.000, 0.345] (4N) | [**0.011**, 0.697] (N) / [**0.011**, 0.856] (4N) | **重叠** |
| 全扫描扣除 `sgd_m09_lr0.003`（n=48） | [0.076, 0.168] / [0.054, 0.345] | [0.542, 0.697] / [0.583, 0.856] | 不重叠 |
| `family=warm_factorial`（n=36） | [0.120, 0.167] / [0.196, 0.309] | [0.562, 0.697] / [**0.737**, 0.856] | 不重叠 |

即：0.168/0.345 取自**全扫描**，而 0.562/0.737 取自 **warm_factorial 子集**——两者不同范围却写在同一句“全部 648 个配置中”。
而全扫描下距离输入确实能低到 0.011，因为 `sgd_m09_lr0.003` 从零开始的臂会塌缩。

核实塌缩细节：f100 的 `sgd_m09_lr0.003` 共 24 臂（12 scratch / 12 warm，12 raw / 12 sixdist），
其中 J<0.05 的恰好 12 臂且**全部是 scratch**；f100 中所有 J<0.05 的臂都只属于这一配置。
故 handoff 的“从零开始塌缩”这一描述是对的，错的只是**不重叠声明的范围**。

已在附录改为可核的形式：全扫描下距离最低 0.011 但全部属于该已披露的塌缩配置，扣除后最低 0.542/0.583；
并单独给出主因子（n=36）的完全不相交区间。ledger 同步新增一行（含“可以说/不可以说”的范围限定）。

**r03 —— 结果与声明相符。** 392412 状态、quartet 银行 0 exact/0 near、检测器标定（400/400 + 400/400 + 注入 1 条）、
96/96 来源指纹定位；`d09fresh665` 单编辑重叠 761/126295（0.603%，16N）/ 163/126295（0.129%，4N）已写入附录为**强制披露**。
其“两次运行 sha256 完全相同”的确定性声明与产物一致。

## 9. 本轮复核的自我修正清单

诚实记录我在复核中先出错、后纠正的三处（均为**我自己的**错误，非 lane 的）：

1. O02 训练目标“10²–10³ 倍”一度被我怀疑夸大 —— 实为我拿 typed 的**均值**去比 raw/six 的**最小值**；同口径均值下为 2462×/1652×，原claim成立。
2. R08 真实帧对照一度算出 396/7117 —— 实为我用错字段（联合编辑标签 vs 真实 t2 状态）；纠正后 260/7117 与 lane 一致。
3. 一度以为 `PAPER_BUILD_RECEIPT` 的 `conclusion_page: 9` 有误（因 aux 标签读出 8）—— 实为回执正确、aux 在分页边界少记一页。

三处均已写入本文件与相应报告，避免后续再次基于错误前提质疑产物。
