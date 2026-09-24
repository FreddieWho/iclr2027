# R08 补验：自然 E 发生率（真实比赛坐标）

状态：自然 E 发生率已独立完成（本文件）；受控搜索可构率沿用 R08 只读产物；旧统计更名与可复现性见第 7 节。本 lane 的完成不等于整个施工包完成。

## 0. 一句话结论

在同一条几何 oracle、同一批 1449 个合法情境上，把两个作用变量换成真实观测位移后，自然 E 情境率为 0.55%（8/1449），而受控搜索在同一批情境上的可构率为 46.24%（670/1449），富集约 84 倍。自然 E 存在，但属于稀有压力测试，不是常见情形。

## 1. 这条补的是什么

施工卡 R08 要求三个分母分开。原产物只有「搜索可构率」：在 cap 内用解析法向引导（49 对）＋随机对照（128 对）**搜索** witness，得到 670/1449。
搜索可以沿法向反复试探并二分边界，因此它是「在给定 cap 内是否存在 witness」的构造率，不是「自然情境中是否出现 E」的发生率。
本文件把同一条 oracle、同一批合法情境上的两个作用变量改成**真实观测值**：
A = 接球候选者在后续真实帧上的位置；B = t1 时刻真正约束通道的防守者（身份固定）在同一后续真实帧上的位置；传球者与其余 10 名防守者保持 t1 坐标。
四个格子中只有 A-only、B-only 是反事实，联合格是两个**真实单人位移**的交叉；没有任何引导、随机搜索或 cap 放大。

## 2. 分子/分母定义（可核）

- 情境集合：`artifacts/e1a933_review/football/events_candidates.jsonl`，1449 条记录（240 个抽样传球事件中通过角色/时间审计的 162 个事件内全部同队 2–45m 场内接球候选）。载入时逐条用标量 oracle 复核 base 标签与 defender_index：不一致 0 / 0 条。
- 自然对：同一 match+period 的真实帧对 (t1,t2)，dt 落在窗口内，接球者与防守者 k 在 t2 都真实存在（身份链接、坐标有限），两人位移都 ≤ cap，且移动后位置仍在场内（|x|≤52.5, |y|≤34）。
- cap 合同：speed_p95*W + 0.5*accel_p95*W^2 from same player/match/period consecutive diffs 0<dt<=0.12s (R08 motion_cap formula)。替代 cap：directly measured identity-linked same-period displacement p95 at lag W。
- 方向：later real frames only (0<dt<=W)；past and later real frames (0<|dt|<=W)。
- 主格（开工前指定，非事后挑选）：窗口 0.2s、合同 cap、forward。
- E 判定：y(x+eA)=y(x+eB)=y(x) and y(x+eA+eB)!=y(x), label-agnostic (y0=0 works too)。
- 真实观测 vs 解析计算：x and both real single-player moves are real observations; the A-only, B-only and joint cell labels are model-independent analytic oracle evaluations; y_real uses the real positions of passer/receiver/all eleven defenders at the later frame。
- 三个分母：(i) 自然 E 率，分母 = 全部合法情境 1449（另给实际被选中接球者的子集）；(ii) 候选抽样富集率，分母 = 被抽样的合法候选 1449，分子 = 受控搜索找到 witness 的候选 670（沿用只读产物）；(iii) E 条件率，分母 = 两个单编辑都保持 base 的编辑对，分子 = 其中联合编辑翻转的编辑对。

## 3. 主表：三个分母（逐场，主格）

| 比赛 / 用途 | 情境数 | (i) 自然E情境率 | (i) 自然E事件率 | (i) 自然E率(实际被选中接球者) | (ii) 搜索可构率 | 富集比 (ii)/(i) | (iii) 搜索E条件率 | (iii) 自然E条件率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| J03WOY / train | 574 | 0.70% (4/574) | 6.15% (4/65) | 0.00% (0/59) | 44.25% (254/574) | 63x | 2.32% (2026/87372) | 0.19% (5/2630) |
| J03WMX / dev | 621 | 0.16% (1/621) | 1.47% (1/68) | 0.00% (0/67) | 45.41% (282/621) | 282x | 2.20% (2059/93705) | 0.04% (1/2788) |
| J03WN1 / test | 254 | 1.18% (3/254) | 10.34% (3/29) | 0.00% (0/29) | 52.76% (134/254) | 45x | 2.74% (981/35847) | 0.37% (4/1067) |
| **合计** | 1449 | 0.55% (8/1449) | 4.94% (8/162) | 0.00% (0/155) | 46.24% (670/1449) | 84x | 2.34% (5066/216924) | 0.15% (10/6485) |

- (i) 的自然 E 情境率 95% 事件簇 bootstrap 区间（按事件重抽，事件是抽样单位）：J03WOY 0.17%–1.41%；J03WMX 0.00%–0.49%；J03WN1 0.00%–2.41%。
- (i) 的自然 E 事件率 95% 区间：J03WOY 1.54%–12.31%；J03WMX 0.00%–4.41%；J03WN1 0.00%–20.69%。
- (ii) 与 (i) 的分子不同、分母也不同：(ii) 的分母是 1449 个被抽样的合法候选，(i) 的分母是同一批情境；两者都用同一批情境与同一条 oracle，差别只在编辑生成方式（搜索 vs 真实位移）。
- 富集比 = (ii)/(i)，即同一批情境上「搜索能找到 witness」与「自然位移里真的出现 E」的比值；它说明搜索是构造性的，不能反过来当自然发生率。
- 阈值裁决：无。本文件不设任何「必须够大」的门槛，也不把低自然率写成不可行；低自然率读作稀有压力测试，分母逐场给出。

## 4. oracle 布尔分解（主格真值表，逐场）

| 比赛 | 窗口内真实帧对 | 入 cap 对 | t2 缺测排除 | 场内排除 | 超 cap 排除 | y0yAyByAB 计数 |
|---|---:|---:|---:|---:|---:|---|
| J03WOY | 2870 | 2821 | 0 | 0 | 49 | 0000:237 0001:4 0010:25 0011:31 0100:22 0101:7 0110:4 1000:9 1010:10 1011:19 1100:46 1101:18 1110:1 1111:2388 |
| J03WMX | 3105 | 3053 | 0 | 0 | 52 | 0000:264 0010:17 0011:100 0100:29 0101:9 0110:2 1011:20 1100:65 1101:23 1110:1 1111:2523 |
| J03WN1 | 1270 | 1243 | 0 | 0 | 27 | 0000:157 0001:4 0010:20 0011:38 0100:15 0101:1 0110:1 0111:17 1000:1 1011:15 1100:52 1101:16 1111:906 |

E 格是 y0=yA=yB 且 yAB≠y0 的模式（1110 与 0001）；其余 14 格是单编辑即翻转、或联合也不翻转。

## 5. 窗口 / cap / 方向敏感性（并列报告，不做取舍）

| 比赛 | 窗口(s) | cap(m) | 方向/cap规则 | 入 cap 对 | 自然E对 | 自然E对率 | 自然E条件率 | 自然E情境率 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| J03WOY | 0.2 | 1.214 | forward/contract | 2821 | 5 | 0.18% | 0.19% | 0.70% |
| J03WOY | 0.2 | 0.931 | forward/empirical | 2663 | 4 | 0.15% | 0.16% | 0.52% |
| J03WOY | 0.2 | 1.214 | both/contract | 5649 | 12 | 0.21% | 0.23% | 1.92% |
| J03WOY | 0.2 | 0.931 | both/empirical | 5326 | 9 | 0.17% | 0.18% | 1.39% |
| J03WOY | 0.4 | 2.986 | forward/contract | 5740 | 47 | 0.82% | 0.91% | 4.36% |
| J03WOY | 0.4 | 1.860 | forward/empirical | 5397 | 37 | 0.69% | 0.76% | 3.83% |
| J03WOY | 0.4 | 2.986 | both/contract | 11472 | 93 | 0.81% | 0.90% | 8.89% |
| J03WOY | 0.4 | 1.860 | both/empirical | 10807 | 70 | 0.65% | 0.72% | 7.32% |
| J03WOY | 1.0 | 11.658 | forward/contract | 14333 | 208 | 1.45% | 1.71% | 9.93% |
| J03WOY | 1.0 | 4.589 | forward/empirical | 13610 | 200 | 1.47% | 1.72% | 9.58% |
| J03WOY | 1.0 | 11.658 | both/contract | 28660 | 462 | 1.61% | 1.90% | 18.82% |
| J03WOY | 1.0 | 4.589 | both/empirical | 27356 | 416 | 1.52% | 1.79% | 17.77% |
| J03WOY | 2.0 | 37.292 | forward/contract | 28674 | 542 | 1.89% | 2.32% | 16.55% |
| J03WOY | 2.0 | 8.855 | forward/empirical | 27398 | 511 | 1.87% | 2.29% | 15.85% |
| J03WOY | 2.0 | 37.292 | both/contract | 57281 | 1230 | 2.15% | 2.63% | 29.97% |
| J03WOY | 2.0 | 8.855 | both/empirical | 54976 | 1182 | 2.15% | 2.63% | 29.44% |
| J03WMX | 0.2 | 1.199 | forward/contract | 3053 | 1 | 0.03% | 0.04% | 0.16% |
| J03WMX | 0.2 | 0.918 | forward/empirical | 2893 | 1 | 0.03% | 0.04% | 0.16% |
| J03WMX | 0.2 | 1.199 | both/contract | 6106 | 3 | 0.05% | 0.05% | 0.48% |
| J03WMX | 0.2 | 0.918 | both/empirical | 5777 | 2 | 0.03% | 0.04% | 0.32% |
| J03WMX | 0.4 | 2.957 | forward/contract | 6199 | 41 | 0.66% | 0.76% | 4.03% |
| J03WMX | 0.4 | 1.832 | forward/empirical | 5873 | 29 | 0.49% | 0.57% | 2.90% |
| J03WMX | 0.4 | 2.957 | both/contract | 12385 | 71 | 0.57% | 0.66% | 6.60% |
| J03WMX | 0.4 | 1.832 | both/empirical | 11708 | 52 | 0.44% | 0.51% | 4.83% |
| J03WMX | 1.0 | 11.584 | forward/contract | 15525 | 297 | 1.91% | 2.34% | 12.24% |
| J03WMX | 1.0 | 4.521 | forward/empirical | 14867 | 267 | 1.80% | 2.19% | 11.43% |
| J03WMX | 1.0 | 11.584 | both/contract | 31036 | 528 | 1.70% | 2.08% | 21.42% |
| J03WMX | 1.0 | 4.521 | both/empirical | 29433 | 473 | 1.61% | 1.96% | 20.13% |
| J03WMX | 2.0 | 37.144 | forward/contract | 31038 | 687 | 2.21% | 2.80% | 16.26% |
| J03WMX | 2.0 | 8.735 | forward/empirical | 29904 | 662 | 2.21% | 2.80% | 16.10% |
| J03WMX | 2.0 | 37.144 | both/contract | 62023 | 1430 | 2.31% | 2.89% | 29.79% |
| J03WMX | 2.0 | 8.735 | both/empirical | 59020 | 1375 | 2.33% | 2.91% | 29.15% |
| J03WN1 | 0.2 | 1.174 | forward/contract | 1243 | 4 | 0.32% | 0.37% | 1.18% |
| J03WN1 | 0.2 | 0.888 | forward/empirical | 1147 | 3 | 0.26% | 0.30% | 0.79% |
| J03WN1 | 0.2 | 1.174 | both/contract | 2479 | 14 | 0.56% | 0.65% | 4.33% |
| J03WN1 | 0.2 | 0.888 | both/empirical | 2288 | 13 | 0.57% | 0.64% | 3.94% |
| J03WN1 | 0.4 | 2.907 | forward/contract | 2534 | 31 | 1.22% | 1.54% | 6.30% |
| J03WN1 | 0.4 | 1.770 | forward/empirical | 2346 | 22 | 0.94% | 1.16% | 4.72% |
| J03WN1 | 0.4 | 2.907 | both/contract | 5064 | 63 | 1.24% | 1.55% | 11.81% |
| J03WN1 | 0.4 | 1.770 | both/empirical | 4679 | 46 | 0.98% | 1.20% | 9.45% |
| J03WN1 | 1.0 | 11.460 | forward/contract | 6350 | 156 | 2.46% | 3.26% | 15.75% |
| J03WN1 | 1.0 | 4.358 | forward/empirical | 5937 | 137 | 2.31% | 3.05% | 14.96% |
| J03WN1 | 1.0 | 11.460 | both/contract | 12700 | 293 | 2.31% | 3.06% | 29.13% |
| J03WN1 | 1.0 | 4.358 | both/empirical | 11781 | 256 | 2.17% | 2.87% | 27.95% |
| J03WN1 | 2.0 | 36.896 | forward/contract | 12685 | 355 | 2.80% | 3.84% | 19.69% |
| J03WN1 | 2.0 | 8.392 | forward/empirical | 11919 | 335 | 2.81% | 3.83% | 19.29% |
| J03WN1 | 2.0 | 36.896 | both/contract | 25364 | 576 | 2.27% | 3.13% | 34.65% |
| J03WN1 | 2.0 | 8.392 | both/empirical | 23550 | 554 | 2.35% | 3.22% | 34.25% |

合同 cap 随窗口按 v95·W+0.5·a95·W² 二次增长，窗口 ≥1s 时已不是有意义的运动学界（表内直接给出 cap 值）；empirical 行用同一窗口上直接测得的同球员位移 p95。两套 cap 与两个方向全部列出，不选一个当结论。

## 6. 真实状态对照

| 比赛 | 入 cap 对 | t2 有完整帧的对 | 真实帧翻转对 | 真实帧翻转率 | 自然E对中真实帧也翻转 | 以真实 t2 帧为联合格的 E 率 |
|---|---:|---:|---:|---:|---:|---:|
| J03WOY | 2821 | 2821 | 90 | 3.19% | 1 | 0.04% |
| J03WMX | 3053 | 3053 | 107 | 3.50% | 0 | 0.00% |
| J03WN1 | 1243 | 1243 | 63 | 5.07% | 3 | 0.24% |

最后一列把联合格换成 t2 的**真实整帧**（传球者、接球者与 t1 身份链接的 11 名对手都在 t2 的真实位置）：这时四格里有三格是真实观测，只有 A-only/B-only 仍是反事实。它要求 t2 帧里这些身份都在，覆盖率见表。

## 7. 旧统计更名与可复现

历史 `paired`（至少一个 keep 且至少一个 flip）在 `experiments/f095_campaign/d06_roles.py` 中已更名为 `supported_single_keep_and_flip`，旧产物 `artifacts/f095_campaign/D06/D06_ROLES_J03WOY.json` 原样保留。
复现检查（`--mode legacy-recheck`，把 HEAD 版 d06_roles.py 复制到临时目录运行，只写临时目录）：committed 与重跑的 `paired`=22 vs 22、`used`=721 vs 721、`paired_rate`=0.0305 vs 0.0305，全部字段一致=True。
该旧统计只能称 supported_single_keep_and_flip，不能称 E 可构率；它也不是自然发生率。

## 8. 证据与复现命令

```
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 .venv/bin/python experiments/e1a933_review/football_repair.py --mode natural --windows 0.2,0.4,1.0,2.0
```
```
OPENBLAS_NUM_THREADS=6 .venv/bin/python experiments/e1a933_review/football_repair.py --mode verify-natural
```
```
OPENBLAS_NUM_THREADS=6 .venv/bin/python experiments/e1a933_review/football_repair.py --mode legacy-recheck
```
```
OPENBLAS_NUM_THREADS=6 .venv/bin/python experiments/e1a933_review/football_repair.py --mode natural-report
```

产物：`artifacts/e1a933_review/football/NATURAL_E_RATE.json`（三分母、逐场计数、真值表、bootstrap 区间）、`natural_pairs_primary.jsonl`（主格逐对标签）、`NATURAL_E_VERIFICATION.json`（标量 oracle 逐 witness 复核）、`LEGACY_STAT_RECHECK.json`。
`experiments/e1a933_review/football_repair.py` 原有 `--mode probe` 路径未改动；旧产物 `artifacts/f095_campaign/`、`experiments/f095_campaign/` 未被写入。

## 9. 未完成与边界

- caps and windows are empirical motion envelopes, not joint dynamic-feasibility proofs
- the joint cell crosses two real single-player moves and is not a real observed frame
- y_real requires passer, receiver and all eleven t1-identified defenders present at the later frame
- forward windows only in the primary cell; situation set is the R08 240-event sample (162 audited events), not every pass of the three matches
- no threshold is applied to call anything infeasible; a low natural rate is a rare-stress-test statement
- reports/e1a933_review/FOOTBALL_E_FEASIBILITY_V2.md is generated by football_summarize.py and still lists the natural rate as NOT_RUN; this lane did not rewrite that generator, so its line is stale
- the situation table is the frozen R08 probe output (re-read, not re-searched); the probe path of this script is unchanged
- NOT_RUN：real pass-success labels for counterfactual cells (the oracle is geometric, not outcome-based)
- NOT_RUN：natural E rate over every pass of the three matches (only the R08 240-event sample is measured)
- NOT_RUN：learned partial-observation relation model and natural-missingness state estimation (O05 scope)
- NOT_RUN：other four available matches (outside the R08 train/dev/test naming; J03WQQ is sealed)
