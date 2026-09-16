# B30 终期交付：逐条检索确认＋推荐开工顺序（task-b30verify）

- 日期：2026-09-16；目标 mu3wl782-yxuns1
- 总轮耗：30 / 30（W1 10＋W2 10＋W3 10，全落盘 reports/research_b30/）
- 只读纪律：本目标全程未做新训练、未改代码、未碰保留场（J03WQQ）/外部（SoccerTrack）；
  唯一重算为 GOAL-MU01（旧目标 artifact，零训练成本）

## 顶级提案文献复核（抽查 18 项主来源，全部命中）

| 提案引用 | 复核结果 | 证据等级终判 |
|---|---|---|
| ArcFace（Deng CVPR 2019） | CVPR openaccess＋IEEE 8953658 双命中 | A（维持） |
| SupCon（Khosla NeurIPS 2020） | arXiv 2004.11362＋NeurIPS poster 双命中 | A（维持） |
| Set Transformer PMA（Lee ICML 2019） | arXiv 1810.00825＋ICML slides 双命中 | A（维持） |
| EGNN（Satorras ICML 2021） | arXiv 2102.09844＋PMLR v139 双命中 | A（维持） |
| Frame Averaging（Puny ICLR 2022 Oral） | arXiv 2110.03336＋ICLR oral 6190 双命中 | A（维持） |
| Othello（ICLR 2023） | arXiv 2210.13382＋GitHub 双命中 | A（维持） |
| ROME（Meng NeurIPS 2022） | NeurIPS proceedings PDF＋GitHub 双命中 | A（维持） |
| Gruver Lie-derivative（ICLR 2023） | mlanthology＋ICLR oral 12763＋arXiv 2210.02984 三命中 | A（维持） |
| Geiger 因果抽象（NeurIPS 2021） | proceedings＋arXiv 2106.02997 双命中 | A（维持） |
| TacticAI（Nature Commun 2024） | Nature s41467-024-45965-x＋DeepMind blog 双命中 | B＋（维持，D2 细节仍须 camera-ready 前复核 PDF 全文） |
| Partial G-CNN（Romero NeurIPS 2022） | NeurIPS proceedings＋arXiv 2110.10211 双命中 | A（维持） |
| TEM（Whittington Cell 2020） | ScienceDirect＋PubMed 33181068 双命中 | A−（维持） |
| Hewitt 结构探针（NAACL 2019） | ACL N19-1419＋Stanford pubs 双命中 | A（维持） |
| Agarwal 统计悬崖（NeurIPS 2021） | arXiv 2108.13264＋NeurIPS proceedings 双命中 | A（维持） |
| Circle loss（Sun CVPR 2020） | CVPR openaccess＋arXiv 2002.10857＋IEEE 双命中 | A（维持） |
| DeepVecFont-v2（CVPR 2023） | arXiv 2303.14585＋CVPR openaccess＋GitHub 三命中 | A（维持） |
| Agaskar-Lu 图不确定性（2013） | arXiv 1206.6356＋APSIPA 双命中 | A（维持） |
| SimCLR 长 schedule（Chen ICML 2020） | 仅命中 Medium/blog＋教程，未命中主 PDF | B（降级：方向可信，量级推断弱；S1 原标注 B+/C 维持不变） |

结论：18 项中 17 项维持原证据等级，1 项（SimCLR）明确降级标注；
S4 中 TacticAI D2 全文细节仍为唯一的 camera-ready 前待办。

## 淘汰项确认（维持淘汰，不复活）
- X1 认证式鲁棒 / X2 DiffPool / X3 grid-metric 正文 / X4 leaderboard 承诺 /
  X5 μ 重做——复核中未见反证，维持禁令。

## 推荐开工顺序（最终）
1. Tier 1 写作六件套（T1-1→T1-6，零训练成本）：标题摘要（S3-A）→ related-work 柔道（S4）→
   claim 分级（M5）→ rebuttal 预演（S5）→ K1–K3 预注册（M9）→ 图表三件套（S2）。
   先做 T1-2 中 TacticAI PDF 全文复核（1 小时）与 T1-3 的五处冻结数 source-check。
2. 训练开火（训练预算剩 8 轮独立记账）：T2-1 margin–SupCon（1 轮）→
   T2-2 attention readout（1 轮，可与 T2-1 并行）→ T2-4 结构等变（占 1–2 轮）。
   T2-3 长 schedule 列为顺手备选。
3. Tier 3 按需取用：W-1 书法 outlook（一段＋定性图）/ W-5 B1 audit 进贡献；
   W-2/W-3/W-4 按 non-claim 口径进 broader impact／related 段落。

## 审计口径
- B30 支出：30 简报文件均在盘（W1-M1–M10 / W2-A1–A10 / W3-S1–S10），
  LEDGER.md 30/30；失败 run 状态不另扣费（文件照常落盘）。
- 训练预算：2/10（R1/R2），BUDGET_10R.md 独立记账；本目标未动训练轮。
- GPU：¥0 / ¥50；CPU：本目标仅只读＋检索，未超 48 核并行上限。
