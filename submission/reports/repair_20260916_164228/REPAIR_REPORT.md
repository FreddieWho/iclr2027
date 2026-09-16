# REPAIR_REPORT（Memory Pilot 修复收口，2026-09-16）

## 实际完成
- F1–F8 全部 FIXED（ledger 见 `FIX_LEDGER.md`）。运行器行为变更三项（F3/F4/F5/F8）均有离线定向回归（`tests/test_v4_repair_regression.py`，13 新＋7 既有＝20/20 通过），覆盖审查复现的三个真实失败。
- 新独立复算包与最小证据包均在干净临时目录验证通过；冻结数值零变化（复算 CSV 逐字节一致；新聚合 288 字段零差异；证据拷贝 8/8 SHA 一致）。
- 当前状态唯一入口 `reports/CURRENT_STATUS.json`；README 首屏直达；8 处旧文件历史标记完成，原文未改。

## 未完成 / 未运行
- P3/C2/自然语义评分/restore/placebo/真实长度操纵/共享起点 rewrite/类型交互新历史：按 prompt 明确不执行，保留 `NOT_RUN`（见 CURRENT_STATUS completion_matrix）。
- C1/R1 原版 runner 快照（`55eabd3a…`）：Git 与 `work/`/`artifacts` 只读查找无果，记 `UNAVAILABLE_ORIGINAL_RUNNER_SNAPSHOT`；未伪造。
- Git 发布：按授权不自动 add/commit/push。待发布路径：`reports/repair_20260916_164228/`、`tests/test_v4_repair_regression.py`、修改的 `scripts/run_v4_fast_pilot.py`、新增 `reports/CURRENT_STATUS.json`/`HISTORICAL_INDEX.md`/附录、两个新 `docs/` 包＋新 ZIP。注意 `docs/PROJECT_REVIEW_20260917/`、`docs/ICLR_MemoryV4_Independent_Reanalysis_20260914.zip`（旧）等仍未跟踪，发布时一并决定范围；不得新增宽泛忽略规则掩盖缺交付。

## 数值是否变化及原因
- 科学数值：无任何变化。所有修复均为 fail-closed 完整性加固；现存 P2 输入本就满足新校验（哈希匹配、键完整），故重算恒等。
- 输出变化仅为新增的诊断字段（`missing_score_ids`、`blocked_cells`、`contract_source`、prompt/SHA 绑定）与文档。

## 科学结论边界
- 仍为 `INCONCLUSIVE_NO_GO_SIGNAL`（`reports/V4_ROUTE_DECISION.json`）。工程修复完成≠科学验证完成；不得切换主线，不得声称等长/无隐藏压缩/novelty 已证。
- R1 staged2−direct −4.545pp（n=11）、R2 −3.125pp（n=12），区间均跨零；pair-specific 与原文一致。

## 旧清单说明
- 根 `SHA256SUMS.txt` 仍是历史清单（119 项一致＋PILOT_REPORT 1 项不一致为已知）；本修复的新交付以各包内 MANIFEST.json 为准（重算包 7 文件、证据包 18 文件），旧清单不追认新文件。
