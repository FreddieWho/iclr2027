# 修复 ledger（F1–F8，owner：本任务单线程执行，无并行写者）

| 项 | 状态 | 改动路径 | 证据 | 残余影响 |
|---|---|---|---|---|
| F1 状态/打包混淆 | FIXED | `reports/CURRENT_STATUS.json`（新）、`reports/HISTORICAL_INDEX.md`（新）、README/MASTER/PILOT_REPORT/V4_PREFLIGHT/VALIDATION/ROBUSTNESS/EVIDENCE_AUDIT/LENGTH_AUDIT 首屏 banner（只加注不改原文） | README 一次导航直达当前结论/输入/范围；旧 JSON 收据原值未动 | 无；旧文件仍可读，需读者先看 banner |
| F2 复算 ZIP 过期 | FIXED | `docs/ICLR_MemoryV4_Independent_Reanalysis_20260916/` ＋同名 `.zip`（SHA `67081501…`）；旧版保留为历史 | 干净目录运行输出与冻结 CSV 逐字节一致；篡改源 fail-closed（exit=1）；`--source` 支持外部 checkout | 旧 ZIP 仍过期 By design（历史件，不删除） |
| F3 评分缺冻结校验 | FIXED | `scripts/run_v4_fast_pilot.py`：`verify_frozen_score_inputs`＋`expected_query_contract`；query/provider/ledger 之前校验 | 4 项回归：篡改正文/文件 SHA/历史 ID 均在零 provider 调用下拒绝；完整样本仍 COMPLETE 且幂等 | 现存 P2 哈希实际匹配，无篡改证据；旧运行目录 resume 会因新增 manifest 绑定键而拒绝（fail-closed，须用新目录） |
| F4 续跑丢评分行 | FIXED | 同上：`_score_cell`＋score-key 完整性跳过＋缓存确定性恢复＋`blocked_cells` | 中断模拟：0 行/部分行均恢复恰好预期键集，再恢复无变化；无缓存时 `BLOCKED_MISSING_RESPONSE_RECEIPT` 停住，不补发不假填 | 现存 P2 四份 reader 产物键完整，未受此缺陷影响 |
| F5 聚合缩分母 | FIXED | `aggregate_history_utilities`/`analyze_phase` 合同化（`queries=` 可选）；COMPLETE 改精确键覆盖＋重复/未知键报错 | 1/8 题臂输出 U=None 而非 1.0；真实 P2 重算 288 字段零差异（R1=11/R2=12 完整） | 默认 8 题/4 类×2 合同；若未来 phase 题型结构变化须显式传 queries |
| F6 证据包未收口 | FIXED | `docs/V4_MINIMAL_EVIDENCE_20260916/`（18 文件＋MANIFEST＋README＋OFFLINE_RECOMPUTE.py）；全部证据拷贝与 live SHA 一致 | 干净目录离线重算 headline 均值与冻结值一致（R1 −4.545pp/R2 −3.125pp）；包可复现 | `work/` 全量/API 缓存/tokenizer 二进制有意排除（README 列明影响：不影响主表复核） |
| F7 rewrite 边界 | FIXED | `reports/V4_FAST_PILOT_REPORT.md` 附录 A/B/C（只追加） | rewrite 整路径策略说明、费用口径、NOT_RUN 声明落字 | 历史 memory 未动；共享起点 rewrite 仍为 NOT_RUN |
| F8 prompt 未绑定 | FIXED | `resolve_compressor_prompt`＋manifest 绑定（compressor/reader/runner code SHA）；`--allow-legacy-prompt`；缺字段 fail-closed | 双 runtime 选中各自模板；改模板后 frozen resume 拒绝；旧 runtime 默认拒绝（须显式 flag）；P2 离线分析不受影响 | 新 flag 仅用于历史只读；live 新跑必须用带 prompt_path 的 runtime |

- 冻结源哈希：三哈希复核与快照一致（见 `input_hashes.json`），修复前后未变。
- 模型调用/费用：新增 0（全离线 fake-provider 回归；20/20 通过含既有 7 项）。
- 未碰：父仓库 Action-Mode、`sources/`、`.env`（未读）、Git 发布（未 commit/push）、V3 续跑、P3/C2/LongMemEval/restore/placebo。
- `NOT_RUN` 项：保持未运行（P3/C2/自然评估/严格等长/共享起点 rewrite/完整 novelty/24 对审计/第三方复现）。
