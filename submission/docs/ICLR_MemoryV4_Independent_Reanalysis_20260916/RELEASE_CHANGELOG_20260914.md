# 独立复算交付记录 — 2026-09-14

## 范围

完成 Memory V4/P2 compact-v2 的离线 history-level 复算与来源核验。范围仅限 `submission/docs/ICLR_MemoryV4_Independent_Reanalysis_20260914/`；未更改 P2 源 JSONL、主科学报告、路线判定、P1/P2 原始 receipts 或父目录中的其他项目。

## 更新

- 将目录 `README.md` 接入主报告和本变更记录。
- 在 `recompute.py` 中加入源 JSONL SHA-256 校验、48 个 reader-history 键核对以及 utility/native-length 转录核对；源缺失、哈希不符、重复键或字段不一致时 fail closed。
- 运行 `python recompute.py`，重算三路径 complete-case 和 pair-specific 下 R1/R2 的 SD、RD、SR，共 12 个结果单元。
- 新增 `PHASE_REPORT_20260914.md`，作为本独立复算的规范读者报告。

## 审计与验证

- 锁定源 JSONL 与提交 `5646f1d69154d2294817a74e1dc5f148265a72f2` 中的 blob 一致；SHA-256：`9a5502b2e00f4f9fcaf2daffeaa820f28e0eeba25cca4eef7d03fc4c3e18497e`。
- 输入 CSV 共 48 个唯一键；R1/R2 的 3 路 utility 与 3 路长度共 288 个字段逐项匹配源 JSONL。
- `python recompute.py` 成功；结果包含预期的 2 readers × 2 纳入规则 × 3 contrasts，共 12 行，每行均记录 seed `2026091303` 与 20,000 次 bootstrap。
- 重算后 `paired_reanalysis.csv` SHA-256 为 `0c0770a9035ea87f070117b313a691a84d386e20150c9fd0ab436f9fda2b4745`，与运行前一致；结果文件内容未漂移。
- 本次未调用模型/API、未访问网络、未新增样本或费用；未运行与本离线任务无关的广泛测试。

## 冻结面与限制

- 三路径完整的结果与 V4 主报告相符；pair-specific 结果作为新增敏感性分析保留，不替代预先报告的 complete-case 估计。
- 主结论保持 `INCONCLUSIVE_NO_GO_SIGNAL`；不声称等价、无效应或 `NO_GO_MEMORY`，不触发 P3。
- 仓库排除的 `work/` 原始 reader rows 与 frozen memory 正文不在本包中；无法独立审计逐题评分、来源证据或长度解析，只能核验已提交 history 汇总的转录和统计计算。
- 当前目录交付已完成；尚未为本次新增文件创建 Git commit 或推送。

## 2026-09-16 (v20260916): self-contained repackage (repair F2)
- Vendored `source_v4_confirmation_rows_compact_v2_20260914.jsonl` (SHA-256 `9a5502b2…84197e`) into the package; `recompute.py` now verifies the in-package copy by default and accepts `--source` for an external checkout. Fail-closed on SHA mismatch, missing source, or duplicate keys.
- No numeric changes: `paired_reanalysis.csv` is byte-identical to the 20260914 output (verified in a clean temp dir; see repair ledger).
- Old `../ICLR_MemoryV4_Independent_Reanalysis_20260914/` + `.zip` are preserved as history.
