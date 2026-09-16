# V4 最小离线证据包（2026-09-16，repair F6）

自足离线复核 P2 主结果：已有 synthetic histories/queries、P2 frozen memories、
R1/R2 rows＋manifests、运行配置/prompt 注明来源副本、复算脚本与 manifest。
无凭据、无 provider header、无隐藏 reasoning、无第三方原始数据、无 tokenizer 大文件。
`work/` 全量与 API 缓存未纳入（有意排除，见下）。

## 离线命令（干净目录同样可用）

```bash
python3 OFFLINE_RECOMPUTE.py [--out recomputed.json]
```

从逐题答案重算 history utility、长度分布、三路径完整与 pair-specific 表、
 headline 对比均值；CI 用独立标注 seed（文件头 `INDEPENDENT_SEED`），
 不得冒充原 runner 随机序列。确定性字段须与 `science_result.json` 完全一致。

## 来源

- `p2_histories.jsonl` / `p2_queries.jsonl` / `p2_data_manifest.json` ← `work/v4_fast_decision_20260914_owner_continuation/p2/data/`
- `p2_memories.jsonl` / `compression_manifest.json` ← `.../compact_v2/p2/live/`
- `r1/r2_reader_rows.jsonl` + `r1/r2_reader_manifest.json` ← `.../compact_v2/p2/r1_low|/r2_low/`
- `confirmation_rows.jsonl` / `science_result.json` ← `artifacts/`
- `runtime.json` ← `configs/v4_compact_v2_continuation_runtime_20260914.json`
- `prompt_*.txt` ← `prompts/`（哈希见 MANIFEST.json）

## 未纳入及其对复核能力的影响

- `work/` 全量步骤与 API 缓存：体积大且含传输收据；逐题行＋manifest 已足够重算主表，不影响结论复核；中断恢复行为由 `tests/test_v4_repair_regression.py` 覆盖。
- P0/P1 行：主结果只需 P2；P0/P1 有各自 artifact，不合并统计。
- tokenizer 大文件：复算只用已记录的 token 计数，不重数。
