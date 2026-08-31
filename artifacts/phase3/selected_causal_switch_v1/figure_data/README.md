# Figure source tables

- prediction and response: `switch_seed_summary.parquet`, `switch_response_by_seed.parquet`
- task outcome: `switch_task_metrics.parquet`
- context robustness: `switch_context_robustness_by_seed.parquet`

The seed-resolved repaired summaries add the missing seed key to the original aggregate tables; the original files are retained and not overwritten. Any figure must show the task and robustness boundary together with the geometry change.
