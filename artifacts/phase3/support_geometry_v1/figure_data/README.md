# Figure source tables

This directory is a contract index; source tables remain in the parent artifact directory.

- observed vs predicted pair effects: `pair_predictions.parquet`, `prediction_metrics.parquet`
- retrospective uncertainty: `match_bootstrap_comparisons.parquet`
- layer/block anatomy: `layerwise_block_summary.parquet`, `layerwise_prediction_metrics.parquet`, `layerwise_jacobian_rank.parquet`

Prospective and causal-switch tables are kept in their own versioned artifact roots. No table is selected after looking at its plotted outcome.
