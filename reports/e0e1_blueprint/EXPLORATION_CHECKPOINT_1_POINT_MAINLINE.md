# Exploration Checkpoint 1 — Point-Set Mainline

- Status: completed
- Study mode: exploratory_discovery
- Mainline modality: direct player point set
- Samples: 250 across 10 SkillCorner matches
- Interventions: 42500 total; 39328 valid; 3172 boundary-invalid
- Maximum energy error: 8.881784197001252e-16
- Minimum target-mode purity: 0.9999999999999998

## Models

- deepsets_ae_seed11: completed
- gat_ae_seed11: completed
- phase_gat_seed11: completed
- deepsets_ae_seed23: completed
- gat_ae_seed23: completed
- phase_gat_seed23: completed
- deepsets_ae_seed47: completed
- gat_ae_seed47: completed
- phase_gat_seed47: completed

## Measurements

- Primary response: normalized cosine distance between baseline and intervened point-set embeddings.
- Transfer curves: six equal-count Laplacian bands across all seeds and energies.
- Preliminary ERRR: common versus role-ordered semantic coalition; not a P2 strict same-frequency result.
- Confidence intervals: match-level bootstrap, not frame-level independent bootstrap.
- Pairing/validity gate: pass; each intervention has one baseline counterpart.

## Exploratory findings

These are candidate response patterns for exploration, not confirmatory claims.

### Preliminary ERRR

JSON:
[
  {
    "model_id": "deepsets_ae_seed11",
    "n_pairs": 2789,
    "errr": 1.0,
    "mean_common_minus_semantic": 0.030361443170299235
  },
  {
    "model_id": "deepsets_ae_seed23",
    "n_pairs": 2789,
    "errr": 1.0,
    "mean_common_minus_semantic": 0.024571240909982642
  },
  {
    "model_id": "deepsets_ae_seed47",
    "n_pairs": 2789,
    "errr": 1.0,
    "mean_common_minus_semantic": 0.026026157473269737
  },
  {
    "model_id": "gat_ae_seed11",
    "n_pairs": 2789,
    "errr": 1.0,
    "mean_common_minus_semantic": 0.029532969158240312
  },
  {
    "model_id": "gat_ae_seed23",
    "n_pairs": 2789,
    "errr": 0.9996414485478666,
    "mean_common_minus_semantic": 0.03237838713487616
  },
  {
    "model_id": "gat_ae_seed47",
    "n_pairs": 2789,
    "errr": 0.9996414485478666,
    "mean_common_minus_semantic": 0.02792574277741373
  },
  {
    "model_id": "phase_gat_seed11",
    "n_pairs": 2789,
    "errr": 0.9146647543922553,
    "mean_common_minus_semantic": 0.013855464112677426
  },
  {
    "model_id": "phase_gat_seed23",
    "n_pairs": 2789,
    "errr": 0.980279670132664,
    "mean_common_minus_semantic": 0.014808565805615726
  },
  {
    "model_id": "phase_gat_seed47",
    "n_pairs": 2789,
    "errr": 0.9781283614198637,
    "mean_common_minus_semantic": 0.003759490050223191
  }
]

### Notch candidates

JSON:
[
  {
    "model_id": "deepsets_ae_seed11",
    "notch_depth_candidate": 0.9160121638781892,
    "low_high_reference": 0.005022650516977037,
    "mid_response": 0.000421841548516996,
    "interpretation": "candidate only; inspect monotonicity and P2 controls"
  },
  {
    "model_id": "deepsets_ae_seed23",
    "notch_depth_candidate": 0.8934424304018618,
    "low_high_reference": 0.004148659092187658,
    "mid_response": 0.00044207102995473535,
    "interpretation": "candidate only; inspect monotonicity and P2 controls"
  },
  {
    "model_id": "deepsets_ae_seed47",
    "notch_depth_candidate": 0.900870879983664,
    "low_high_reference": 0.004363247935252265,
    "mid_response": 0.0004325249282346522,
    "interpretation": "candidate only; inspect monotonicity and P2 controls"
  },
  {
    "model_id": "gat_ae_seed11",
    "notch_depth_candidate": 0.9835987400727406,
    "low_high_reference": 0.005229776234601535,
    "mid_response": 8.577491938510374e-05,
    "interpretation": "candidate only; inspect monotonicity and P2 controls"
  },
  {
    "model_id": "gat_ae_seed23",
    "notch_depth_candidate": 0.9752678091120308,
    "low_high_reference": 0.006145088560697956,
    "mid_response": 0.000151981503306658,
    "interpretation": "candidate only; inspect monotonicity and P2 controls"
  },
  {
    "model_id": "gat_ae_seed47",
    "notch_depth_candidate": 0.9850651553303817,
    "low_high_reference": 0.005131704154641123,
    "mid_response": 7.664120443999956e-05,
    "interpretation": "candidate only; inspect monotonicity and P2 controls"
  },
  {
    "model_id": "phase_gat_seed11",
    "notch_depth_candidate": 0.9748763599136323,
    "low_high_reference": 0.003462896235384101,
    "mid_response": 8.700055867422795e-05,
    "interpretation": "candidate only; inspect monotonicity and P2 controls"
  },
  {
    "model_id": "phase_gat_seed23",
    "notch_depth_candidate": 0.9538620196824389,
    "low_high_reference": 0.003879058416906397,
    "mid_response": 0.0001789719208898972,
    "interpretation": "candidate only; inspect monotonicity and P2 controls"
  },
  {
    "model_id": "phase_gat_seed47",
    "notch_depth_candidate": 0.8289089052794523,
    "low_high_reference": 0.0024706643228549047,
    "mid_response": 0.0004227086636842465,
    "interpretation": "candidate only; inspect monotonicity and P2 controls"
  }
]

## Artifacts

- artifacts/phase1/point_mainline/figures/transfer_curves.png
- artifacts/phase1/point_mainline/figures/preliminary_reversal.png
- artifacts/phase1/point_mainline/figures/support_sweep.png
artifacts/phase1/point_mainline/spectrum_response.parquet
artifacts/phase1/point_mainline/embedding_manifest.parquet
artifacts/phase1/point_mainline/spectrum_summary.parquet
artifacts/phase1/point_mainline/pipeline_summary.json
artifacts/phase1/point_mainline/execution_contract.json

## Scope boundary

The direct point-set branch is the P1/C1 mainline because the source data already represent players as points. The minimap/vision run is preserved as an auxiliary rendering-robustness result and is excluded from this mainline checkpoint.
P1 does not establish that common motion is more salient, that a mid-frequency notch exists, or that organization exceeds frequency. Those interpretations require the matched controls and alternative graph constructions planned for P2.
