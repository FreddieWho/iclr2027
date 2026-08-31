# Exploration Checkpoint 1 — Full Run Record

> P1/C1 主线报告为 [EXPLORATION_CHECKPOINT_1_POINT_MAINLINE.md](EXPLORATION_CHECKPOINT_1_POINT_MAINLINE.md)。本文件保留本次完整运行记录；其中视觉模型仅作为辅助渲染稳健性分支，不纳入主线结论。

- **Status**: `completed`
- **Study mode**: `exploratory_discovery`
- **Samples**: 250 across 10 SkillCorner matches
- **Interventions**: 42500 total; 39328 valid; 3172 boundary-invalid
- **Maximum energy error**: `8.881784197001252e-16`
- **Minimum target-mode purity**: `0.9999999999999998`

## Models

- `deepsets_ae_seed11`: **completed**
- `gat_ae_seed11`: **completed**
- `phase_gat_seed11`: **completed**
- `deepsets_ae_seed23`: **completed**
- `gat_ae_seed23`: **completed**
- `phase_gat_seed23`: **completed**
- `deepsets_ae_seed47`: **completed**
- `gat_ae_seed47`: **completed**
- `phase_gat_seed47`: **completed**
- `dinov2_vits14`: **completed**
- `openclip_vit_b32`: **completed**

## Measurements

- Primary response: normalized cosine distance between baseline and intervened embeddings.
- Transfer curves: six equal-count Laplacian bands across all seeds and energies; exact modes are a primary-energy/seed-11 accessibility diagnostic.
- Preliminary ERRR: common versus role-ordered semantic coalition; not a P2 strict same-frequency result.
- Confidence intervals: match-level bootstrap, not frame-level independent bootstrap.
- Pairing/validity gate: `pass`; observation key is `match_id:sample_id`, and each intervention has one baseline counterpart.
- Vision inputs are deterministic minimap rasters generated from the same tracking state; they are modality-specific sensitivity measurements, not video-based architecture comparisons.

## Exploratory findings

The numerical tables below are generated from the response artifact. A positive or null result is retained as an exploration outcome; no mechanical positive threshold is applied.

### Preliminary ERRR

```json
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
    "model_id": "dinov2_vits14",
    "n_pairs": 283,
    "errr": 0.8586572438162544,
    "mean_common_minus_semantic": 0.00571504919773277
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
    "model_id": "openclip_vit_b32",
    "n_pairs": 283,
    "errr": 0.8833922261484098,
    "mean_common_minus_semantic": 0.0071486450547464326
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
```

### Notch candidates

```json
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
    "model_id": "dinov2_vits14",
    "notch_depth_candidate": 0.08043182324837883,
    "low_high_reference": 0.011789572570059036,
    "mid_response": 0.010841315752930113,
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
    "model_id": "openclip_vit_b32",
    "notch_depth_candidate": 0.11144801362940171,
    "low_high_reference": 0.01717283891306983,
    "mid_response": 0.015258960127830505,
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
```

## Artifacts

- `artifacts/phase1/figures/transfer_curves.png`
- `artifacts/phase1/figures/preliminary_reversal.png`
- `artifacts/phase1/figures/support_sweep.png`
- `artifacts/phase1/spectrum_response.parquet`
- `artifacts/phase1/embedding_manifest.parquet`
- `artifacts/phase1/pipeline_summary.json`
- `artifacts/phase1/model_manifest.json`
- `artifacts/phase1/execution_contract.json`
- `artifacts/phase1/intervention_validation.json`

## Boundary and next action

P1 is a response-measurement checkpoint. It does not claim that common motion is more salient, that a mid-frequency notch exists, or that organization exceeds frequency. Those interpretations require the matched controls and alternative graph constructions planned for P2.

Runtime: 7479.3 seconds.
