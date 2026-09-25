# mechanism_transfer_v3 reproduction (v3 numbers only)

Scope: regenerates every v3 number cited in `M1_MECHANISM.md`, `M2_TRANSFER.md`,
and the v3 portions of the route2/route3 reports. Uses only frozen banks and
archived predictions where noted. Never pools denominators across banks.
Seeds are training replicates, not independent datasets.

Repo root for all commands: `/home/huyudi/012_conference/iclr2027`.

## M1.0 collision boundary (CPU, no training)

Artifacts: `artifacts/mechanism_transfer_v3/m1/FEATURE_CONTRACT.json`
(shared triangle 1/0, margins 0.09375, legacy sorted gap 0.0 float32+float64,
unsorted/whole-orbit gap 0.3719),
`artifacts/mechanism_transfer_v3/m1/source_search.json`
(bank SHA `15f5bf18…`, 4096 scenes; nearest opposite-label distance 0.0471;
3000-step local search 0.0335; equal-label control 0.0213),
`artifacts/mechanism_transfer_v3/m1/collision_pair.npz`,
`artifacts/mechanism_transfer_v3/m1/reference_eval.json`.

```bash
python3 experiments/mechanism_transfer_v3/m1/collision.py
python3 experiments/mechanism_transfer_v3/m1/source_search.py
python3 experiments/mechanism_transfer_v3/m1/reference_eval.py
python3 -m unittest discover -s experiments/mechanism_transfer_v3/m1 -p 'test_m1.py' -v
```

## M1.2 source matrix (CPU, 108 cells, ~110 s train + ~20 s eval)

Bank (frozen, never pooled): `artifacts/e832_focus/structure/bank_quartets.npz`
(2188 E / 871 parents, sha256 `15f5bf18…`).
Artifacts: `artifacts/mechanism_transfer_v3/m1/training/`
(`metrics_3seed.json`, `SUMMARY_3seed.json`, `selection.json`, `capacity.json`,
`probe.json`, `orbit_seam.json`, `MANIFEST.json`, `runs/*/model.pt|receipt.json|train_curve.json|dev_curve.json`).
Recipe: 300 full-batch Adam steps, lr {0.01, 0.003, 0.001}, 4 threads,
supervision clean or clean + all 1534 mined single flips, selection by final
pooled static+single dev BCE (smaller lr breaks ties), `logit > 0`.
Numbers reproduced: flip J3 raw ~0.08-0.11, unsorted ~0.44-0.46,
sorted ~0.52-0.54, orbit ~0.50-0.59, segment_moment ~0.33-0.34,
repaired_segment_rho ~0.02-0.03; pre-stated flip contrasts
orbit-minus-sorted +0.0654/+0.0005/-0.0311 (last two CIs cross 0),
orbit-minus-unsorted all positive excluding 0; clean orbit beats
sorted/unsorted/raw in all 3 seeds (CIs exclude zero).
Full detail: `artifacts/mechanism_transfer_v3/m1/training/RESULTS.md`.

```bash
python3 experiments/mechanism_transfer_v3/m1/training/run_matrix.py --stage all
python3 -m unittest discover -s experiments/mechanism_transfer_v3/m1/training -p 'test_*.py' -v
```

Partial stages (same defaults: all 6 arms, seeds 11/23/47):

```bash
python3 experiments/mechanism_transfer_v3/m1/training/run_matrix.py --stage train
python3 experiments/mechanism_transfer_v3/m1/training/run_matrix.py --stage select
python3 experiments/mechanism_transfer_v3/m1/training/run_matrix.py --stage eval
python3 experiments/mechanism_transfer_v3/m1/training/run_matrix.py --stage summary
```

## M2 CPU prep steps 0-3 (no GPU, no frozen head, no sealed pool)

Artifacts: `artifacts/mechanism_transfer_v3/m2/bank/`
(`data.npz` archive SHA `69c4bb5c…`, `data_manifest.json` SHA `50060815…`;
692 quartets / 351 parents; splits 1024/128/384 test parents),
`artifacts/mechanism_transfer_v3/m2/mask_audit.json`,
`artifacts/mechanism_transfer_v3/m2/analytic_baseline.json`
(analytic image baseline J3 = 0.886 [0.857, 0.917]),
`artifacts/mechanism_transfer_v3/m2/PREP_RECEIPT.json`.
Detail: `artifacts/mechanism_transfer_v3/m2/M2_PREP_REPORT.md`.

```bash
python3 experiments/mechanism_transfer_v3/m2/generate_bank.py --out artifacts/mechanism_transfer_v3/m2/bank
python3 experiments/mechanism_transfer_v3/m2/mask_audit.py --bank artifacts/mechanism_transfer_v3/m2/bank --out artifacts/mechanism_transfer_v3/m2/mask_audit.json
python3 experiments/mechanism_transfer_v3/m2/analytic_image_baseline.py --bank artifacts/mechanism_transfer_v3/m2/bank
python3 -m unittest discover -s experiments/mechanism_transfer_v3/m2 -p 'test_m2_prep.py' -v
```

## M2 GPU transfer (16 cells; requires CUDA; RTX 2080 Ti, torch 2.5.1+cu121, ResNet18 IMAGENET1K_V1, FP32, 20 epochs, batch 32, seeds 803/805/806, singleton-dev selection)

Frozen head: M1 `orbit_distance` CoordMLP seed 11, checkpoint SHA `8b9c43f8…`
(`EXPECTED_HEAD_SHA256` in `gpu_transfer.py`), 6-dim whole-orbit features
standardized with source-train stats (never refit).
Remote outputs: `artifacts/mechanism_transfer_v3/m2/gpu_run_remote/out/`
(16 cells, each `result.json` + `predictions.npz` + histories/checkpoints).
Results: direct_full J3 0.919/0.915/0.921; direct_clean 0.863/0.870/0.845;
true-geometry + frozen head 0.796 (no training); random/wrong head
0.000/0.006/0.013; geometry-loss-only frontend ~0.19-0.22
(dev MSE ~0.51 frozen or fine-tuned).
All 16 cells independently recomputed from `predictions.npz`.

```bash
python3 -m unittest discover -s experiments/mechanism_transfer_v3/m2 -p 'test_m2_gpu.py' -v
# one cell (example: direct_full seed 803); repeat per mode/seed for all 16:
CUDA_VISIBLE_DEVICES=0 python3 experiments/mechanism_transfer_v3/m2/gpu_transfer.py \
  --bank artifacts/mechanism_transfer_v3/m2/bank \
  --out artifacts/mechanism_transfer_v3/m2/gpu_run_remote/out/direct_full_803 \
  --mode direct_full --seed 803
# other modes: --mode direct_clean | --mode true_geometry --head-checkpoint <m1-orbit-s11.pt> \
#   | --mode random_head --head-checkpoint <m1-orbit-s11.pt> | --mode geom_front --head-checkpoint <m1-orbit-s11.pt> [--unfreeze-backbone]
```

## route2 v3 area-mask-pooling (fresh 28-quartet/19-parent bank; corrected predictions only)

Data (frozen archive SHA `8e5128e5…`, manifest SHA `fd148616…`):
`artifacts/e832_focus/route2/data/`.
v2 formal contract fix + v3 single-factor change (`--mask-pool area`;
direct arm mathematically unaffected, v3 direct == v2 per seed as control).
Results: `artifacts/e832_focus/route2/gpu_run_v3_remote_20260925/gpu_run_v3/`
(v2 corrected reference:
`artifacts/e832_focus/route2/gpu_run_v2_remote_20260925/corrected_results/`).
Numbers: v3 mean J3 direct 0.571 / additive 0.464 / representation 0.619 /
interaction 0.476; interaction-minus-direct -0.250/-0.036/+0.000.
Contract-broken v1 J3 field is retained but excluded from all conclusions.

```bash
python3 experiments/e832_focus/route2_visual/data_generator.py --out artifacts/e832_focus/route2/data
python3 -m unittest discover -s experiments/e832_focus/route2_visual -p 'test_*.py' -v
python3 experiments/e832_focus/route2_visual/run_route2.py
# GPU host, v3 (12 arm/seed cells + 3 clean-only baselines):
CUDA_VISIBLE_DEVICES=0 python3 experiments/e832_focus/route2_visual/gpu_run.py \
  --manifest artifacts/e832_focus/route2/gpu_bundle/manifest.json \
  --data artifacts/e832_focus/route2/data \
  --out artifacts/e832_focus/route2/gpu_run_v3 --mask-pool area
```

## route3 parser + three evidence-driven rounds (CPU, 4 threads, 100/300 epochs)

Baseline: `artifacts/e832_focus/route3/results.json`
(parser J3 = 1.0 on 230 E / 65 parents; learned seeds J3 raw 0.026/0.022/0.013,
segment DeepSets/relational 0.022/0.017/0.052, G8 group-average raw 0.017/0.009/0.000).
Round outputs: `artifacts/e832_focus/route3/round1_optim.json`
(300-epoch train BCE ~0.18-0.29 yet J3 0.004-0.065: not underfitting),
`artifacts/e832_focus/route3/round2_repr.json`
(raw ~0.01-0.03, orbit_canonical ~0.30-0.37, still far below parser 1.0),
`artifacts/e832_focus/route3/round3_freshbank.json`
(fresh E=171/56: parser 1.0, rich_sorted ~0.21-0.24, orbit_canonical ~0.24-0.29).
Selection by dev BCE only; test J never used for selection; no sealed pools.

```bash
python3 experiments/e832_focus/route3_baselines/strong_baselines.py --out artifacts/e832_focus/route3
python3 experiments/e832_focus/route3_baselines/route3_rounds.py --out artifacts/e832_focus/route3 --rounds r1 r2 r3
```

## Paper build

Current receipt: `reports/e832_focus/PAPER_BUILD_RECEIPT.json`
(27 pages, conclusion page 9, 0 overfull, 0 undefined;
v3 migration is appendix-only route1/route2/M2-frozen-module notes, no
main-text promotion). Appendix tail carrying the v3 numbers:
`paper/sections/app_l015.tex`.
