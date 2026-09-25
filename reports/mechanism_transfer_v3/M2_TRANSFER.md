# M2 transfer report — frozen relation head to a new visual frontend

> **Superseded interpretation, 2026-09-25.** The runner actually used for these
> historical results disabled backbone gradients in the tuned arm and updated
> BatchNorm buffers in the nominally frozen arm. Initialization also preceded
> seeding. These results cannot establish the stated frozen-versus-tuned contrast,
> exclude capacity, or prove the interface unlearnable. Corrected computation and
> allowed wording are recorded in `reports/submission_audit_20260925/mechanism/`.
> The historical values below are retained for traceability.

Bank: expanded blinded visual bank, 692 quartets / 351 parents, archive SHA
`69c4bb5c…` (`artifacts/mechanism_transfer_v3/m2/bank/`).
Frozen head: M1 orbit_distance CoordMLP seed 11, checkpoint SHA
`8b9c43f8…`, 6-dim whole-orbit features standardized with source-train stats.
GPU: RTX 2080 Ti, torch 2.5.1+cu121, ResNet18 IMAGENET1K_V1, FP32, 20 epochs,
batch 32, seeds 803/805/806, singleton-dev selection, no test-J selection.
All 16 cells independently recomputed from `predictions.npz`: match.

## Results (J3 row mean, parent-cluster 95% CI)

| cell | 803 | 805 | 806 |
|---|---|---|---|
| direct_full | 0.919 [0.896, 0.942] | 0.915 [0.893, 0.939] | 0.921 [0.899, 0.942] |
| direct_clean | 0.863 [0.835, 0.893] | 0.870 [0.845, 0.897] | 0.845 [0.818, 0.876] |
| true_geometry + frozen head | 0.796 [0.764, 0.831] (no training) | — | — |
| random/wrong head + true geometry | 0.000 | 0.006 | 0.013 |
| geom_front, frozen backbone, geometry loss only | 0.215 (dev MSE 0.514) | 0.186 (0.516) | 0.194 (0.512) |
| geom_front, fine-tuned backbone, geometry loss only | 0.205 (dev MSE 0.516) | 0.205 (0.504) | 0.207 (0.508) |

Reference ceilings on the same bank: analytic image baseline J3 = 0.886
[0.857, 0.917]; pilot-bank v2 learned best 0.583.

## Verdict: negative with precise attribution (M2.5 阴性分支)

1. The frozen relation head is competent: 0.796 on true geometry of a new
   bank, 0.000–0.013 with a random head of the same class. Specificity holds.
2. The geometry-only frontend fails at perception, not at relation:
   dev geometry MSE ≈ 0.51 whether the backbone is frozen or fine-tuned.
   The lexicographically canonicalized orbit interface switches branches
   piecewise-constantly and is not learnable by smooth MSE regression from
   these images. Unfreezing changes nothing (0.21 vs 0.20), so probe capacity
   is excluded as the cause.
3. Direct end-to-end vision is best (0.918 ≥ analytic 0.886). There is no
   transfer gain from the frozen module on this bank.

This is adaptation-vs-transfer resolved against transfer: stop adding heads.
The M1 source-representation result (orbit reference removes the risky sort
without accuracy cost) stands independently and does not depend on this
negative.

## Non-claims
- 692-bank and 28-bank denominators never pooled.
- No layered/transparent isolation renderer was run; the opaque-bank result
  is a role-conditioned readout setting only.
- No cross-renderer (M2.4) validation; not opened because the main cell failed.
- No sealed pool read at any step.

Remote outputs: `artifacts/mechanism_transfer_v3/m2/gpu_run_remote/out/`
(16 cells, each `result.json` + `predictions.npz` + histories/checkpoints).
Code: `experiments/mechanism_transfer_v3/m2/gpu_transfer.py` (+ `test_m2_gpu.py`).
