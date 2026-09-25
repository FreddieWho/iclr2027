# FIX_IMPACT — contract repairs and measured impact (mechanism_transfer_v3)

Scope: every repair below is a contract/implementation fix from pack
`docs/2c3e2d9_mechanism_transfer_pack/01_CRITICAL_REPAIRS.md` (R1–R6).
Each entry gives the measured impact and what it did NOT change.
No new claims; all numbers trace to the artifact path cited.

Shared-module home: `experiments/mechanism_transfer_v3/common/`
(`bank.py`, `geom_features.py`, `metrics.py`, `source_typed.py`, `visual_pool.py`)
with 13 fail-closed tests in `experiments/mechanism_transfer_v3/tests/test_shared_contracts.py`
(per closeout commit `be5ce3e`).

## 1. J3 definition (route2 v1 → v2)

- Defect: `gpu_run.metrics` wrote a `J3` field that equaled atomic-joint;
  correct contract is J3 = all(A, B, AB) correct.
  Source: `reports/e832_focus/route2/REPORT.md` ("Final v2 result" §1).
- Fix: postprocessor `experiments/e832_focus/route2_visual/recompute_from_predictions.py`
  derives corrected J3/J4 from archived predictions; predictions themselves were uncorrupted.
- Measured impact (fresh bank 28 test quartets / 19 parents, seeds 803/805/806,
  12/12 runs, independently recomputed):
  `artifacts/e832_focus/route2/gpu_run_v2_remote_20260925/corrected_results/`
  — mean J3 direct 0.571 / additive 0.417 / representation 0.274 / interaction 0.583;
  interaction−direct seed deltas +0.179 / −0.143 / 0.000 (direction unstable).
- Did NOT change: archived prediction files (recomputation matched);
  data bank and seeds; clean-only direct baseline flow
  (J3 0.107 / 0.250 / 0.107, mean 0.155); the bounded-negative verdict
  (interaction shows no stable gain over direct).
- Old v1 outputs retained but marked contract-wrong and excluded from conclusions.

## 2. Native-RGB input (route2 v1 → v2)

- Defect: the formal runner passed the normalized tensor as the "image" for red/blue
  mask generation instead of the native RGB original.
  Source: `reports/e832_focus/route2/REPORT.md` ("Final v2 result" §2).
- Fix: runner passes native 64 RGB originals into the model; resize/normalize happens
  inside the model. Four arms share the same ResNet18 backbone, native64→224 input,
  and train exposure (singleton labels only, singleton-dev selection).
- Measured impact: part of the v2 rerun whose corrected numbers are listed in §1
  above (same bank/seeds/epochs as the v1 run it replaces).
- Did NOT change: the bank (same 28/19 fresh bank, archive SHA `8e5128e5…`,
  `artifacts/e832_focus/route2/data/data_manifest.json`);
  per-arm head parameter counts (direct/additive 513, representation 1025,
  interaction 131329); the v3 direct-control identity (see §3).

## 3. Mask area-pooling (route2 v2 → v3)

- Defect (R4): `visible_segment_features` downsampled the native64 binary mask to 7×7
  with nearest-neighbor, silently zeroing genuinely visible masks.
  Local audit: 18/112 quartet red-channel, 24/319 test-red, 19/319 test-blue masks
  erased to zero vectors.
  Source: `reports/e832_focus/route2/REPORT.md` ("GPU追加第1輪 v3 area池化").
- Fix: `mask_pool` nearest→area (`visual_mechanism.py` new parameter, default nearest
  locks legacy behavior; `gpu_run.py` new `--mask-pool`); same data hash, seeds,
  epochs, metrics; output `artifacts/e832_focus/route2/gpu_run_v3_remote_20260925/gpu_run_v3/`.
  12-arm predictions independently recomputed, all matched.
- Measured impact (J3 seed 803/805/806, mean):
  direct 0.571/0.607/0.536 = 0.571 (bitwise identical to v2 — control arm, global
  pooling only, mathematically unaffected);
  additive 0.429/0.500/0.464 = 0.464 (now stable-negative vs direct);
  representation 0.536/0.643/0.679 = 0.619 (erasure had suppressed it from 0.274);
  interaction 0.321/0.571/0.536 = 0.476 (deltas vs direct −0.250/−0.036/+0.000,
  still direction-unstable).
- Did NOT change: the overall verdict (bounded negative — interaction advantage
  still not established); training health (all 12 arms fit, BCE 0.01–0.06 range,
  no NaN/OOM); rounds 2–3 of the authorized GPU budget were retained unspent
  per the stop rule (no remaining fixable contract/implementation factor).

## 4. T1/T2 feature bugs (R2 — contract-level repair, no v3 retraining)

- Defects: T2 `rich` used `norm(axis=1)` on [B,2 endpoints,2 coords] (aggregates over
  x/y, not center-to-endpoint distances); `sort_features(T2)` sorted segment length
  against an unrelated quantity (not an endpoint swap); T1 independently sorted the
  edge bag and the query-distance bag, which aliases opposite labels.
  Source: `docs/2c3e2d9_mechanism_transfer_pack/01_CRITICAL_REPAIRS.md` (R2).
- Fix in shared module: `experiments/mechanism_transfer_v3/common/geom_features.py`
  provides fixed T2 features `[|A-B|, min(|C-A|,|C-B|), max(|C-A|,|C-B|), R]`
  (`axis=-1` norms, sorted center-endpoint distances) and transcribes the legacy
  formulas as negative controls (`legacy_t1_features`, legacy T2 wrong-axis form).
- Measured impact (construction, not a training delta):
  `artifacts/mechanism_transfer_v3/m1/FEATURE_CONTRACT.json` formalizes the T1
  opposite-label witness (shared triangle, margins 0.09375): legacy sorted gap
  exactly 0.0 in float32 and float64 (bitwise equal), unsorted/whole-orbit gap
  0.3719. Any deterministic classifier reading only the legacy representation caps
  at half on the two witness states. This says the legacy sort is not a universally
  sufficient representation; per `docs/2c3e2d9_mechanism_transfer_pack/00_REVIEW.md` §3
  it does not prove the same Bayes-error rate in the random bank, does not attribute
  the full T1 negative to this cause, and says nothing about the source 8-dim sort.
- Did NOT change: no T1/T2 arm was retrained in v3
  (per `reports/mechanism_transfer_v3/M1_MECHANISM.md`: "T1/T2 not run in v3");
  source M1.2 arms and route1 historical T1/T2 numbers stand as history, not
  overwritten; the `sine=area2/(edge1*edge2*edge3)` legacy quantity is accurately
  named/kept in the legacy transcription, not silently overwritten.

## 5. J4 (R5 — metric contract)

- Defect: `J4` equaled `J3`, ignoring the P state.
  Source: `docs/2c3e2d9_mechanism_transfer_pack/01_CRITICAL_REPAIRS.md` (R5).
- Fix: `experiments/mechanism_transfer_v3/common/metrics.py` (and the route2
  postprocessor) compute J3 = all(A,B,AB) and J4 = all(P,A,B,AB) independently,
  missing-P → null; the old `J3_atomic_joint` alias is compatibility-only.
- Measured impact: J4 now differs from J3 where P differs, e.g.
  `artifacts/mechanism_transfer_v3/m1/training/RESULTS.md` §2:
  `raw` s11 flip J3 0.0809 vs J4 0.0804.
- Did NOT change: any J3 value; denominators (2188/871 source bank;
  28/19 pilot visual bank; 692/351 M2 bank — never pooled);
  the claim level of M1/M2 (J4 is a descriptor, not a replacement denominator).

## 6. Shared modules (R1 source head, R3 bank, R6 schema)

- R1 repaired head: legacy `Typed` source branch `sum(phi(points[:2])) +
  sum(phi(points[2:]))` is a four-point sum that discards segment grouping even
  with a nonlinear rho. `experiments/mechanism_transfer_v3/common/source_typed.py`
  extracts the valid `RepairedSegmentRho` (intra-segment psi, cross-segment
  nonlinearity) as the single shared module.
  Measured: fail-closed test transcribes the legacy defect (regroup gap < 1e-12,
  i.e. blind to regrouping) and shows the repaired head distinguishes the
  regroup witness (gap > 1e-6) while staying segment-swap invariant (< 1e-12);
  v3 replication of the frozen source matrix matches legacy within ~0.005 with
  params/MACs exact (`artifacts/mechanism_transfer_v3/m1/training/RESULTS.md` §1);
  `repaired_segment_rho` flip J3 ~0.02–0.03, at/below raw in all seeds
  (frozen negative reproduced with the v3 implementation).
- R3 bank contract: `experiments/mechanism_transfer_v3/common/bank.py` fixes the
  dedup-axis audit (whole-scene Linf: max over point and xy axes, then min across
  reference parents) and records endpoint/role/edit-family/radius coverage, not
  just draw counts. M1.2 runs on the frozen 2188/871 bank (`15f5bf18…`); M2 runs
  on the expanded blinded 692/351 bank (archive SHA `69c4bb5c…`).
- R6 fail-closed schema: training/eval isolation, input visibility, and
  label/role semantics are hard errors; weak signals and boundary effects stay
  explorable (no arbitrary gates).
- D1 follow-on repair (recorded in
  `experiments/mechanism_transfer_v3/m1/training/DECISION_NOTES.md`):
  `segment_moment` now pools one statistics vector across the two exchangeable
  blocks; independent per-block statistics broke exact invariance (measured swing
  ~12 pre-fix → 0 post-fix on 64 frozen A states). Pre-fix run preserved under
  `artifacts/mechanism_transfer_v3/m1/training/superseded/blockwise_stats/` and
  excluded from results.
- Did NOT change: bank, seeds, lrs, epochs, supervision, selection rule, and every
  other arm (D1); the M1 claim level (whole-orbit reference removes the risky sort
  without accuracy cost; flip-side gain over sorted stays unstable:
  +0.0654/+0.0005/−0.0311, last two CIs cross 0); the M2 negative
  (frozen head 0.796 on true geometry vs 0.000–0.013 random-head;
  geometry-only frontend dev MSE ≈ 0.51 frozen or fine-tuned, J3 ≈ 0.20;
  direct end-to-end 0.918 ≥ analytic 0.886 — see
  `reports/mechanism_transfer_v3/M2_TRANSFER.md`).
