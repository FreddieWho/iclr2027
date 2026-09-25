# Route and structural evidence audit

**Checkout:** `3fc9757398ebc88652362b906a2a98710bf17650`
**Scope:** e832_focus Routes 1–4 and structural claims. No sealed holdout or confirmation assets were read. No training was run in this lane. Route 2's matched clean baseline was run by the coordinating lane on the authorized GPU; this audit recomputed its metrics from the returned prediction arrays.

## Route 2: corrected metrics and matched clean baseline

The fresh visual bank has **28 quartets from 19 parents** (112 images), with 323 train singleton images, 81 dev singleton images, and no AB training images. The train/test data archive SHA-256 is `8e5128e51499b32bc5082a010a27d0e19542a1979325c31969780594f1e221ee`. Independent recomputation from v2, v3, and matched-static NPZ predictions verified labels and parent IDs against that bank; all logits were finite.

Correct definitions used here are `J3 = A & B & AB correct` and `J4 = P & A & B & AB correct`. The preserved v2 correction summary copied J3 into J4. Its J4 differs from prediction-level PABC correctness for all three representation seeds and interaction seed 806; the other values match only by coincidence. The v3 direct arm has P accuracy 1.0, so its J3 and J4 happen to be numerically equal. The old outputs were left intact. The new runner, static baseline, and prediction postprocessor now calculate J4 with P correctness included ([gpu_run.py](/home/huyudi/012_conference/iclr2027/experiments/e832_focus/route2_visual/gpu_run.py:30), [static_baseline.py](/home/huyudi/012_conference/iclr2027/experiments/e832_focus/route2_visual/static_baseline.py:47), [recompute_from_predictions.py](/home/huyudi/012_conference/iclr2027/experiments/e832_focus/route2_visual/recompute_from_predictions.py:20)).

The native RGB path thresholds visible-color masks before ImageNet normalization, and the v3 run manifest records area pooling. Area pooling preserves narrow line masks that nearest downsampling erased in v2. The future runner now defaults to area while retaining explicit `--mask-pool nearest` replay, and checkpoints/results record the selected pool ([visual_mechanism.py](/home/huyudi/012_conference/iclr2027/experiments/e832_focus/route2_visual/visual_mechanism.py:45), [gpu_run.py](/home/huyudi/012_conference/iclr2027/experiments/e832_focus/route2_visual/gpu_run.py:63)).

Prediction-level J3 means for direct/additive/representation/interaction were **0.571/0.417/0.274/0.583 in v2** and **0.571/0.464/0.619/0.476 in v3**. Interaction-minus-direct J3 by seed was `+0.179, −0.143, 0.000` in v2 and `−0.250, −0.036, 0.000` in v3. The direction is not consistently positive; the visual interaction-advantage claim remains unsupported. Representation recovered under area pooling, which is an implementation sensitivity on this bank, not a general structure result.

The old clean-only reference (mean J3 0.155) is not a comparable training-budget control: it saw 128 unique clean examples per epoch for 20 epochs (2,560 presentations) and selected on 32 clean dev examples, versus 323 train presentations per epoch (6,460 total) and 81 singleton dev examples for the main arm. It remains a low-exposure historical description.

The returned matched clean-only baseline used the same seeds, initial shared-backbone hashes, 20-epoch schedule, data hash, and all 81 singleton dev labels for selection. It repeated its 128 unique clean training rows to 323 presentations per epoch (6,460 total). Recomputed clean-baseline J3 was `0.143, 0.250, 0.214` (mean **0.202**); direct v3 J3 was `0.571, 0.607, 0.536` (mean **0.571**). Paired direct-minus-clean J3 differences were `+0.429, +0.357, +0.321`. The 10,000-draw parent-cluster percentile intervals were `[0.222, 0.633]`, `[0.172, 0.556]`, and `[0.138, 0.517]`, respectively. Conditional composition misses (CCM, among atomically correct A/B pairs) fell from `0.800, 0.632, 0.667` to `0.273, 0.150, 0.250`; paired CCM-difference intervals were `[-0.723, -0.306]`, `[-0.746, -0.217]`, and `[-0.644, -0.184]`.

Among each clean baseline's `110` cases (A and B correct, AB wrong), direct v3 produced:

| Seed | Baseline 110 | Full repairs | Migrations | Endpoint gains | Full-repair fraction (parent CI) | Migration fraction (parent CI) |
|---|---:|---:|---:|---:|---:|---:|
| 803 | 16 | 10 | 1 | 11 | 0.625 `[0.353, 0.882]` | 0.063 `[0.000, 0.188]` |
| 805 | 12 | 9 | 1 | 10 | 0.750 `[0.538, 1.000]` | 0.083 `[0.000, 0.250]` |
| 806 | 12 | 5 | 3 | 8 | 0.417 `[0.100, 0.727]` | 0.250 `[0.000, 0.600]` |

For every seed, endpoint gains equal full repairs plus migrations. This supports a bounded same-bank contrast under matched total presentations and dev selection. The comparison still has only 19 parent clusters and 28 quartets; the clean arm repeats a smaller set of unique examples, and the three fitted seeds share one bank. Do not generalize this estimate or interpret it as evidence that a nonlinear interaction head wins.

The rerunnable prediction recomputation and full per-seed metrics/intervals are in [recomputed.json](/home/huyudi/012_conference/iclr2027/artifacts/submission_audit_20260925/routes/recomputed.json); the script is [recompute_routes.py](/home/huyudi/012_conference/iclr2027/experiments/e832_focus/submission_audit_20260925/routes/recompute_routes.py). Focused Route 2 checks passed: 11 tests, with the CUDA-only input-path test skipped on this CPU checkout. The remote matched-baseline outputs were returned by the coordinator and prediction arrays were available; this lane recomputed predictions; the coordinator subsequently collected and hash-verified all 16 returned files, including the three checkpoints.

## Route 1: source ordering and cross-task boundary

The preserved Round 2 summary JSON reports 1,414 E pairs / 389 parents for source, 417 / 170 for T1, and 912 / 361 for T2. Arithmetic on the saved per-seed summary rows reproduces the stated pattern: rich-sorted exceeds rich-unsorted for all three source seeds under both supervision settings; T1 contrasts include zero or are mixed, and T2 has mixed signs. T1/T2 therefore do not confirm transfer from source.

This is summary arithmetic, not a raw prediction recomputation: Route 1 retains aggregate seed metrics but not per-case logits/correctness arrays. The feature map itself is provably G8-invariant under the implemented permutations: source rich features calculate two segment lengths, four cross distances, midpoint distance, and absolute sine; `sort_features` sorts the two lengths and four cross distances while keeping the other two quantities ([common_runner.py](/home/huyudi/012_conference/iclr2027/experiments/e832_focus/route1_cross_task/common_runner.py:131)). The focused test applies each declared group element ([test_common_runner.py](/home/huyudi/012_conference/iclr2027/experiments/e832_focus/route1_cross_task/test_common_runner.py:14)).

That operation uses the same eight feature quantities, but the audit found no proof that it preserves all their assignments or is information-equivalent to the unsorted vector. Allowed wording is limited to this particular sorted invariant processing and its source-task results. The persisted positive source ordering does not establish T1/T2 transfer.

## Route 3: parser and denominator

The archived first-bank result says parser J3 is 1.0 on 230 E pairs. Its stored metadata and the per-arm JSON rows give **65 unique eligible parent indices**. `mine_E` assigns each E pair the source test-scene row index and the saved `parents` count is the number of unique such indices ([common_runner.py](/home/huyudi/012_conference/iclr2027/experiments/e832_focus/route1_cross_task/common_runner.py:77), [strong_baselines.py](/home/huyudi/012_conference/iclr2027/experiments/e832_focus/route3_baselines/strong_baselines.py:57)). The alternate `142` in the initial Route 3 report line was inconsistent with the artifact; the correct denominator is 230 E / 65 parents. The separately generated Round 3 summary is 171 / 56, with parser J3 1.0.

The parser implements strict orientation-sign segment intersection and the Route 3 score is scoped to the source coordinate task, not image parsing. Route 3 did not archive its test bank states or per-case parser predictions, so the formal 230-pair parser score was **NOT_RUN** by this independent recomputation. Existing small-case parser checks and source-code inspection are diagnostic only and do not replace that missing prediction archive.

## Route 4 and structural claims

Route 4's saved aggregate reports 207 E / 66 parents and 45 rows covering canonical, translation, and 7° rotation conditions. It reports the sorted-versus-unsorted direction as positive at low budget (60 epochs), with the same scores under the three rigid transforms. The transform implementation is coordinate-only and preserves the oracle label on a focused crossing case; the focused regression now exercises label invariance and fails closed for the unimplemented noise branch. Route 4 did not retain per-case predictions, so its formal performance recomputation is **NOT_RUN**. Noise (`sigma=0.003`) and visual-renderer performance remain **NOT_RUN**; rigid coordinate robustness does not imply visual robustness.

The structural archive supports a scoped single-flip sorted-minus-unsorted J3 increase of about `+0.061` to `+0.092` across five seeds, with all five parent intervals excluding zero, on 2,188 E / 871 parents. It is a result for this concrete sort, not proof of equal information or a learned set network. The nonlinear-rho repair improves clean J3 over the additive arm by about `+0.012` to `+0.032`; under flip supervision three of five intervals include zero and repaired J3 remains about `0.017–0.046`. This does not explain the larger geometry gain. These structural predictions are present only as aggregate metrics/intervals, not as archived per-quartet arrays, so their formal prediction recomputation is **NOT_RUN**.

The repository expressivity audit is a four-case algebraic witness for a two-segment sum with linear readout. It does not cover TypedTri, TypedDisk, or vision. Applying that witness to those architectures is forbidden. T1/T2 structural transfer and a vision-structure intervention remain **NOT_RUN**; no vision structure result is claimed.

## Fixes made in this lane

- Corrected Route 2 J4 to require P, A, B, and AB; added a regression example where J3 and J4 differ.
- Made area pooling the future default, while preserving explicit nearest replay and recording the selected pool in run metadata/checkpoints.
- Changed the future clean-only baseline to match the main arm's per-epoch presentation count and singleton dev-selection set by default; retained explicit `--selection-dev clean-only` for historical selection. Metadata records unique clean rows, repeated exposure indices, exposure totals, and dev selection.
- Made Route 4's unimplemented noise transform fail closed and strengthened the rigid-transform test to check oracle labels.

The fixes do not rewrite any historical prediction, result, or report artifact. Claims and their exact evidence boundaries are indexed in [claims.json](/home/huyudi/012_conference/iclr2027/reports/submission_audit_20260925/routes/claims.json).

Final focused Route 4 check: `python -m pytest experiments/e832_focus/route4_robustness/test_robustness.py -q` — **2 passed**.
