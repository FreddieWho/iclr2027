# M1.2 decision notes

Short records of onsite implementation choices in the M1.2 source matrix, with
the information contract or learning evidence that justifies each one. Nothing
here changes the task, the denominator, the seeds, or the supervision.

## D1 — Block-shared statistics for `segment_moment` (2026-09-25)

**What changed.** The two 5-column segment blocks of the `segment_moment`
descriptor initially received independent per-column train statistics. They now
share one 5-vector of statistics, pooled across both blocks and tiled onto each
block.

**Why.** The arm is pre-declared as a role-preserving reference whose symmetry
is exact: the descriptor is endpoint-swap invariant and segment-swap covariant,
and the fusion `0.5*(q(s0,s1)+q(s1,s0))` is symmetric. Independent per-block
statistics do not commute with the block swap, so the standardized features
changed under a legal segment swap and the trained model was not invariant.
Measured on 64 frozen A states before the fix: max abs logit swing ~12. The
frozen recipe already applies one shared xy statistic to all four points for
exactly this reason (`experiments/f095_campaign/u01_models.py` docstring: the
typed model is invariant to the group "when inputs are scalar-standardized").
Pooling statistics across the exchanged blocks is the same choice for a
two-block descriptor.

**Evidence.** New fail-closed tests:
`test_fitted_statistics_are_block_shared`,
`test_standardized_features_are_exactly_group_invariant`, and
`GroupSwingTest.test_reference_arms_have_zero_swing_and_raw_does_not`
(reference arms <= 1e-4 on real A states; raw and `rich8_unsorted` > 1e-3).

**What is preserved.** The pre-fix run is kept under
`artifacts/mechanism_transfer_v3/m1/training/superseded/blockwise_stats/` (nine
`segment_moment` cells plus the pre-fix `metrics_3seed.json`,
`SUMMARY_3seed.json`, `selection.json`). Those numbers are a preliminary
variant and are not the M1.2 result.

**What is not changed.** The bank, the seeds, the lrs, the epochs, the
supervision, the selection rule, and every other arm.

## D2 — `orbit_distance` standardization order (2026-09-25)

The lexicographic orbit representative is computed on raw coordinates and then
standardized per column. Because the representative is identical under every
legal permutation, per-column standardization cannot break the group action;
the measured swing on real A states is exactly 0. No change was needed here;
the check is recorded so the two role-preserving references are not silently
treated as equivalent.

## D3 — Arms kept exactly as pre-stated

The matrix keeps the six pre-stated arms. The legacy `raw_matched` capacity
control and the `sixdist` and `additive` controls from the frozen report are
not re-run: `raw_matched` is a capacity control, `sixdist` is a subset of the
`rich8` quantities, and `additive` is the legacy segment-sum negative control
already covered by `repaired_segment_rho` on the same eight coordinates. Adding
them would change the matrix after seeing results.
