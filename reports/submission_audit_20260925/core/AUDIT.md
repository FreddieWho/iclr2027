# Core claim audit

Date: 2026-09-25
Baseline: 3fc9757

## Findings

- **Fresh composition:** the active 151/176 (85.8%) headline is supported by the archived n04_clean aggregate. Eligibility requires both component edits to preserve the oracle label and both component predictions to be correct; the composition changes the oracle label. The aggregate does not establish 176 independent parents.
- **P1 confidence:** archived summaries support the reported direction when confidence is identified as pre-change/start confidence and endpoint error is described conditional on a correct start plus true semantic flip. Frozen dev cutoffs and ties mean bins need not have equal counts.
- **P2 decision:** refusal and coverage rates use 93 oracle-feasible scenes, not all 171 test scenes. Non-refusal precision is similar. Affine-control wording is limited to the tested grid, operating point, and seeds.
- **P3 repair/migration:** the full-repair plus migration identity is supported among clean-baseline hard cases. Migration accounts for about 79–94% of endpoint repair in the audited dev summaries. Parent-cluster intervals resample parents while retaining quartets, but point estimates are quartet-row weighted, not equal-parent estimates. Fresh 109/141 and P3 110/120/110 use different conditioning sets.
- **WP4:** archived six-arm correlations are unchanged by the Spearman implementation fix because saved vectors have no ties. Current code uses average ranks, pairwise finite observations, and returns no coefficient for constant or underspecified inputs.

## Reproduction and scope

Exact denominators, code/evidence paths, numeric values, allowed wording, and limitations are in reports/submission_audit_20260925/core/claims.json.

Targeted regression command: python3 -m pytest tests/test_submission_audit_core_spearman.py -q

The P1/P2/P3 confirmation and fresh-holdout values above are read-only archived-summary/prediction checks, not new inference. No sealed raw inputs were reopened. No training, broad testing, downloading, external transfer, or additional mechanism exploration was performed.

## Remaining limits

The fresh aggregate lacks unique-parent counts and a parent-level uncertainty interval for its 176 eligible cases. The P2 decision slice is a single seeded split. P1 bins are conditional diagnostics. P3 cluster intervals do not change the row-weighted estimand of point estimates. Keep these boundaries explicit in the submission.
