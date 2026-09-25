# Final decision — mechanism_transfer_v3

> **Historical decision; superseded by the 2026-09-25 submission audit.**
> The M2 implementation did not perform its labeled frozen/tuned interventions.
> The M2 attribution and exclusion claims below are withdrawn. M1's bounded
> search gives an observed minimum, not a positive global lower bound; flip-side
> noninferiority was not established. Use `reports/submission_audit_20260925/CLAIMS.csv`
> and the mechanism audit for current supported wording and corrected results.

Date: 2026-09-25. Evidence: `reports/mechanism_transfer_v3/M1_MECHANISM.md`,
`reports/mechanism_transfer_v3/M2_TRANSFER.md`,
`reports/e832_focus/route2/REPORT.md`, `reports/e832_focus/route3/REPORT.md`.
Claim ledger: `reports/mechanism_transfer_v3/CLAIM_LEDGER.csv`.
Reproduction: `reports/mechanism_transfer_v3/REPRODUCE.md`.
Paper status: appendix-only, no main-text promotion
(`reports/e832_focus/PAPER_BUILD_RECEIPT.json`).

## Supported contributions (2, both scoped)

1. **Whole-orbit reference removes the risky independent-bag sort without an
   accuracy price on the frozen source bank (M1, source-only).**
   On 2188 E / 871 parents (`artifacts/e832_focus/structure/bank_quartets.npz`,
   SHA `15f5bf18…`), orbit_distance beats sorted/unsorted/raw on clean
   supervision in all 3 seeds with CIs excluding zero
   (`artifacts/mechanism_transfer_v3/m1/training/SUMMARY_3seed.json`;
   detail `.../training/RESULTS.md`).
   Exact invariance is neither necessary nor sufficient for the ordering gain:
   rich8_sorted, orbit_distance, and repaired_segment_rho are all exactly
   invariant, yet their flip J3 spans 0.02-0.59 (same artifacts).
2. **Frozen module transfer fails with precise attribution, closing the
   adaptation-vs-transfer question against transfer (M2, negative).**
   On the expanded blinded 692-quartet/351-parent bank
   (`artifacts/mechanism_transfer_v3/m2/bank/`, SHA `69c4bb5c…`), the frozen M1
   orbit head scores J3 0.796 on true geometry while a random/wrong head of the
   same class scores 0.000-0.013 (competent and specific), but the
   geometry-loss-only frontend stalls at J3 ~0.19-0.22 with dev geometry
   MSE ~0.51 whether the backbone is frozen or fine-tuned, and direct
   end-to-end vision reaches J3 ~0.92, above the analytic image baseline 0.886
   (`artifacts/mechanism_transfer_v3/m2/gpu_run_remote/out/`,
   `artifacts/mechanism_transfer_v3/m2/analytic_baseline.json`).
   The lexicographically canonicalized orbit interface is not learnable by
   smooth MSE regression from these images; probe capacity is excluded as the
   cause. Stop adding heads.

## Explicit negative boundaries (what is NOT claimed)

- **No flip-side gain over the sorted bag.** M1 flip orbit-minus-sorted is
  +0.0654/+0.0005/-0.0311 with the last two CIs crossing zero (same
  SUMMARY_3seed.json). Direction unstable; do not promote.
- **No source information-collision claim.** M1.0 bounded G8 search
  (`artifacts/mechanism_transfer_v3/m1/source_search.json`) found no exact
  merge (nearest opposite-label 0.0471; 3000-step search 0.0335;
  equal-label control closer at 0.0213). Lower bound only.
- **No cross-task transfer.** T1/T2 not run in v3; the standing route1 result
  (T1 uninformative 417 E/170 parents; T2 negative/unstable 912 E/361 parents;
  `artifacts/e832_focus/route1/round2_analysis.json`) is unchanged.
- **No visual interaction method.** Route2 v2+v3 on the fresh 28-quartet/
  19-parent bank (`artifacts/e832_focus/route2/gpu_run_v3_remote_20260925/gpu_run_v3/`):
  v3 recovers representation (mean J3 0.274 → 0.619) and makes additive stably
  slightly negative, but interaction-minus-direct stays unstable
  (-0.250/-0.036/+0.000). Bounded negative; structure advantage not established.
  Contract-broken v1 J3 numbers excluded.
- **No "complex relation model required" for the source task.** The analytic
  segment-intersection parser achieves J3 = 1.0 on 230 E / 65 parents and again
  on the fresh 171-E/56-parent bank
  (`artifacts/e832_focus/route3/results.json`,
  `artifacts/e832_focus/route3/round3_freshbank.json`); learned comparisons
  (best orbit_canonical ~0.30-0.37) are parser-bounded diagnostics. R1 excludes
  simple underfitting (same bank/arms, 300 epochs, train BCE ~0.18-0.29 yet J3
  0.004-0.065; `artifacts/e832_focus/route3/round1_optim.json`).
- **No pooled denominators, ever.** 2188/871, 692/351, 230/65 (+171/56),
  28/19 stay separate; no comparison of the parser 1.0 with unrun visual
  numbers; no layered/transparent-renderer, cross-renderer (M2.4), or sealed-pool
  claims (none run — see `reports/mechanism_transfer_v3/M2_TRANSFER.md`
  Non-claims).
- **No main-text promotion.** All six appendix notes stay in
  `paper/sections/app_l015.tex` tail. Conclusion stays on page 9.

## Decision

Accept the two scoped contributions above; enforce all seven boundaries.
No further GPU rounds on the M2 frozen-module line (attribution is precise and
unfreezing changed nothing: 0.21 vs 0.20). No optimization rounds to chase the
route2 interaction arm (v3 exhausted the single evidenced fix; user GPU
rounds 2-3 retained unused per route2 report stop rule).
