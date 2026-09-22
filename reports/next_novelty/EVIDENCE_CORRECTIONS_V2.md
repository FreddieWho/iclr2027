# EVIDENCE_CORRECTIONS_V2 (2026-09-22, HEAD 578d0f0 + worktree)

## V2-1. paired_compare preprocess scare: INVESTIGATED, CLEARED
Suspected missing standardization (raw model call). Found: old code did
manual (x-mu)/sd inline; ckpt has no featurize flag, so preprocess() is
exactly equivalent. Re-ran with preprocess(): bit-identical results
(same cells, same rates). No bug, no void. Refactor kept for clarity.

## V2-2. P2 eval n=96, not 170 (REPORT TYPO, fixed)
P2_REPORT said "Eval = 170"; actual eval feasible scenes = 96 (139 feasible
minus 86 order-fixed dev). Corrected in place. Numbers unchanged.

## V2-3. WP-B seed11/seed47 target duplication (carried note)
Transfer report already documents computationally identical runs reported
as one table. No information lost.

## V2-4. Unit-test tolerances are rounding-aware (5e-4), documented in test.
