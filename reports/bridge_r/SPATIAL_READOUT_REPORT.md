# Bridge-R Task 2: Fine Spatial Readout Diagnostic — Report (2026-09-20)

Am03. Same v2 train (first 2000 quartets, shared IDs) / dev (760).
Fixed recipe everywhere: StandardScaler + L2 logistic regression
(C=1.0, lbfgs, tol=1e-4, max_iter=5000). R0/R1/R2 recomputed from cached
features under this recipe (bitwise match to published baselines).
Dev diagnostic only. Holdout sealed. No composition metric computed.

## Primary table (dev atomic_mean; full per-state accuracies in CSV)

| model | R0 | R1 | R2 2x2 | R3 4x4 | R4 7x7 | R5 oracle-ROI |
|---|---|---|---|---|---|---|
| DINOv3-S | 0.7678 | 0.7474 | 0.7243 | 0.7118 | 0.7013 | 0.7914 |
| DINOv3-B | 0.7408 | 0.7125 | 0.6967 | 0.6822 | 0.6487 | 0.7743 |
| DINOv3-L | 0.6954 | 0.7000 | 0.6546 | 0.6395 | 0.6336 | 0.7421 |

All 18 cells: Gate B1 FAIL (no ATOMIC_ACCESSIBILITY_RESCUED; Task 2 stops here).

## Answers to the five questions

1. **Does 4x4 rescue over 2x2? No — it hurts.** R3-R0 paired bootstrap
   (10k, quartet units): S -0.056 [-0.082,-0.030], B -0.058 [-0.088,-0.031],
   L -0.056 [-0.084,-0.027]. Finer uniform pooling strictly loses.
2. **Does 7x7 continue recovery? No.** R4-R0: S -0.066, B -0.092, L -0.062,
   all CIs below zero. Under the fixed linear readout and fixed training
   budget, naively keeping more spatial bins does not improve readability
   and hurts generalization instead. This is partly a high-dimensional
   statistics effect (16C/49C dims on 8000 states at fixed C=1), not purely
   a visual-organization mechanism.
3. **Does privileged ROI restore competence? Partially, never fully.**
   R5-R0: S +0.024 [-0.002,+0.049] (marginal), B +0.034 [+0.007,+0.061],
   L +0.047 [+0.018,+0.075]. Oracle localization rescues 2-5pp with the
   largest rescue on the largest model — but R5 absolute (0.74-0.79) stays
   far below B1. Information is present locally yet even perfect
   localization does not restore competence.
4. **Do size gaps shrink with finer readout? No — they persist.**
   A_S-A_L paired CI: R0 +0.072 [+0.041,+0.104], R1 +0.047, R2 +0.070,
   R3 +0.072, R4 +0.068, R5 +0.049 — every CI excludes zero. The inverse
   size trend survives all six readouts.
5. **Does changed-region response weaken S->B->L? No — but cross-size
   magnitudes are not calibrated.** Within each representation space,
   Large shows larger cosine displacement (AB: S 0.017, B 0.047, L 0.078);
   feature geometry differs across sizes, so `0.078 > 0.017` must NOT be
   read as a calibrated "L is 4.6x more sensitive than S". The firmer
   half: all three sizes show a stable changed-vs-unchanged contrast
   (S 4.9-7.2x, B 5.9-8.6x, L 6.2-8.7x across edits, backgrounds now
   exactly locked to feature inputs). The naive "large models don't
   register micro-edits" story is rejected; a universal-invariance claim
   is not licensed.

## Closure controls (audit-requested): R5 rescue was geometric selection

R5 recomputed ROI per state from full geometry, so ROI position itself
could carry label information. Two controls on the same patch cache:

- **R5b position-only** (normalized ROI coords, same LR, no DINO
  features): 0.7882 identically for S/B/L — at or above the DINO-ROI
  scores (S 0.7914, B 0.7743, L 0.7421). R5-minus-R5b paired CI covers
  zero for S/B and is significantly NEGATIVE for L (-0.046). DINO ROI
  features add nothing over bare position; for L they add noise.
- **R5c fixed-base ROI** (one window per quartet from the base state,
  leakage-free within quartet): S 0.7684, B 0.7368, L 0.7039 — all
  within noise of R0 (paired CIs cover zero). Without per-state
  re-aiming, rescue disappears.

Conclusion: the R5 lift came from privileged geometric selection, not
from visual relation information decoded out of patches. The earlier
R5-based "local information present" reading is withdrawn. What stands:
(1)-(2) pooling-hurt results, (3) the size gap, (4) the sensitivity
contrast (response, not task-usable information).

## Interpretation (post-controls)

- **SPATIAL_POOLING_EXPLAINS is rejected**: uniform finer pooling hurts,
  it does not rescue (with the fixed-budget statistical caveat above).
- **The R5-based localization claim is withdrawn**: see closure controls.
- What remains standing: ordinary readouts cannot access the relation
  (R3/R4 losses); the S->B->L accessibility gap survives all readouts;
  edited regions evoke a strong, specific patch response (~5-9x
  contrast) whose absolute scale grows with size within each geometry.
  I.e. patches register the edits, but no linear readout — global,
  uniform-spatial, or leakage-free localized — converts that response
  into task information. That is a weak-local-information (for this
  task) finding, not a pooling artifact and not dead patches.

## Base-vs-atomic pattern (tracked per §19)

Base accuracy rises with size (S 0.888 / B 0.909 / L 0.925 at R0) while
A/B atomic falls (S 0.787/0.749, B 0.746/0.736, L 0.701/0.690). Larger
models are better at the unedited state and worse at edited states —
consistent with a size-dependent reduction in linear accessibility on
these abstract edits; stronger invariance is one hypothesis to be tested.
