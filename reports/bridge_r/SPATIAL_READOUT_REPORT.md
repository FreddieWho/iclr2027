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
   all CIs below zero. More spatial bins + more parameters = worse linear
   accessibility, at every scale.
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
5. **Does changed-region response weaken S->B->L? No — absolute response
   grows, relative ratio holds.** Mean d_cos on changed patches (edit AB):
   S 0.0161, B 0.0442, L 0.0740; R_local (changed/unchanged): S 4.6-6.0,
   B 5.6-7.0, L 5.9-7.2 across edits. Larger models' patches respond MORE
   in absolute terms; the changed-vs-unchanged contrast (~6x) is
   scale-invariant. The attention/invariance story in its naive form
   ("large models don't register micro-edits") is rejected: they register
   them more strongly, but organize them less accessibly.

## Interpretation

- **SPATIAL_POOLING_EXPLAINS is rejected**: uniform finer pooling hurts,
  it does not rescue.
- **LOCAL_INFORMATION_WEAK_OR_OOD is rejected as stated**: changed-region
  contrast ~6x at all scales; local features are not dead.
- What remains: local information is present (sensitivity ratios, R5
  rescue), ordinary readouts cannot localize/use it (R3/R4 losses), and
  linear accessibility declines with size even though absolute local
  response grows. I.e. a localization/organization failure with a
  size-dependent accessibility gradient — not a pooling artifact, not
  dead patches.

## Base-vs-atomic pattern (tracked per §19)

Base accuracy rises with size (S 0.888 / B 0.909 / L 0.925 at R0) while
A/B atomic falls (S 0.787/0.749, B 0.746/0.736, L 0.701/0.690). Larger
models are better at the unedited state and worse at edited states —
consistent with a size-dependent reduction in linear accessibility on
these abstract edits; stronger invariance is one hypothesis to be tested.
