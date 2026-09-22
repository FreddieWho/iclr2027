# P1_FINAL_REPORT (2026-09-22): confidence for current vs future reliability

## Unified population (bank_dev512: 474 paths + 20480 singles/model; frozen bank)
Rank policy: ranks over path starts + one row/parent from singles; absolute
thresholds frozen per model on dev; ties merged (tie share 0.30 deduped base);
parent-cluster bootstrap throughout.

## B: three risks, same rank policy (s11 clean shown; 6/6 same shape)
- current: 0.43 -> 0.05 across bins (classic: confidence predicts correctness)
- preserve: 0.40 -> 0.04 (same direction as current)
- flip (start-correct): 0.58 -> 1.00 (REVERSED)
Flip risk 0.85-0.88 clean / 0.69-0.76 repair; preserve-spurious 1-2%.

## C/D: trust selection + environment composition
Retention top25% (dev-frozen thresholds): flip_err 1.00 (clean) with
preserve_err 0.04-0.06. R(tau,rho) = (1-rho)*preserve + rho*flip:
at top25, R = 0.28 (rho=.25), 0.52 (rho=.5), 1.0 (rho=1).
Whether trusting high confidence is good policy depends entirely on the
change composition rho of the deployment environment. No overall-deployment
claim beyond the retained population.

## E: confound control
Regularized logistic (conf + margins + distance + norm + class + family):
AUC conf-only 0.94 vs no-conf 0.67; full 0.96; AUC-gain CI [0.23,0.36]
excludes 0 (dev s11; same direction all models). Max |corr| 0.47 noted.
No single variable absorbs confidence. Unresolved: unmeasured difficulty
beyond oracle margin/distance (stated, not controlled away).

## F: threshold-distance control
Normalized |f|/|Df|: AUC 0.989 vs raw 0.926 (dev s11); joint coef
[0.75, 3.50] — confidence keeps independent value. P1 NARROWED, not
explained away: most of the effect rides threshold distance, with a
residual confidence contribution. Temperature T*=9.6 preserves ranks
exactly; static calibration does not touch the ordering.

## G: repair residual, strictly paired (clean-defined high-risk, no repair-start filter)
High-risk cells: repair-start-correct (n~50): repair err 0.83-0.92;
repair-start-wrong (n~15): repair end err 0.0 (boldness again).
Paired delta CI: s11 [0.0,0.25], s23 [0.0,0.31], s47 [0.04,0.37] (dev).
Common-correct secondary consistent. Means improve; residual dominates.

## H: confirm (confirm1007 bank, frozen thresholds/settings)
Flip curves: Q0 0.53-0.61 -> Q3 1.00, 6/6 models. Confound gain CIs exclude
0 all 3 seeds. P1-G paired deltas exclude 0 (s11 [0.19,0.43], s23
[0.28,0.64], s47 [0.04,0.29]). Grade: P1-STRONG (coordinate).

## Risk demo (minimal, held-out confirm): NO independent value
Risk score vs conf-only vs geometry at fixed 10/20% budgets: conf alone
already captures 1.00 (base rate 0.86); geometry 0.88-0.91; random 0.86.
Simple confidence saturates the budget demonstration. P1-C retained;
no method extension.
