# P2_REPORT (2026-09-23): repair gain decomposition, 3 seeds

Immutable table p2_table.npz (sha e7ee314b, U02 cache frames/feas + 6 logit
vectors). Eval = 96 feasible scenes (139 feasible total; dev = first 86 of all 256
scenes order-fixed; eval = feasible scenes not in dev).

## Lexicographic rule, lam=2 (refusal gate)
Symmetric split of net success gain (total +0.03..+0.08):
- s11: D_coverage +0.095 / D_quality -0.043
- s23: D_coverage +0.099 / D_quality -0.016
- s47: D_coverage +0.061 / D_quality -0.030
Strata (both/clean-only/repair-only/neither): 50/5/34/7, 45/5/38/8, 49/6/33/8.
Common-act paired (same scenes both act): clean_ok > repair_ok 16v8, 12v9,
9v7 — on identical acted scenes repair is worse, 3/3.
Candidate sets: precision flat/down (0.37->0.31-0.38), recall up
(0.40-0.42->0.60-0.68), 3/3. Repair finds more feasible candidates without
ranking them better.
Interface 4-cell: collapses to set-only by definition (lexico score stage =
cheapest cost, model-independent) — verified, reported as definitional fact.
Equal-coverage: infeasible — repair dev coverage at tau=0.95 (0.50-0.63)
still exceeds clean at tau=0.5 (0.37-0.43); closest-match eval mixed 2/3.
Recorded as instrument limitation ("bolder" model saturates thresholds).

## Score rule, lam=2 (forced choice, no refusal)
Success rate: s11 0.25->0.30, s23 0.19->0.34, s47 0.15->0.28 — consistent
quality gain 3/3. Rule-dependent picture: lexico gain = coverage with
quality drag; score gain = better forced-choice ranking.

## Temperature separation
Lexico feasible sets provably invariant under T rescaling (verified);
score-rule selections shift 5-14% (T=0.5/2.0). Scale effects separated
from representation effects.

## P1 linkage (analogue, tiny-n honest)
High-confidence scenes (top quartile clean selected-P0, n=13-14): success
near 0 both arms (s11 repair 0.0 vs clean 0.07; s23 tie; s47 repair 0.14 vs
clean 0.0). Coverage gains do NOT reach the high-confidence group.
Supports: "repair lets more feasible actions into scope, but that does not
mean already-high-risk states are fixed."

## Static accuracy (separate metric, not decomposed here)
Fresh static improves 0.180->0.107. The decomposition concerns action
success only; no blanket precision denial (see C3).
