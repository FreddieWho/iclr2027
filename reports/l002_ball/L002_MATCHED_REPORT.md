# WP-D Report: motion explains S2 (2026-09-22)

## Matched test: INFEASIBLE (structural)
468 event/control pairs (WOY+WMX); SMD on motion covariates 0.6–2.7
(ball_speed 2.65, accel 1.75, mean disp 1.87) ≫ 0.10 gate. Contested events
are intrinsically higher-motion; matched non-event windows with equal motion
barely exist. The pre-registered matched estimand has no empirical support.

## Residual check (exploratory): motion explains 70–90%
Logistic P(turn|motion) fit on non-event windows, predicted on event windows:

| match | obs | motion-predicted | excess | 95% CI |
|---|---|---|---|---|
| WOY | 0.249 | 0.175 | +0.074 | [0.065, 0.081] |
| WMX | 0.187 | 0.170 | +0.017 | [0.009, 0.026] |

Excess CIs exclude 0 but are small and heterogeneous (+7pp vs +2pp); with
only 4 motion covariates, residual confounding cannot be excluded either.

## Consequence
- L002 S2 (+7~8pp contested presence) is downgraded from keeper to
  **motion-confounded observation**: most of it is "duels move fast",
  not event selectivity.
- D5 cover-selectivity, D6 curves-as-claim, D7 expansion: all dead.
  No more matches/seeds (would only shrink an already-confounded excess).
- Event curves computed and archived (matched_pairs.json) for appendix honesty.
