# Event Updater Report (2026-09-22)

Frozen clean encoder (z=32) + 2-layer MLP event head on [z_t,z_1,dz,|dz|].
Train 4000 / dev 2000 single transitions (1:1 stay/flip, parent-disjoint).
Seeds 11/23/47.

## G2 gate: FAIL decisively
- Dev Acc_event: s11 0.673 (best ep8) / s23 0.679 (ep3) / s47 0.716 (ep0).
  All below the 0.80 STOP line, far from 0.90. Flat after best epoch.
- Train fit: 0.903 / 0.822 / 0.762 — s11 memorizes train yet dev stalls:
  frozen z supports memorizing transitions on seen parents but carries no
  generalizable transition signal. The failure is not "lack of trying":
  explicit update supervision cannot install what the representation lacks.

## Verdict chain (no further main-line work)
G3/G4/G5/G6 not opened. No arch expansion per protocol.

## Side C: SECOND_ORDER_KILL
Quartet interaction deficit D_AB f (logit + hidden norm), 3 mining seeds:
hit medians 0.75/0.42/0.47 vs miss 0.61/0.55/0.58 (logit);
hidden 0.32/0.18/0.20 vs 0.26/0.23/0.25 — direction flips by seed
(s11 hit>miss, s23/s47 hit<miss), magnitudes overlap. No stable deficit.
CLOSED permanently.

## Side D: radius signal SURVIVES (retained diagnostic)
R = r_m / r_s over 24 directions, transition-miss vs hit groups, 3 seeds:

| seed | R_hit q25/med/q75 | R_miss q25/med/q75 |
|---|---|---|
| 11 | 0.39 / 0.64 / 1.44 | 1.03 / 1.90 / 4.06 |
| 23 | 0.28 / 0.81 / 1.71 | 1.21 / 1.87 / 3.76 |
| 47 | 0.23 / 0.94 / 1.74 | 1.03 / 1.87 / 6.47 |

Decomposition: oracle r_s quartiles hit≈miss (med 0.18 vs 0.20 — task
difficulty matched); model r_m medians 0.16–0.21 (hit) vs 0.41–0.46 (miss).
The whole separation comes from the model's own decision radius, straddling
1.0: hits from over-eager regions, misses from under-sensitive ones.
Caveat: probe flip shares directions with R_m (partial circularity); the
r_s-matched design + 1.0-straddle keep it interpretable. No loss built
(loss development explicitly not authorized by this verdict).
