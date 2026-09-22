# P3_CLOSURE_REPORT (2026-09-22): paired repair comparison, parent-clustered

## State-transition table (dev512, 237 quartets, clean -> flipmine)
Baseline-110 subset (n=110-120): R_full 0.03-0.10, R_endpoint 0.43-0.48,
M 0.36-0.40, all parent-cluster CIs. Dominant repair move is 110->001 (26/40/39 across seeds: AB fixed, BOTH
atomics broken); full repair 110->111 is rare (7/5?/11). Of every 10 endpoint gains, ~8-9 are
AB-right-with-atomics-wrong.
Joint consistency J: clean 0.051-0.055 -> flipmine 0.059-0.076 (+1-2pp).
Common-correct set: direction inconsistent incl. s23 reversal
(0.920 -> 0.947).

## keepbal fair rerun (§12 formula): still negative
fliprep vs fliphalf vs keepbal, 3 seeds: J all 0.03-0.11, no arm
consistently best (keepbal wins 1/3); R_full mixed directions; M 0.23-0.40
all arms. Preserve balancing per the fair formula does not reduce
migration. Old unfair keepbal numbers superseded (kept as history).

## RelFeat closure + conditional extension (§13-15)
relfeat clean vs raw clean, 3 seeds: J 0.13/0.17/0.21 vs 0.05 (mean Delta-J
+0.11 >= 5pp); static 0.06 vs 0.15-0.17 (improved, not degraded); atomic
pass 0.68-0.71 vs 0.52-0.56 (gain source: primarily atomic, AB secondary).
Confirm: J 0.16-0.23 vs 0.04-0.06. Gate passes -> RELFEAT_J_SUPPORTED.
relfeat+flipmine: dev J 0.38-0.42, R_full 0.36-0.44, M 0.32-0.42; confirm
J 0.44-0.47, R_full 0.45-0.50, M 0.25-0.30. Full repair now exceeds
migration. Grade: P3-METHOD (bounded per §16 wording).

## Grade: P3-PHENOMENON for raw flipmine; P3-METHOD for relflip
No further repair variants without new authorization.
