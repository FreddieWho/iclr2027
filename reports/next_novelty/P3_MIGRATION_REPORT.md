# P3_MIGRATION_REPORT (2026-09-22): repair moves errors, rarely removes them

## State-transition table (dev512 bank, 237 quartets, clean -> flipmine)
Baseline-110 subset (n=110-120): R_full 0.03-0.10, R_endpoint 0.43-0.48,
M (migration) 0.36-0.40, all with quartet-paired CIs. Of every 10 endpoint
"gains", ~8-9 are AB-right-with-atomics-wrong.
Joint consistency J: clean 0.051-0.055 -> flipmine 0.059-0.076 (+1-2pp).
Cross-checkpoint breadth (s11): clean 0.055, flipmine 0.068, fliprand 0.055,
supmine 0.042, auxmargin 0.021, relfeat 0.131 (exploratory: relational
features double J single-seed; needs multi-seed before any claim).

## Preserve-balanced pilot (P3-D): negative
fliprep vs fliphalf vs keepbal (1:1 flip/preserve BCE), 3 seeds:
- J: no arm consistently best (all 0.03-0.11); keepbal wins 1/3 seeds.
- R_full: keepbal vs fliprep s11 0.073>0.018, s23 0.075<0.10, s47 tie.
  Inconsistent direction.
- M: 0.30-0.38 in ALL 9 runs — preserve-balancing does not reduce migration.
- static/single/FA: indistinguishable across arms.
Scope: BCE-balance only; margin/teacher variants not tried (recorded).

## Grade: P3-PHENOMENON
Error migration is dominant and robust (survives preserve-balancing);
no better repair method found here. Novelty, if any, must come from
measurement + tailored balance ideas — not from preserve loss itself
(which just failed its pilot).
