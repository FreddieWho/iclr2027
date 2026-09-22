# P2_CLOSURE_REPORT (2026-09-22): operating point vs capability

Code: experiments/repair_decomposition/p2_operating.py (frozen wide grid:
alpha 0.25..4, b -8..+8; committed BEFORE this run) + p2_affine_atoms.py.
Table: p2_table.npz (sha e7ee314b). Dev = first 86 scenes (fit only);
eval = remaining feasible scenes.

## Affine fits (dev)
s11 (1.0,-8.0, EDGE), s23 (0.5,-3.0), s47 (0.5,-3.0); dev loss 0.08-0.15.
s11 edge noted: even extreme shift cannot match repair coverage there.

## Action lexico (eval): affine does NOT reproduce repair
success: clean 0.14-0.20 / affine 0.11-0.17 / flip 0.17-0.24.
Paired CI affine-clean: s11 [-0.14,0.07], s23 [-0.14,0.05], s47 [-0.06,0.11]
(all overlap 0 or negative); flip-clean: all overlap 0 too (small-n honest).

## Action score: affine partial, never full
success: clean 0.15-0.25 / affine 0.18-0.23 / flip 0.28-0.34, 3/3 seeds.

## Equal coverage (own staircases, gaps reported, paired CIs)
At matched levels repair quality > clean in most levels (s11 L3/L4 CI
[0.21,0.45]; s47 L2 [0.09,0.36]; s23 mixed), at higher cost.
Quality|same-coverage favors repair: not explainable by coverage alone.

## Behavior metrics (dev bank, analytic affine on stored logits)
static: clean 0.15-0.17 / affine 0.18-0.20 (WORSE) / flip 0.13-0.17.
single: clean 0.67-0.70 / affine 0.58-0.62 (partial) / flip 0.47-0.54.
preserve FA flat ~0.01-0.03 all. comp: clean 0.84-0.86 / affine 0.67-0.69 /
flip 0.73-0.79. joint: 0.05-0.06 / 0.02-0.06 / 0.06-0.08.
atomic pass: clean 0.52-0.56 / affine 0.36-0.47 / flip 0.37-0.46.
Affine reproduces single-flip and comp-endpoint direction, fails static,
joint, and action-lexico. NOT a pure operating-point shift.

## Rule dependence (kept, not merged)
Lexico gain = coverage with quality drag; score gain = better forced-choice
ranking. Temperature: lexico sets invariant under T rescaling (verified);
score-rule selections shift 5-14% (scale, not representation).

## Grade: P2-MAIN_SCOPED
Equal-coverage advantage + affine failure on core behaviors + reproducible
code. Scope: dev scenes/bank; static/single gains are component-level.
