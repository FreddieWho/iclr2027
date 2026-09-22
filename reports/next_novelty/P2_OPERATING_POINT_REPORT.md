# P2_OPERATING_POINT_REPORT (2026-09-22): threshold or capability?

## Equal coverage (rank-based thresholds, dev-matched, eval-compared)
At matched coverage levels with both arms acting (n~65-82), repair quality
exceeds clean quality 3/3 seeds (0.27-0.45 vs 0.11-0.20) at higher action
cost (0.10-0.13 vs 0.06-0.07). Ties prevent exact matching; nearest levels
reported with gaps. Quality|same-coverage favors repair — not explainable
by coverage alone.

## Affine-clean control (dev-fit, eval-compared)
Wide-grid best (alpha 0.5-0.75, b -4..-5, loss 0.07-0.15): even extreme
flattening cannot match repair coverage+prevalence well, and where it
approaches coverage it destroys quality.
- Action lexico success: clean 0.14-0.20 / affine 0.08-0.16 / flip 0.17-0.24.
  Paired CI affine-clean: s11 [-0.20,-0.03] (worse), s23/s47 overlapping 0.
- Action score success: clean 0.15-0.25 / affine 0.22-0.23 / flip 0.28-0.34.
  Affine partial, never full.
- Behavior (dev bank): static clean 0.15-0.17 / affine 0.18-0.20 (WORSE) /
  flip 0.13-0.17; single clean 0.67-0.70 / affine 0.58-0.62 (partial) /
  flip 0.47-0.54; preserve FA flat ~0.01-0.03 all; comp clean 0.84-0.86 /
  affine 0.67-0.69 / flip 0.73-0.79; joint 0.05-0.06 / 0.05-0.06 / 0.06-0.08.
Affine reproduces single-flip and comp-endpoint direction, fails static,
joint, and action-lexico. NOT a pure operating-point shift.

## Rule dependence (kept, with paired CIs)
Lexico gain = coverage with quality drag; score gain = better forced-choice
ranking. Not contradictory: different interfaces price the same
representation change differently. No merging.

## Reading
Repair = genuine component capability (static/single/recall/ranking) +
operating-point shift (action coverage), but NO joint compositional
capability (P3: J flat 0.05-0.08 all arms). Unified with P3: capability
gains are component-level; composition-level unchanged.
