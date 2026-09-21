# P2 Draft: Coverage, Not Precision (Discussion段草稿, 2026-09-22)

All repair gains in this paper come from expanding WHERE the model decides,
never from improving HOW WELL it decides at a fixed point. Four independent
lines converge:

1. **U02 decision interface**: flipmine's gain is feasible-set coverage
   (refusal 0.43 → 0.12, 3.6× set size) with non-refusal precision
   unchanged (0.23 vs 0.22); the gain survives score/lexicographic/calibrated
   rules because the feasible set itself grew.
2. **N06 cost sweep**: at λc=0 (pure judgment, no cost pressure) clean vs
   flipmine are identical (0.309 vs 0.331); the +9pp at λc=2 comes from
   higher flip confidence surviving cost pressure — bolder, not wiser.
3. **WP2 boundary diagnostic**: flipmine raises local boundary coverage
   (9.6% → 16.9%, CI excludes 0) while signed alignment (≈0) and offset
   medians do not move — more roots near semantic boundaries, same
   orientation.
4. **WP-B transfer denominators**: repair sets shift which quartets enter
   the conditional denominator; transition behavior itself is flat
   (0.42–0.56) across all 45 retrained models.

Statement kept narrow: "every repair in this work expands decision coverage;
none improves pointwise precision; the remaining defect (confident blind
spots, §P1) is coverage-resistant." No causal verb beyond the data.
Placement: Discussion synthesis paragraph + honesty appendix pointer.
