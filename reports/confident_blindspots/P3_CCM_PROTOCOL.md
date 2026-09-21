# P3: Conditional-Composition-Miss (CCM) Measurement Protocol (2026-09-22)

A reusable protocol for measuring DECISION UPDATING (not static accuracy),
validated across coordinate / football / pixel in this paper.

## Definitions
1. **Conditional composition miss**: P(ŷ_AB ≠ y_AB | ŷ_A = y_A, ŷ_B = y_B,
   y_A = y_B = y_0, y_AB ≠ y_0). Headline: 151/176 = 85.8% (fresh holdout).
2. **Matched-single control**: displacement-matched single-edit flips;
   composition-specific gap = conditional miss − matched-single miss
   (+29pp coordinate, ±11pp).
3. **Transition-miss pair**: report BOTH denominators — random-screen
   single-turn paths (108/128 = 84.4%, boundary-hugging subpopulation) AND
   targeted fresh paths (28/64 = 43.8%). Never average, never generalize
   across denominators.
4. **Repair reporting**: static + atomic A/B + both-fraction + composition
   miss (triple denominators: own / baseline-fixed / common-correct) +
   matched-single + preserve FA. No denominator-shopping.

## Adoption criteria (when CCM applies to a new task)
- Oracle (or trusted labels) for base, atomic edits, AND joint outcome.
- Atomic effects known per-edit; joint effect known and sometimes ≠ parts.
- A matched-single control constructible (displacement/norm-matched).
- Fresh-split discipline for the headline number (one-shot).

## Known limits (ours)
- Pixel matched-single cell incomplete (CNN checkpoints not retained;
  endpoint numbers only) — flagged, not filled post-hoc.
- Composition constructions saturate near 1.0 in small-parent screens;
  use targeted fresh paths (E1-style) for the headline, screens for shape.
- Single-seed retraining comparisons are non-claims (±15pp floor, L-006).

## Status
Packaging complete from existing evidence; zero new experiments.
Placement: Methods/Appendix protocol box + reproducibility section pointer.
