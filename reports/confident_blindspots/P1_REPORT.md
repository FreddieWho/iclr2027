# P1 Report: Confident Blind Spots (2026-09-22)

Claim: **within states the model judges correctly, its confidence
anti-predicts whether it will update when the world changes.** The states
the model "knows" best are the least updatable.

## Evidence
1. Coordinate emergent quartets (n=237 primary paths, clean): start-confidence
   Q0 miss 0.41 vs Q1–Q3 0.89–1.00; within every oracle-margin tertile,
   high-confidence starts miss 1.00 vs low-confidence 0.59–0.73.
2. Coordinate general single-cross paths (n=3508): attenuated but positive
   (+15pp within-margin-bin mean, 62% bins positive; margin dominates).
   Pattern: the harder the population, the stronger the reversal.
3. Football zone turns (WOY/WMX/WOH, n≈2400/match, frozen): confidence Q3
   miss 0.97 vs Q0 0.60–0.64; near-boundary high 0.91 vs low 0.66; far
   0.99 vs 0.89–1.00. 3/3 matches same direction.
4. Repair resistance: flipmine (coordinate: hi 0.97/1.00/0.88) and cover
   (football: Q3 0.97–0.98) preserve the curve shape exactly — repair moves
   average endpoints, never the confidence-blindness phenotype.

## Honest alternatives (paired design)
- Anticipation: football low-conf turns may reflect updating already in
  progress (turn is immediate). Coordinate emergent starts sit far before
  the crossing (no anticipation possible) — same phenotype. The pair
  brackets the interpretation.
- Degenerate confidence ties (football conf≈1.0 masses) handled by
  near/far distance splits; effect persists within near-boundary.
- Margin is the dominant population predictor; confidence dominates WITHIN
  the hard subpopulation. Both statements kept, never averaged.

## Figure素材
`artifacts/confident_blindspots/FOOTBALL_CONF.json` + WP1 rows
(confidence quartile × miss with margin bands, clean vs repair curves).
Kill rule passed (no reversal in 3 matches × 2 models + coordinate 2 pops).

## Multi-seed closure (s23/s47, same 512 parents, zero retraining)
| seed | clean hi / lo | flip hi / lo |
|---|---|---|
| s11 | 1.00 / 0.59-0.73 | 0.88-1.00 / ~0.7 |
| s23 | 0.98-1.00 / 0.58-0.77 | 0.85-0.98 / 0.53-0.57 |
| s47 | 1.00 / 0.65-0.76 | 0.96-1.00 / 0.53-0.57 |
3/3 seeds same direction; repair shifts the curve down (s23: 0.84→0.73)
but never flattens it. Single-seed objection closed.

Verdict: P1成立, 正文级候选 (new subsection + Figure panel).
