# Competence-First Report (2026-09-20)

Frozen DINOv3-L final 14×14 patch tokens → static conv head
(1×1 1024→128, 2× Conv3×3, AvgPool, Linear). Static state supervision
only (12000 states), dev-accuracy selection (patience 5), seeds 11/23/47.
Eval: repair compdev base/A/B (AB never extracted, never opened).

## Results

| seed | dev state | base | A | B | atomic_mean | min | both_frac | both_n |
|---|---|---|---|---|---|---|---|---|
| 11 | 0.740 | 0.866 | 0.711 | 0.669 | 0.690 | 0.669 | 0.570 | 570 |
| 23 | 0.748 | 0.883 | 0.698 | 0.653 | 0.676 | 0.653 | 0.547 | 547 |
| 47 | 0.760 | 0.889 | 0.708 | 0.662 | 0.685 | 0.662 | 0.554 | 554 |

C1 needs atomic_mean≥0.85 + min≥0.82 + both≥0.60 in ≥2/3 seeds → **0/3**.

## Reading

The signature is unchanged from every prior readout: unedited base
states read well (0.87–0.89) while edited A/B states stall at 0.65–0.71
— now confirmed with a nonlinear spatial head on full patch tokens.
The wall is edited-state readability in the frozen representation, not
pooling, not linearity, not adapter capacity. H0 (R0 adapter 0.70–0.76)
vs H1 (conv head 0.68–0.69): the spatial head adds nothing.
