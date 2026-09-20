# Semantic-Delta Decision: NEGATIVE — STOP (2026-09-20)

## Dev results (DINOv3-L final R0, fixed LR, 4000 train / 1000 dev pairs)

| Input | Target | Acc | AUROC | Train |
|---|---|---|---|---|
| C1 geometry-only | flip/no-flip | 0.543 | 0.536 | 0.518 |
| C2 pixel-summary | flip/no-flip | 0.568 | 0.586 | 0.555 |
| T0 endpoint z1 | flip/no-flip | 0.598 | 0.661 | 0.812 |
| **D0 delta z1−z0** | **flip/no-flip PRIMARY** | **0.497** | **0.505** | 0.747 |
| D1 |delta| | flip/no-flip | 0.595 | 0.616 | 0.779 |
| T1 concat | flip/no-flip | 0.589 | 0.628 | 0.925 |
| A endpoint z1 | endpoint state | 0.732 | 0.805 | 0.889 |
| D0 permuted-labels | flip/no-flip | 0.523 | — | — |

Dataset: 1:1 balanced, all |SMD|<0.08, parents disjoint, permutation ≈ chance.
Confirm split (1000 pairs) NEVER OPENED. Bridge-R holdout 895 untouched.

## Gate outcome

DEV GATE: AccΔ=0.497 (need ≥0.82), margin −0.071 (need ≥+0.08) → **FAIL**.
Kill pattern Kill A: AccΔ<0.80 → **`SEMANTIC_DELTA_NEGATIVE`**. STOP.
No confirm. No Stage 2. No paper-body change.

## Six questions (§38)

1. Dataset balanced? **Yes** — 1:1, |SMD|<0.08, exact base-label strata.
2. Controls? C1 0.543 / C2 0.568 — matching worked, no low-level shortcut.
3. D0 delta? 0.497 acc / 0.505 AUROC — chance.
4. Better than controls? **No** — worse than both (−0.05/−0.07).
5. Updated-state still weak? A_state 0.732 — below B1, consistent with prior.
6. Mechanism rescued? **No.** Stage 2? **No.** Paper entry? **No.**

Verdict stands: `LOCAL_INFORMATION_WEAK_OR_OOD`. No further mechanism rescue.
