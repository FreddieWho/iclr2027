# Balanced Transition Pilot — Report (2026-09-20, corrected rerun)

Frozen DINOv3-L final R0 + residual adapter (1024→128→ReLU→1024, α=1),
3 arms × 3 seeds, equal budget (8000 states / 4000 pairs, 800 steps,
AdamW 1e-3, 20 epochs fixed). Train: single-transition only (AB never
seen). Eval: independent 1000-quartet compdev + matched single-dev.
Holdout 895 sealed.

**Correction note:** the first run (commit ed8b7e0) had a `load_pairs`
indentation bug using only the last feature shard (1000 pairs, unequal
budgets). All adapters were retrained on the full 4000 pairs with
count asserts; numbers below supersede the earlier ones.

## Atomic competence on compdev (Layer 1)

| arm | s11 | s23 | s47 |
|---|---|---|---|
| static | 0.760 | 0.712 | 0.701 |
| flip-only | 0.634 | 0.644 | 0.680 |
| balanced | 0.689 | 0.722 | 0.631 |

P1 needs balanced ≥0.85 in ≥2/3 seeds → **FAIL** (best 0.722).
Full data removed the seed instability (static now 0.70–0.76) but no
arm approaches competence: the adapter cannot install atomic
readability on frozen L-R0, with or without transition supervision.

## Composition (Layer 2, uninterpretable — prerequisite failed)

| arm | M_comp | M_single | Δ_comp | preserve FA |
|---|---|---|---|---|
| static | 0.64–0.70 | 0.55–0.59 | +0.08–+0.12 | 0.10–0.12 |
| flip-only | 0.51–0.52 | 0.48–0.52 | +0.01–+0.04 | 0.16–0.17 |
| balanced | 0.54–0.58 | 0.50–0.52 | +0.03–+0.06 | 0.12–0.14 |

Directional secondary (not a claim): flip-only lowers M_comp but lowers
M_single in parallel (Δ_comp shrinks toward zero) while FA rises — the
predicted oversensitivity signature, without the competence that would
make it interpretable. Balanced sits between static and flip-only.

## Reading

Failure is at atomic installation, not composition generalization —
consistent with the frozen-readout record (B1 FAIL everywhere,
semantic-delta D0 at chance).
