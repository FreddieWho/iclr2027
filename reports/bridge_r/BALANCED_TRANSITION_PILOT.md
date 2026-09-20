# Balanced Transition Pilot — Report (2026-09-20)

Frozen DINOv3-L final R0 + residual adapter (1024→128→ReLU→1024, α=1),
3 arms × 3 seeds, equal budget (8000 states, 800 steps, AdamW 1e-3, 20
epochs fixed). Train: single-transition only (AB never seen). Eval:
independent 1000-quartet compdev + matched single-dev. Holdout 895 sealed.

## Atomic competence on compdev (Layer 1)

| arm | s11 | s23 | s47 |
|---|---|---|---|
| static | 0.353 | 0.728 | 0.689 |
| flip-only | 0.238 | 0.597 | 0.387 |
| balanced | 0.598 | 0.684 | 0.493 |

P1 needs balanced ≥0.85 in ≥2/3 seeds → **FAIL** (best 0.684).
No arm reaches 0.85; seed instability (static 0.35–0.73) shows the
adapter cannot reliably install even atomic readability.

## Composition (Layer 2, uninterpretable — prerequisite failed)

M_comp 0.60–0.92 across runs; both_n 212–633 (powered eval, failed models).
Δ_comp scatters around zero (−0.07…+0.14). Preserve FA 0.06–0.17, no arm
shows the flip-only-oversensitivity pattern (§28) because no arm learned
the transition either.

## Reading

The failure is at atomic installation, not at composition generalization:
a 128-wide residual adapter on frozen L-R0 cannot lift edited-state
readability to competence, with or without transition supervision.
This is consistent with the frozen-readout record (B1 FAIL everywhere,
semantic-delta D0 at chance): the limitation is in what the frozen
representation exposes, not in the supervision mix.
