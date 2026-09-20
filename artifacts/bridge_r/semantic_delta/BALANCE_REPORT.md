# Semantic-Delta Balance Report (M3, pre-freeze)

Dataset: `artifacts/bridge_r/semantic_delta/{train,dev,confirm}.npz`
(v2 pools; v1 pools discarded for SMD>0.10 on margins).

## Sizes (1:1 flip/no-flip)

| split | rows | flip | no-flip |
|---|---|---|---|
| train | 4000 | 2000 | 2000 |
| dev | 1000 | 500 | 500 |
| confirm | 1000 | 500 | 500 |

## Standardized mean differences (flip − no-flip)

| split | edit_norm | dpix | m0 (base margin) | m1 (endpoint margin) |
|---|---|---|---|---|
| train | +0.001 | +0.002 | −0.023 | −0.029 |
| dev | +0.002 | −0.002 | −0.037 | −0.078 |
| confirm | −0.001 | +0.003 | −0.022 | −0.058 |

All |SMD| < 0.10 → confirm entry allowed (§6). Residual margin lean
(flip endpoints slightly nearer-boundary) is structural and documented,
not a shortcut: geometry-only and pixel-only controls (§13) exist
precisely to absorb it.

## Base-label strata (exact)

train 1023/977, dev 272/228, confirm 280/220 — matched exactly within
strata by construction.

## Leakage audit (DATA_AUDIT.json)

- Parent IDs: zero overlap across splits.
- (base, edit) pairs: 6000/6000 unique cross-split.
- Edit-orientation circular resultant 0.01–0.08 in both classes → no
  direction shortcut.
- Family mix: flips concentrate in move_node/rot families (expected:
  crossing needs node displacement); geometry control absorbs difficulty.
