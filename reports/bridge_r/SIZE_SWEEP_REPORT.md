# DINOv3 S/B/L Size Sweep — Final (2026-09-20)

Same batch (v2 train 2164 / dev 760), same renderer v1a, official processors,
R0/R1/R2 x LR/MLP-128 x {230,500,1000,2000}, plateau rule -> size selection.

## Best dev atomic per scale (plateau-selected)
| scale | best config | atomic | minAB | both% | B1 |
|---|---|---|---|---|---|
| S | R0/lr/n2000 | 0.768 | — | 0.622 | FAIL |
| B | R0/mlp128/n2000 | 0.750 | — | 0.629 | FAIL |
| L | R0/mlp128/n2000 | 0.726 | — | 0.592 | FAIL |
| (ref) DINOv2-B | R0/lr/n1000 | 0.728 | 0.724 | 0.609 | FAIL |

## Reading
Monotone INVERSE size trend within v3 (S>B>L); DINOv2-B lands with L.
Bigger frozen encoders expose small abstract displacements LESS reliably
under linear/MLP probes — consistent with a size-dependent reduction in
linear accessibility on these abstract edits; stronger invariance is one
hypothesis to be tested.
All four points fail Gate B1 (0.85); composition testing stays forbidden;
holdout 895 remains sealed.

## Claim consequence
- No size point earns holdout opening. The sweep converges as a cross-scale
  replication of atomic-inaccessibility, strengthening (not weakening) the
  Bridge-R OOD_INCONCLUSIVE verdict.
- Paper language unchanged from BRIDGE_R_CLAIM_DECISION: small-model headline
  stands; foundation composition question open; title stays off Level 3.
