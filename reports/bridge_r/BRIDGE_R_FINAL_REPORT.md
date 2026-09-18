# BRIDGE_R_FINAL_REPORT (2026-09-19)

Question: does a composition-specific update gap
(M_comp >> M_single, conditioned on correct atomics) persist in strong
frozen visual foundation representations?

## Protocol executed
- Fresh v2 splits: train 2164 / dev 760 / holdout 895 quartets, disjoint
  parents/RNG (lock `cce7cd18`, committed pre-holdout; Am01 renderer-only).
- Backbones: DINOv2-B (official BitImageProcessor, rev `f9e44c81`) +
  SigLIP2-L/384 (official SiglipImageProcessor, rev `1b426889`).
  DINOv3-L: BLOCKED_EXTERNAL_ACCESS (gated, no token; fallback per protocol).
- Readouts R0/R1/R2 x probes LR(C=1.0)/MLP-128 x sizes {230,500,1000,2000},
  plateau rule -> n=1000 nearly everywhere. All backbones fully frozen.
- Holdout: NEVER opened (no render, no feature, no read). HOLDOUT_USED absent.

## Dev competence (the gate that decided everything)
| backbone | best config | atomic | minAB | both% | B1 |
|---|---|---|---|---|---|
| DINOv2-B | R0/lr/n1000 | 0.728 | 0.724 | 0.609 | FAIL |
| SigLIP2-L/384 | R1/mlp128/n1000 | 0.638 | ~0.63 | ~0.55 | FAIL |

Pattern in both families: base states ~0.8-0.9, edited atomics ~0.6-0.7.
MLP train acc -> 1.0 with no dev gain (readout not the bottleneck).
Features verified healthy (variance, alignment, shapes, pinned revisions).

## R10 decision: Outcome D
Both backbones fail Gate B1 (atomic_mean>=0.85). Per frozen Gate B1, the only
permitted readout is **atomic-accessibility-insufficient**; composition
discussion is forbidden. The matched-single control (R6) was correctly NOT
executed: without a B1-passing backbone it could serve no confirmatory
purpose, and building it would require touching sealed holdout geometry.
The one-shot holdout is deliberately preserved unopened.

## Interpretation
- This is NOT "foundation models resolve the gap" (that would require B1 pass
  + delta~0). It is OOD_INCONCLUSIVE: on abstract line-render stimuli at
  canonical scale, frozen DINOv2-B and SigLIP2-L/384 do not expose atomic
  displacements reliably enough to test compositional updating.
- Cross-family replication (self-supervised + vision-language, 224px + 384px)
  rules out single-backbone/single-resolution explanations for the B1 failure.
- Small-model headline (151/176) is untouched: it lives in coordinate space
  with full atomic competence, a different and valid measurement regime.

## Six direct answers
1. DINOv2-B under corrected protocol: best dev atomic 0.728 (R0/lr/n1000);
   B1 FAIL; no composition claim permitted.
2. Strong model (SigLIP2-L/384 fallback; DINOv3 blocked): best 0.638; B1 FAIL.
3. Did models "see" atomic states? No — not reliably (0.64-0.73 vs 0.85 floor).
4. Is comp miss above matched single? UNTESTED (correctly so — gate forbids it).
5. Does spatial readout fix it? No: R1/R2 pattern matches R0 in both families.
6. How big can the paper's claim be? Unchanged small-model claims (Level 1-2)
   + an honest foundation boundary: atomic readability of abstract
   displacements is itself scale/renderer-sensitive; title must stay off Level 3.
