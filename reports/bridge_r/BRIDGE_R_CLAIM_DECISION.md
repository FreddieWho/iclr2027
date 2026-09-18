# BRIDGE_R_CLAIM_DECISION

## Verdict: OOD_INCONCLUSIVE

(DINOv2-B and SigLIP2-L/384 both fail Gate B1 on canonical abstract renders;
composition-specific testing correctly never started; holdout preserved sealed.)

## Consequences for the paper (advisory; no manuscript file touched)
- KEEP: headline 151/176 compositional miss (coordinate regime, full atomic
  competence, fresh holdout) and all Level 1-2 claims.
- KEEP OUT: any Level-3 phrasing ("learned representations" in general,
  "modern practical models share the defect", "scaling reduces 85.8% to X%"
  as a strict curve, "DINOv2 has a composition-specific residual").
- ADD (one honest paragraph, Discussion/Limitations): frozen DINOv2-B and
  SigLIP2-L/384 expose small abstract displacements unreliably (dev atomic
  0.73/0.64 vs 0.85 floor, cross-family, readout-independent); therefore the
  foundation-model composition question remains open and needs either
  naturalistic stimuli or a readability-validated rendering regime. The sealed
  895-quartet holdout is retained for that future test.
- Do NOT open the holdout to "confirm B1 failure": dev n=760 already settles
  the precondition, and spending the sealed asset adds zero information.

## Why this is a success under section 26
Fresh frozen evaluation: yes (2 backbones). Competence verdict: explicit
failure, cross-family. Global-vs-spatial diagnosis: yes (R0/R1/R2 same
pattern). DINOv2 + one stronger backbone: yes. Numerator/denominator/CI
discipline: held (dev-stage; no holdout numbers exist). Claim restriction:
this document. The reviewer sentence "just a 7k-parameter MLP artifact" is
answered: the coordinate headline stands, and the foundation boundary is
measured rather than hand-waved.
