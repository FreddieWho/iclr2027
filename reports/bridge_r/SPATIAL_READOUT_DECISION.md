# Bridge-R Task 2: Decision (2026-09-20)

## Primary verdict

**LOCAL_INFO_PRESENT_LOCALIZATION_FAILURE**

## Why this label and not the others

- Not SPATIAL_POOLING_EXPLAINS: R3/R4 significantly hurt at all scales.
- Not SIZE_DEPENDENT_LOCAL_ACCESSIBILITY: absolute local response grows
  with size (L>B>S); the size gradient is in accessibility, not response.
- Not LOCAL_INFORMATION_WEAK_OR_OOD: ~6x changed/unchanged contrast at
  every scale; R5 rescues significantly for B and L.
- Not MIXED_INCONCLUSIVE: the mechanism direction is consistent across
  S/B/L (pooling hurts, ROI rescues, gaps persist, ratios hold).

## Secondary observations (not verdicts)

1. R5 rescue is partial (0.74-0.79, below B1): even perfect localization
   does not restore competence — organization failure runs deeper than
   "where to look".
2. The S->B->L accessibility gap survives all six readouts (paired CIs
   exclude zero everywhere).
3. Base accuracy rises with size while edited-state accuracy falls.

## Consequences

- Bridge-R Task 2 stops. Holdout stays sealed (895 quartets).
- A third task (natural-image local positive control) is justified ONLY
  if the goal is testing whether the accessibility gradient generalizes
  beyond abstract renders — not decided here.
- Paper body untouched per ban. SIZE_SWEEP_REPORT wording downgraded to
  the hypothesis phrasing (§23) in the same commit.
