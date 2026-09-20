# Bridge-R Task 2: Decision (2026-09-20; closure controls 2026-09-20)

## Primary verdict

**LOCAL_INFORMATION_WEAK_OR_OOD**

(post-controls; supersedes the pre-control LOCAL_INFO_PRESENT read.)

## Why this label

- Not SPATIAL_POOLING_EXPLAINS: R3/R4 significantly hurt at all scales
  (fixed-budget statistical caveat recorded in REPORT).
- Not LOCAL_INFO_PRESENT_LOCALIZATION_FAILURE: its evidence (R5 rescue)
  was demolished by closure controls — R5b position-only matches or beats
  DINO-ROI at every scale; R5c leakage-free ROI equals R0. The rescue came
  from privileged geometric selection, not from decoded patch content.
- Not SIZE_DEPENDENT_LOCAL_ACCESSIBILITY: absolute local response grows
  with size (L>B>S within each geometry); the gradient is in task
  accessibility, not in response strength.
- Not MIXED_INCONCLUSIVE: the direction after controls is consistent —
  response present, task-usable local information weak, gaps stable.

## What "WEAK" means here (narrow reading)

"Weak" = no linear readout — global, uniform-spatial, or leakage-free
localized — converts patch response into relation-task information on
these abstract edits. It does NOT mean unresponsive patches: edited
regions show ~5-9x changed/unchanged contrast at all scales (backgrounds
exactly locked). The OOD alternative (abstract edits off-manifold for the
encoder's relation organization) remains open and is not distinguished here.

## Secondary observations (not verdicts)

1. R5b 0.7882 identical across backbones confirms the leakage is purely
   geometric (no backbone involved).
2. R5-minus-R5b significantly negative for L (-0.046): L's ROI patch
   features are worse than bare position — 1024-dim ROI at fixed C=1
   overfits and/or misleads.
3. The S->B->L accessibility gap survives all eight readouts (paired CIs
   exclude zero everywhere).
4. Base accuracy rises with size while edited-state accuracy falls.

## Consequences

- Bridge-R Task 2 closed. Holdout stays sealed (895 quartets).
- A third task (natural-image local positive control) stays parked: the
  controls decide it should test the OOD-abstract-task accessibility
  gradient, not a general scale-organization problem — not opened here.
- Paper body untouched per ban.
