# L-002 Pilot Report: ball + events (2026-09-22)

WOY (valid views, 5702 snaps) + WMX (train views, 5840 snaps) confirmation.
Frozen + cover, seed 11. Model turn = predicted-zone change in ±2-snap window.

## Data audit (the real find)
- Ball: continuous full-match trajectory in canonical frames (WOY 142k rows).
- Events: ~1500/match with outcome labels (Pass 743, Tackle 195, ...).
- DEAD ends: ball_possession all-NaN (S1-as-designed impossible);
  contested-event outcomes all None (S2-as-designed impossible);
  SoccerTrack side has neither.
- Salvage: S1→BallStatus flips; S2→contested presence vs random baseline.

## Results (WOY / WMX)
- Main passes ok vs bad: 0.146/0.153 vs 0.176/0.195 (WOY);
  0.172/0.171 vs 0.201/0.195 (WMX). Same direction 2/2, ~1se. WEAK NULL.
- S1 ballstatus flip vs stable: 0.022 vs 0.030 (WOY); 0.005 vs 0.033 (WMX).
  BURY 2/2 null.
- S2 contested vs baseline: 0.251/0.265 vs 0.169/0.192 (WOY, +0.08);
  0.186/0.194 vs 0.120/0.116 (WMX, +0.07). Same direction 2/2 matches ×
  2 models, ~2se each. KEEPER (modest).
- S3 ball-speed Q0→Q3: 0.020→0.048 (WOY); 0.013→0.047 (WMX). Monotone 2/2,
  confounded (fast ball co-occurs with real transitions). Observation only.

## Reading
Duels move the zone model (~+7pp over baseline, replicated); pass outcomes
and play-state flags do not. First ball/event-grounded (non-oracle) evidence
in the real-data column — appendix-grade, not headline (2 matches, 1 seed).
