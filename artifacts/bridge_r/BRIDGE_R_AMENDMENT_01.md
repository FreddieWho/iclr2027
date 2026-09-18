# BRIDGE_R Amendment 01 — renderer sensitivity repair (pre-holdout)

Status: ALLOWED (pre-holdout, holdout never rendered, no holdout read anywhere).
Lock reference: `BRIDGE_R_PROTOCOL_LOCK.json` PRE-amendment sha 1f245d52.

## Observation (train/dev only)
DINOv2 dev atomic peaks at 0.706 (n=2000, all readouts/probes), below B1
floor 0.85. Base-state acc 0.90 but edited atomics 0.61-0.73. Quartet
difficulty ruled out: v2 dev vs U01-eval oracle margins and edit norms match
to 3 decimals. Rendering ruled in: B01 full-canvas 224px renders reached
atomic 0.85 on the same backbone family; canonical v1 shrinks geometry
(safe-scale 0.80) then downsamples 512->256->crop224, thinning strokes to
~5px and diluting displacement signal.

## Change (instrument only)
- `SAFE_SCALE`: 0.80 -> 0.88 (worst-case train/dev |coord| 0.978*0.88 = 0.861
  < 0.875 crop bound; zero clipping verified, script `verify_canvas.py` logic
  in amendment check below).
- `LINE_WIDTH`: 10 -> 16 px @512 (8px @256 post-resize). Pure rendering
  choice; oracle-invariant (segment relations unchanged).
- `renderer_version`: bridge_r_canonical_v1 -> v1a.

## Firewall (unchanged, restated)
- Selection among renderer variants uses DEV ATOMIC COMPETENCE ONLY.
- Composition miss on dev is reported but NEVER a selection criterion.
- Quartet mining, splits, seeds, models, processors, probes, matching, gates,
  metrics untouched. Holdout parents still sealed (no render, no feature, no read).

## Why this is repair, not tuning
The locked protocol itself demands atomic readability as a precondition (Gate
B1); an instrument that cannot render decodable stimuli cannot test anything.
B01 already proved this domain decodable at 0.85 with larger-geometry renders.
We restore decodability, then let gates judge.
