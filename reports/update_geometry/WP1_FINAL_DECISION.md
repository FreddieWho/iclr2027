# WP1 Final Decision: UPDATE_GEOMETRY_NOT_SUPPORTED (2026-09-21)

## Verdict
**`UPDATE_GEOMETRY_NOT_SUPPORTED`** for the incidence law; composition-as-update
retained as a supported Level-A piece (111/111 attribution, §8-Level-A first
clause holds; the "depend systematically on approach angle" second clause is
dropped).

## Branch consequences (§40 NO branch)
- No fresh-confirm burn on the matched test: `geometry_confirm` stays sealed
  (a confirm cannot fix structural non-overlap; spending one-shot on a test
  that cannot pass its own SMD gate violates §10).
- WP3 killed (no geometry to generalize; no 2nd family).
- WP5 permanently parked (G1 fails).
- WP2 → minimal boundary-mismatch diagnostic only (discovery-grade, no confirm).
- WP4 proceeds (propagation does not require the incidence law; uses frozen
  checkpoints + existing evals only).
- Paper body untouched. Claim ceiling for this round: Level A-partial →
  possibly Level D if WP4 delivers (no angle clause anywhere).
