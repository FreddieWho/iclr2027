# EVIDENCE_CORRECTIONS (2026-09-23, HEAD 41079c3)

Steering rule: old numbers stay in history; corrected numbers supersede for any
future claim. Fixes are code + wording; no conclusion is assumed before re-run.

## C1. P1 football mapping bug (CONFIRMED, results VOID)
`experiments/confident_blindspots/football_conf.py` maps
`{"defense_third":0,...}` but canonical labels are `defensive_third`
(r01_t5r3.py:83, prepare_idsse_t5r2.py:712-717). All defensive snapshots became
NaN; NaN!=NaN is True, so every defensive-adjacent pair counted as an oracle
turn. WOY has 1180 defensive snapshots (21%) — FOOTBALL_CONF.json numbers are
VOID. Same bug in the mining-turn inline analysis. Rewrite required (P1-A):
correct zmap + start-correct filter + arrival-at-new-class + tie handling.

## C2. P1 coordinate recomputability gap (CONFIRMED)
`wp1_paths.py` rows lack x/e vectors and input hashes — incidence and curves
cannot be recomputed from rows.json. New sample bank (bank_dev512, sha
2178613957f4, 474 paths + 126151 singles with x/e stored) fixes this.
Old quartile numbers stand as exploration only until P1-A rerun.

## C3. P2 universal "no pointwise improvement" (CONFIRMED contradiction)
P2 draft said "none improves pointwise precision" but fresh static error is
0.1796875 -> 0.107421875 (E1 result.json). Universal deleted. Replacement:
repair improves static accuracy AND expands coverage; the open question is the
decomposition (P2-B/C/D), not a blanket denial.

## C4. +29pp gap is not causal composition-specific error (CONFIRMED)
+29pp = dev 93.9% - 65.3% (different screens, unmatched). Fresh arithmetic:
0.8579545 - 0.6095618 = 24.8393pp (clean own-conditional vs own-matched,
different denominators: 176 vs 251). Neither is a causal "pure composition"
effect. P3-C replaces it with paired same-quartet comparison.

## C5. 109/141 vs 151/176 denominator split (CONFIRMED)
E1: clean emergent_cond 151/176 = 0.8579; flipmine 109/141 = 0.7730
(141 x 0.7730496 = 109.0). Evidence table row FIXED to 109/141 with explicit
non-paired status. History 151/176 (single model) kept; repair delta never
described as paired.

## C6. E1 rng4 non-paired clean/flipmine edits (CONFIRMED)
`e1_battery.py` section 4 loops over models sharing rng4;
`atomic_edits`/`candidates_for_scene` consume rng per call -> same parents,
DIFFERENT edits (and different matched singles). Common-parent is not
common-edit. Single-model numbers retained with history; rate differences are
NOT paired repair effects on the same quartets. New work uses frozen bank.

## C7. 130/248 -> 86/248 is the C1 single-edit test (CONFIRMED)
n09 C1 = single label-changing edits (margin >= 0.005, n=248, PAIRED same
samples both models). NOT the headline N04 conditional composition metric.
Paper "compositional C1 miss 52.4%->34.7%" is a misnomer -> corrected wording:
"single-flip C1 miss" vs "headline conditional composition 151/176 -> 109/141
(non-paired denominators)".

## C8. wp1_analyze.py stat bugs (CONFIRMED, FIXED 2026-09-23)
(a) matched-delta CI used .mean(0) on [10000,n] resamples -> .mean(1)
(that CI fed no claim). (b) `audit_rel or 1` excluded legitimate rel == 0.0
-> explicit None check. (c) binned CIs treated same-parent paths as
independent -> parent-cluster bootstrap now. Old WP1 binned CIs void;
incidence branch stays closed regardless.

## Paper prose already contradicted by our verdicts
- phenomenon "depends on how a change approaches the semantic boundary" +
  "precisely the shape the model misses most": WP1 killed the incidence law.
  Rewrite to sampling/denominator language (PAPER_INSERTS).
- "compositional C1 miss": see C7.

## Checkpoints in use (sha256 of model.pt)
- r04b_s11/clean 8b20dfee / flipmine 8b566fa3
- r04b_s23/clean cf09fd3a / r04b_s47/clean d50ee487
- T5R3 base seed11: phase3/.../t5r3_sanity_v3/models/raw_single_channel_phase_gat_team_mean_seed11.pt
- T5R3 cover s11: discovery_campaign/r04b_t5r3_s11/cover.pt
