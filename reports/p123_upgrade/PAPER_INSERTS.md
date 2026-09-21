# PAPER_INSERTS (one question + one result + one scope + figure each)

## INSERT-1 (P1, new subsection candidate)
Q: Within states the model already judges correctly, does its confidence tell
us anything about whether it will track the next change?
R: On start-correct paths whose oracle label flips, endpoint error rises from
~0.4 in the lowest start-confidence quartile to ~1.0 in the highest, in all
three training seeds and both clean and repaired models; the same models'
static error falls from ~0.36 to ~0.02 across the same quartiles.
Scope: coordinate single-crossing paths, dev pool, recomputable bank; natural
football turns do not show it (floor effects after correction).
Figure: dual rank curves (static vs update) + margin-tertile bands.

## INSERT-2 (P2, Discussion-anchored + main hook sentence)
Q: Does repair make the same decisions more reliable, or more decisions available?
R: On a fixed scene-action table the net success gain splits into positive
coverage (+0.06..+0.10) and negative quality (-0.02..-0.04) in all seeds;
on identically acted scenes the repaired model is worse (16v8, 12v9, 9v7);
candidate recall rises (0.41->0.64) at flat precision; forced-choice ranking
improves under the score rule.
Scope: dev scenes, lexicographic rule primary; equal-coverage infeasible
(repair saturates thresholds) — stated as instrument limit.

## INSERT-3 (P3, Methods box)
Q: How should compositional repair be measured so denominator shifts do not
pose as fixes?
R: Report own, baseline-fixed, and common-correct together: on one frozen
quartet bank the paired (baseline-fixed) repair effect is ~35-40pp while
own-vs-own mixes genuine fixes with denominator churn (51-57 exits, 16-33
entries); a synthetic unit test moves own-rates 60pp with zero AB change.
Scope: coordinate bank; literature: conditional estimand after Press 2022,
closest art Composition Collapse double-gate (preprint) — deltas listed in
NOVELTY_MATRIX, no first-conditional claim.

## Wording corrections (must-fix in current draft)
- "compositional C1 miss 52.4%->34.7%" -> "single-flip C1 miss ..."; headline
  repair stated as 151/176 -> 109/141 with explicit non-paired denominators.
- "depends on how a change approaches the semantic boundary" -> sampling /
  denominator language (incidence law dead per own WP1 verdict).
- Evidence table C5 row already fixed to 109/141.
