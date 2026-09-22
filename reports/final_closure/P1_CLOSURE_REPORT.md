# P1_CLOSURE_REPORT (2026-09-22): confidence for current vs future reliability

## Unified population (bank_dev512 + bank_confirm1007; frozen bank)
Rank policy: ranks over path starts + one row/parent from singles; absolute
thresholds frozen per model on dev (p1_freeze.json); ties merged (tie share
0.30 deduped base); parent-cluster bootstrap throughout. Confirm run with
frozen thresholds/settings exactly once.

## B: three risks, same rank policy (6/6 models dev + 6/6 confirm)
- current: 0.43 -> 0.05 (classic)
- preserve: 0.40 -> 0.04 (same direction; start-correct preserve spurious
  flips concentrate at LOW confidence, Q0 3.8% vs Q1-Q3 ~0%)
- flip (start-correct): 0.58 -> 1.00 dev; 0.53-0.61 -> 1.00 confirm (REVERSED)

## C/D: trust selection + environment composition (uniform conditioning)
Retention top25% (dev-frozen): flip_err 1.00, preserve-update err 0.00
(start-correct preserve never spuriously flips at top confidence).
R_update(tau,rho) = (1-rho)*preserve + rho*flip; at top25 R = rho exactly.
Joint R_joint reported on retained population. No overall-deployment claim
beyond retained sets.

## E: confound control (P1-C1 margins audited: m_start verified = scan margins)
Regularized logistic + grouped 5-fold CV by parent:
- conf-only pooled OOF AUC 0.93-0.95; difficulty-only 0.53-0.61 (~chance);
  full 0.93-0.95. Oracle difficulty/geometric covariates carry almost no
  held-out signal. Max |corr| ~0.47 noted, not controlled away beyond this.
- Matched (margin×norm×class cells): delta +0.19..+0.46, n~500 dev /
  ~1300 confirm, support OK (no thin-support flag).
- Unresolved: unmeasured difficulty beyond oracle margin/distance (stated).

## F: threshold-distance control (diagnostic, not causal)
Normalized |f|/|Df| AUC 0.989 vs raw 0.926 (dev s11); joint coef keeps
confidence positive. Most failure consistent with score distance an edit
must overcome; static confidence retains additional predictive value after
grouped held-out analysis. Temperature T*~=8 preserves ranks exactly;
static NLL/Brier improve while ordering (hence P1) is untouched.

## G: repair residual, strictly paired (clean-defined high-risk)
High-risk cells: repair-start-correct (n~140-160): repair err 0.53-0.84;
repair-start-wrong: repair end err 0.0-0.03. Paired delta CIs exclude 0
(dev marginal s11/s23 touch 0; confirm all exclude 0). Common-correct
secondary consistent. Means improve; residual dominates.

## H: confirm integrity
confirm_1007: 1000 parents, sha 28365ae3, parent-disjoint audited (0 overlap
train/eval), frozen thresholds/covariates/metrics (p1_freeze.json +
p1_freeze_meta.json), primary = confidence-rank update-error curve.
Results: shape, confound-gain, paired-delta all replicate. No post-confirm
primary redefinition.

## Grade: P1-STRONG (coordinate only)
Wording gate: "Among currently correct states undergoing a true semantic
change, higher pre-change confidence predicts a substantially greater risk
of failing to update, despite predicting lower error on static and
semantics-preserving cases." Football excluded (null/floor).
