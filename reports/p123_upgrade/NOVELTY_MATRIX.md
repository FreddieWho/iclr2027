# NOVELTY_MATRIX (2026-09-23; peer-reviewed vs preprints separated)

## Peer-reviewed
### Press et al. 2022/2023 Findings EMNLP — compositionality gap
Known: P(correct all sub-problems but not overall solution); multi-hop QA;
gap persists with scale; CoT/self-ask narrows it.
Different: static QA, no controlled change, no semantic boundary, no
matched-single controls, no denominator discipline, no repair decomposition.
Needed: nothing new — cite as the conditional estimand's origin. DO NOT
claim first conditional-on-atomics.

### Guo et al. 2017 (calibration), Geifman & El-Yaniv 2017 (selective classification)
Known: confidence~accuracy miscalibration; coverage-risk tradeoff with reject.
Different: both concern CURRENT correctness; neither tests updating under
change. P1's reversal (confidence anti-predicts update success) is outside
both. Cite as the baseline our P1 contrasts with.

### Jacobsen et al. 2019 (Excessive Invariance)
Known: overly-robust features cause adversarial vulnerability; accuracy/robustness tension.
Different: perturbation-robustness of fixed decisions; ours is failure to
CHANGE decisions across known semantic transitions. Adjacent mechanism
flavor, distinct object. One-line cite.

## 2026 preprints (not peer-reviewed; check versions before citing)
### Composition Collapse (2605.26789v1) — CLOSEST prior art
Known: double-gate (atomic stability across paraphrases + all-atoms-known
filter) -> residual composition failure; recipe matters 40+pp at matched atoms.
Different: (1) static multi-hop QA vs our continuous controlled change with
known boundary; (2) no update/transition framing; (3) no matched-single
displacement controls; (4) no repair decomposition or consequence chain;
(5) no denominator/selection-artifact analysis.
Needed: state the above deltas precisely; never claim first conditional.
If reviewers equate us, the boundary + update + repair chain is the defense.

### ATOM-Bench (2606.16826) — robotics manipulation
Known: atomic skills -> compositional generalization gap in policies.
Different domain, different estimand (no per-example conditioning reported
in abstract; verify full text before any AS/CFS comparison claim).
Use: one-line convergent-domain cite only.

## Metamorphic/counterfactual evaluation
Nearest: counterfactual augmentation literature (improves robustness via
targeted examples). Ours differs IF P3-C shows paired same-quartet repair
with denominator decomposition; otherwise our repair is an instance of it.
Pending P3-C.

## Verdicts for P3
- "First conditional" claim: DEAD (Press 2022). Never write it.
- Defensible novelty (pending P3-C): controlled-change update protocol +
  denominator discipline + repair decomposition + consequence chain.
- If P3-C shows no sorting change: METHODS_PROTOCOL_ONLY.
