# P1_REPORT (2026-09-23): confidence vs update risk, corrected

## Coordinate (bank_dev512, recomputable, 3 seeds x clean/flipmine)
Dual test, same models:
- (1) Static reference (bank singles endpoints): err by confidence rank
  Q0 0.36 -> Q3 0.02 (s11 clean). Classic direction: confidence predicts
  current correctness.
- (2) Update subset (start-correct + oracle flipped, n~200-290/model):
  endpoint-wrong by confidence rank Q0 0.34-0.54 -> Q3 0.93-1.00, 6/6
  model-seeds same direction; margin-tertile stratified hi-lo gaps persist;
  parent-cluster bootstrap CIs.
- Preserve stratum (4000 stays, s11 clean): false-update 1.2% overall,
  concentrated in LOW confidence (Q0 3.8%, Q1-Q3 ~0%). Symmetric picture:
  high confidence = stays put (no spurious flips) AND never updates.
- Selection (P1-C): retaining top-25% most confident start-correct paths
  keeps conditional update-error at 1.00 (clean) / 0.93 (flip) while the
  same screen's static error reads ~2%. Stricter trust lowers current error
  and concentrates update failures. Joint risk reported on the retained
  population; no claim about overall deployment risk beyond it.
- Repair residual (P1-E): clean-fixed high-risk group err 1.00 ->
  repair 0.83-0.92, paired delta CI excludes 0 all 3 seeds. Means improve;
  residual stays high. No "same stubborn errors" claim.
- Temperature: binary |logit|/T strictly monotonic -> ranks and argmax
  invariant by construction; curves unchanged. P1 does not depend on
  saturation.
- Analytic oracle control: y=oracle(x), conf=margin updates perfectly at
  any confidence -> reversal is not a logical necessity (sanity only).
Grade: 校正后支持 (coordinate).

## Football (rewrite with start-correct + arrival + time windows)
Corrected analysis: update miss 0.6-2.8% (vs old 86% artifact). Confidence
separation at floor (adjQ3 0-6% vs adjQ0 0-1%), lead definitions inconsistent
across seeds. Jitter 30/match excluded+counted; lead exclusions <10.
Grade: 不支持 — natural snapshot turns are tracked when starting correct;
the phenomenon needs hard constructions (E2 aimed sweeps stand separate).
Old football P1 numbers void (C1).

## P1-D warning: NOT pursued
Selection consequence (P1-C) already demonstrates the practical cost without
a model; a small risk score adds deployment machinery the paper does not
need. Recorded, not built.

## Claim status
Coordinate-only finding. No cross-domain P1 claim. Figure: dual rank curves
(static vs update) + selection panel, coordinate s11 with s23/s47 overlay.
