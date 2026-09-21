# WP2 Minimal Diagnostic Report (2026-09-21, discovery-grade, no confirm)

1301 oracle single-crossings (eval_202, aimed+random), identical points for
clean vs flipmine (`r04b_s11`), search radius r=0.025 frozen (= 0.25 × median
atomic-edit norm). Frozen checkpoints, zero training.

## Clean mismatch: stark and quantitative
- Metric A signed alignment: mean −0.003, median −0.005 (≈ 0; |A| mean 0.59).
- Coverage R_boundary: **0.096** — 90% of oracle boundary points have NO model
  boundary within r.
- Metric C (where a root exists, n=125): Ab mean −0.049 (≈ 0).
- Reading: the model's decision boundary is mostly elsewhere, and where it
  exists nearby its orientation is unrelated to the semantic normal.
  Consistent with the N02 null (gradient↔outcome), now at the boundary.

## Flipmine shift (paired, same points)
- Coverage: 0.096 → **0.169** (Δ=+0.073, parent-cluster bootstrap CI
  [0.020, 0.128]) — flipmine puts decision roots near semantic boundaries
  more often.
- Offset median: 0.0121 → 0.0114 (no change).
- Signed alignment: mean −0.003 → −0.007 (no change; flip median −0.237 with
  |A| 0.76 vs 0.59 — distributional skew noted, not claimed).
- Ab: −0.049 → +0.038 (both ≈ 0).
- Reading: repair-associated shift is **coverage-only** — more boundary
  presence, no better orientation. Mirrors the N06/U02 mechanism line
  (coverage, not precision).

## Limits
Discovery-grade (eval_202, viewed pool); no fresh confirm; no per-path
miss↔mismatch link tested (N02 pre-null). Appendix-grade evidence.
