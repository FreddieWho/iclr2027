# EVIDENCE_CORRECTIONS_FINAL (2026-09-22; appends EVIDENCE_CORRECTIONS.md + V2)

## F-1. paired_compare preprocess scare: INVESTIGATED, CLEARED (no-op refactor kept)
Old inline (x-mu)/sd == preprocess() for non-featurize ckpts; reran: bit-identical.
## F-2. P2 eval n=96 (was "170" typo in one report draft): fixed; numbers unchanged.
## F-3. P2 affine grid mismatch RESOLVED: report now matches frozen wide-grid code
(alpha 0.25..4, b -8..+8); s11 fit at grid edge reported as limitation, not widened.
## F-4. WP-B seed11/seed47 target duplication: documented; single table reported.
## F-5. P1 football old numbers VOID (mapping bug); rewritten analysis is floor/null.
## F-6. Unit-test tolerances are rounding-aware (5e-4); `or True` removed everywhere.
## F-7. keepbal loss-weight confound CONFIRMED and FIXED by fair rerun (§12 formula);
old keepbal numbers retained as history, superseded by p3d_fair_eval.
## F-8. relflip/relfeat checkpoints: trained this round with manifests; eval on dev+confirm.
