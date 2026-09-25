# Paper migration map — mechanism_transfer_v3 (appendix-only)

Date: 2026-09-25. Status: appendix-only notes in `paper/sections/app_l015.tex`
tail; no main-text promotion. Build receipt: `reports/e832_focus/PAPER_BUILD_RECEIPT.json`
(27 pages, conclusion page 9, 0 overfull, 0 undefined;
`v3_migration: "appendix-only route1/route2/M2-frozenmodule notes; no main-text promotion"`).

## What entered the appendix (`app_l015.tex` tail)

1. Source ordering diagnostic (scoped): the same eight quantities / same MLP /
   existing within-segment sort raises single-flip J by +0.061 to +0.092 vs the
   unsorted copy on 2188 E / 871 parents (all five parent intervals exclude
   zero) — framed as one specific invariant processing, not identical
   information and not a learned set network. Source: `artifacts/e832_focus/structure/metrics_continued.json`.
2. Repaired-readout negative: nonlinear readout after the segment sum does not
   recover the geometry gain (repaired J ~0.017-0.046; flip contrast vs additive
   at most ~+0.015, three of five intervals include zero). Source: same metrics file.
3. Parser-bounded diagnostic: analytic segment-intersection parser solves the
   source E bank (230 E / 65 parents, J3 = 1.0), so learned source comparisons
   are parser-bounded representation diagnostics, not evidence a complex
   relation model is required. Sources: `artifacts/e832_focus/route3/results.json`,
   `reports/e832_focus/route3/REPORT.md`.
4. Route1 scoped direction + cross-task boundary: rich-sorted exceeds
   rich-unsorted on the fresh 1414-E/389-parent source bank under both
   supervisions, while T1 (417 E / 170 parents) is uninformative and T2
   (912 E / 361 parents) is negative/unstable. Source: `artifacts/e832_focus/route1/round2_analysis.json`.
5. Visual bounded negative: contract-corrected fresh 28-quartet/19-parent
   screen, mean J3 direct/additive/representation/interaction
   .571/.417/.274/.583 (v2), interaction-minus-direct +.179/-.143/.000 —
   not a stable interaction advantage; clean-only direct mean .155.
   Source: `artifacts/e832_focus/route2/gpu_run_v2_remote_20260925/corrected_results/`.
   (v3 area-pooling recovery of representation to mean 0.619 with interaction
   still unstable lives in `reports/e832_focus/route2/REPORT.md` and
   `artifacts/e832_focus/route2/gpu_run_v3_remote_20260925/gpu_run_v3/`;
   it sharpens but does not overturn this paragraph.)
6. M2 frozen-module negative: frozen source relation head scores J3 = .796 on
   true geometry of the 692-quartet/351-parent bank (random head 0-.013, so
   competent and specific); geometry-loss-only vision frontend reaches
   J3 ≈ .20 with dev geometry MSE ≈ .51 frozen or fine-tuned, while direct
   end-to-end vision reaches .92. No module-transfer gain on this bank.
   Source: `artifacts/mechanism_transfer_v3/m2/gpu_run_remote/out/`.

## What was deliberately held out of the main text

- Any flip-side claim that whole-orbit beats the sorted bag (M1 flip
  orbit-minus-sorted +0.0654/+0.0005/-0.0311, last two CIs cross zero:
  direction unstable). Appendix wording is "removes the risky sort without an
  accuracy price," not a gain.
- Any source information-collision claim (M1.0 bounded search found no exact
  merge: nearest opposite-label 0.0471, 3000-step search 0.0335; lower bound
  only, not a completeness proof). Source: `artifacts/mechanism_transfer_v3/m1/source_search.json`.
- Any cross-task transfer (T1/T2 not run in v3; route1 T1/T2 negative/unstable
  stands). No M1→vision transfer gain (M2 negative is the result).
- Any visual interaction method claim (route2 v2+v3: bounded negative/structure
  advantage not established).
- Any "complex relation model required" reading of the source task (parser
  J3 = 1.0 forbids it; learning numbers are diagnostics).
- Any pooled denominator: 2188/871 (M1 source), 692/351 (M2), 230/65 + 171/56
  (route3), 28/19 (route2 v2+v3), 237/106 and 509 (historical dev/confirm),
  1414/389-417/170-912/361 (route1) stay separate. Seeds are training
  replicates throughout.
- Figure, title, or conclusion-page changes: none in this round. Conclusion
  stays on page 9; no pixels invented; no A/B/C numbers inserted.

## Why

Main-text promotion would overclaim: the supported v3 content is one scoped
source-representation parity plus three bounded/negative boundaries. Those
belong in the appendix as guardrails on the existing mechanism section, exactly
where `app_l015.tex` already keeps the "not identical information / not a
learned set network / not pooled" caveats. The pending A/B/C program is
untouched and uncited as finished.
