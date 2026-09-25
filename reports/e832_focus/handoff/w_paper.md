# w_paper handoff

Implemented the question-centered migration and the ten decided wording fixes. No unfinished A/B/C result was inserted. No repaired-model success was claimed.

## What changed

The abstract and introduction now open on one question: when single-change supervision improves full compositional correctness rather than only moving the error, and what input representation and cross-object interaction do. P1 is a coordinate-only side result. The 151/176 fresh conditional composition miss is the one failure example. The ResNet-18 comparison is stated as a boundary, not as another case of migration.

Historical banks are not pooled. The 237-quartet development bank and the 509-quartet confirmation bank have separate captions. The 1,229, 159, and 178 banks stay in their own paragraphs or appendix tables. Seeds are called training replicates, not datasets.

Figure 1, the missing quartet illustration, was not drawn. The selection rule is in `reports/e832_focus/PAPER_MIGRATION.md`. `fig:p1` is not that figure.

The L014 radius sentence was not added. Before this round the conclusion already started on page 9.

## Ten wording fixes

1. Applied. Source-task typed-pair / segment-set negative is not used to exclude learnable relational ability. Segment-sum plus linear readout cannot represent a general crossing. Pointer: `artifacts/e832_focus/structure/EXPRESSIVITY_REPO_AUDIT.json`. TypedTri/TypedDisk are named as not covered. No repaired-model success.
2. Applied. L015 handcrafted arm is an 8-d feature plus MLP, not a learned set network. It adds midpoint distance and an absolute direction cross. Zero swing is by construction.
3. Applied. T2 joint-scale interaction stays negative. Input advantage is separated from positive interaction transfer. No new T2 number.
4. Applied. Discussion no longer says a native VLM was untested. The sentence is limited to the completed Qwen2.5-VL-3B-Instruct run already in the appendix. A pre-existing sentence that broader nonlinear readouts remain untested was left alone; it is not a VLM claim.
5. Applied. The integer-ID leakage charge against dev512 is not repeated. The bank is called a repeatedly analyzed historical bank, not a sealed confirmation bank. The existing 0/237 lineage rows are cited, not recomputed.
6. Applied. Matched total exposure is not matched per-parent exposure.
7. Applied. CCM is defined as a miss. J and atomic-joint accuracy are conditional joint accuracy and are not abbreviated CCM. The N02 0.728 to 0.888 rates are named conditional joint success.
8. Applied. The paper does not say every repair mostly migrates. The vision comparison is the boundary.
9. Applied. The path-screen explanation is no longer called a sampling mechanism. N02 keeps the 12.25× MAC confound. Unchanged information is not said to prove a sampling mechanism.
10. Applied. The AI-use comment was replaced by a rendered draft. It says the author must verify it and does not claim item-by-item review.

None of the ten was blocked by a missing source.

## Build

TinyTeX `pdflatex`, two-pass equivalent final run, exit 0.

- PDF: `paper/main.pdf`
- Pages: 25 total
- Conclusion: starts and ends on page 9
- References: start on page 10
- Overfull boxes: 0
- Undefined references: 0

Figure widths were reduced only to keep that page-9 ending (`fig3` 0.90 to 0.78, `fig4` 0.95 to 0.82). The two bank tables share one float and have separate captions, so the second table does not land inside a later section.

The previous on-disk PDF was a 15-page build. Rebuilding the current appendix source is longer. That length is not a new result.

## Not cited

No number from `STRUCTURE_DECISION.md`, `VISUAL_DECISION.md`, or `SAMPLING_DECISION.md`. Pending A/B/C rows in the claim ledger have status PENDING and no hoped-for numbers.

## Files

Changed: `paper/main.tex` and the section files listed in git status for `paper/sections/`.

Written: `reports/e832_focus/PAPER_MIGRATION.md`, `CLAIM_LEDGER.csv`, `CLAIM_LEDGER.json`.

Not edited: TODO, STATUS, DECISIONS, PLAN, ROADMAP, LEADS, `experiments/`.

## Risks

- `paper/sections/app_f095.tex` is generated. A later run of `experiments/f095_campaign/gen_app_tables.py` can overwrite the wording fixes in that file. The generator was not edited.
- Page 9 is tight. Another main-text sentence can push the conclusion onto page 10.
- Figure 1 is still missing by rule. Do not treat the confidence panel as the quartet illustration.
- The AI-use draft is not an author certification.

## Next step

Do not fill A/B/C claims until those lanes finish. When they finish, add only rows whose denominators are in their decision files, then choose the title. Render Figure 1 from the fixed `bank_dev512` / seed-11 rule, or report an empty eligible set. Do not use that figure for effect estimation.
