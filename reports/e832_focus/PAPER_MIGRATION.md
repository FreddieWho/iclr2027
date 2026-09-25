# Paper migration map (writing round)

Date: 2026-09-25. Writer scope only. A, B, and C were not edited and are not cited as finished results.

## Question

When does supervision of a single change improve full compositional correctness, rather than only moving the error? What role do the input representation and cross-object interaction play?

P1 stays a coordinate-only side result. It is no longer the center of the abstract or the introduction.

## What this round changed

Applied in `paper/`:

- Abstract and introduction now open on that question. The 151/176 fresh conditional composition miss is the one failure example. The structural comparison is relational input under the same flip coverage, with input advantage separated from positive interaction transfer. The ResNet-18 comparison is the competent-vision boundary. Confidence is one scoped sentence, not the contribution list's center.
- Historical banks are not pooled. The 237-quartet development bank and the 509-quartet confirmation bank are separate tables (`tab:fourarmdev`, `tab:fourarmconfirm`). The 1,229-quartet pool, the 159-quartet canonical visual bank, and the 178-quartet sampling bank stay in their own paragraphs or appendix tables. Seeds are labeled as training replicates, not datasets.
- The ten wording fixes below are applied. No repaired-model success is claimed. No A/B/C number is inserted.
- Figure 1, the missing quartet illustration, is not drawn. The current `fig:p1` confidence panel is not that figure.
- The L014 radius sentence was not added to the discussion. Before this round the conclusion already started on page 9, the submission page limit.

Not done, on purpose:

- Title not changed. Candidates in `docs/e832887_focus_pack/05_PAPER_RESTRUCTURING.md` wait for finished A/B evidence.
- Sections were not mechanically reordered into the suggested §1–§7 skeleton. The center moved; the failure catalogue remains in §3 and the appendix so existing denominators stay findable.
- No pixels, no placeholder figure environment, no estimated effect from a displayed quartet.

## Target reading order

| Target | Current file | This round |
|---|---|---|
| Question, why an endpoint rate is not enough | `paper/sections/01_introduction.tex` | Applied |
| Scene, legal edit, J, CCM, banks | `paper/sections/02_instrument.tex` | Wording only |
| One failure example; P1 as side result | `paper/sections/03_phenomenon.tex` | Scoped, not deleted |
| Repair flow: raw migration, relational input, T1/T2 sign split | `paper/sections/04_mechanism.tex` | Wording and split tables |
| Expressivity limit on the source-task typed negative | `04_mechanism.tex`, `app_l015.tex`, maintained `app_f095.tex` | Applied from the existing audit file |
| A interaction contrast after a repaired model is actually run | not in the paper | PENDING. Do not fill. |
| Competent vision boundary | `paper/sections/04b_boundary.tex` | Exposure caveat kept; B transfer not added |
| Related work, discussion, conclusion | `05_repair.tex`, `discussion.tex`, `conclusion.tex` | O06 and migration-boundary wording only |
| Failure catalogue and finished appendices | `06_external_confirmation.tex`, maintained `app_f095.tex`, `app_l015.tex` | Catalogue kept; wording fixes only |

`app_f095.tex` is now a maintained appendix. The numeric generator writes the separate reproducible reference `app_f095_generated.tex`; it no longer overwrites the maintained manuscript file. This prevents routine regeneration from deleting audited wording.

## Figure 1 selection rule

Do not invent pixels. Do not add a figure environment until the two quartets are rendered from archived predictions. Do not use the selected quartets for effect estimation. Effect estimates stay in the existing tables.

Named bank: `bank_dev512`, 237 quartets from 106 parents. It is a repeatedly analyzed historical bank, not a sealed confirmation bank. Do not read `holdout_909` or `bank_confirm1007` to choose the picture.

Fixed rule, seed 11 only, because that is the seed already used for the fresh-holdout headline model. No outcome-based tie break.

1. Endpoint-only repair. Arm: raw coordinates, clean versus flip coverage, on the baseline-broken subset H for seed 11. Eligible quartet: AB endpoint becomes correct and at least one atomic becomes wrong. Choose the eligible quartet with the smallest parent id, then the smallest quartet id in the archived bank order. If none exists for seed 11, stop and report the empty set. Do not switch seeds after looking.
2. Full repair. Same bank, same seed, relational-input plus flip coverage, on that arm's baseline-broken subset H. Eligible quartet: the quartet becomes fully correct (both atomics and AB). Same parent-id then quartet-id rule. If none exists, stop and report the empty set.
3. Display only those two quartets, with oracle state and the archived prediction. No third panel until A finishes. A structural positive from an unfinished run is not a substitute.
4. The figure caption must say the pair is an illustration from `bank_dev512`, seed 11, under this rule, and is not an effect estimate. Do not pool it with the 509, 1,229, 159, or 178 banks.

`fig:p1` remains the confidence side-result panel. It is not Figure 1 for this rule, even if LaTeX currently numbers it first.

## Ten wording fixes

| # | Fix | Status | Where |
|---|---|---|---|
| 1 | Source-task typed negative does not exclude learnable relational ability. Segment-sum plus linear readout cannot represent a general crossing. Point at `artifacts/e832_focus/structure/EXPRESSIVITY_REPO_AUDIT.json`. Do not claim the repaired model succeeds. | Applied. The audit file exists and is an algebraic witness with no bank. TypedTri/TypedDisk are named as not covered. | `04_mechanism.tex`, `app_l015.tex`, `app_f095.tex` |
| 2 | L015 G8 arm is an 8-d feature plus MLP, not a learned set network. It adds midpoint distance and an absolute direction cross. Zero swing is by construction. | Applied | `app_l015.tex`, `discussion.tex` limitation (xi) |
| 3 | Keep the T2 interaction negative. Separate input advantage from positive interaction transfer. | Applied. Existing negative sign kept; no new T2 number. | abstract, introduction, `04_mechanism.tex`, conclusion |
| 4 | O06 ran. Do not write that a native VLM was untested. Limit the sentence to the single Qwen2.5-VL-3B run. | Applied. Discussion no longer says native multimodal generality is untested. The existing O06 paragraph is unchanged in its numbers. | `discussion.tex`; numbers remain in `app_f095.tex` |
| 5 | Do not repeat the old integer-ID leakage charge against dev512. It is a repeatedly analyzed historical bank, not a sealed confirmation bank. | Applied. The charge is withdrawn by reference to the existing 0/237 lineage rows, not by a new audit. | `02_instrument.tex`, `08_reproducibility.tex`, `app_f095.tex` |
| 6 | Matched total exposure is not matched per-parent exposure. | Applied. Existing O03 caveat kept and stated in those words. | `04b_boundary.tex`, discussion, abstract |
| 7 | CCM means miss. J and atomic joint accuracy are conditional joint accuracy. Do not use one abbreviation for both. | Applied. N02's rising 0.728 to 0.888 rates are named conditional joint success, not CCM. | `02_instrument.tex`, `app_f095.tex` |
| 8 | Do not write that every repair mostly migrates. The vision system is a boundary. | Applied as a scoped negation plus the existing ResNet full-repair-exceeds-migration sentence. | abstract, introduction, `04_mechanism.tex`, `04b_boundary.tex`, discussion, conclusion |
| 9 | Unchanged information does not prove a sampling mechanism. Keep the MAC confound. | Applied. The path-screen explanation is renamed so it is not that claim. N02 keeps the 12.25× MAC confound already in the text. | `03_phenomenon.tex`, `appendix.tex`, `app_f095.tex` |
| 10 | AI-use draft must say the author has to verify it, and must not claim item-by-item review. | Applied as a rendered draft after the appendix. It is a draft, not an author certification. | `paper/main.tex` |

No fix was blocked by a missing source. The expressivity file was present. O06 text was already in the appendix. Lineage rows for dev512 were already in the appendix table.

## Numbers not used

- No result from `reports/e832_focus/STRUCTURE_DECISION.md`, `VISUAL_DECISION.md`, or `SAMPLING_DECISION.md`.
- No hoped-for A/B/C accuracy, interaction, or page-count of a future figure.
- L014 radius medians were not copied into the discussion.
- The algebraic witness is not reported as bank generalization. Its own file says four-case nonlinear success is expressivity, not bank generalization.

## Claim map

This round's allowed and pending wording is `reports/e832_focus/CLAIM_LEDGER.csv` and `CLAIM_LEDGER.json`. Denominators for allowed rows are copied from `reports/final_closure/MASTER_CLAIM_LEDGER.md` or from text that was already in the paper. Pending A/B/C rows have no numbers.

## Build receipt

Rebuilt with TinyTeX `pdflatex` on 2026-09-25. `paper/main.pdf` is 25 pages. Conclusion starts and ends on page 9. References start on page 10. Overfull boxes: 0. Undefined references: 0. The AI-use draft is rendered after the appendix; the style file does not remove it from the PDF page count, and it is not part of the 9-page main text.

The 237 and 509 banks are separate captions (Tables 2 and 3), not pooled. They share one float so the second table does not land inside a later section. Seeds are labeled as replicates, not datasets. Figure widths were reduced only enough to keep the conclusion on page 9 (`fig3` 0.90 to 0.78, `fig4` 0.95 to 0.82). No pixels were invented.

The pre-edit PDF in the tree was a 15-page build. Rebuilding the current appendix source is what produces the longer total. That length is not a new result.

## 路线5首轮更新（2026-09-25）

- 真实 Figure 1 已按 `FIGURE1_SELECTION_RECEIPT.json` 接入正文；confidence 侧结果移至附录。
- 两个部分四臂表已替换为 `MAIN_TABLE_FALLBACK` 统一测量表；缺失的 `J4`、base、parent-CI、成本字段显式标为 missing。
- source 强基线解析器在 E=230/65 上 J3=1.0；source 学习式比较改写为 parser-bounded diagnostic，不再暗示复杂关系模型是必要的。
- 路线1 T1/T2 首轮 E=8/17，不进入跨任务主张；固定8192 parent可行性轮待结果。
- 路线2 CPU smoke 只标 `PILOT_ONLY`；GPU bundle 正式状态为 `BLOCKED_GPU`。
- 当前主文不再把 cross-object interaction 写成已解决的方法结论；只保留输入/修复流向的范围性结论和视觉边界。
- 最新构建：TinyTeX `pdflatex`，26页，结论第9页，0 overfull，0 undefined references。
