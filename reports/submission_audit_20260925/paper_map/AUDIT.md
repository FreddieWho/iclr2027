# Paper claim map and hostile review

**Baseline:** `3fc9757398ebc88652362b906a2a98710bf17650` (`/home/huyudi/012_conference/iclr2027`). The shared worktree changed during this review; this report separates baseline findings from fixes the root agent reported or made afterward. Our ownership is limited to this report, `INVENTORY.csv`, and `artifacts/submission_audit_20260925/paper_map/gaps.json`.

## Scope and limits

Read the full active LaTeX assembly (`paper/main.tex`, every section, active maintained appendix, generated appendix reference, fallback table), the paper evidence map/checklist, root README/STATUS/TODO and report index, and current/historical claim ledgers plus reports for final closure, e1a933, e832 routes 1–4, and mechanism-transfer M2. Used CodeGraph for implementation entry points, checked the four current manuscript figures visually, inspected Figure 1 selection/receipt code, and checked paper/reproduction path existence and PDF/receipt metadata. No sealed source data were opened. No experiment, test suite, web search, or manuscript edit was performed by this lane.

`INVENTORY.csv` is a claim-to-evidence map, not an independent numerical reanalysis. It groups claims by their scientific conclusion and retains the important denominator, reuse/freshness, conditioning, and lane boundary; it intentionally does not list every table cell. Other lanes own numerical/code acceptance.

## Main scientific story and evidence map

The draft's principal claim is a fresh coordinate result: even when both atomic edits are correct, the joint change is missed in 151/176 cases. The paper distinguishes this conditional miss (CCM) from joint accuracy and from a paired repair delta. It keeps random-screen and targeted-path denominators separate, describes path-level zero-transition evidence without a causal propagation claim, and confines the confidence reversal to coordinate scenes. The core lane should confirm the exact E1 and P1 artifacts and frozen estimands.

The repair story is split correctly across populations: fresh-holdout component metrics; raw-coordinate historical-dev endpoint repair that mostly migrates; separate 509-quartet confirmation full repair; relational input evidence on the 237-quartet development bank; and the later rendered ResNet18 boundary. The source parser result narrows the learned source-task comparison to a parser-bounded diagnostic. T1/T2 results separate input advantage from task-specific interaction, and the appendix labels seeds as training replicates rather than independent tasks. These should stay bank-separated; no pooled denominator is supported.

The active appendix also records a number of scoped diagnostics: matched-budget retained-example efficiency, label-renaming analyses, a restricted segment-sum expressivity negative, route2's corrected bounded visual comparison, N02 grid/compute evidence with its unresolved mechanism, N01 start-confidence limits, a classical image parser, N03/O05/O06 boundaries, O04's predeclared MISS, and R03 lineage/audit corrections. Their source candidates and correct review lanes are in the inventory. The M1 transfer/ordering result is not a current-paper claim; do not add it merely because it exists in the newer lane ledger.

## P0/P1 items for the coordinating review

No P0 central-result falsifier was independently established in this document-only pass. The following P1 items were present at baseline or were reported by the root agent and must be resolved before treating every appendix claim as accepted:

1. **M2 result validity (open; `CM-21`).** The root agent reports that `GeomFront.forward` places each branch under `no_grad`, while BatchNorm state still updates. That prevents the fine-tuned condition from supporting the interpretation in `M2_TRANSFER.md` and `app_l015.tex:18`. The M2 negative/"not learnable" conclusion is withdrawn pending valid independent acceptance. Current status from the root agent: `DEFERRED_PENDING_ACCEPTANCE`; do not convert an invalid run into a mechanism-closed result.
2. **Source-parser parent denominator (open; `CM-11`).** `reports/e832_focus/route3/REPORT.md:9` gives 230 E/142 parents, while the final decision, active paper, and appendix cite 230/65. Check `artifacts/e832_focus/route3/results.json` metadata and synchronize the report and manuscript mapping. Keep the independent 171/56 bank separate.
3. **Reproducibility and build identity (final snapshot pending; `CM-32`, `CM-36`, `CM-38`).** At baseline, `paper/main.pdf` was 27 pages with SHA-256 `8a558080f680ecbceae61f45d8b3f1283c37973da92f043538e480c1ae5c7b99`; the e832 receipt named a different SHA (`8582e85b…`), the older e1a933 receipt described a 25-page PDF with source hashes that no longer matched, and the checklist said 26 pages. Recompile once from the final source, then publish one matching receipt and synchronize the checklist. The current paper says assets are versioned with it, but its cited reproduction files do not give a single complete recipe for all active e832/M1/M2-era claims. At baseline 359/497 files under `artifacts/f095_campaign/` and 29/50 under `artifacts/mechanism_transfer_v3/m2/` were untracked; 28 untracked M2 run outputs were absent from `artifacts/SHA256SUMS_UNTRACKED.txt`. The path scan found no missing literal paper paths and all listed reproduction script paths existed; availability/hash coverage is the gap.
4. **Cross-ledger authority (root agent editing; `CM-33`, `CM-37`).** Baseline README/INDEX/EVIDENCE_MAP called the 2026-09-22 final-closure ledger the sole authority, while current route and mechanism claims live in `reports/e832_focus/CLAIM_LEDGER.csv` and `reports/mechanism_transfer_v3/CLAIM_LEDGER.csv`. Publish one cross-ledger map or name an authority per claim; do not imply the old master ledger covers all current manuscript claims.

## Additional wording and navigation findings

At baseline, the active maintained appendix contradicted later completed evidence in two status sentences: O04 at `app_f095.tex:106` said the prospective intervention and downstream echo were not run, despite `:80` and `O04_FORWARD_INTERVENTION.md` reporting execution, primary MISS, and a completed D10 echo; the R03 table caption at `:136` said statewise duplicate checking was not run, despite `:149` and `R03_LINEAGE_REPORT.md` reporting the 392,412-state audit. The root agent reported that both active-paper sentences were corrected. The unselected `app_f095_generated.tex` still contains historical copies of these NOT_RUN statements and should either be regenerated/annotated as a non-canonical reference or explicitly excluded from the reproduction package.

Figure 1 originally plotted archived correctness indicators as if they were predicted class labels; its caption said predictions were shown. The root agent corrected the renderer to recover binary baseline/flip labels and regenerated the figure; the new receipt is `reports/submission_audit_20260925/FIGURE1_SELECTION_RECEIPT.json`. Keep the selection examples fixed and do not revert this change. Figure 1's P state displays oracle label only; the caption should continue to say so. Fig. 2 is an appendix confidence side result, Fig. 3 is the dev migration plot, and Fig. 4 combines dev/confirmation four-arm measurements; their concepts match their captions at a high level.

For the metric contract, the paper already says CCM always denotes a miss and warns that conditional joint success is not CCM. It should define `J_3` once as the current A/B/AB joint correctness measure corresponding to `J`, and state that `J_4`/base-state quantities are unavailable where the source artifact lacks them. Route 1 explicitly leaves undefined J4 as null; the fallback table correctly marks it missing.

The root agent reported updates to README/STATUS/INDEX/master-ledger history pointers, active O04/R03 text, Figure 1, and the reproduce/assets notes after these baseline observations. Build receipt and final PDF identity are intentionally left pending the coordinating agent's last compilation snapshot. Our files do not replace that final acceptance.

## Recommended minimum fixes

- Keep the M2 interpretation withdrawn until the implementation and outputs pass independent acceptance; no new scientific result is inferred here.
- Resolve the route3 230-E parent count at its source and update all copied prose.
- Rebuild one final 27-page PDF, hash it, hash the exact manuscript source set, and point the checklist/build receipt to that snapshot.
- Preserve current per-lane claim lineage in the new top-level claims register; label old evidence maps and generated appendix copies as historical/non-canonical.
- Standardize `J`/`J_3`/`J_4` and CCM language once; retain explicit `NOT_RUN`, `BLOCKED`, `MISS`, and `DEFERRED_PENDING_ACCEPTANCE` states at their exact scope.

## 最终整合说明

本报告主体记录初查状态。后续修复状态已逐项回写 `artifacts/submission_audit_20260925/paper_map/gaps.json`：Route3 分母、PDF 回执、Figure1、O04/R03 文案和证据入口已修正；指标口径见本轮 METRIC_CONTRACT。`app_f095_generated.tex` 是未参与主稿编译的历史参考，不作为写作依据。M2 验收按用户要求后置，公开大资产发布与最终文章仍未执行。本次只保留本地交付，不提交或推送 GitHub。
