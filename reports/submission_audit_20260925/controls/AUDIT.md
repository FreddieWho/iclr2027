# Controls and lineage audit

Baseline: `3fc9757398ebc88652362b906a2a98710bf17650`. Scope: O01, O02, O04, R03, R08 and R09. This overlay reads existing summaries and checkpoints only. It does not retrain, use a GPU, call an API, or read the sealed datasets named in the task. The R03 sensitivity used CPU inference on the local `d09fresh665` bank.

## Findings

### O01: retained labels and continuation

The six-distance 25% vs raw 100% table is a finite-budget retained-label comparison after the flip pools were mined in full. It supports the stated per-cell J ordering. It does not show oracle-query savings: N generated 1,534 mined flips in 2,801 attempts and retained 383 for the 25% arm; 4N generated 6,390 flips in 11,955 attempts and retained 1,597. The evaluation denominator is 1,229 quartets from 250 eligible parents; the 256 development parents are separate. The 3 seeds and two training sizes share this same evaluation pool.

The parent bootstrap samples parent clusters while retaining quartets. Its estimate is row-weighted across quartets; it is not an equal-parent estimate. The per-seed intervals must remain separate. The 648-arm continuation analysis (432 warm-start, 216 robust) uses the same 1,229/250 evaluation bank, so its arms and seeds are not independent data replications.

### O02: matched capacity and budget

The primary artifacts show positive six-minus-typed_m flip contrasts for every seed: T1 `0.514/0.529/0.486`, T2 `0.731/0.701/0.755`, on 2,724/322 and 3,443/369 quartet/parent banks. A separate two-thread run also keeps all six contrasts positive, but changes the numbers: T1 `0.525/0.522/0.448`, T2 `0.741/0.691/0.732`. Do not describe the exact estimates as thread-invariant. The result bounds one typed width and a 3-LR by 2-step recipe grid; it does not prove that a representation or architecture class cannot work.

### O04: prospective intervention did not meet its primary prediction

From the stored per-cell seed-mean deltas, P1 is true in 7/8 cells and misses at T1/raw_clean; P2 is true in 8/8. The registered primary verdict is therefore `MISS`. The banks and trunks were generated earlier; the prediction is prospective relative to the previous readout analysis, not independent confirmation on fresh banks. The baseline appendix said the prospective intervention and downstream echo were not run; the coordinating audit has corrected the active appendix.

### R03: single-state overlap sensitivity completed; quartet result remains untouched

The audit identified 21 affected checkpoints: 18 trained on 4N and 3 on 16N. Each checkpoint was scored on the 126,295 `Sx+Se` endpoint states in `bank_d09fresh665.npz`; the start `Sx` was not scored as a separate endpoint. Total: 2,652,195 CPU state evaluations, 4 threads, 3.57 seconds. The exact-ID lists are pool-specific: 4N removes 163 eids from 30 parents; 16N removes 761 from 127 parents. The 4N list is a subset of the 16N list (163 shared eids), so each checkpoint was filtered against its own pool rather than the union.

Across the 21 checkpoints, row-weighted single-endpoint accuracy changes by `-0.109` to `+0.010` percentage points after exclusion; parent-equal changes by `-0.252` to `-0.002` pp. The hit subsets have accuracy `0.723` to `1.000`; all hit entries are label-flip states. No CI was calculated for this descriptive, parent-nested diagnostic. The result is not a D09 quartet J re-estimate. The audited D09 quartet denominator stays 236, with 0 exact and 0 near state overlap; no quartet was removed.

### R08: natural and searched E use different numerators

Under the registered 0.2-second, forward-only contract cap, natural E occurs in 8/1,449 situations (0.55%), 8/162 events (4.94%) and 10/7,117 frame pairs (0.14%). Controlled search constructs E in 670/1,449 sampled candidate opportunities (46.2%); 48/155 eligible selected passes is also a search result. Neither search rate is natural E incidence. These are geometric-oracle labels, not observed pass-success labels.

### R09: the feasible-only table omits 117 evaluation scenes

The full universe has 256 scenes and 49 candidate actions per scene. There are 139 feasible and 117 infeasible scenes overall. The evaluation split has 213 scenes: 96 feasible and 117 infeasible. Thus the current 96-scene conditional table omits 54.9% of evaluation scenes (45.7% of the full universe). At the nominal 0.5 development target, full-scene evaluation coverage is 0.1455–0.3239. The actual evaluation coverage differs from the development target, so quality contrasts are policy extrapolations, not equal-coverage effects. The decomposition is undefined when all candidates are infeasible; it cannot explain or recover that 117-scene share.

## Minimal appendix replacement text

Locations below are in `paper/sections/app_f095.tex`; the paper was not modified in this overlay.

- **Line 106, O04.** Replace the stale “partial / not run” paragraph with:

  > `\paragraph{Prospective intervention diagnostic (O04).} A preregistered head-supervision change predicted that the largest mean change would be in $J$ in all eight task-by-encoder cells (P1). This primary prediction missed in one cell, T1/raw\_clean, and held in 7/8; the secondary prediction that $|\Delta J^\ast|<|\Delta J|$ held in 8/8. The evaluation used 2,724/322 T1 and 3,443/369 T2 quartets/parents and three model seeds. These were previously generated banks and frozen trunks, so this is prospective relative to the earlier readout analysis, not an independent fresh-bank confirmation. The downstream echo also has a domain boundary: 117/256 football scenes have all 49 candidates infeasible, where the decomposition is undefined.`

- **Line 109, R09.** Add after the current conditional-table description:

  > `The full universe contains 256 scenes (43 development and 213 evaluation); 117/256 scenes have all 49 candidates infeasible. The 213-scene evaluation split contains 96 feasible and 117 infeasible scenes, so the 96-scene table is conditional and omits 54.9% of evaluation scenes. At the nominal 0.5 development coverage target, measured full-scene evaluation coverage is 0.1455--0.3239. These are synthetic oracle/cost diagnostics on a shared scene universe, not equal-coverage effects or real deployment utility.`

- **Line 108, R08.** Append the natural-rate denominator beside the search result:

  > `At the registered 0.2-second forward contract cap, natural E occurred in 8/1,449 situations (0.55%), 8/162 events (4.94%) and 10/7,117 frame pairs (0.14%). The 670/1,449 search rate (46.2%) is controlled constructibility, not natural incidence.`

- **Line 149, R03.** Add after the single-bank overlap count:

  > `A new CPU-only sensitivity scored all 126,295 Sx+Se endpoints for the 21 affected 4N/16N checkpoints and excluded each model's own exact-overlap eids (4N: 163; 16N: 761). Row-weighted accuracy moved by -0.109 to +0.010 pp; this is a descriptive single-endpoint diagnostic, not a D09 quartet J re-estimate. The 236-quartet denominator remains unchanged, with 0 exact/near overlaps.`

- **Line 103, O02.** Add a compact sensitivity note:

  > `A separate two-thread reproduction preserves all six positive six-minus-typed_m directions but changes the estimates (T1: 0.525/0.522/0.448; T2: 0.741/0.691/0.732), so exact effect sizes are not thread-invariant.`

## Not run / limits

- O01 oracle-query savings were not tested; the full candidate pools were mined before retaining 25% labels.
- R03 uncertainty intervals for the newly scored single-bank accuracy sensitivity were not computed. This does not change the unchanged 236-quartet denominator.
- R08 observed pass-success prediction was not run; no pass-success labels are present.
- R09 real deployment utility was not run; reward/cost values are illustrative.
- O04 independent fresh-bank replication and a general causal-mechanism test were not run.
- O02 expansion to other widths, depths or optimizer families was not run.

## Reproducible artifacts

- `experiments/submission_audit_20260925/controls_r03_sensitivity.py` writes per-checkpoint hashes, pool-specific excluded eids, full/excluded/hit metrics and denominator receipts to `artifacts/submission_audit_20260925/controls/r03_single_sensitivity.json`.
- `experiments/submission_audit_20260925/controls_summary_recompute.py` is the summary-only replay script for O01/O02/O04/R08/R09 and the R03 sensitivity output. It reads stored result summaries and emits JSON; it does not train or access sealed source data.
- Machine-readable claim dispositions are in `reports/submission_audit_20260925/controls/claims.json`.

## Final integration

The coordinating audit updated the maintained appendix for O04, R03, O02 numerical sensitivity and R09 conditional denominators. Replacement paragraphs above document the initial finding and suggested wording, rather than outstanding publication work. No Git commit or push was performed.
