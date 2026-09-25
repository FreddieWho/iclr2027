# Route 1: cross-task feasibility (source/T1/T2)

## Scope and contract

This route tests whether the source task's representation ordering is informative enough to motivate a task comparison. It does **not** replace the official E rule or the J3/A/B/AB/CCM metric. E remains an unordered pair for which both individual edits preserve the label and their sum flips the label. Ambiguous states do not change the denominator. J4 remains `null` when it is not defined.

No sealed pool, dev512, or test J was read. Test J was not used for checkpoint or hyperparameter selection. Model seeds were frozen as 11, 23, and 47.

Implementation:

- runner: `experiments/e832_focus/route1_cross_task/common_runner.py`
- focused tests: `experiments/e832_focus/route1_cross_task/test_common_runner.py`
- precommit: `artifacts/e832_focus/route1/round2_manifest.json` (SHA-256 `82d93712449a76fc89fdfe78376fc363c4a1cc3cab05315cd66d14f668659976`)
- post-run denominator/bootstrap manifest: `artifacts/e832_focus/route1/round2_analysis.json`
- per-task outputs: `round2_source.json`, `round2_T1.json`, and `round2_T2.json`
- merged task view: `results.json`

The runner exposes CLI and environment equivalents for all requested execution controls:

| Setting | CLI | Environment |
|---|---:|---|
| train parents | `--train-parents` | `ROUTE1_TRAIN_PARENTS` |
| dev parents | `--dev-parents` | `ROUTE1_DEV_PARENTS` |
| test parents | `--test-parents` | `ROUTE1_TEST_PARENTS` |
| edit cap | `--edit-cap` | `ROUTE1_EDIT_CAP` |
| deterministic candidate draws per parent | `--candidate-draws` | `ROUTE1_CANDIDATE_DRAWS` |
| epochs | `--epochs` | `ROUTE1_EPOCHS` |

The legal candidate families and their 24 random draws per family are unchanged. `--task-prefix round2_` keeps the original per-task Round 1 files intact. `results.json` is updated by merge, so a later single-task invocation cannot erase results for another task; per-task files are also retained.

## Round 0 — contract/setup

Round 0 has no independent score denominator in the Route 1 artifacts. It establishes the task oracles, legal action families, E rule, and model/metric contract. Its test denominator is therefore **N/A**. It is not used as evidence for a positive or negative cross-task claim.

## Round 1 — heterogeneous feasibility run

The preserved files are `source.json`, `T1.json`, and `T2.json`.

| Task | Train | Dev | Test parents | Official E denominator | E parents | Model epochs | Use |
|---|---:|---:|---:|---:|---:|---:|---|
| source | 512 | 128 | 1,536 | 2,313 pairs | **513 in preserved JSON** | 120 | feasibility/reference |
| T1 | 256 | 64 | 256 | **8 pairs** | 5 | 20 | uninformative |
| T2 | 256 | 64 | 256 | **17 pairs** | 10 | 20 | uninformative |

The earlier prose handoff described source as `E=2313/871`, but the preserved `source.json` records 2,313 pairs from 513 unique test parents in its top-level `E_parents` and every arm metric. This report does not silently alter the original JSON; 513 is the auditable Round 1 denominator.

Because T1 and T2 had only 8 and 17 official pairs at 20 epochs, Round 1 is a feasibility diagnosis, not a negative cross-task result. The single-flip means in that run could be dominated by individual pairs and were not used to select Round 2 settings or checkpoints.

## Round 2 — precommitted targeted feasibility

The settings and fresh seeds were written to `round2_manifest.json` before any Round 2 score was viewed. T1/T2 each received exactly one new 8,192-parent pool; no Round 2 seed reuses the old test seed. Source uses a fresh 1,536-parent test seed and 300 epochs. All tasks use 300 epochs, edit cap 8, and three legal candidate draws per parent.

| Task | Train/dev/test parents | Seeds (train, dev, **test**, edit) | E pairs | E parents | Status |
|---|---:|---|---:|---:|---|
| source | 512 / 128 / 1,536 | 832501, 832502, **932503**, 932504 | **1,414** | 389 | COMPLETE |
| T1 | 256 / 64 / **8,192** | 832511, 832512, **932513**, 932514 | **417** | 170 | COMPLETE |
| T2 | 256 / 64 / **8,192** | 832521, 832522, **932523**, 932524 | **912** | 361 | COMPLETE |

T1 and T2 have nonzero E, so the predeclared three-seed arms were run once. Neither task received another parent draw. No success threshold was introduced.

### Single-flip J3 by predeclared arm

Each cell is the mean across model seeds 11/23/47. Every J3 cell has the exact denominator shown in the preceding table; exact per-seed values and model records are in the per-task JSON and `round2_analysis.json`.

| Arm | source clean / flip | T1 clean / flip | T2 clean / flip |
|---|---:|---:|---:|
| raw | 0.0354 / 0.0431 | 0.0032 / 0.0136 | 0.0227 / 0.0482 |
| additive | 0.0000 / 0.0113 | 0.0000 / 0.0048 | 0.0029 / 0.0044 |
| repaired | 0.0000 / 0.0125 | 0.0056 / 0.0104 | 0.0161 / 0.0322 |
| six | 0.1808 / 0.2079 | 0.0328 / 0.0560 | 0.2039 / 0.4697 |
| rich-unsorted | 0.1643 / 0.2008 | 0.0192 / 0.0416 | 0.0577 / 0.0698 |
| rich-sorted | 0.2211 / 0.2914 | 0.0200 / 0.0336 | 0.0479 / 0.0742 |

### Parent-bootstrap representation contrast

The predeclared primary representation contrast is `rich-sorted − rich-unsorted` in J3. Bootstrapping resamples test parents with replacement, gives each parent equal weight, and reports a 1000-resample percentile 95% interval. Round 1 mixed a pair-weighted point estimate with parent-cluster resampling; Round 2 fixes that estimand mismatch. This does not alter the official J3 metric or E rule.

**Source — stable positive ordering.** All six model-seed/supervision contrasts are positive and their intervals exclude zero:

- clean: +0.0424 [0.0041, 0.0813], +0.0774 [0.0346, 0.1199], +0.0685 [0.0273, 0.1108];
- flip: +0.0694 [0.0201, 0.1137], +0.1055 [0.0604, 0.1519], +0.0734 [0.0300, 0.1178].

**T1 — uninformative ordering.** Clean deltas are −0.0025, −0.0034, +0.0008; flip deltas are +0.0019, −0.0255, +0.0056. Every interval includes zero. The expanded denominator removes the Round 1 eight-pair degeneracy, but it does not show that sorted features improve J3.

**T2 — negative/unstable ordering.** Clean deltas are +0.0027, +0.0027, −0.0276; flip deltas are +0.0216, −0.0021, +0.0040. Signs and intervals vary by seed; only the clean seed-47 interval excludes zero, on the negative side. This is not a stable sorted advantage.

Thus the source ordering is stable on its new source pool, but the cross-task interaction is not stable: T1 is uninformative and T2 is negative/unstable.

## Wording boundary

**Allowed:**

- “On the predeclared fresh source pool, rich-sorted exceeded rich-unsorted in all three model seeds under both supervision conditions.”
- “T1 and T2 now have 417 and 912 official E pairs from new 8,192-parent pools, respectively.”
- “The sorted interaction is uninformative on T1 and negative/unstable on T2.”
- “The source interaction is stable within this round, but cross-task confirmation is not established.”

**Prohibited:**

- “The representation ordering is confirmed across tasks.”
- “T1/T2 replicate the source result.”
- “A repaired interaction is universally beneficial.” In Round 2, repaired versus additive remains mixed.
- Treating 8/17 Round 1 pairs as a negative task result, or treating 417/912 pairs as automatic success.
- Changing E, using an easier task, continuing parent draws until desired, or using test J for selection.
- Calling group-average predictions independent samples.

## Effect on the plan

Round 2 resolves the original denominator-feasibility concern for T1/T2 but does not add cross-task evidence for the sorted interaction. The hypothesis is therefore supported only on the source task and remains unconfirmed elsewhere.
