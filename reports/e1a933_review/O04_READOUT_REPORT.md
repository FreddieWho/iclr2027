# O04 frozen readout reanalysis

Status: EXECUTED_BOUNDED_REANALYSIS; full O04 remains PARTIAL.

12 frozen encoders (4 arms × seeds 11/23/47), 6 selected heads each, all 237 legacy dev512 A/B/AB joint events from 106 parents. This is the original U04 three-state criterion, not four-state accuracy. No new external data. Old outputs remained read-only.

The clean train_101 parents were deterministically split into 409 head-fit and 103 head-dev parents. All edits followed their parent. Frozen encoders had already seen the head-dev parents: this split protects head fitting and selection, not independence of the representation. Clean head-fit moments alone standardize the frozen feature cache; the same moments apply to static, flip, and evaluation. Model/hash lineage cannot be inferred from integer parent IDs: see the separate R03 lineage audit before asserting train/evaluation disjointness.

Static/flip logistic heads start from the identical zero vector; each receives the same three L2 candidates (0.0001, 0.001, 0.01) and L-BFGS-B convergence tolerances. MLP heads start identically for each width/seed, use widths 16 and 64, and receive 300 Adam epochs, with head-dev BCE selection every 10 epochs. Static selection uses static head-dev states; flip selection uses a 50:50 clean/flip head-dev mixture, matching each training objective. Thus supervision regimes differ intentionally; selection is not based on evaluation J. MLP results concern these two budgets only.

The ranking intervention adds logistic positive/negative comparisons across different head-fit parents (4096 candidate pairs, same-parent pairs removed), weight 0.2, to the same state-labelled BCE. Both BCE and ranking use the same convex solver/iteration cap and L2 search; actual iterations are logged, not forced equal. This extends supervision using existing train state labels and is not the original unseen-combination training claim. Every prediction remains a single-state function: no qid, quartet threshold, or evaluation label enters inference.

## Results

Means below average three model seeds on the SAME exposed bank; seeds are not independent tasks. J uses fixed zero logit threshold. J_dev uses a threshold selected for weighted single-state accuracy on head-dev states. J* is the evaluation-truth oracle threshold bound, never deployable performance.

| Frozen encoder | Head/regime | S | J* | J | J_dev |
|---|---|---:|---:|---:|---:|
| raw_clean | logistic_static | 0.5513 | 0.0703 | 0.0563 | 0.0253 |
| raw_clean | logistic_flip | 0.5527 | 0.0844 | 0.0197 | 0.0394 |
| raw_clean | ranking_static | 0.5513 | 0.0703 | 0.0563 | 0.0253 |
| raw_clean | ranking_flip | 0.5527 | 0.0886 | 0.0267 | 0.0281 |
| raw_clean | mlp_static | 0.5513 | 0.0689 | 0.0253 | 0.0267 |
| raw_clean | mlp_flip | 0.4937 | 0.0816 | 0.0197 | 0.0295 |
| raw_flipmine | logistic_static | 0.6962 | 0.1252 | 0.0759 | 0.0731 |
| raw_flipmine | logistic_flip | 0.6990 | 0.1210 | 0.0816 | 0.0774 |
| raw_flipmine | ranking_static | 0.6962 | 0.1238 | 0.0774 | 0.0802 |
| raw_flipmine | ranking_flip | 0.6990 | 0.1224 | 0.0816 | 0.0788 |
| raw_flipmine | mlp_static | 0.6906 | 0.1224 | 0.0774 | 0.0703 |
| raw_flipmine | mlp_flip | 0.6624 | 0.1294 | 0.0703 | 0.0816 |
| relfeat | logistic_static | 0.8481 | 0.2925 | 0.1730 | 0.0872 |
| relfeat | logistic_flip | 0.8439 | 0.3122 | 0.2644 | 0.2518 |
| relfeat | ranking_static | 0.8481 | 0.2925 | 0.1730 | 0.0872 |
| relfeat | ranking_flip | 0.8439 | 0.3136 | 0.2757 | 0.2560 |
| relfeat | mlp_static | 0.8509 | 0.2940 | 0.1646 | 0.0816 |
| relfeat | mlp_flip | 0.8411 | 0.3052 | 0.2700 | 0.2574 |
| relflip | logistic_static | 0.8917 | 0.4248 | 0.3404 | 0.3305 |
| relflip | logistic_flip | 0.8917 | 0.4276 | 0.3854 | 0.3826 |
| relflip | ranking_static | 0.8917 | 0.4248 | 0.3418 | 0.3305 |
| relflip | ranking_flip | 0.8917 | 0.4276 | 0.3854 | 0.3826 |
| relflip | mlp_static | 0.8917 | 0.4233 | 0.3235 | 0.3826 |
| relflip | mlp_flip | 0.8917 | 0.4304 | 0.3910 | 0.3826 |

All 144 convex candidate fits report solver success; maximum final gradient infinity norm 1.17e-06. The successful termination reason and full objective traces are saved; success is numerical optimization evidence, not scientific validation.

## Interpretation and limits

Frozen relational encoders retain the highest S/J under these refits. However S changes when the head changes: e.g. raw_clean seed 11 logistic-flip S=0.5738 versus MLP-flip S=0.4810. Therefore S is a property of the final encoder-plus-head scalar score. Relative stability for some relational encoders supports only a bounded empirical observation, not encoder-only or all-head invariance.

The BCE-versus-ranking flip comparison gives limited, seed/encoder-dependent changes; it does not close the large S−J* gap. No universal readout or probability failure is established. Head-dev threshold changes do not equal evaluation-oracle gains.

`summary.json` contains same-parent paired deltas for S, J*, J, both within encoder across heads and across encoder arms at fixed head/seed. It reports quartet-weighted means and parent-equal means with parent bootstrap intervals. Each row uses the same parents; no seed/parent/task denominator inflation. J* paired intervals hold the sample-chosen oracle threshold fixed and are descriptive, not uncertainty for a deployable decision rule. `parent_metrics.csv` and each `*_scores.npz` retain parent IDs and per-event flags/scores. Paired diagnostics are exploratory and not multiplicity-adjusted.

NOT_RUN: broader nonlinear architecture search; context-conditioned correction; fresh-bank independent confirmation. The new-task prospectively specified intervention prediction and the D10 downstream echo were subsequently completed (see `O04_FORWARD_INTERVENTION.md`): primary verdict MISS (P1 7/8), and the echo adds only a domain-of-definition boundary. This reanalysis still cannot make the decomposition a validated prospective intervention diagnostic. Training-bank exposure and mining selection risks remain bounded by the separate lineage review; no split here retroactively repairs historical encoder leakage.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python experiments/e1a933_review/readout_objective_check.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python experiments/e1a933_review/readout_refit.py --output artifacts/e1a933_review/readout_refit_FRESH_DIRECTORY
python experiments/e1a933_review/readout_report.py --run artifacts/e1a933_review/readout_refit_FRESH_DIRECTORY --report reports/e1a933_review/O04_readout_FRESH.md
```

The actual executed run is `artifacts/e1a933_review/readout_refit_20260924/`; its stdout is the sibling `.log` file. The fitter refuses an existing directory. SciPy is imported before torch because this environment otherwise selects a system libstdc++ lacking GLIBCXX_3.4.29. The failed import occurred before artifact creation; no computation was silently dropped. Checks cover finite-difference BCE/ranking gradients, convex convergence, common AB reconstruction, and unchanged frozen backbone hashes.
