# M2 scientific acceptance and fixed conclusions

Status: **SCIENTIFIC_ACCEPTANCE_COMPLETE_SCOPED**. Acceptance of the collected 16-cell run is complete; the stronger original M2 proposal is not declared complete.

## Primary population and duplicate correction

The archive contains 692 rows from 351 parents. `quartet_keys` are parent-lineage keys, not unique quartet identifiers. Seven rows repeat the same parent and unordered pair of exact edits, with identical rendered images up to A/B ordering. The model-blind primary rule retains the first occurrence: **685 distinct quartets / 351 parents**. All 16 predictors agree on the duplicate correctness patterns. Original files are immutable; both 692-row and 685-quartet analyses are retained. The largest J3 shift from deduplication is 0.209 percentage points.

## Verified execution and evidence

- All 16 declared cells succeeded; 60 output hashes and 7 launch-input hashes match.
- Independent oracle evaluation agrees on all 2,768 stored states; all quartet labels satisfy the atomic-preserving/joint-changing contract.
- Train/dev/test parent keys are disjoint; no exact training-image/quartet-image matches were found.
- All saved labels, parent order, finite logits and P/A/B/AB, atomic, J3 and J4 summaries agree with independently calculated values.
- Frozen runs have zero changed backbone parameter tensors, zero changed BatchNorm buffers, and no backbone gradient.
- Tuned runs have 60 changed parameter tensors and nonzero first-batch backbone gradients. Selected-checkpoint tensor hashes match the recorded post-selection hashes. All probes changed, and paired frozen/tuned probe and backbone initializations match.
- Every trained cell has 20 epochs and 50,640 image presentations. Clean-only uses 1,024 unique training images resampled to 2,532 presentations/epoch; full-singleton uses 2,532 distinct training rows. Both select on the same 333 singleton dev images.
- Checkpoint selection uses minimum singleton-dev BCE for direct models or dev geometry MSE for frontends; no test-J selection path was found.
- Independently reconstructed legal-orbit features and a CPU replay of the frozen source head reproduce all 2,768 true-geometry decisions (maximum CPU/GPU logit difference 4.58e-5).
- 419 executable acceptance checks passed. These checks include engineering/provenance checks; they are not 419 independent scientific experiments.

## Accepted joint-correctness results

J3 requires A, B and AB to be correct together; it is not ordinary single-image accuracy. All following numbers use 685 distinct quartets.

| Condition | Seed 803 | Seed 805 | Seed 806 |
|---|---:|---:|---:|
| Frozen visual backbone + learned geometry probe | 0.1825 | 0.1533 | 0.1781 |
| Tuned visual backbone + learned geometry probe | 0.4555 | 0.3416 | 0.3825 |
| Clean-only direct visual model, matched exposures | 0.8642 | 0.8730 | 0.8920 |
| Full-singleton direct visual model | 0.9095 | 0.8934 | 0.9168 |
| Random head on true geometry | 0.0000 | 0.0058 | 0.0131 |
| Frozen trained head on true geometry | 0.7942 | Deterministic reference | Deterministic reference |

The random-head controls receive true geometry. They establish a trained-head reference under that input; they do not constitute a matched random-head test on the learned frontend outputs. All source-head reuse is for the same predicate across observation types.

## Paired contrasts and uncertainty

10,000 parent-cluster percentile bootstrap draws; seed 20260925; quartet-weighted point estimates; 95% per-seed intervals. These are post-run acceptance analyses without multiplicity adjustment, not new preregistered superiority tests. The three training seeds share one bank.

| Contrast | Seed | Delta J3 | 95% interval |
|---|---:|---:|---|
| Tuned minus frozen | 803 | +0.2730 | [+0.2230, +0.3216] |
| Tuned minus frozen | 805 | +0.1883 | [+0.1420, +0.2343] |
| Tuned minus frozen | 806 | +0.2044 | [+0.1554, +0.2540] |
| Full singleton minus clean-only | 803 | +0.0453 | [+0.0203, +0.0714] |
| Full singleton minus clean-only | 805 | +0.0204 | [-0.0072, +0.0471] |
| Full singleton minus clean-only | 806 | +0.0248 | [+0.0058, +0.0449] |
| Direct minus tuned geometry pipeline | 803 | +0.4540 | [+0.4108, +0.4971] |
| Direct minus tuned geometry pipeline | 805 | +0.5518 | [+0.5089, +0.5953] |
| Direct minus tuned geometry pipeline | 806 | +0.5343 | [+0.4899, +0.5796] |

## Repair flows: tuning the geometry frontend

Baseline set H contains frozen-frontend cases with A/B correct and AB wrong. It is fixed before the tuned comparison.

| Seed | H quartets / parents | Full repairs | Relocations | Endpoint repairs | Joint gains / regressions on full bank |
|---|---|---:|---:|---:|---|
| 803 | 246 / 168 | 114 | 31 | 145 | 234 / 47 |
| 805 | 251 / 168 | 85 | 44 | 129 | 183 / 54 |
| 806 | 230 / 159 | 88 | 38 | 126 | 207 / 67 |

Full repair plus relocation equals endpoint repair in every pair. Tuning has genuine gains but also regressions; it is not regression-free. Dev geometry MSE improves from 0.394–0.403 to 0.159–0.164.

## Fixed scientific decision

On 685 distinct quartets from 351 parents, unfreezing the geometry-prediction backbone raises J3 from 0.153–0.182 to 0.342–0.455. Paired per-seed parent-cluster intervals for gains of 0.188–0.273 exclude zero. The fixed source-head pipeline remains substantially below direct visual classification (J3 0.893–0.917) under the tested recipes. These results support a learnable adaptation benefit, not module superiority or an unlearnable interface.

The direct full-singleton versus clean-only point differences are positive for all seeds (+2.0–4.5 pp), but the seed-805 interval includes zero. Do not describe this as significant improvement in every seed.

## What this does not close

- Training the frontend uses privileged geometric targets. Direct classifiers use relation labels. Matching image exposure does not match supervision information, loss functions or all compute costs.
- The tuned intervention bundles parameter learning and BatchNorm adaptation. No separate BN-only intervention identifies their causal contributions.
- The six-distance output is unconstrained and the canonical interface is lexicographic. Physical validity, matched geometry-error interventions and causal attribution to interface discontinuity were not tested.
- Cross-renderer validation, source-predicate changes, object-token swaps, matched auxiliary classifiers and other extensions from the original M2 proposal remain unrun.
- Full ResNet replay of all 16 cells is not performed; checkpoint tensors, training receipts and prediction artifacts are verified, and the true-geometry head is independently replayed.
- No main-paper/PDF numerical results are changed by this acceptance. This report is the current M2 conclusion source.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 experiments/submission_audit_20260925/m2_scientific_acceptance.py
```

Outputs: `artifacts/submission_audit_20260925/m2_acceptance/ACCEPTANCE.json`, `DECISION.json`, and `PREDICTIONS.csv`. CSV includes `included_in_primary` so both original and deduplicated estimates can be reproduced. Large model weights and image arrays remain local; hashes alone do not make them publicly downloadable.
