# Mechanism evidence audit — 2026-09-25

Baseline: `3fc9757`. Scope: `experiments/mechanism_transfer_v3/**` and the archived M1/M2 evidence named below. Historical result files remain unchanged. Sealed pools were not read. The corrected 16-cell GPU run is `DEFERRED_PENDING_ACCEPTANCE`; its outputs are not used here.

## Findings

### M1: source orbit and T1 witness

- The formal T1 pair is a valid, specific counterexample: the current T1 oracle gives opposite labels with the archived 0.09375 margin; the legacy independently sorted feature vectors are exactly equal in float32 and float64, while the unsorted and legal whole-orbit representations separate the pair. This supports a collision for this constructed pair only.
- The source G8 search reports an exact nearest opposite-label distance within its finite 4,096-scene bank of 0.0471. Its 3,000-step local search reports a best sampled distance of 0.0335. The old source and JSON call this a “lower bound”; that is incorrect. A positive best-observed distance does not bound all unsampled points away from zero and does not establish global completeness.
- The whole-orbit source feature uses one legal group element across all six distances. Existing tests compare its batched result with the scalar legal-orbit implementation and test invariance under each legal source permutation. The saved seam diagnostic has a code defect: it compared candidates with the componentwise minimum instead of counting duplicates of the lexicographic minimum. This can miss ties (a square is a concrete counterexample). The current diagnostic code and its regression test are repaired. This does not alter archived training features or checkpoints; the old seam-rate receipt is superseded for interpretation.
- The archive reports clean-setting orbit-distance J3 gains against sorted, unsorted, and raw arms across three seeds. For flip supervision, orbit-minus-sorted point contrasts are `+0.0654`, `+0.0005`, and `−0.0311`; the latter two archived intervals cross zero. There is no stable flip-side improvement over sorted features. Existing M1 intervals pair a row-weighted point estimate with an equal-parent bootstrap interval. The current evaluator now labels that estimand and emits a separate row-weighted cluster interval. This audit does not present the historical mixed-estimand intervals as inferential support.

### M2: frozen head and visual frontend

- The remote runner used for the archived M2 results is byte-identical to the baseline runner: `/root/e832_route2/m2_v3/code/m2/gpu_transfer.py`, 16,151 bytes, SHA-256 `fab5ffde7e0e6bf10303cc40a370f516c6cf36a0f8dd8094f05fa20618b26d28`; the baseline Git blob has the same hash. That implementation wrapped the backbone in `torch.no_grad()` for both frontend modes. Its “fine-tuned” cells therefore had no backbone gradients. `model.train()` also left BatchNorm in training mode for the nominally frozen backbone, so its running statistics changed. The old frozen-versus-tuned comparison is `INVALID_IMPLEMENTATION` and cannot exclude probe capacity or a learnable frontend.
- The archive reports direct-full J3 `0.915–0.921`, clean-only direct J3 `0.845–0.870`, true-geometry plus frozen-head J3 `0.796`, random-head J3 `0.000–0.013`, and geometry-frontend J3 near `0.19–0.22` with dev MSE near `0.51`. These remain descriptive archived numbers pending a fresh prediction-level acceptance. In particular, the geometry MSE is evidence about the old recipe execution, not proof that smooth regression cannot learn the interface.
- The old `direct_clean` loop exposed each clean singleton only once per epoch, while `direct_full` exposed the full singleton bank. Their training budgets were unequal. The corrected runner resamples clean rows with replacement to match full per-epoch row exposure and records unique rows and total exposures.
- J3 is joint correctness on A, B, and AB; J4 additionally requires P. The old runner stored row-weighted J3 together with an equal-parent CI estimate, mixing point-estimate and interval estimands. The corrected runner stores row-weighted cluster CI and equal-parent CI separately and writes parent IDs into prediction NPZ files.
- Code inspection finds no test-J selection path: trained cells select the best epoch by singleton-dev loss and evaluate quartet J only afterward. The geometry transform uses the frozen source-head mean/SD for train, dev, and test, and all image paths use the same `/255`, resize, and ImageNet normalization contract.
- The analytic image baseline (`artifacts/mechanism_transfer_v3/m2/analytic_baseline.json`, reported J3 `0.886`) is a separate same-bank reference, not a ceiling. The pilot-bank value is not pooled with the 692-quartet bank.

## Engineering work and verification

- Fixed M2 backbone freeze/unfreeze semantics, seeded model construction before probe initialization, recorded per-tensor backbone and BatchNorm before/after hashes plus first-batch gradient evidence, matched `direct_clean` exposure, separated J3 interval estimands, and added prediction parent IDs.
- Fixed M1 tie counting and corrected the bounded-search wording. M1 row-weighted confidence intervals are now emitted under an explicit field; old JSON remains immutable.
- M2 CPU contract tests: **8/8 passed** (`experiments.mechanism_transfer_v3.m2.test_m2_gpu`, 1.3 s). The new M1 square-tie regression test was added but not run in this closeout.
- `artifacts/submission_audit_20260925/mechanism/recompute_archive.py` is the CPU-only recomputation entry point. It reads archived predictions and banks, never trains, and writes only under the audit artifact directory. A complete numeric recomputation receipt was not produced in this closeout; archived M2 values above remain explicitly unaccepted pending the corrected GPU predictions.
- Remote corrected 16-cell run: `DEFERRED_PENDING_ACCEPTANCE`. It must be independently checked from prediction NPZ files and its backbone update receipts before any revised M2 scientific conclusion is stated.

## Allowed summary

The M1 archive contains a valid constructed T1 feature collision and source-bank evidence for the role-preserving orbit representation, with clean-setting gains reported on the frozen bank; global information loss and stable flip-side advantage are unestablished. M2's archived direct, true-geometry, and random-head outputs are descriptive only. The frontend contrast is invalid under the archived runner implementation, and the corrected run remains pending acceptance.

## Immediate local closeout

An M1-only numeric output was subsequently produced at `artifacts/submission_audit_20260925/mechanism/archive_recomputation_m1.json`. It was not independently accepted before the user requested immediate closeout; no new inferential claim is promoted from it. The combined archive recomputation and new square-tie test have no completed acceptance receipt in this handoff. The M2 repaired-run results remain excluded.
