# M1.2 source representation matrix — results

Frozen recipe (`reports/e832_focus/STRUCTURE_DECISION.md`): 300 full-batch Adam
steps, lr 0.01/0.003/0.001, selection by final pooled static+single dev BCE
(smaller lr breaks ties), `logit > 0`, 4 threads, supervision clean or
clean + all 1534 mined single flips. Bank: `artifacts/e832_focus/structure/bank_quartets.npz`
(`15f5bf18…`, 2188 quartets / 871 parents), never pooled with any other bank.
Seeds 11/23/47 are training replicates; each seed keeps its own parent-cluster
interval. 108/108 cells trained on CPU in ~110 s; evaluation ~20 s.

Artifacts: `artifacts/mechanism_transfer_v3/m1/training/` (`metrics_3seed.json`,
`SUMMARY_3seed.json`, `selection.json`, `capacity.json`, `probe.json`,
`orbit_seam.json`, `MANIFEST.json`, `runs/*/model.pt|receipt.json|train_curve.json|dev_curve.json`).
Code: `experiments/mechanism_transfer_v3/m1/training/` (+ `DECISION_NOTES.md`).
Verify: `python3 -m unittest discover -s experiments/mechanism_transfer_v3/m1/training -p 'test_*.py' -v`.

## 1. Replication of the frozen report (same bank, same recipe, v3 modules)

| arm | supervision | legacy J3 (11/23/47) | v3 J3 (11/23/47) | max abs diff |
|---|---|---|---|---|
| `rich8_sorted` | flip | 0.5215 / 0.5425 / 0.5329 | 0.5206 / 0.5425 / 0.5343 | 0.0014 |
| `rich8_unsorted` | flip | 0.4561 / 0.4538 / 0.4410 | 0.4607 / 0.4525 / 0.4388 | 0.0046 |
| `raw` | clean | 0.0425 / 0.0585 / 0.0452 | 0.0425 / 0.0580 / 0.0407 | 0.0046 |
| `repaired_rho` → `repaired_segment_rho` | flip | 0.0174 / 0.0288 / 0.0219 | 0.0151 / 0.0283 / 0.0315 | 0.0096 |

`sorted_minus_unsorted` under flip reproduces as well: legacy +0.0654 / +0.0887 /
+0.0919, v3 +0.0599 / +0.0900 / +0.0955, all intervals excluding zero in both.
Parameter and MAC counts match the frozen table exactly (raw/rich8 6849 params
and 6688 MAC, orbit 6721/6560, repaired 8001/19600). Legacy and v3 trajectories
differ only by initialization/threading consumption order, consistent with the
report's own warning that weights from different thread regimes are not
bit-comparable.

## 2. Arm results (J3 row mean, parent-cluster 95% interval)

| arm | clean J3 (11/23/47) | flip J3 (11/23/47) |
|---|---|---|
| `raw` | 0.0425 / 0.0580 / 0.0407 | 0.0809 / 0.1101 / 0.0763 |
| `rich8_unsorted` | 0.1901 / 0.1682 / 0.1677 | 0.4607 / 0.4525 / 0.4388 |
| `rich8_sorted` | 0.2153 / 0.2317 / 0.2518 | 0.5206 / 0.5425 / 0.5343 |
| `orbit_distance` | 0.2925 / 0.2605 / 0.3044 | 0.5859 / 0.5430 / 0.5032 |
| `segment_moment` | 0.1682 / 0.1696 / 0.1728 | 0.3364 / 0.3332 / 0.3446 |
| `repaired_segment_rho` | 0.0242 / 0.0256 / 0.0425 | 0.0151 / 0.0283 / 0.0315 |

All 36 intervals exclude zero. `J4` is computed independently from P and is
never copied from `J3` (e.g. `raw` s11 flip: `J3` 0.0809, `J4` 0.0804). G8
max-abs-logit swing on 64 real A states: exactly 0 for `orbit_distance`,
`rich8_sorted`, `repaired_segment_rho`, and `segment_moment`; 11–26 for `raw`
and `rich8_unsorted`.

## 3. Pre-stated contrasts (delta J3, flip supervision)

| contrast | s11 | s23 | s47 |
|---|---|---|---|
| `orbit_distance − rich8_sorted` | +0.0654 | +0.0005 (CI crosses 0) | −0.0311 (CI crosses 0) |
| `orbit_distance − rich8_unsorted` | +0.1252 | +0.0905 | +0.0644 |
| `orbit_distance − raw` | +0.5050 | +0.4328 | +0.4269 |
| `segment_moment − rich8_sorted` | −0.1842 | −0.2093 | −0.1897 |
| `segment_moment − orbit_distance` | −0.2495 | −0.2098 | −0.1586 |
| `segment_moment − raw` | +0.2555 | +0.2230 | +0.2683 |
| `rich8_sorted − rich8_unsorted` | +0.0599 | +0.0900 | +0.0955 |
| `repaired_segment_rho − raw` | −0.0658 | −0.0818 | −0.0448 |

Under clean supervision every `orbit_distance` contrast against
`rich8_sorted`, `rich8_unsorted`, and `raw` is positive in all three seeds with
intervals excluding zero (`orbit_distance − rich8_sorted`: +0.0772 / +0.0288 /
+0.0526).

## 4. Repair flow against the fixed raw-clean baseline (flip supervision)

Baseline 110 cells are A and B correct with AB wrong (909 / 843 / 892 of 2188).

| arm | R_endpoint (11/23/47) | R_full | M | regression111 |
|---|---|---|---|---|
| `orbit_distance` | 0.809 / 0.804 / 0.776 | 0.589 / 0.524 / 0.500 | 0.220 / 0.280 / 0.276 | 0.282 / 0.344 / 0.335 |
| `rich8_sorted` | 0.788 / 0.789 / 0.776 | 0.528 / 0.554 / 0.548 | 0.260 / 0.235 / 0.228 | 0.362 / 0.290 / 0.293 |
| `rich8_unsorted` | 0.672 / 0.705 / 0.650 | 0.433 / 0.432 / 0.401 | 0.239 / 0.273 / 0.249 | 0.356 / 0.411 / 0.359 |
| `segment_moment` | 0.620 / 0.636 / 0.633 | 0.338 / 0.343 / 0.361 | 0.283 / 0.293 / 0.272 | 0.475 / 0.494 / 0.395 |
| `repaired_segment_rho` | 0.492 / 0.434 / 0.407 | 0.013 / 0.036 / 0.037 | 0.479 / 0.399 / 0.370 | 0.983 / 0.946 / 0.922 |

`R_endpoint = R_full + M` holds to floating-point residual. The segment head
repairs the AB bit without preserving A and B, so almost all of its endpoint
repairs are migrations; `orbit_distance` has the highest full-repair share and
the lowest regression of the baseline-111 cells among the non-raw arms.

## 5. What this does and does not show

Shown on this bank and recipe:

* The frozen source-ordering effect reproduces under the v3 shared modules, so
  the M1.2 matrix is run on a validated implementation.
* Exact group invariance is neither necessary nor sufficient for the ordering
  gain: `rich8_sorted`, `orbit_distance`, and `repaired_segment_rho` are all
  exactly invariant, and their flip `J3` ranges from 0.02 to 0.53.
* The whole-orbit distance reference is the strongest arm on clean supervision
  in all three seeds, and on flip it matches or exceeds the sorted bag
  (+0.0654, +0.0005, −0.0311 with the last two intervals crossing zero). It
  therefore removes the dependence on the risky independent-bag sort without
  paying an accuracy price, but the flip-side direction is not stable enough to
  claim a gain over the sorted bag.
* The continuous segment-moment reference with symmetric nonlinear fusion is
  better than raw in all seeds but clearly worse than both distance-feature
  arms in all seeds. Under this recipe the "keep the segment, drop the sort"
  representation is not sufficient.
* The corrected segment-preserving head is at or below raw in all seeds
  (`−0.066 / −0.082 / −0.045`) and below raw on clean in two of three seeds;
  this reproduces the frozen negative with the v3 implementation.

Not shown, and not claimable from this matrix:

* No cross-task transfer. T1/T2 with these representations was not run in v3;
  per-seed source intervals say nothing about another task.
* No information-collision claim on the source task. `source_search.json`
  reports no exact opposite-label merge for the source G8 features under a
  bounded local search; the M1.0 exact collision is a T1 (triangle/query)
  witness, not a source witness. The ordering effects here are a finite-data
  representation/optimization effect on this bank until a source-side collision
  is exhibited.
* No claim that J4 or the margin slice replaces the full denominator; the
  margin slice and `CCM` are descriptors on the same 2188/871 denominator.
* The bank is the frozen development bank already used by earlier reports, and
  `dev512`, the confirm and holdout pools were not read.
* The group-mean reference is a zero-train control and was not trained.
* `orbit_distance` is a canonical representative and therefore only piecewise
  smooth. Its tie fraction at 1e-9 is 0.0 on the 512 training scenes and on the
  2188 bank P and A states, so no sample in this experiment sits on a seam; the
  caveat is recorded rather than exercised.

## 6. Decision note referenced

`DECISION_NOTES.md` D1: the `segment_moment` two-block descriptor now uses one
statistics vector pooled across the two exchangeable blocks. Independent
per-block statistics did not commute with the segment swap, so the standardized
features moved under a legal permutation and the model was not exactly
invariant (measured swing ~12 before the fix, 0 after). The pre-fix run is
preserved under `superseded/blockwise_stats/` and is not part of the result.
