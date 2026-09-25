# M1 mechanism report — role-preserving invariance (source phase)

## M1.0 collision boundary
- T1 opposite-label witness formalized in `artifacts/mechanism_transfer_v3/m1/FEATURE_CONTRACT.json`: shared triangle, inside/outside labels 1/0, margins 0.09375, legacy sorted gap exactly 0.0 (float32 and float64), unsorted/whole-orbit gap 0.3719.
- Source G8 bounded search (`source_search.json`, bank SHA `15f5bf18`, 4096 scenes): nearest opposite-label pair distance 0.0471; 3000-step local search reached 0.0335; no exact merge. Equal-label control is closer (0.0213). This is a lower bound, not a completeness proof.

## M1.2 source matrix (frozen 2188/871 bank, 6 arms x clean/flip x 3 seeds)
- Replication: v3 `sorted-minus-unsorted` flip matches legacy within ~0.005; params/MACs exact.
- Flip J3: raw ~0.08-0.11, unsorted ~0.44-0.46, sorted ~0.52-0.54, orbit ~0.50-0.59, segment_moment ~0.33-0.34, repaired_rho ~0.02-0.03.
- Pre-stated flip contrasts: `orbit-minus-sorted` +0.0654/+0.0005/-0.0311 (last two CIs cross 0, direction unstable); `orbit-minus-unsorted` all positive excluding 0; `segment_moment` worse than both distance arms in all seeds; `repaired_rho` at/below raw.
- Clean: orbit beats sorted/unsorted/raw in all 3 seeds with CIs excluding zero.
- Flow: orbit has highest full-repair share and lowest 111-regression among non-raw arms; repaired head repairs AB mostly via migration.
- D1 accepted: `segment_moment` now uses block-shared statistics (pre-fix swing ~12, post-fix 0); pre-fix run preserved under `superseded/blockwise_stats/` and excluded.

## Claim level
- Supported: the whole-orbit reference removes the risky independent-bag sort without an accuracy price on this bank; exact invariance is neither necessary nor sufficient for the ordering gain.
- Not supported: flip-side gain over the sorted bag (unstable); source information-collision claim (no exact merge found); any cross-task transfer (T1/T2 not run in v3).
- Artifacts: `artifacts/mechanism_transfer_v3/m1/`; code: `experiments/mechanism_transfer_v3/m1/`; training details: `artifacts/mechanism_transfer_v3/m1/training/RESULTS.md`.
