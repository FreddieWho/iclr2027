# WP-B Report: no stable transfer (2026-09-22)

Round-1 only. Sources: s11/s23 clean (kept 149/143 of K=256 — shortfall
documented; all arms equal-N). Targets: source-seed / unseen-seed47 /
archB(32/16). 45 models, r04b recipe. Fixed-denominator comp (n=32).

## Design correction (honest)
`seed11` vs `seed47` targets are computationally IDENTICAL runs (same arch,
init seed, data, recipe — the flag only renamed outputs). There is one
CoordMLP(64,32) transfer table below, not two. The real transfer dimensions
are: fresh-init models (inits 23/47 differ from source r04b_s11's init 11)
repaired with source-CE sets, plus the archB(32/16) table.

## comp_basefixed means (3 training inits)
| target | random | flipmine | src11-CE | src23-CE | diverse |
|---|---|---|---|---|---|
| 64/32 fresh-init | 0.875 | 0.688 | 0.594 | 0.823 | 0.802 |
| archB (x-arch) | 0.854 | 0.802 | 0.771 | 0.792 | 0.740 |

## Transfer reading
- src11-CE vs flipmine on fresh-init 64/32: 0.594 vs 0.688 (Δ=9.4pp), direction
  2/3 with one strong reversal (s47: 0.875 vs 0.500). Below the ≥10pp +
  stability bar; within retraining noise floor (±15pp, L-006).
- src23-CE: no advantage anywhere. Diverse: no advantage.
- archB: all arms 0.74–0.85, nothing ≥5pp over flipmine.
- trans: flat 0.42–0.56 across all 45 models — repair sets move composition
  denominators, not transition behavior.

## Oracle-query note
Mining spent 59–82k queries/source for ~150 kept. No efficiency race was
run (budgets fixed); nothing suggests ≤0.5× flipmine is reachable.
