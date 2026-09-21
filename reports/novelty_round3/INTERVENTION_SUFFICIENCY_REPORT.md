# WP-A Report: intervention head adds nothing (2026-09-22)

A0 static / A1 +endpoint augmentation / A2 +intervention loss (λ=1),
same base + same pairs, 3 seeds, r04b recipe. Dev = eval_202.

## Behavior (trans n=217 discriminates; comp saturated ~1.0, no signal)
- static: A1 0.180/0.148/0.201 vs A2 0.160/0.168/0.195 — gate (≥A1−2pp) passes.
- trans: A1 0.442/0.484/0.424 (mean 0.450) vs A2 0.465/0.456/0.442
  (mean 0.454) — identical, no gain.
- comp: A1 0.96/1.00/0.92 vs A2 1.00/1.00/1.00 — saturated construction,
  ΔM ≈ 0 both ways. Primary FAILS (needs ≥10pp reduction).

## A6 representation diagnostic (n=256 pairs/arm/seed)
Median ||h(x1)−h(x2)||, same label + same edit + different outcome:
- s11: A0 4.49 / A1 3.81 / A2 4.78 (A2 best)
- s23: A0 4.51 / A1 2.79 / A2 3.99 (A0 best)
- s47: A0 5.97 / A1 2.71 / A2 5.39 (A0 best)
A2 beats A1 3/3 but beats A0 only 1/3 → criterion 5 FAILS.
Curiosity (non-claim): endpoint augmentation (A1) consistently COLLAPSES
future-distinguishing separation below both A0 and A2 — more endpoint data
makes same-label states more similar, not more predictive.

## Reading
"State-sufficient ⇏ intervention-sufficient" as a trainable gap is not
supported: the intervention loss installs neither behavioral gain nor
representation separation beyond the static baseline. The failure is not
the λ or the head size (frozen) — A2 matches A1 everywhere, i.e., the extra
supervision is simply inert.
