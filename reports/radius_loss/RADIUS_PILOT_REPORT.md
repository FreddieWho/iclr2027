# Radius-Loss Pilot Report (2026-09-22): BURIED

3 arms x 3 seeds, CoordMLP(64,32) from scratch, r04b recipe (Adam 1e-2,
300ep full-batch), equal budgets (750 flips + 512 base). Dev = eval_202.

## trans_miss (primary)
- clean:   0.558 / 0.535 / 0.562 (mean 0.551)
- flipcov: 0.442 / 0.438 / 0.470 (mean 0.450)
- radius:  0.484 / 0.502 / 0.512 (mean 0.499)
- radius vs flipcov: +4.2 / +6.5 / +4.2pp — WORSE on 3/3 seeds.
  Gate needed mean < -10pp same direction. FAIL, wrong direction.

## Secondary
- static: radius 0.143–0.154 vs flipcov 0.145–0.168 (no cost, moot).
- comp: all arms 0.83–1.00, n≈25–39 — saturated construction here, no
  discrimination (different denominator family from E1 C1; not comparable).
- action: mixed within noise (0.73–0.76 vs 0.69–0.83).
- stay_FA: radius_s23 0.193 vs 0.02–0.09 elsewhere — margin term makes one
  seed twitchy on stays (red flag, not pursued).
- R mechanism: rm_miss medians clean 0.41–0.47 → flipcov 0.29–0.31,
  radius 0.28–0.34. Radius loss moves R the right way but NO better than
  plain endpoint coverage.

## Reading
Endpoint BCE already moves the boundary; explicit Δf margin-shaping is
redundant at best, destabilizing at worst. Side D's R describes where the
defect lives but prescribing crossing margins via this loss does not fix it
beyond coverage. The R signal stays a diagnostic (L-014), never a loss.
