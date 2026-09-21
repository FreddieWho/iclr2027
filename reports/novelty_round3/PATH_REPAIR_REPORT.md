# WP-C Report: partition works, path repair loses (2026-09-22)

## Feasibility (engineering): PASS
64 paths (only 4 single-cross from random candidates — thin), median 5
linear regions/path, prediction-constancy verified 100% via
activation-pattern enumeration + bisection. The tooling is sound.

## Repair pilot: path loses to point, even on seen paths
- seen_point 0.041 vs seen_path 0.235 (4 worst paths, equal 8-sample budget).
- unseen: no held-out singles available (NaN) — cannot even test generalization.
- Mechanism note: path-arm samples at boundary ±0.01 are near-ambiguous
  low-margin points — the U03-pixquartet lesson repeats (near-boundary
  equal-weight supervision distorts). Point arm's ±0.02–0.20 endpoints are
  cleaner supervision. The failure is consistent, not noise.
