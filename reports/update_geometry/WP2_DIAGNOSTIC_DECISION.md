# WP2 Decision: BOUNDARY_ASSOCIATION_ONLY (2026-09-21)

- Criterion 1 (clean mismatch): PASS, starkly (A≈0, coverage 0.096, Ab≈0).
- Criterion 2 (mismatch↔miss link): ABSENT (N02 null stands; not re-tested).
- Criterion 3 (repair improves ≥1 metric): PASS directionally on coverage
  (+7.3pp, CI excludes 0); offset/alignment unchanged.
- Criterion 4 (fresh confirm): NOT DONE (minimal-diagnostic scope).

Verdict **`BOUNDARY_ASSOCIATION_ONLY`**: flipmine's mechanism gains a
boundary-coverage footnote (more roots near semantic boundaries, same
orientation) — consistent with "coverage not precision" but not a boundary-
alignment repair story. WP2 does not upgrade any claim above Level A/D
support; it constrains the repair narrative honestly. No further WP2 work.
