# Superseded: per-block statistics for `segment_moment`

The first run of the M1.2 matrix fitted independent per-column statistics for
the two segment blocks of the `segment_moment` descriptor. The descriptor
itself is exactly endpoint-swap invariant and block-swap covariant, but
independent per-block statistics do not commute with the block swap, so the
standardized features changed under a legal segment swap and the trained model
was not exactly G8 invariant (measured max |logit| swing ~12 on 64 A states).

The freeze recipe applies one shared xy statistic to all four points precisely so
that coordinate standardization does not break the task group. The analogous
choice for an exchangeable two-block descriptor is one 5-vector of statistics
pooled across both blocks and tiled onto each block. That makes standardization
commute with the block swap, so the symmetrized model is exactly G8 invariant.

Preserved here: the nine superseded `segment_moment` cells (3 seeds x 3 lrs)
plus the pre-fix `metrics_3seed.json`, `SUMMARY_3seed.json`, and
`selection.json`. These numbers are the preliminary block-wise variant and must
not be quoted as the M1.2 result.
