#!/usr/bin/env python3
"""Exact diagnostic for joint binary correctness of labelled multi-state cases.

Prediction convention: class 1 iff score > threshold.
S and J_star use evaluation labels: they are diagnostic oracle bounds, NOT
out-of-sample prediction accuracies. No model or original project is run here.
Requires NumPy only. Input NPZ arrays: scores[n,k], labels[n,k], optional weights[n].
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class Certificate:
    n_cases: int
    n_states: int
    S_local_separable: float
    J_star_global_oracle: float
    J_at_threshold: float
    ordering_failure: float
    global_incompatibility: float
    operating_point_gap: float
    threshold: float
    diagnostic_optimal_threshold: float | None


def threshold_certificate(scores, labels, threshold: float = 0.0, weights=None) -> Certificate:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    if s.ndim != 2 or min(s.shape) == 0 or y.shape != s.shape:
        raise ValueError('scores and labels must be nonempty matching [n_cases,n_states] arrays')
    if not np.isfinite(s).all() or not np.isfinite(threshold):
        raise ValueError('scores and threshold must be finite')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('labels must be binary 0/1')
    pos = y.astype(bool)
    if not (pos.any(axis=1) & (~pos).any(axis=1)).all():
        raise ValueError('Each case must have at least one positive and one negative state')
    if weights is None:
        w = np.ones(s.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    if w.shape != (s.shape[0],) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('weights must be finite nonnegative [n_cases] with positive sum')
    w = w / w.sum()
    lo = np.max(np.where(~pos, s, -np.inf), axis=1)
    hi = np.min(np.where(pos, s, np.inf), axis=1)
    valid = lo < hi
    S = float(w[valid].sum())
    J = float(w[(lo <= threshold) & (threshold < hi)].sum())
    # Intervals are [lo,hi); at a common endpoint removals and additions both
    # take effect before evaluating the point. Invalid zero-length intervals
    # must not create events.
    events: dict[float, float] = {}
    for a, b, wi in zip(lo[valid], hi[valid], w[valid]):
        events[float(a)] = events.get(float(a), 0.0) + float(wi)
        events[float(b)] = events.get(float(b), 0.0) - float(wi)
    active = best = 0.0
    best_t = None
    for t in sorted(events):
        active += events[t]
        if active > best + 1e-14:
            best, best_t = active, t
    best = float(np.clip(best, 0, S))
    if J > best + 1e-10:
        raise ArithmeticError('J(threshold) exceeded oracle optimum; check implementation')
    parts = [1-S, S-best, best-J]
    if min(parts) < -1e-10 or abs(sum(parts)-(1-J)) > 1e-10:
        raise ArithmeticError('Decomposition failed numerical invariant')
    return Certificate(s.shape[0],s.shape[1],S,best,J,
                       *[float(max(0,x)) for x in parts],float(threshold),best_t)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=0.0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    with np.load(args.input, allow_pickle=False) as d:
        result = threshold_certificate(d['scores'],d['labels'],args.threshold,
                                       d['weights'] if 'weights' in d else None)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(asdict(result),indent=2,allow_nan=False)+'\n',encoding='utf-8')

if __name__ == '__main__':
    main()
