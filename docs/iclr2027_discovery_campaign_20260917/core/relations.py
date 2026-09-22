"""A small analytic relational task. It is NOT a validated natural benchmark.

Four visible endpoint identities A/B/C/D define two segments AB and CD.
Only proper crossings count as positive. Collinear/touching examples are
returned as ambiguous and excluded by a response-independent margin.
"""
from __future__ import annotations
import numpy as np


def segment_relation(positions: np.ndarray, tolerance: float = 1e-12) -> dict[str, float | bool | int]:
    x = np.asarray(positions, dtype=float)
    if x.shape != (4, 2) or not np.all(np.isfinite(x)):
        raise ValueError("Expected four finite 2-D points A/B/C/D")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    def orient(a, b, c):
        u, v = b-a, c-a
        return float(u[0]*v[1] - u[1]*v[0])
    a, b, c, d = x
    values = np.array([orient(a,b,c), orient(a,b,d), orient(c,d,a), orient(c,d,b)])
    margin = float(np.abs(values).min())
    ambiguous = margin <= tolerance
    positive = (values[0]*values[1] < 0) and (values[2]*values[3] < 0)
    return {"label": int(positive), "margin": margin, "ambiguous": bool(ambiguous)}


def make_relational_scenes(n: int, seed: int = 0, min_margin: float = .02) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate balanced classes with independent coordinates and fixed roles.

    This balances labels only, not every potentially exploitable image cue.
    Renderer/style and marginal-cue comparisons remain agent tasks.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not np.isfinite(min_margin) or min_margin < 0 or min_margin >= 1:
        raise ValueError("min_margin must be finite and in [0,1)")
    rng = np.random.default_rng(seed)
    targets = {0: n//2, 1: n - n//2}
    counts = {0: 0, 1: 0}
    positions, labels, margins = [], [], []
    for _ in range(max(10_000, n*2000)):
        x = rng.uniform(-.8, .8, size=(4,2))
        relation = segment_relation(x)
        label = int(relation['label'])
        if relation['ambiguous'] or relation['margin'] < min_margin or counts[label] >= targets[label]:
            continue
        positions.append(x); labels.append(label); margins.append(float(relation['margin']))
        counts[label] += 1
        if len(positions) == n:
            order = rng.permutation(n)
            return np.asarray(positions)[order], np.asarray(labels, dtype=np.int64)[order], np.asarray(margins)[order]
    raise RuntimeError("Could not generate requested data; reduce min_margin or n")
