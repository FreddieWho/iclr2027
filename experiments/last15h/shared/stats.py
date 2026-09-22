"""Shared statistics helpers for L007-series diagnostics.

spearman() uses AVERAGE RANKS for ties (SciPy rankdata semantics). The
earlier `argsort(argsort(x))` variant produced ordinal ranks, which assigns
arbitrary distinct ranks to equal values and can inflate or deflate rho.
"""
import numpy as np


def _avg_rank(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y, min_n=3):
    """Average-rank Spearman rho over pairwise-finite entries.

    Returns (rho, n) with rho None when degenerate (n < min_n, or either
    side constant, which makes rho undefined rather than 0)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < min_n:
        return None, n
    rx = _avg_rank(x[m])
    ry = _avg_rank(y[m])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    if d <= 0:
        return None, n
    return round(float((rx * ry).sum() / d), 4), n
