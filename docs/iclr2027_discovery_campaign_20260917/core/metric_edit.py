"""A conservative frozen-feature metric initializer, not a novel solver.

Use equal information budgets against standard metric learning. Exact
feature collisions cannot be undone by this or any deterministic metric.
"""
from __future__ import annotations
import numpy as np


def second_moment(deltas: np.ndarray) -> np.ndarray:
    x = np.asarray(deltas, dtype=float)
    if x.ndim != 2 or min(x.shape) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("deltas must be nonempty finite [pairs,features]")
    return x.T @ x / len(x)


def fit_metric_initializer(semantic_deltas: np.ndarray, nuisance_deltas: np.ndarray,
                           rank: int = 4, ridge: float = 1e-4,
                           max_gain: float = 1.) -> dict[str, np.ndarray]:
    """Generalized signal/nuisance directions with bounded PSD identity edit.

    Ridge and gain are engineering starting values, not empirically optimal.
    M has eigenvalues >= 1; this initializer enhances selected directions
    while retaining all other input-feature directions.
    """
    s, n = second_moment(semantic_deltas), second_moment(nuisance_deltas)
    if s.shape != n.shape:
        raise ValueError("Semantic and nuisance feature dimensions must match")
    if not isinstance(rank, (int, np.integer)) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if not np.isfinite(ridge) or ridge <= 0 or not np.isfinite(max_gain) or max_gain < 0:
        raise ValueError("ridge>0 and max_gain>=0 must be finite")
    d = len(s)
    n_regularized = n + ridge * np.eye(d)
    eigen_n, basis_n = np.linalg.eigh(n_regularized)
    invsqrt = (basis_n * (1. / np.sqrt(eigen_n))) @ basis_n.T
    contrast = invsqrt @ s @ invsqrt
    contrast = (contrast + contrast.T) / 2
    eigen_s, basis_s = np.linalg.eigh(contrast)
    indices = np.argsort(eigen_s)[::-1][:min(rank, d)]
    values = np.maximum(eigen_s[indices], 0)
    generalized = invsqrt @ basis_s[:, indices]
    # QR is just a bounded Euclidean editing basis, not the original
    # generalized eigenbasis. Returned values are diagnostic scores only.
    directions, _ = np.linalg.qr(generalized, mode='reduced')
    gains = max_gain * np.maximum(values - 1., 0.) / (1. + values)
    metric = np.eye(d) + (directions * gains) @ directions.T
    return {"metric": (metric + metric.T) / 2, "directions": directions,
            "diagnostic_eigenvalues": values, "gains": gains}


def squared_distance(first: np.ndarray, second: np.ndarray, metric: np.ndarray) -> np.ndarray:
    x, y, m = np.asarray(first, float), np.asarray(second, float), np.asarray(metric, float)
    if x.shape != y.shape or x.ndim not in (1, 2) or m.shape != (x.shape[-1], x.shape[-1]):
        raise ValueError("Inputs must have equal [d] or [batch,d] shapes and metric [d,d]")
    if not all(np.all(np.isfinite(a)) for a in (x, y, m)):
        raise ValueError("All arrays must be finite")
    if not np.allclose(m, m.T, atol=1e-9, rtol=1e-9) or np.linalg.eigvalsh(m).min() < -1e-9:
        raise ValueError("metric must be symmetric positive semidefinite")
    diff = x - y
    return np.einsum('...i,ij,...j->...', diff, m, diff)
