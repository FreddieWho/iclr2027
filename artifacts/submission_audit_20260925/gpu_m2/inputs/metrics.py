"""Joint-state metrics with parent-cluster uncertainty and table schema.

J3 means A, B, and AB are all correct. J4 additionally requires P. A missing P
state yields J4=None; it is never silently copied from J3. Parents are the
resampling unit; both quartet-weighted and equal-parent estimands are explicit.
Seeds are training replicates, not new data.
"""
from __future__ import annotations

import numpy as np

TABLE_REQUIRED_FIELDS = (
    "n_quartets",
    "n_eligible_parents",
    "n_sampled_parents",
    "row_mean",
    "parent_mean",
)
UNIT = "test parent (equal weight)"


def _check_state_arrays(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(logits)
    y = np.asarray(labels)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] not in (3, 4):
        raise ValueError("expected matching [n,4] P,A,B,AB or [n,3] A,B,AB arrays")
    if not np.all(np.isfinite(x)):
        raise ValueError("expected finite logits")
    if not set(np.unique(y)).issubset({0, 1, 0.0, 1.0}):
        raise ValueError("expected binary labels")
    return x, y


def joint_metrics(logits: np.ndarray, labels: np.ndarray, p_available: bool = True) -> dict:
    """Correct J3/J4 contract; P-missing evaluations return J4=None."""
    x, y = _check_state_arrays(logits, labels)
    correct = (x > 0) == (y > 0.5)
    if x.shape[1] == 4:
        p, a, b, ab = (correct[:, 0], correct[:, 1], correct[:, 2], correct[:, 3])
        has_p = bool(p_available)
    else:
        p = None
        a, b, ab = (correct[:, 0], correct[:, 1], correct[:, 2])
        has_p = False
    joint3 = a & b & ab
    return {
        "n_quartets": int(len(y)),
        "P": None if p is None or not has_p else float(p.mean()),
        "A": float(a.mean()),
        "B": float(b.mean()),
        "AB": float(ab.mean()),
        "atomic_joint": float((a & b).mean()),
        "J3": float(joint3.mean()),
        "J4": None if p is None or not has_p else float((p & joint3).mean()),
        "p_available": has_p,
    }


def parent_cluster_ci(
    values, parents, seed: int, n_boot: int = 2000
) -> dict:
    """Parent-cluster bootstrap CI for a quartet-level binary or score vector."""
    v = np.asarray(values, dtype=float).ravel()
    p = np.asarray(parents).ravel()
    if len(v) != len(p):
        raise ValueError("values and parents must have the same length")
    if n_boot < 100:
        raise ValueError("bootstrap needs at least 100 draws")
    unique = np.unique(p)
    if len(unique) < 2:
        raise ValueError("need at least two parents for a cluster CI")
    rng = np.random.default_rng(int(seed))
    group_means = np.asarray([v[p == parent].mean() for parent in unique])
    draws = [
        float(group_means[rng.integers(0, len(unique), len(unique))].mean())
        for _ in range(int(n_boot))
    ]
    return {
        "estimate": float(group_means.mean()),
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "n": int(len(v)),
        "parents": int(len(unique)),
        "unit": UNIT,
    }


def row_weighted_cluster_ci(values, parents, seed: int, n_boot: int = 2000) -> dict:
    """Cluster bootstrap for a quartet-weighted mean, resampling whole parents."""
    v = np.asarray(values, dtype=float).ravel()
    p = np.asarray(parents).ravel()
    if len(v) != len(p):
        raise ValueError("values and parents must have the same length")
    if n_boot < 100:
        raise ValueError("bootstrap needs at least 100 draws")
    unique = np.unique(p)
    if len(unique) < 2:
        raise ValueError("need at least two parents for a cluster CI")
    groups = [v[p == parent] for parent in unique]
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(n_boot)):
        sampled = rng.integers(0, len(unique), len(unique))
        rows = np.concatenate([groups[index] for index in sampled])
        draws.append(float(rows.mean()))
    return {
        "estimate": float(v.mean()),
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "n": int(len(v)),
        "parents": int(len(unique)),
        "unit": "test parent cluster bootstrap; quartet-weighted estimand",
    }


def validate_table_record(record: dict) -> dict:
    """Require both quartet/parent denominators and both row/parent means."""
    missing = [field for field in TABLE_REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"table record is missing {missing}")
    checked = {field: record[field] for field in TABLE_REQUIRED_FIELDS}
    for field in ("n_quartets", "n_eligible_parents", "n_sampled_parents"):
        if not isinstance(checked[field], int) or checked[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    for field in ("row_mean", "parent_mean"):
        value = float(checked[field])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be in [0,1]")
    return checked
