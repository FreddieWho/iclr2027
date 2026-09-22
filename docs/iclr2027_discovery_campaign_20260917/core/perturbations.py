"""Perturbation-distribution kernels for the discovery campaign.

Each row of a pattern bank represents an antithetic pair (+s, -s), NOT a
single attack. Any mixture over such pairs retains identical one-node
Rademacher marginals. Geometry/label validity must be checked for both arms
by a task adapter before selecting weights; these kernels do not certify it.
"""
from __future__ import annotations

from itertools import combinations, product
from math import comb
from typing import Sequence
import numpy as np

Array = np.ndarray


def _finite(a: Array, name: str) -> Array:
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def validate_patterns(patterns: Array) -> Array:
    s = _finite(patterns, "patterns")
    if s.ndim != 2 or min(s.shape) == 0 or not np.all(np.isin(s, [-1., 1.])):
        raise ValueError("patterns must be a nonempty [modes, active_nodes] +/-1 array")
    return s


def balanced_sign_patterns(group_sizes: Sequence[int], max_patterns: int = 64,
                           seed: int = 0) -> Array:
    """Unique +/- equivalence classes, balanced inside every contiguous group.

    Returns at most max_patterns rows; the first active sign is always +1.
    A full reference bank may contain fewer modes than requested. Sampling
    is not an assertion that the nodes are independent: balance couples them.
    """
    groups = tuple(group_sizes)
    if not groups or any(not isinstance(g, (int, np.integer)) or g <= 0 or g % 2 for g in groups):
        raise ValueError("group_sizes must contain positive even integers")
    if not isinstance(max_patterns, (int, np.integer)) or max_patterns <= 0:
        raise ValueError("max_patterns must be a positive integer")
    total = 1
    for g in groups:
        total *= comb(int(g), int(g)//2)
    target = min(max_patterns, total // 2)
    rng = np.random.default_rng(seed)
    rows: dict[tuple[int, ...], None] = {}

    def canonical(parts: list[Array]) -> tuple[int, ...]:
        s = np.concatenate(parts).astype(np.int8)
        if s[0] < 0:
            s = -s
        return tuple(int(x) for x in s)

    if total <= 200_000 and target > total // 8:
        banks = []
        for g in groups:
            bank = []
            for plus in combinations(range(g), g//2):
                s = -np.ones(g, dtype=np.int8)
                s[list(plus)] = 1
                bank.append(s)
            banks.append(bank)
        for parts in product(*banks):
            rows[canonical(list(parts))] = None
        keys = list(rows)
        choose = rng.permutation(len(keys))[:target]
        return np.asarray([keys[i] for i in choose], dtype=np.float64)

    attempts = 0
    while len(rows) < target and attempts < max(1000, 80 * target):
        parts = []
        for g in groups:
            s = -np.ones(g, dtype=np.int8)
            s[rng.choice(g, size=g//2, replace=False)] = 1
            parts.append(s)
        rows[canonical(parts)] = None
        attempts += 1
    if len(rows) < target:
        raise RuntimeError("Could not sample requested bank; reduce max_patterns")
    return np.asarray(list(rows), dtype=np.float64)


def antithetic_fields(patterns: Array, epsilon: float, direction: Array) -> Array:
    """Return [modes, 2, active_nodes, dimension], each arm norm=epsilon.

    Active nodes have equal amplitude. Use an external fixed support mapping
    to scatter these fields to all nodes. No clipping, centering or rescaling
    should subsequently change one arm independently of another.
    """
    s = validate_patterns(patterns)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    v = _finite(direction, "direction")
    if v.ndim != 1 or v.size == 0 or np.linalg.norm(v) == 0:
        raise ValueError("direction must be a nonzero vector")
    v = v / np.linalg.norm(v)
    one_arm = s[:, :, None] * v[None, None, :] * (epsilon / np.sqrt(s.shape[1]))
    return np.stack([one_arm, -one_arm], axis=1)


def normalize_weights(weights: Array, size: int) -> Array:
    w = _finite(weights, "weights")
    if w.shape != (size,) or np.any(w < 0) or not (w.sum() > 0):
        raise ValueError("weights must have one nonnegative value per mode and positive sum")
    return w / w.sum()


def weighted_sign_law(patterns: Array, weights: Array) -> dict[str, Array]:
    """Exact moments and marginal probabilities of a pair-symmetric law.

    Not an empirical estimate. Actual neural runs must evaluate both arms
    equally or use valid Monte Carlo estimates with their uncertainty.
    """
    s = validate_patterns(patterns)
    w = normalize_weights(weights, len(s))
    plus = 0.5 * ((s == 1).T @ w + ((-s) == 1).T @ w)
    mean = 0.5 * (s.T @ w + (-s).T @ w)
    covariance = s.T @ (w[:, None] * s)
    return {"plus_probability": plus, "mean": mean, "covariance": covariance,
            "weights": w}


def keep_complete_pairs(fields: Array, arm_is_valid: Array) -> tuple[Array, Array]:
    """Keep a mode iff BOTH arms are valid, once for ALL compared laws."""
    fields = _finite(fields, "fields")
    valid = np.asarray(arm_is_valid)
    if fields.ndim != 4 or fields.shape[1] != 2 or valid.shape != fields.shape[:2]:
        raise ValueError("Expected fields [modes,2,n,d] and validity [modes,2]")
    if valid.dtype != np.bool_:
        raise ValueError("arm_is_valid must be boolean")
    keep = valid.all(axis=1)
    return fields[keep], keep


def directional_gram(jacobian: Array, direction: Array) -> Array:
    """J is [latent_dim, nodes, coordinate_dim]; return K[i,j].

    This is a local proposal score, NOT a task-risk certificate or proof of
    learned semantic interaction.
    """
    j = _finite(jacobian, "jacobian")
    v = _finite(direction, "direction")
    if j.ndim != 3 or v.shape != (j.shape[2],) or np.linalg.norm(v) == 0:
        raise ValueError("J shape must be [latent,n,d], direction [d] nonzero")
    v = v / np.linalg.norm(v)
    projected = np.einsum('lnd,d->ln', j, v)
    return projected.T @ projected


def expected_quadratic(gram: Array, covariance: Array, amplitude: float) -> float:
    g, c = _finite(gram, "gram"), _finite(covariance, "covariance")
    if g.ndim != 2 or g.shape[0] != g.shape[1] or c.shape != g.shape:
        raise ValueError("gram and covariance must be same-sized square matrices")
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("amplitude must be finite and nonnegative")
    return float(0.5 * amplitude**2 * np.trace(g @ c))


def spectral_sign_fields(basis: Array, amplitudes: Array, signs: Array,
                         direction: Array) -> Array:
    """Equal graph-spectral power in ONE fixed orthonormal basis.

    Output [modes,nodes,d]. This does NOT preserve per-node amplitudes,
    support, or group centroids. Set DC amplitude to zero if appropriate.
    """
    u, r, s = _finite(basis, "basis"), _finite(amplitudes, "amplitudes"), validate_patterns(signs)
    v = _finite(direction, "direction")
    n = s.shape[1]
    if u.shape != (n, n) or r.shape != (n,) or np.any(r < 0):
        raise ValueError("basis/amplitudes/signs shapes incompatible or negative amplitudes")
    if not np.allclose(u.T @ u, np.eye(n), atol=1e-8, rtol=1e-8):
        raise ValueError("basis must be orthonormal")
    if v.ndim != 1 or len(v) == 0 or np.linalg.norm(v) == 0:
        raise ValueError("direction must be a nonzero vector")
    v = v / np.linalg.norm(v)
    return ((s * r) @ u.T)[:, :, None] * v[None, None, :]


def worst_case_weights(pair_losses: Array, regularization: float = 1.,
                       prior: Array | None = None) -> Array:
    """Solve max_w w.L - regularization*KL(w||prior) over pair weights.

    pair_losses are averages of the two arms. Reweighting only pairs keeps
    each active-node marginal fixed. This is standard entropic DRO.
    """
    loss = _finite(pair_losses, "pair_losses")
    if loss.ndim != 1 or len(loss) == 0 or regularization <= 0 or not np.isfinite(regularization):
        raise ValueError("Need nonempty loss vector and positive regularization")
    p = np.full(len(loss), 1./len(loss)) if prior is None else normalize_weights(prior, len(loss))
    active = p > 0
    log_weight = np.full(len(loss), -np.inf)
    # Subtracting max before dividing avoids overflow for small regularizers.
    log_weight[active] = np.log(p[active]) + (loss[active] - loss[active].max()) / regularization
    w = np.exp(log_weight - np.max(log_weight))
    return w / w.sum()


def greedy_mode_cover(success: Array, k: int) -> tuple[list[int], Array]:
    """Source-only set cover heuristic; columns are independent source scenes.

    Use the selected indices unchanged on target scenes. This function does
    not ensure the matrix was created without target leakage.
    """
    s = np.asarray(success)
    if s.ndim != 2 or min(s.shape) == 0 or s.dtype != np.bool_ or not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("success must be nonempty boolean [modes,scenes], k>=0 integer")
    covered = np.zeros(s.shape[1], dtype=bool)
    chosen: list[int] = []
    for _ in range(min(k, len(s))):
        gains = np.sum(s & ~covered[None, :], axis=1).astype(float)
        gains[chosen] = -np.inf
        index = int(np.argmax(gains))
        if gains[index] <= 0:
            break
        chosen.append(index)
        covered |= s[index]
    return chosen, covered
