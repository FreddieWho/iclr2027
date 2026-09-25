"""Feature schemas for the M1.2 source representation matrix.

Six representations are compared on one frozen recipe and one frozen bank:

  raw                      eight standardized coordinates (strong baseline)
  rich8_unsorted           the eight continuous quantities, fixed endpoint order
  rich8_sorted             the same quantities with the existing G8 bag sort
  orbit_distance           whole-orbit lexicographic distance representative
  segment_moment           centered segment moments with a symmetric nonlinear q
  repaired_segment_rho     segment-preserving head (common.source_typed)

Only ``orbit_distance`` and ``segment_moment`` are role-preserving references.
``rich8_sorted`` keeps the legacy independent-bag sort that the M1.0 T1 witness
shows can merge two opposite-label states; it is retained as the comparison
arm, not as a candidate repair. ``rich8_unsorted`` keeps endpoint order and is
therefore role-sensitive but not permutation-corrected.

No function here reads labels, oracles, or sealed pools. Statistics are fit on
the training scenes only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _v3common import geom_features  # noqa: E402

SOURCE_PERMS = geom_features.SOURCE_PERMS
source_regroup_witness = geom_features.source_regroup_witness

ARMS = (
    "raw",
    "rich8_unsorted",
    "rich8_sorted",
    "orbit_distance",
    "segment_moment",
    "repaired_segment_rho",
)

# Shared-xy arms consume standardized coordinates; feature arms consume a
# fitted per-column standardization of their own descriptor.
COORDINATE_ARMS = ("raw", "repaired_segment_rho")

FEATURE_COLUMNS = {
    "raw": ("x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"),
    "rich8_unsorted": (
        "length_ab",
        "length_cd",
        "cross_ac",
        "cross_ad",
        "cross_bc",
        "cross_bd",
        "midpoint_distance",
        "direction_sine",
    ),
    "rich8_sorted": (
        "within_sorted_1",
        "within_sorted_2",
        "cross_sorted_1",
        "cross_sorted_2",
        "cross_sorted_3",
        "cross_sorted_4",
        "midpoint_distance",
        "direction_sine",
    ),
    "orbit_distance": ("d01", "d02", "d03", "d12", "d13", "d23"),
    "segment_moment": (
        "mid0_x",
        "mid0_y",
        "q0_xx",
        "q0_xy",
        "q0_yy",
        "mid1_x",
        "mid1_y",
        "q1_xx",
        "q1_xy",
        "q1_yy",
    ),
    "repaired_segment_rho": ("x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"),
}

EPS = 1e-8


def _scenes(x) -> np.ndarray:
    scenes = np.asarray(x, dtype=np.float64).reshape(-1, 4, 2)
    if not np.all(np.isfinite(scenes)):
        raise ValueError("expected finite scene coordinates")
    return scenes


def rich8_quantities(x) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The eight legacy quantities before any sort: lengths, crosses, mid, sine."""
    s = _scenes(x)
    length_ab = np.linalg.norm(s[:, 0] - s[:, 1], axis=-1)
    length_cd = np.linalg.norm(s[:, 2] - s[:, 3], axis=-1)
    cross = np.stack(
        [
            np.linalg.norm(s[:, 0] - s[:, 2], axis=-1),
            np.linalg.norm(s[:, 0] - s[:, 3], axis=-1),
            np.linalg.norm(s[:, 1] - s[:, 2], axis=-1),
            np.linalg.norm(s[:, 1] - s[:, 3], axis=-1),
        ],
        axis=1,
    )
    midpoint = np.linalg.norm(
        0.5 * (s[:, 0] + s[:, 1]) - 0.5 * (s[:, 2] + s[:, 3]), axis=-1
    )
    u0 = s[:, 1] - s[:, 0]
    u1 = s[:, 3] - s[:, 2]
    sine = np.abs(u0[:, 0] * u1[:, 1] - u0[:, 1] * u1[:, 0]) / (
        np.linalg.norm(u0, axis=-1) * np.linalg.norm(u1, axis=-1) + EPS
    )
    return length_ab, length_cd, cross, midpoint, sine


def rich8_unsorted_features(x) -> np.ndarray:
    length_ab, length_cd, cross, midpoint, sine = rich8_quantities(x)
    return np.concatenate(
        [length_ab[:, None], length_cd[:, None], cross, midpoint[:, None], sine[:, None]],
        axis=1,
    )


def rich8_sorted_features(x) -> np.ndarray:
    """The existing G8 sort: sort the length pair and the four crosses separately.

    This is exactly the legacy operation that the M1.0 T1 witness can collapse.
    It is a controlled comparison, not a role-preserving reference.
    """
    length_ab, length_cd, cross, midpoint, sine = rich8_quantities(x)
    within = np.sort(np.stack([length_ab, length_cd], axis=1), axis=1)
    return np.concatenate(
        [within, np.sort(cross, axis=1), midpoint[:, None], sine[:, None]], axis=1
    )


def batch_orbit_representative(x, perms=SOURCE_PERMS) -> np.ndarray:
    """Whole-orbit lexicographic representative of the six point distances.

    One legal permutation acts on every distance at once; the six distances of
    the representative therefore keep point roles up to the task group. This is
    the same construction as ``common.geom_features.orbit_representative``,
    batched, and it is non-smooth at ties (the representative switches branch).
    """
    scenes = _scenes(x)
    n = len(scenes)
    if not perms:
        raise ValueError("need at least one legal permutation")
    for perm in perms:
        if tuple(sorted(perm)) != (0, 1, 2, 3):
            raise ValueError(f"invalid source permutation: {perm}")
    rows, cols = np.triu_indices(4, 1)
    candidates = np.empty((n, len(perms), len(rows)), dtype=np.float64)
    for k, perm in enumerate(perms):
        moved = scenes[:, list(perm), :]
        candidates[:, k, :] = np.linalg.norm(moved[:, rows, :] - moved[:, cols, :], axis=-1)
    flat = candidates.reshape(n * len(perms), len(rows))
    keys = tuple(flat[:, col] for col in range(flat.shape[1] - 1, -1, -1))
    order = np.lexsort(keys)
    position = np.empty(n * len(perms), dtype=np.int64)
    position[order] = np.arange(n * len(perms))
    best = position.reshape(n, len(perms)).argmin(axis=1)
    return candidates[np.arange(n), best]


def orbit_tie_fraction(x, perms=SOURCE_PERMS, atol: float = 1e-9) -> float:
    """Fraction of rows where the lexicographic minimum is not unique.

    The representative is discontinuous at these rows; a nonzero rate is a
    known property of canonical representatives, not a bug, but the rate has to
    be reported before the arm's numbers are read.
    """
    scenes = _scenes(x)
    rows, cols = np.triu_indices(4, 1)
    tied = 0
    for k in range(len(scenes)):
        cand = np.array(
            [
                np.linalg.norm(
                    scenes[k, list(perm), :][rows] - scenes[k, list(perm), :][cols], axis=-1
                )
                for perm in perms
            ]
        )
        best = cand.min(axis=0)
        near = np.abs(cand - best).max(axis=1) <= atol
        if int(near.sum()) > 1:
            tied += 1
    return float(tied / len(scenes))


def segment_moment_descriptors(x, scale: float) -> np.ndarray:
    """Centered, scale-normalized segment moments (midpoint and direction outer product).

    A segment is described by m=(A+B)/2 and Q=(A-B)(A-B)^T. Q is symmetric and
    sign-free, so the descriptor is invariant to endpoint swap by construction.
    Subtracting the scene centroid removes the only translation the label does
    not use; the global scale is a fixed monotone rescaling fitted on train.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    s = _scenes(x)
    centroid = s.mean(axis=1, keepdims=True)
    centered = (s - centroid) / scale
    blocks = []
    for q, r in ((0, 1), (2, 3)):
        midpoint = 0.5 * (centered[:, q] + centered[:, r])
        u = centered[:, q] - centered[:, r]
        outer = np.stack([u[:, 0] * u[:, 0], u[:, 0] * u[:, 1], u[:, 1] * u[:, 1]], axis=1)
        blocks.append(np.concatenate([midpoint, outer], axis=1))
    return np.concatenate(blocks, axis=1)


def raw_coordinate_features(x, mu, sd) -> np.ndarray:
    s = _scenes(x)
    return ((s - np.asarray(mu).reshape(1, 1, 2)) / np.asarray(sd).reshape(1, 1, 2)).reshape(
        len(s), 8
    )


def fit_stats(arm: str, train_positions) -> dict:
    """Fit preprocessing on training scenes only; no test labels are touched."""
    if arm not in ARMS:
        raise KeyError(arm)
    if arm in COORDINATE_ARMS:
        flat = _scenes(train_positions).reshape(-1, 2)
        return {
            "kind": "shared_xy",
            "mu": flat.mean(axis=0),
            "sd": flat.std(axis=0) + EPS,
        }
    if arm == "segment_moment":
        scenes = _scenes(train_positions)
        scale = float(scenes.reshape(-1, 2).std())
        descriptors = segment_moment_descriptors(scenes, scale)
        # The two descriptor blocks are exchanged by a legal group element, so
        # they must share one 5-vector of statistics. Independent per-block
        # statistics would break the group action between the descriptor and
        # the standardization, exactly as per-point statistics would break the
        # coordinate arms. Pooling is fitted on training scenes only.
        pooled = descriptors.reshape(len(descriptors), 2, -1).reshape(-1, 5)
        mu5 = pooled.mean(axis=0)
        sd5 = pooled.std(axis=0) + EPS
        return {
            "kind": "train_feature_block_shared",
            "scale": scale,
            "mu": np.tile(mu5, 2),
            "sd": np.tile(sd5, 2),
        }
    raw = describe(arm, train_positions)
    return {"kind": "train_feature", "mu": raw.mean(axis=0), "sd": raw.std(axis=0) + EPS}


def describe(arm: str, x) -> np.ndarray:
    """Raw (unstandardized) descriptor for one arm, before statistics."""
    if arm == "rich8_unsorted":
        return rich8_unsorted_features(x)
    if arm == "rich8_sorted":
        return rich8_sorted_features(x)
    if arm == "orbit_distance":
        return batch_orbit_representative(x)
    raise KeyError(f"{arm} has no standalone descriptor; pass fitted stats")


def featurize(arm: str, x, stats: dict) -> np.ndarray:
    """Standardize scenes with previously fitted train statistics."""
    if arm not in ARMS:
        raise KeyError(arm)
    if arm in COORDINATE_ARMS:
        if stats["kind"] != "shared_xy":
            raise ValueError(f"{arm} expects shared_xy statistics")
        return raw_coordinate_features(x, stats["mu"], stats["sd"]).astype(np.float32)
    if arm == "segment_moment":
        if stats["kind"] != "train_feature_block_shared" or "scale" not in stats:
            raise ValueError(
                "segment_moment expects block-shared train statistics with scale"
            )
        raw = segment_moment_descriptors(x, stats["scale"])
    else:
        raw = describe(arm, x)
    return ((raw - stats["mu"]) / stats["sd"]).astype(np.float32)


def in_dim(arm: str) -> int:
    if arm not in ARMS:
        raise KeyError(arm)
    return len(FEATURE_COLUMNS[arm])


def feature_schema(arm: str) -> dict:
    """Auditable feature contract: names, dimension, and construction kind."""
    if arm not in ARMS:
        raise KeyError(arm)
    kind = {
        "raw": "standardized_coordinates",
        "rich8_unsorted": "legacy_quantities_fixed_order",
        "rich8_sorted": "legacy_quantities_g8_bag_sort",
        "orbit_distance": "whole_orbit_lexicographic_distance_representative",
        "segment_moment": "centered_midpoint_and_direction_outer_product",
        "repaired_segment_rho": "standardized_coordinates_segment_preserving_head",
    }[arm]
    return {
        "arm": arm,
        "columns": list(FEATURE_COLUMNS[arm]),
        "in_dim": in_dim(arm),
        "construction": kind,
        "role_preserving_reference": arm in ("orbit_distance", "segment_moment"),
        "statistics": stats_contract(arm),
    }


def stats_contract(arm: str) -> str:
    if arm in COORDINATE_ARMS:
        return "train shared xy mean/std"
    if arm == "segment_moment":
        return (
            "train global coordinate scale, then train statistics pooled across the two "
            "exchangeable segment blocks and tiled onto each block"
        )
    return "train per-column mean/std"
