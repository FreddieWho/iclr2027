"""Role-preserving geometric features and legal-permutation references.

Fixed functions use only legal regroupings: source uses the eight G8 pair
permutations, T1 permutes triangle vertices while Q stays fixed, and T2 swaps
only the two endpoints. Legacy wrappers preserve defective behavior for
controlled comparisons and must not be called "same information reordered".
"""
from __future__ import annotations

import itertools

import numpy as np

T1_PERMS = [tuple(p) + (3,) for p in itertools.permutations(range(3))]
SOURCE_PERMS = [
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
]
T2_PERMS = [(0, 1, 2), (1, 0, 2)]
T1_LEGACY_COLUMNS = (
    "edge01",
    "edge02",
    "edge12",
    "query_distance0",
    "query_distance1",
    "query_distance2",
    "centroid_distance",
    "legacy_area_ratio",
)
T2_ROLE_COLUMNS = ("segment_length", "query_distance_min", "query_distance_max")


def _check_points(points: np.ndarray, n_points: int) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64).reshape(-1, n_points, 2)
    if not np.all(np.isfinite(x)):
        raise ValueError("expected finite coordinates")
    return x


def _check_perms(perms: list[tuple[int, ...]], n_points: int) -> None:
    for perm in perms:
        if tuple(sorted(perm)) != tuple(range(n_points)):
            raise ValueError(f"invalid {n_points}-point permutation: {perm}")


def orbit_representative(points: np.ndarray, perms: list[tuple[int, ...]]) -> np.ndarray:
    """Lexicographic representative over one legal orbit; diagnostic, not smooth."""
    x = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if x.ndim != 2 or x.shape[1] != 2 or not np.all(np.isfinite(x)):
        raise ValueError("expected finite [n_points,2]")
    _check_perms(perms, len(x))
    rows, cols = np.triu_indices(len(x), 1)
    candidates = []
    for perm in perms:
        moved = x[list(perm)]
        candidates.append(np.linalg.norm(moved[rows] - moved[cols], axis=1))
    return min(candidates, key=lambda row: tuple(row.tolist())).copy()


def t1_orbit_representative(points: np.ndarray) -> np.ndarray:
    return orbit_representative(_check_points(points, 4).reshape(4, 2), T1_PERMS)


def source_orbit_representative(points: np.ndarray) -> np.ndarray:
    return orbit_representative(_check_points(points, 4).reshape(4, 2), SOURCE_PERMS)


def t2_role_distances(points: np.ndarray, radius=None) -> np.ndarray:
    """Fixed T2 features: segment length plus sorted center-endpoint distances."""
    p = _check_points(points, 3)
    center_to_ends = np.linalg.norm(p[:, 2, None, :] - p[:, :2, :], axis=-1)
    features = np.c_[
        np.linalg.norm(p[:, 0] - p[:, 1], axis=-1),
        np.sort(center_to_ends, axis=1),
    ]
    if radius is None:
        return features
    r = np.broadcast_to(np.asarray(radius, dtype=np.float64), (len(p),))
    if not np.all(np.isfinite(r)):
        raise ValueError("expected finite radius")
    return np.c_[features, r]


def legacy_t1_features(points: np.ndarray, sorted_features: bool = True) -> np.ndarray:
    """Exact legacy T1 formula, including the independent bag sorts."""
    p = _check_points(points, 4)
    v, q = p[:, :3], p[:, 3]
    edges = np.stack(
        [np.linalg.norm(v[:, i] - v[:, j], axis=1) for i, j in ((0, 1), (0, 2), (1, 2))],
        axis=1,
    )
    query = np.linalg.norm(q[:, None] - v, axis=2)
    a, b = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    area_ratio = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / (
        np.prod(edges, axis=1) + 1e-8
    )
    if sorted_features:
        edges = np.sort(edges, axis=1)
        query = np.sort(query, axis=1)
    return np.c_[edges, query, np.linalg.norm(q - v.mean(1), axis=1), area_ratio]


def legacy_t2_features(points: np.ndarray, sorted_features: bool = False) -> np.ndarray:
    """Exact legacy T2 formula, including the endpoint-axis norm defect."""
    p = _check_points(points, 3)
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    wrong_axis = np.linalg.norm(c[:, None] - p[:, :2], axis=1)
    features = np.c_[
        np.linalg.norm(a - b, axis=1),
        wrong_axis,
        np.linalg.norm((a + b) / 2 - c, axis=1),
        wrong_axis.min(1),
        wrong_axis.prod(1),
    ]
    if sorted_features:
        return np.c_[np.sort(features[:, :2], axis=1), features[:, 2:]]
    return features


def t1_collision_pair() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array([[-0.75, 0.0], [0.75, 0.0], [-0.375, 0.75]])
    return np.vstack([vertices, [0.1875, 0.09375]]), np.vstack(
        [vertices, [-0.1875, -0.09375]]
    )


def source_regroup_witness() -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[-0.6, -0.6], [0.6, 0.6], [-0.6, 0.6], [0.6, -0.6]])
    return x, x[[0, 2, 1, 3]]
