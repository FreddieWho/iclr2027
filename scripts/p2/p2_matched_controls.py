#!/usr/bin/env python3
"""Pure P2 matching primitives.

The module is deliberately independent of the P1 model code.  P2-A/B only
constructs interventions and checks their comparability; model inference is a
separate later step.  Every invalid construction is represented by an
explicit status in the caller rather than silently replaced.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import hashlib
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import Delaunay, QhullError


class GraphInvalid(RuntimeError):
    """The requested graph cannot be constructed without a fallback."""


@dataclass(frozen=True)
class TopologyFeatures:
    component_count: int
    induced_density: float
    cut_weight: float
    rq_normalized: float
    band_power: tuple[float, ...]


@dataclass(frozen=True)
class SpectralRetryResult:
    success: bool
    delta: np.ndarray | None
    signs: np.ndarray | None
    attempt_count: int
    boundary_slacks: tuple[float, ...]


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def candidate_id(
    sample_id: str,
    graph_id: str,
    coalition_id: str,
    anchor_support: Iterable[int],
    candidate_support: Iterable[int],
) -> str:
    """Stable candidate identifier independent of input ordering."""
    anchor = ",".join(str(int(value)) for value in sorted(set(anchor_support)))
    candidate = ",".join(str(int(value)) for value in sorted(set(candidate_support)))
    payload = f"{sample_id}|{graph_id}|{coalition_id}|{anchor}|{candidate}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def frequency_ratio(
    rq_difference: float,
    band_l1_difference: float,
    rq_reference: float,
    band_reference: float,
) -> float:
    """Response-blind distance relative to reference matching scales."""
    if rq_reference <= 0 or band_reference <= 0:
        raise ValueError("frequency reference scales must be positive")
    if rq_difference < 0 or band_l1_difference < 0:
        raise ValueError("frequency differences must be non-negative")
    return max(float(rq_difference) / float(rq_reference), float(band_l1_difference) / float(band_reference))


def normalize_direction(direction: Sequence[float]) -> np.ndarray:
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError("direction must be a finite 2-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("direction cannot be zero")
    return vector / norm


def equal_energy(delta: np.ndarray, energy: float) -> np.ndarray:
    """Scale a vector-valued intervention to the requested L2 energy."""
    array = np.asarray(delta, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
        raise ValueError("delta must be a finite (n_nodes, 2) array")
    requested = float(energy)
    if not math.isfinite(requested) or requested < 0:
        raise ValueError("energy must be finite and non-negative")
    norm = float(np.linalg.norm(array))
    if requested == 0:
        return np.zeros_like(array)
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero intervention")
    return array * (requested / norm)


def rigid_subset_translation(
    n_nodes: int,
    support: Iterable[int],
    energy: float,
    direction: Sequence[float],
) -> np.ndarray:
    """Translate every selected node by the same vector, with total energy fixed."""
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    indices = np.asarray(sorted(set(int(index) for index in support)), dtype=int)
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= n_nodes):
        raise ValueError("support must contain valid, non-empty node indices")
    vector = normalize_direction(direction)
    delta = np.zeros((n_nodes, 2), dtype=np.float64)
    delta[indices] = vector[None, :] * (float(energy) / math.sqrt(float(indices.size)))
    return delta


def boundary_valid(positions: np.ndarray, delta: np.ndarray, bound: float = 1.0, tolerance: float = 1e-9) -> bool:
    positions = np.asarray(positions, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    return bool(
        positions.shape == delta.shape
        and np.all(np.isfinite(positions))
        and np.all(np.isfinite(delta))
        and np.all(np.abs(positions + delta) <= float(bound) + float(tolerance))
    )


def boundary_slack(positions: np.ndarray, delta: np.ndarray, bound: float = 1.0) -> float:
    """Minimum remaining coordinate margin; negative means out of bounds."""
    positions = np.asarray(positions, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if positions.shape != delta.shape or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions and delta must have equal (n_nodes, 2) shape")
    moved = positions + delta
    if not np.all(np.isfinite(moved)):
        raise ValueError("moved positions must be finite")
    return float(np.min(float(bound) - np.abs(moved)))


def spectral_coefficients(eigenvectors: np.ndarray, delta: np.ndarray) -> np.ndarray:
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    perturbation = np.asarray(delta, dtype=np.float64)
    if vectors.ndim != 2 or perturbation.ndim != 2 or vectors.shape[0] != perturbation.shape[0]:
        raise ValueError("eigenvectors and delta have incompatible shapes")
    return vectors.T @ perturbation


def mode_power(eigenvectors: np.ndarray, delta: np.ndarray) -> np.ndarray:
    coefficients = spectral_coefficients(eigenvectors, delta)
    return np.sum(coefficients * coefficients, axis=1)


def band_ranges(n_modes: int, n_bands: int) -> list[np.ndarray]:
    if n_modes <= 0 or n_bands <= 0:
        raise ValueError("n_modes and n_bands must be positive")
    edges = np.linspace(0, n_modes, n_bands + 1, dtype=int)
    ranges: list[np.ndarray] = []
    for band in range(n_bands):
        start, end = int(edges[band]), int(edges[band + 1])
        if end <= start:
            end = min(n_modes, start + 1)
        ranges.append(np.arange(start, end, dtype=int))
    return ranges


def normalized_band_power(power: np.ndarray, n_bands: int) -> np.ndarray:
    values = np.asarray(power, dtype=np.float64)
    total = float(np.sum(values))
    if values.ndim != 1 or not np.all(np.isfinite(values)) or total <= 1e-15:
        raise ValueError("mode power must be finite and non-zero")
    result = np.array([float(np.sum(values[indexes])) for indexes in band_ranges(len(values), n_bands)])
    return result / total


def normalized_rayleigh(eigenvalues: np.ndarray, power: np.ndarray) -> float:
    values = np.asarray(eigenvalues, dtype=np.float64)
    weights = np.asarray(power, dtype=np.float64)
    denominator = float(np.sum(weights))
    if values.ndim != 1 or values.shape != weights.shape or denominator <= 1e-15:
        raise ValueError("eigenvalues and power must have equal non-zero shape")
    raw = float(np.sum(values * weights) / denominator)
    scale = max(float(np.max(np.abs(values))), 1e-12)
    return raw / scale


def randomized_sign_spectral_delta(
    eigenvectors: np.ndarray,
    delta: np.ndarray,
    rng: np.random.Generator,
    preserve_dc: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomize modal signs while preserving every mode's vector power."""
    coefficients = spectral_coefficients(eigenvectors, delta)
    signs = rng.choice(np.array([-1.0, 1.0]), size=coefficients.shape[0])
    if preserve_dc and len(signs):
        signs[0] = 1.0
    randomized = np.asarray(eigenvectors, dtype=np.float64) @ (coefficients * signs[:, None])
    return randomized, signs.astype(np.int8)


def randomized_sign_spectral_delta_with_retries(
    eigenvectors: np.ndarray,
    delta: np.ndarray,
    positions: np.ndarray,
    seed: int,
    max_attempts: int = 32,
    preserve_dc: bool = True,
) -> SpectralRetryResult:
    """Try deterministic sign draws until an exact-spectrum control is feasible."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    rng = np.random.default_rng(int(seed))
    slacks: list[float] = []
    for attempt in range(1, int(max_attempts) + 1):
        candidate, signs = randomized_sign_spectral_delta(eigenvectors, delta, rng, preserve_dc=preserve_dc)
        slack = boundary_slack(positions, candidate)
        slacks.append(slack)
        if slack >= -1e-9:
            return SpectralRetryResult(
                success=True,
                delta=candidate,
                signs=signs,
                attempt_count=attempt,
                boundary_slacks=tuple(slacks),
            )
    return SpectralRetryResult(
        success=False,
        delta=None,
        signs=None,
        attempt_count=int(max_attempts),
        boundary_slacks=tuple(slacks),
    )


def induced_component_count(support: Iterable[int], adjacency: np.ndarray) -> int:
    indices = tuple(sorted(set(int(index) for index in support)))
    if not indices:
        return 0
    graph = np.asarray(adjacency, dtype=np.float64)[np.ix_(indices, indices)] > 0
    seen: set[int] = set()
    count = 0
    for start in range(len(indices)):
        if start in seen:
            continue
        count += 1
        stack = [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            for neighbor in np.flatnonzero(graph[node]):
                neighbor = int(neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return count


def topology_features(
    support: Iterable[int],
    adjacency: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    delta: np.ndarray,
    n_bands: int,
) -> TopologyFeatures:
    indices = tuple(sorted(set(int(index) for index in support)))
    matrix = np.asarray(adjacency, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be square")
    support_mask = np.zeros(matrix.shape[0], dtype=bool)
    support_mask[list(indices)] = True
    induced = matrix[np.ix_(support_mask, support_mask)]
    n = len(indices)
    possible = n * (n - 1) / 2
    edge_count = float(np.count_nonzero(np.triu(induced > 0, 1)))
    density = edge_count / possible if possible else 0.0
    cut = float(matrix[support_mask][:, ~support_mask].sum())
    power = mode_power(eigenvectors, delta)
    return TopologyFeatures(
        component_count=induced_component_count(indices, matrix),
        induced_density=float(density),
        cut_weight=cut,
        rq_normalized=normalized_rayleigh(eigenvalues, power),
        band_power=tuple(float(x) for x in normalized_band_power(power, n_bands)),
    )


def enumerate_same_team_supports(
    team_slots: Sequence[int],
    team: int,
    size: int,
    exclude: Iterable[int] | None = None,
    max_overlap: int | None = None,
) -> list[np.ndarray]:
    """Enumerate all same-team supports in stable lexicographic order."""
    slots = np.asarray(team_slots)
    eligible = [int(index) for index in np.flatnonzero(slots == int(team))]
    excluded = set() if exclude is None else {int(index) for index in exclude}
    if size < 1 or size > len(eligible):
        return []
    supports: list[np.ndarray] = []
    for values in combinations(eligible, int(size)):
        candidate = set(values)
        if candidate == excluded:
            continue
        if max_overlap is not None and len(candidate & excluded) > int(max_overlap):
            continue
        supports.append(np.asarray(values, dtype=int))
    return supports


def knn_adjacency(position: np.ndarray, k: int = 4) -> np.ndarray:
    """Build the P1 weighted symmetric kNN graph without importing P1 code."""
    points = np.asarray(position, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise GraphInvalid("positions must be a non-empty 2D point set")
    if k < 1 or k >= len(points):
        raise GraphInvalid("k must be between 1 and n_nodes-1")
    diff = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(diff * diff, axis=2))
    nonzero = distance[distance > 0]
    if not len(nonzero):
        raise GraphInvalid("all positions are identical")
    scale = max(float(np.median(nonzero)), 1e-12)
    adjacency = np.zeros((len(points), len(points)), dtype=np.float64)
    for index in range(len(points)):
        for neighbor in np.argsort(distance[index])[1 : k + 1]:
            weight = math.exp(-float(distance[index, neighbor] ** 2) / (scale**2))
            adjacency[index, neighbor] = max(adjacency[index, neighbor], weight)
            adjacency[neighbor, index] = max(adjacency[neighbor, index], weight)
    return adjacency


def delaunay_adjacency(position: np.ndarray) -> np.ndarray:
    """Build a weighted Delaunay graph; degeneracy is explicit, with no fallback."""
    points = np.asarray(position, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise GraphInvalid("Delaunay requires at least three 2D points")
    if len(np.unique(points, axis=0)) != len(points):
        raise GraphInvalid("duplicate positions make Delaunay ambiguous")
    try:
        triangulation = Delaunay(points)
    except QhullError as exc:
        raise GraphInvalid(f"Delaunay construction failed: {exc}") from exc
    edge_pairs: set[tuple[int, int]] = set()
    for simplex in triangulation.simplices:
        for left, right in combinations(sorted(int(x) for x in simplex), 2):
            edge_pairs.add((left, right))
    if not edge_pairs:
        raise GraphInvalid("Delaunay produced no edges")
    diff = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(diff * diff, axis=2))
    nonzero = distance[distance > 0]
    scale = max(float(np.median(nonzero)), 1e-12)
    adjacency = np.zeros((len(points), len(points)), dtype=np.float64)
    for left, right in sorted(edge_pairs):
        weight = math.exp(-float(distance[left, right] ** 2) / (scale**2))
        adjacency[left, right] = weight
        adjacency[right, left] = weight
    return adjacency


def normalized_laplacian(adjacency: np.ndarray) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.allclose(matrix, matrix.T):
        raise GraphInvalid("adjacency must be square and symmetric")
    degree = matrix.sum(axis=1)
    if np.any(degree <= 0):
        raise GraphInvalid("graph contains an isolated node")
    inv_sqrt = 1.0 / np.sqrt(degree)
    return np.eye(len(matrix)) - inv_sqrt[:, None] * matrix * inv_sqrt[None, :]


def topology_match_score(
    anchor: TopologyFeatures,
    candidate: TopologyFeatures,
    density_scale: float = 0.10,
    cut_scale: float = 1.0,
) -> float:
    band_distance = float(np.abs(np.asarray(anchor.band_power) - np.asarray(candidate.band_power)).sum())
    return (
        abs(anchor.rq_normalized - candidate.rq_normalized)
        + band_distance
        + abs(anchor.induced_density - candidate.induced_density) / max(float(density_scale), 1e-12)
        + abs(anchor.cut_weight - candidate.cut_weight) / max(float(cut_scale), 1e-12)
    )
