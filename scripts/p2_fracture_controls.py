#!/usr/bin/env python3
"""Pure operators for P2-FC-A independent-role fracture controls."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import hashlib
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def stable_seed(*parts: object) -> int:
    """Derive a reproducible uint32 seed without depending on Python hashing."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


derive_seed = stable_seed


def _support(support: Iterable[int], n_nodes: int, name: str) -> tuple[int, ...]:
    values = tuple(int(index) for index in support)
    if len(set(values)) != len(values) or not values:
        raise ValueError(f"{name} must be non-empty and contain unique indices")
    if min(values) < 0 or max(values) >= n_nodes:
        raise ValueError(f"{name} contains an invalid node index")
    return tuple(sorted(values))


def validate_fracture_supports(
    team_slots: Sequence[int], source_support: Iterable[int], target_support: Iterable[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate equal-sized, same-team, disjoint source and target supports."""
    teams = np.asarray(team_slots)
    if teams.ndim != 1:
        raise ValueError("team_slots must be one-dimensional")
    source = _support(source_support, len(teams), "source_support")
    target = _support(target_support, len(teams), "target_support")
    if len(source) != len(target):
        raise ValueError("source and target supports must have equal size")
    if set(source) & set(target):
        raise ValueError("source and target supports must be disjoint")
    if len(set(teams[list(source)])) != 1 or not np.array_equal(teams[list(source)], teams[list(target)]):
        raise ValueError("source and target supports must belong to the same team")
    return source, target


def enumerate_target_supports(
    team_slots: Sequence[int], source_support: Iterable[int], team: int | None = None
) -> list[np.ndarray]:
    """Enumerate every disjoint target support of the source size in lexicographic order."""
    from itertools import combinations

    teams = np.asarray(team_slots)
    source = _support(source_support, len(teams), "source_support")
    selected_team = int(teams[source[0]]) if team is None else int(team)
    eligible = [int(i) for i in np.flatnonzero(teams == selected_team) if int(i) not in source]
    return [np.asarray(values, dtype=int) for values in combinations(eligible, len(source))]


def enumerate_vector_bijections(
    source_support: Iterable[int], target_support: Iterable[int]
) -> list[dict[int, int]]:
    """Return all stable source-node -> target-node bijections."""
    source = tuple(sorted(int(i) for i in source_support))
    target = tuple(sorted(int(i) for i in target_support))
    if len(source) != len(target) or len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError("bijection supports must be unique and equal-sized")
    return [dict(zip(source, perm)) for perm in permutations(target)]


def scale_displacement_field(base_field: np.ndarray, epsilon: float) -> np.ndarray:
    """Scale a unit field by epsilon, retaining float64 precision."""
    field = np.asarray(base_field, dtype=np.float64)
    value = float(epsilon)
    if field.ndim != 2 or field.shape[1] != 2 or not np.all(np.isfinite(field)):
        raise ValueError("base_field must be a finite (n_nodes, 2) array")
    if not math.isfinite(value) or value < 0:
        raise ValueError("epsilon must be finite and non-negative")
    return field * value


def reassigned_vector_field(
    base_field: np.ndarray, source_support: Iterable[int], target_support: Iterable[int],
    bijection: Mapping[int, int], epsilon: float,
) -> np.ndarray:
    """Move the source vector multiset to target endpoints, then apply epsilon."""
    field = np.asarray(base_field, dtype=np.float64)
    source = tuple(sorted(int(i) for i in source_support))
    target = tuple(sorted(int(i) for i in target_support))
    if field.ndim != 2 or field.shape[1] != 2 or not np.all(np.isfinite(field)):
        raise ValueError("base_field must be a finite (n_nodes, 2) array")
    if set(bijection) != set(source) or set(bijection.values()) != set(target):
        raise ValueError("bijection must map every source to every target exactly once")
    result = np.zeros_like(field)
    for source_index, target_index in bijection.items():
        result[int(target_index)] = field[int(source_index)]
    return scale_displacement_field(result, epsilon)


def nonzero_vector_multiset(field: np.ndarray) -> tuple[tuple[float, float], ...]:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("field must be a finite (n_nodes, 2) array")
    return tuple(sorted((float(x), float(y)) for x, y in values if np.linalg.norm((x, y)) > 1e-12))


def boundary_valid(positions: np.ndarray, delta: np.ndarray, bound: float = 1.0, tolerance: float = 1e-9) -> bool:
    points, field = np.asarray(positions, dtype=np.float64), np.asarray(delta, dtype=np.float64)
    return bool(points.shape == field.shape and points.ndim == 2 and points.shape[1] == 2
                and np.all(np.isfinite(points)) and np.all(np.isfinite(field))
                and np.all(np.abs(points + field) <= float(bound) + float(tolerance)))


def _source_support(team_slots: Sequence[int], source_support: Iterable[int]) -> tuple[int, ...]:
    teams = np.asarray(team_slots)
    if teams.ndim != 1:
        raise ValueError("team_slots must be one-dimensional")
    source = _support(source_support, len(teams), "source_support")
    if len(set(teams[list(source)])) != 1:
        raise ValueError("source_support must belong to one team")
    return source


def unit_base_displacement_field(
    n_nodes: int, source_support: Iterable[int], seed_parts: Sequence[object] = ()
) -> np.ndarray:
    """Generate an independent unit-energy 2D fracture field on source nodes."""
    count = int(n_nodes)
    if count < 1:
        raise ValueError("n_nodes must be positive")
    source = _support(source_support, count, "source_support")
    rng = np.random.default_rng(stable_seed("fracture_anchor", *seed_parts))
    values = rng.normal(size=(len(source), 2))
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        raise ValueError("random fracture field has zero norm")
    field = np.zeros((count, 2), dtype=np.float64)
    field[list(source)] = values / norm
    return field


@dataclass(frozen=True)
class FractureAnchor:
    source_support: tuple[int, ...]
    target_support: tuple[int, ...] | None
    bijection: tuple[tuple[int, int], ...] | None
    base_field: np.ndarray
    displacement: np.ndarray
    seed: int


def make_fracture_anchor(
    positions: np.ndarray, team_slots: Sequence[int], source_support: Iterable[int],
    target_support: Iterable[int] | None = None, epsilon: float = 1.0,
    seed_parts: Sequence[object] = (),
    bijection: Mapping[int, int] | None = None,
) -> FractureAnchor:
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("positions must have shape (n_nodes, 2)")
    source = _source_support(team_slots, source_support)
    target: tuple[int, ...] | None = None
    chosen: dict[int, int] | None = None
    if target_support is not None:
        source, target = validate_fracture_supports(team_slots, source, target_support)
        chosen = dict(zip(source, target)) if bijection is None else dict(bijection)
        if set(chosen) != set(source) or set(chosen.values()) != set(target):
            raise ValueError("bijection must map every source to every target exactly once")
    base = unit_base_displacement_field(len(points), source, seed_parts)
    displacement = scale_displacement_field(base, epsilon)
    return FractureAnchor(source, target, None if chosen is None else tuple(sorted(chosen.items())), base, displacement,
                          stable_seed(*seed_parts))


@dataclass(frozen=True)
class SpectralTopologyFeatures2D:
    power_x: tuple[float, ...]
    power_y: tuple[float, ...]
    power_total: tuple[float, ...]
    band_power_x: tuple[float, ...]
    band_power_y: tuple[float, ...]
    band_power_total: tuple[float, ...]
    rq_x: float
    rq_y: float
    rq_total: float
    component_count: int
    induced_density: float
    cut_weight: float


def spectral_topology_features_2d(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray, delta: np.ndarray,
    support: Iterable[int], adjacency: np.ndarray, n_bands: int,
) -> SpectralTopologyFeatures2D:
    """Compute complete response-blind 2D modal, band, RQ and topology features."""
    values, vectors, field = (np.asarray(x, dtype=np.float64) for x in (eigenvalues, eigenvectors, delta))
    graph = np.asarray(adjacency, dtype=np.float64)
    if values.ndim != 1 or vectors.ndim != 2 or field.ndim != 2 or vectors.shape[0] != len(field):
        raise ValueError("spectral inputs have incompatible shapes")
    if vectors.shape[1] != len(values) or field.shape[1] != 2 or len(graph) != len(field) or graph.shape != (len(field), len(field)):
        raise ValueError("spectral inputs have incompatible shapes")
    if n_bands < 1 or len(values) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues and n_bands are invalid")
    source = _support(support, len(field), "support")
    coeff = vectors.T @ field
    powers = coeff * coeff
    total = powers.sum(axis=1)
    if float(total.sum()) <= 1e-15:
        raise ValueError("delta must have non-zero spectral power")
    edges = np.linspace(0, len(values), int(n_bands) + 1, dtype=int)
    def bands(power: np.ndarray) -> tuple[float, ...]:
        denom = float(power.sum())
        return tuple(float(power[edges[i]:edges[i + 1]].sum() / denom) for i in range(n_bands))
    def rq(power: np.ndarray) -> float:
        return float(np.dot(values, power) / max(float(power.sum()), 1e-15) / max(float(np.max(np.abs(values))), 1e-12))
    matrix = graph[np.ix_(source, source)]
    possible = len(source) * (len(source) - 1) / 2
    induced_density = float(np.count_nonzero(np.triu(matrix > 0, 1)) / possible) if possible else 0.0
    mask = np.zeros(len(field), dtype=bool); mask[list(source)] = True
    seen: set[int] = set(); components = 0
    for start in range(len(source)):
        if start in seen: continue
        components += 1; stack = [start]; seen.add(start)
        while stack:
            node = stack.pop()
            for neighbor in np.flatnonzero(matrix[node] > 0):
                if int(neighbor) not in seen: seen.add(int(neighbor)); stack.append(int(neighbor))
    return SpectralTopologyFeatures2D(tuple(map(float, powers[:, 0])), tuple(map(float, powers[:, 1])),
        tuple(map(float, total)), bands(powers[:, 0]), bands(powers[:, 1]), bands(total),
        rq(powers[:, 0]), rq(powers[:, 1]), rq(total), components, induced_density,
        float(graph[mask][:, ~mask].sum()))


def deterministic_tie_break(items: Iterable[object], seed: int = 0) -> list[object]:
    """Stable ordering based only on canonical text and seed; never reads responses."""
    return sorted(items, key=lambda item: hashlib.sha256(f"{int(seed)}|{item!r}".encode()).hexdigest())
