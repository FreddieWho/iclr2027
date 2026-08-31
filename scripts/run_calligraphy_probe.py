#!/usr/bin/env python3
"""Controlled Chinese-character probe for the exploratory action-mode study.

The pilot uses Make Me a Hanzi stroke medians as a vector-only, controlled
geometry source.  A stroke is an ordered five-point polyline: its endpoints
are retained and three equal-arc interior points preserve local curvature.
The graph is symmetric for spectral comparability; direction is represented
by ordered-point features rather than a directed Laplacian.

This module deliberately has no bioinformatics imports or paths.  It supports
small smoke runs and a separately requested pilot run.  The ResNet-18 branch
is auxiliary: a local checkpoint is required for scientific use, while an
untrained interface smoke is explicitly marked as non-evidence.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw" / "calligraphy" / "makemeahanzi"
DEFAULT_OUT = ROOT / "artifacts" / "calligraphy_pilot"
REPORT_ROOT = ROOT / "reports"
STROKE_POINTS = 5
NODE_FEATURE_DIM = 9
LATENT_DIM = 128
ENERGIES = [0.10, 0.25, 0.50, 1.00]
PRIMARY_ENERGY = 0.25
N_BANDS = 6
DEFAULT_SEEDS = [11, 23, 47]
BOUND = 1.0
INTERVENTION_CHUNK_SIZE = 128

INTERVENTION_COLUMNS = [
    "intervention_id",
    "pair_id",
    "source_sample_id",
    "character",
    "split",
    "seed",
    "kind",
    "coalition_id",
    "mode_rank",
    "q_rank",
    "band_id",
    "target_modes",
    "epsilon",
    "energy",
    "energy_error",
    "support_type",
    "support_indices",
    "support_size",
    "actual_support_size",
    "stroke_count",
    "delta",
    "spectral_coefficients",
    "rayleigh_quotient",
    "target_mode_purity",
    "boundary_status",
    "valid",
    "rejection_attempts",
    # Optional metadata emitted by particular intervention families.
    "direction",
    "matched_target_rayleigh",
    "matched_rayleigh_error",
    "stroke_id",
    "point_index",
]

RESPONSE_COLUMNS = [
    "model_id",
    "intervention_id",
    "pair_id",
    "source_sample_id",
    "character",
    "split",
    "seed",
    "kind",
    "coalition_id",
    "mode_rank",
    "q_rank",
    "band_id",
    "epsilon",
    "energy_error",
    "rayleigh_quotient",
    "target_mode_purity",
    "response_distance",
    "normalized_response",
]


@dataclass(frozen=True)
class HanziRecord:
    character: str
    decomposition: str
    matches: tuple[tuple[int, ...] | None, ...]
    medians: tuple[np.ndarray, ...]
    stroke_component_labels: tuple[int, ...]
    component_supports: dict[int, np.ndarray]

    @property
    def n_strokes(self) -> int:
        return len(self.medians)


@dataclass
class CalligraphySample:
    sample_id: str
    character: str
    split: str
    points: np.ndarray
    stroke_points: np.ndarray
    stroke_ids: np.ndarray
    point_indices: np.ndarray
    arc_positions: np.ndarray
    tangents: np.ndarray
    curvature: np.ndarray
    is_endpoint: np.ndarray
    stroke_order: np.ndarray
    component_labels: np.ndarray
    component_supports: dict[int, np.ndarray]
    adjacency: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    stroke_quadratic: np.ndarray
    node_features: np.ndarray
    decomposition: str

    @property
    def n_strokes(self) -> int:
        return int(self.stroke_points.shape[0])


@dataclass
class PaddedBatch:
    features: Any
    adjacency: Any | None
    mask: Any
    targets: Any


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _component_index(path: Any) -> tuple[int, ...] | None:
    if path is None or not isinstance(path, list):
        return None
    try:
        return tuple(int(value) for value in path)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=8)
def load_makemeahanzi_records(
    data_root: Path = DATA_ROOT,
    min_strokes: int = 4,
    max_strokes: int = 24,
) -> list[HanziRecord]:
    """Load only complete, component-addressable vector characters.

    The dictionary and graphics files are joined by character.  A component is
    identified by the first index in Make Me a Hanzi's decomposition-tree path,
    so nested components remain in the same top-level coalition.  Records with
    missing matches or unusable decompositions are excluded before sampling.
    """
    dictionaries = {row["character"]: row for row in _jsonl(data_root / "dictionary.txt")}
    graphics = {row["character"]: row for row in _jsonl(data_root / "graphics.txt")}
    records: list[HanziRecord] = []
    for character in sorted(set(dictionaries) & set(graphics)):
        dictionary = dictionaries[character]
        graphic = graphics[character]
        decomposition = str(dictionary.get("decomposition") or "")
        raw_matches = dictionary.get("matches")
        raw_medians = graphic.get("medians")
        raw_strokes = graphic.get("strokes")
        if not decomposition or decomposition.startswith("？"):
            continue
        if not isinstance(raw_matches, list) or not isinstance(raw_medians, list):
            continue
        if not isinstance(raw_strokes, list) or len(raw_medians) != len(raw_strokes):
            continue
        n_strokes = len(raw_medians)
        if not min_strokes <= n_strokes <= max_strokes or len(raw_matches) != n_strokes:
            continue
        matches = tuple(_component_index(path) for path in raw_matches)
        if any(path is None or len(path) == 0 for path in matches):
            continue
        medians: list[np.ndarray] = []
        valid = True
        for median in raw_medians:
            try:
                points = np.asarray(median, dtype=np.float64)
            except (TypeError, ValueError):
                valid = False
                break
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.all(np.isfinite(points)):
                valid = False
                break
            if np.linalg.norm(np.diff(points, axis=0), axis=1).sum() <= 1e-8:
                valid = False
                break
            medians.append(points)
        if not valid:
            continue
        stroke_component_labels = tuple(int(path[0]) for path in matches if path is not None)
        component_supports: dict[int, np.ndarray] = {}
        for component in sorted(set(stroke_component_labels)):
            support = np.flatnonzero(np.asarray(stroke_component_labels) == component).astype(int)
            if 2 <= len(support) <= n_strokes - 2:
                component_supports[component] = support
        if len(component_supports) == 0:
            continue
        records.append(
            HanziRecord(
                character=character,
                decomposition=decomposition,
                matches=matches,
                medians=tuple(medians),
                stroke_component_labels=stroke_component_labels,
                component_supports=component_supports,
            )
        )
    return records


def split_records(
    records: Sequence[HanziRecord], n_chars: int = 800, seed: int = 20260828
) -> dict[str, list[HanziRecord]]:
    """Select and split characters before any model fitting or intervention."""
    if n_chars < 3:
        raise ValueError("n_chars must be at least 3")
    if len(records) < n_chars:
        raise ValueError(f"requested {n_chars} characters but only {len(records)} are eligible")
    ordered = sorted(records, key=lambda record: record.character)
    rng = np.random.default_rng(seed)
    selected_indices = rng.permutation(len(ordered))[:n_chars]
    selected = [ordered[int(index)] for index in selected_indices]
    # Floor gives the contract's exact 560/120/120 split for the 800-character
    # pilot and leaves the remainder to heldout for small smoke runs.
    n_train = int(math.floor(0.70 * n_chars))
    n_dev = int(round(0.15 * n_chars))
    n_train = min(max(1, n_train), n_chars - 2)
    n_dev = min(max(1, n_dev), n_chars - n_train - 1)
    return {
        "train": selected[:n_train],
        "dev": selected[n_train : n_train + n_dev],
        "heldout": selected[n_train + n_dev :],
    }


def normalize_median_points(points: np.ndarray) -> np.ndarray:
    """Map Make Me a Hanzi's 1024 canvas to approximately [-1, 1]^2."""
    points = np.asarray(points, dtype=np.float64)
    normalized = np.empty_like(points)
    normalized[:, 0] = points[:, 0] / 512.0 - 1.0
    # The source coordinate system has an upward y-axis and display transform
    # y_display = 900 - y.  This keeps geometry and rendering consistent.
    normalized[:, 1] = (900.0 - points[:, 1]) / 512.0 - 1.0
    return normalized


def resample_polyline(points: np.ndarray, n_points: int = STROKE_POINTS) -> np.ndarray:
    """Resample by equal arc length while retaining the exact endpoints."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("polyline must have shape (n>=2, 2)")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 1e-12:
        raise ValueError("polyline has zero length")
    targets = np.linspace(0.0, total, n_points)
    result = np.column_stack(
        [np.interp(targets, cumulative, points[:, dimension]) for dimension in range(2)]
    )
    result[0] = points[0]
    result[-1] = points[-1]
    return result


def _polyline_geometry(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangents = np.zeros_like(points)
    curvature = np.zeros(len(points), dtype=np.float64)
    for index in range(len(points)):
        if index == 0:
            vector = points[1] - points[0]
        elif index == len(points) - 1:
            vector = points[-1] - points[-2]
        else:
            vector = points[index + 1] - points[index - 1]
        norm = float(np.linalg.norm(vector))
        tangents[index] = vector / max(norm, 1e-12)
        if 0 < index < len(points) - 1:
            first = points[index] - points[index - 1]
            second = points[index + 1] - points[index]
            denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-12)
            curvature[index] = float(np.cross(first, second) / denominator)
    return tangents, curvature


def _edge_weight(distance: float, scale: float) -> float:
    return float(np.exp(-((distance / max(scale, 1e-8)) ** 2)))


def _components(adjacency: np.ndarray) -> list[np.ndarray]:
    n_nodes = len(adjacency)
    unseen = set(range(n_nodes))
    components: list[np.ndarray] = []
    while unseen:
        start = unseen.pop()
        queue = [start]
        component = [start]
        while queue:
            current = queue.pop()
            neighbors = np.flatnonzero(adjacency[current] > 0)
            for neighbor in neighbors:
                neighbor = int(neighbor)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
        components.append(np.asarray(component, dtype=int))
    return components


def graph_is_connected(adjacency: np.ndarray) -> bool:
    return len(_components(np.asarray(adjacency))) == 1


def build_symmetric_graph(
    points: np.ndarray,
    stroke_ids: np.ndarray,
    cross_k: int = 4,
) -> np.ndarray:
    """Build a fixed, symmetric graph from the baseline geometry."""
    points = np.asarray(points, dtype=np.float64)
    stroke_ids = np.asarray(stroke_ids, dtype=int)
    n_nodes = len(points)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    nonzero = distances[distances > 1e-10]
    scale = float(np.median(nonzero)) if len(nonzero) else 1.0
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    def add_edge(left: int, right: int) -> None:
        if left == right:
            return
        weight = _edge_weight(float(distances[left, right]), scale)
        adjacency[left, right] = max(adjacency[left, right], weight)
        adjacency[right, left] = max(adjacency[right, left], weight)

    for stroke in sorted(set(stroke_ids.tolist())):
        indices = np.flatnonzero(stroke_ids == stroke)
        for left, right in zip(indices[:-1], indices[1:]):
            add_edge(int(left), int(right))
    for left in range(n_nodes):
        candidates = [
            int(right)
            for right in np.argsort(distances[left])
            if right != left and stroke_ids[int(right)] != stroke_ids[left]
        ]
        for right in candidates[:cross_k]:
            add_edge(left, right)

    # Add the closest inter-component edge until the graph is connected.  This
    # is an explicit connectivity guarantee, not a directed edge convention.
    while not graph_is_connected(adjacency):
        components = _components(adjacency)
        best: tuple[float, int, int] | None = None
        for first in range(len(components)):
            for second in range(first + 1, len(components)):
                block = distances[np.ix_(components[first], components[second])]
                position = np.unravel_index(int(np.argmin(block)), block.shape)
                candidate = (
                    float(block[position]),
                    int(components[first][position[0]]),
                    int(components[second][position[1]]),
                )
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise RuntimeError("unable to connect graph")
        add_edge(best[1], best[2])
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _node_features(
    points: np.ndarray,
    stroke_ids: np.ndarray,
    point_indices: np.ndarray,
    stroke_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del stroke_ids, point_indices, stroke_order
    batch = _node_features_batch(np.asarray(points, dtype=np.float64)[None, ...], len(points) // STROKE_POINTS)
    features, tangents, curvature, arc_positions, is_endpoint = batch
    return features[0], tangents[0], curvature[0], arc_positions[0], is_endpoint[0]


def _node_features_batch(
    points_batch: np.ndarray, n_strokes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized geometry features for a same-length batch of characters."""
    points_batch = np.asarray(points_batch, dtype=np.float64)
    if points_batch.ndim != 3 or points_batch.shape[1] != n_strokes * STROKE_POINTS or points_batch.shape[2] != 2:
        raise ValueError("points_batch must have shape (batch, n_strokes*5, 2)")
    batch_size = len(points_batch)
    strokes = points_batch.reshape(batch_size, n_strokes, STROKE_POINTS, 2)
    tangents = np.empty_like(strokes)
    tangents[:, :, 0] = strokes[:, :, 1] - strokes[:, :, 0]
    tangents[:, :, -1] = strokes[:, :, -1] - strokes[:, :, -2]
    tangents[:, :, 1:-1] = strokes[:, :, 2:] - strokes[:, :, :-2]
    tangent_norm = np.linalg.norm(tangents, axis=-1, keepdims=True)
    tangents = tangents / np.maximum(tangent_norm, 1e-12)
    curvature = np.zeros((batch_size, n_strokes, STROKE_POINTS), dtype=np.float64)
    first = strokes[:, :, 1:-1] - strokes[:, :, :-2]
    second = strokes[:, :, 2:] - strokes[:, :, 1:-1]
    cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    denominator = np.maximum(np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1), 1e-12)
    curvature[:, :, 1:-1] = cross / denominator
    arc_positions = np.broadcast_to(
        np.linspace(0.0, 1.0, STROKE_POINTS, dtype=np.float64)[None, None, :],
        (batch_size, n_strokes, STROKE_POINTS),
    )
    is_endpoint = np.zeros((batch_size, n_strokes, STROKE_POINTS, 2), dtype=np.float64)
    is_endpoint[:, :, 0, 0] = 1.0
    is_endpoint[:, :, -1, 1] = 1.0
    stroke_order = np.broadcast_to(
        np.linspace(0.0, 1.0, n_strokes, dtype=np.float64)[None, :, None, None]
        if n_strokes > 1
        else np.zeros((1, 1, 1, 1), dtype=np.float64),
        (batch_size, n_strokes, STROKE_POINTS, 1),
    )
    features = np.concatenate(
        [strokes, tangents, arc_positions[..., None], is_endpoint, curvature[..., None], stroke_order],
        axis=-1,
    ).reshape(batch_size, n_strokes * STROKE_POINTS, NODE_FEATURE_DIM)
    return (
        features,
        tangents.reshape(batch_size, -1, 2),
        curvature.reshape(batch_size, -1),
        arc_positions.reshape(batch_size, -1),
        is_endpoint.reshape(batch_size, -1, 2),
    )


def build_calligraphy_sample(record: HanziRecord, split: str) -> CalligraphySample:
    stroke_points = np.stack(
        [resample_polyline(normalize_median_points(median)) for median in record.medians]
    )
    points = stroke_points.reshape(-1, 2)
    stroke_ids = np.repeat(np.arange(record.n_strokes, dtype=int), STROKE_POINTS)
    point_indices = np.tile(np.arange(STROKE_POINTS, dtype=int), record.n_strokes)
    stroke_order = np.repeat(
        np.linspace(0.0, 1.0, record.n_strokes) if record.n_strokes > 1 else np.zeros(1),
        STROKE_POINTS,
    )
    component_labels = np.repeat(np.asarray(record.stroke_component_labels, dtype=int), STROKE_POINTS)
    component_supports = {
        component: np.flatnonzero(np.isin(stroke_ids, strokes)).astype(int)
        for component, strokes in record.component_supports.items()
    }
    features, tangents, curvature, arc_positions, is_endpoint = _node_features(
        points, stroke_ids, point_indices, stroke_order
    )
    adjacency = build_symmetric_graph(points, stroke_ids)
    degree = adjacency.sum(axis=1)
    inverse_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    laplacian = np.eye(len(points)) - inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    spectral_operator = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
    stroke_quadratic = np.zeros((record.n_strokes, record.n_strokes), dtype=np.float64)
    for left in range(record.n_strokes):
        left_indices = np.flatnonzero(stroke_ids == left)
        for right in range(record.n_strokes):
            right_indices = np.flatnonzero(stroke_ids == right)
            stroke_quadratic[left, right] = float(
                spectral_operator[np.ix_(left_indices, right_indices)].sum()
            )
    return CalligraphySample(
        sample_id=f"hanzi_{ord(record.character):x}",
        character=record.character,
        split=split,
        points=points,
        stroke_points=stroke_points,
        stroke_ids=stroke_ids,
        point_indices=point_indices,
        arc_positions=arc_positions,
        tangents=tangents,
        curvature=curvature,
        is_endpoint=is_endpoint,
        stroke_order=stroke_order,
        component_labels=component_labels,
        component_supports=component_supports,
        adjacency=adjacency,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        stroke_quadratic=stroke_quadratic,
        node_features=features,
        decomposition=record.decomposition,
    )


def _features_for_points(sample: CalligraphySample, points: np.ndarray) -> np.ndarray:
    features, _, _, _, _ = _node_features_batch(
        np.asarray(points, dtype=np.float64)[None, ...], sample.n_strokes
    )
    return features[0]


def band_ranges(n_modes: int, n_bands: int = N_BANDS) -> list[np.ndarray]:
    """Split nonzero q-ranks into bands; the common operator is separate."""
    modes = np.arange(1, n_modes, dtype=int) if n_modes > 1 else np.arange(n_modes, dtype=int)
    return [np.asarray(part, dtype=int) for part in np.array_split(modes, n_bands) if len(part)]


def _random_orientation(rng: np.random.Generator) -> np.ndarray:
    direction = rng.normal(size=2)
    return direction / max(float(np.linalg.norm(direction)), 1e-12)


def _support_intervention(
    n_nodes: int,
    support: np.ndarray,
    rng: np.random.Generator,
    direction: np.ndarray | None = None,
) -> np.ndarray:
    delta = np.zeros((n_nodes, 2), dtype=np.float64)
    delta[np.asarray(support, dtype=int)] = _random_orientation(rng) if direction is None else direction
    return delta


def _spectral_intervention(
    eigenvectors: np.ndarray, modes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    coefficients = rng.normal(size=(len(modes), 2))
    return eigenvectors[:, modes] @ coefficients


def _rayleigh(sample: CalligraphySample, support: np.ndarray) -> float:
    support = np.asarray(support, dtype=int)
    selected_strokes = np.unique(sample.stroke_ids[support])
    full_stroke_support = np.flatnonzero(np.isin(sample.stroke_ids, selected_strokes))
    if np.array_equal(np.sort(support), full_stroke_support):
        indicator = np.zeros(sample.n_strokes, dtype=np.float64)
        indicator[selected_strokes] = 1.0
        return float(
            indicator @ sample.stroke_quadratic @ indicator
            / max(float(len(support)), 1e-12)
        )
    indicator = np.zeros(len(sample.points), dtype=np.float64)
    indicator[support] = 1.0
    coefficients = sample.eigenvectors.T @ indicator
    return float(
        np.sum(sample.eigenvalues * coefficients**2) / max(float(np.sum(coefficients**2)), 1e-12)
    )


def _matched_random_support(
    sample: CalligraphySample, target_support: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    target_strokes = np.unique(sample.stroke_ids[target_support])
    all_strokes = np.arange(sample.n_strokes, dtype=int)
    target_rayleigh = _rayleigh(sample, target_support)
    # Vectorized 256-candidate search over complete strokes.  This preserves
    # equal stroke/point support while avoiding repeated N-by-N projections.
    n_candidates = 256
    candidate_strokes = np.argsort(
        rng.random((n_candidates, sample.n_strokes)), axis=1
    )[:, : len(target_strokes)]
    candidate_strokes.sort(axis=1)
    indicators = np.zeros((n_candidates, sample.n_strokes), dtype=np.float64)
    indicators[np.arange(n_candidates)[:, None], candidate_strokes] = 1.0
    candidate_rayleigh = np.einsum(
        "bi,ij,bj->b", indicators, sample.stroke_quadratic, indicators
    ) / max(float(len(target_support)), 1e-12)
    same = np.all(candidate_strokes == np.sort(target_strokes)[None, :], axis=1)
    candidate_rayleigh[same] = np.inf
    best_index = int(np.argmin(np.abs(candidate_rayleigh - target_rayleigh)))
    if not np.isfinite(candidate_rayleigh[best_index]):
        candidate_strokes = np.roll(target_strokes, 1)
        return np.flatnonzero(np.isin(sample.stroke_ids, candidate_strokes)).astype(int)
    return np.flatnonzero(np.isin(sample.stroke_ids, candidate_strokes[best_index])).astype(int)


def _normalize_energy(delta: np.ndarray, epsilon: float) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-12:
        return np.zeros_like(delta)
    return delta * (float(epsilon) / norm)


def _valid_delta(sample: CalligraphySample, delta: np.ndarray, epsilon: float) -> bool:
    if not np.all(np.isfinite(delta)):
        return False
    if epsilon > 0 and abs(float(np.linalg.norm(delta)) - epsilon) > 1e-7:
        return False
    return bool(np.all(np.abs(sample.points + delta) <= BOUND + 1e-9))


def _record_intervention(
    sample: CalligraphySample,
    seed: int,
    epsilon: float,
    kind: str,
    delta: np.ndarray | None,
    target_modes: np.ndarray | None,
    support_type: str,
    requested_support: np.ndarray | None,
    coalition_id: str | None,
    attempts: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    valid = delta is not None
    if valid:
        delta = np.asarray(delta, dtype=np.float64)
        coefficients = sample.eigenvectors.T @ delta
        energy = float(np.linalg.norm(delta))
        energy_error = abs(energy - epsilon)
        support = np.flatnonzero(np.linalg.norm(delta, axis=1) > 1e-10).astype(int)
        rayleigh = float(
            np.sum(sample.eigenvalues[:, None] * coefficients**2)
            / max(float(np.sum(coefficients**2)), 1e-12)
        )
        purity = (
            None
            if target_modes is None
            else float(
                np.sum(coefficients[np.asarray(target_modes, dtype=int)] ** 2)
                / max(float(np.sum(coefficients**2)), 1e-12)
            )
        )
        delta_json = json.dumps(np.round(delta, 10).tolist(), separators=(",", ":"))
        coefficient_json = json.dumps(np.round(coefficients, 10).tolist(), separators=(",", ":"))
    else:
        energy = energy_error = rayleigh = purity = None
        support = np.asarray(requested_support if requested_support is not None else [], dtype=int)
        delta_json = coefficient_json = None
    if requested_support is None:
        requested_support = support
    mode_rank = int(target_modes[0]) if target_modes is not None and len(target_modes) == 1 else None
    band_id = int(kind.rsplit("_", 1)[1]) if kind.startswith("band_") else None
    q_rank = None if mode_rank is None else float(mode_rank / max(len(sample.points) - 1, 1))
    band_modes = None
    if target_modes is not None and band_id is not None:
        band_modes = np.asarray(target_modes, dtype=int)
    return {
        "intervention_id": f"{sample.sample_id}:{kind}:{coalition_id or 'none'}:eps{epsilon:g}:seed{seed}",
        "pair_id": f"{sample.sample_id}:eps{epsilon:g}:seed{seed}:{kind}:{coalition_id or 'none'}",
        "source_sample_id": sample.sample_id,
        "character": sample.character,
        "split": sample.split,
        "seed": int(seed),
        "kind": kind,
        "coalition_id": coalition_id,
        "mode_rank": mode_rank,
        "q_rank": q_rank,
        "band_id": band_id,
        "target_modes": None if target_modes is None else json.dumps(np.asarray(target_modes, dtype=int).tolist()),
        "epsilon": float(epsilon),
        "energy": energy,
        "energy_error": energy_error,
        "support_type": support_type,
        "support_indices": json.dumps(np.asarray(requested_support, dtype=int).tolist(), separators=(",", ":")),
        "support_size": int(len(requested_support)),
        "actual_support_size": int(len(support)),
        "stroke_count": int(len(np.unique(sample.stroke_ids[requested_support]))) if len(requested_support) else 0,
        "delta": delta_json,
        "spectral_coefficients": coefficient_json,
        "rayleigh_quotient": rayleigh,
        "target_mode_purity": purity,
        "boundary_status": "valid" if valid else "invalid_after_retries",
        "valid": bool(valid),
        "rejection_attempts": int(attempts),
        # Kept in memory for the execution path and removed from the public
        # parquet schema.  This avoids reparsing large JSON deltas during
        # inference while retaining the JSON audit field in the manifest.
        "_delta_array": None if delta is None else np.asarray(delta, dtype=np.float32),
        **metadata,
    }


def _build_raw_intervention(
    sample: CalligraphySample,
    kind: str,
    rng: np.random.Generator,
    component_id: int | None = None,
    direction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, str, np.ndarray, dict[str, Any]]:
    n_nodes = len(sample.points)
    if kind == "common":
        return (
            np.ones((n_nodes, 1), dtype=np.float64)
            @ (_random_orientation(rng) if direction is None else direction)[None, :],
            None,
            "global",
            np.arange(n_nodes, dtype=int),
            {},
        )
    if kind == "semantic_component":
        if component_id is None:
            raise ValueError("semantic component requires component_id")
        support = sample.component_supports[component_id]
        return _support_intervention(n_nodes, support, rng, direction), None, "semantic_component", support, {},
    if kind == "matched_random_component":
        if component_id is None:
            raise ValueError("matched random component requires component_id")
        target = sample.component_supports[component_id]
        support = _matched_random_support(sample, target, rng)
        return (
            _support_intervention(n_nodes, support, rng, direction),
            None,
            "matched_random_component",
            support,
            {"matched_target_rayleigh": _rayleigh(sample, target), "matched_rayleigh_error": abs(_rayleigh(sample, support) - _rayleigh(sample, target))},
        )
    if kind == "stroke_rigid":
        stroke = int(rng.integers(0, sample.n_strokes))
        support = np.flatnonzero(sample.stroke_ids == stroke).astype(int)
        return _support_intervention(n_nodes, support, rng, direction), None, "stroke_rigid", support, {"stroke_id": stroke},
    if kind == "endpoint_local":
        candidates = np.flatnonzero(sample.is_endpoint.sum(axis=1) > 0).astype(int)
        support = np.asarray([int(rng.choice(candidates))], dtype=int)
        return _support_intervention(n_nodes, support, rng, direction), None, "endpoint_local", support, {"point_index": int(sample.point_indices[support[0]])},
    if kind == "interior_local":
        candidates = np.flatnonzero((sample.point_indices > 0) & (sample.point_indices < STROKE_POINTS - 1)).astype(int)
        support = np.asarray([int(rng.choice(candidates))], dtype=int)
        return _support_intervention(n_nodes, support, rng, direction), None, "interior_local", support, {"point_index": int(sample.point_indices[support[0]])},
    if kind.startswith("band_"):
        band_id = int(kind.rsplit("_", 1)[1])
        modes = band_ranges(len(sample.eigenvalues), N_BANDS)[band_id]
        return _spectral_intervention(sample.eigenvectors, modes, rng), modes, "spectral_band", np.arange(n_nodes, dtype=int), {},
    if kind.startswith("exact_mode_"):
        mode = int(kind.rsplit("_", 1)[1])
        modes = np.asarray([mode], dtype=int)
        return _spectral_intervention(sample.eigenvectors, modes, rng), modes, "spectral_exact", np.arange(n_nodes, dtype=int), {},
    raise ValueError(f"unknown intervention kind: {kind}")


def _make_one_intervention(
    sample: CalligraphySample,
    seed: int,
    epsilon: float,
    kind: str,
    component_id: int | None = None,
    direction: np.ndarray | None = None,
    max_retries: int = 24,
) -> dict[str, Any]:
    rng = np.random.default_rng(stable_int(sample.character, kind, component_id, seed, epsilon))
    chosen: tuple[np.ndarray, np.ndarray | None, str, np.ndarray, dict[str, Any]] | None = None
    attempts = 0
    for attempts in range(1, max_retries + 1):
        raw_delta, target_modes, support_type, support, metadata = _build_raw_intervention(
            sample, kind, rng, component_id, direction
        )
        delta = _normalize_energy(raw_delta, epsilon)
        if _valid_delta(sample, delta, epsilon):
            chosen = (delta, target_modes, support_type, support, metadata)
            break
    if chosen is None:
        raw_delta, target_modes, support_type, support, metadata = _build_raw_intervention(
            sample, kind, rng, component_id, direction
        )
        chosen = (None, target_modes, support_type, support, metadata)  # type: ignore[assignment]
    if direction is not None:
        chosen[4]["direction"] = json.dumps(np.round(direction, 10).tolist(), separators=(",", ":"))
    return _record_intervention(
        sample,
        seed,
        epsilon,
        kind,
        chosen[0],
        chosen[1],
        chosen[2],
        chosen[3],
        None if component_id is None else f"component_{component_id}",
        attempts,
        chosen[4],
    )


class _ParquetChunkWriter:
    """Write bounded pandas chunks without materializing the complete table."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: Any | None = None
        self._schema: Any | None = None

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self.path, self._schema, compression="zstd")
        else:
            # Normalize metadata and cast nullable columns to the first chunk's
            # stable schema.  This is needed when an early chunk contains only
            # null values for an optional intervention field.
            table = table.cast(self._schema, safe=False)
            table = table.replace_schema_metadata(self._schema.metadata)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "_ParquetChunkWriter":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _public_intervention_chunk(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Create a fixed-schema public chunk after runtime arrays are removed."""
    frame = pd.DataFrame(rows).drop(columns=["_delta_array"], errors="ignore")
    frame = frame.reindex(columns=INTERVENTION_COLUMNS)
    string_columns = {
        "intervention_id",
        "pair_id",
        "source_sample_id",
        "character",
        "split",
        "kind",
        "coalition_id",
        "target_modes",
        "support_type",
        "support_indices",
        "delta",
        "spectral_coefficients",
        "boundary_status",
        "direction",
    }
    integer_columns = {
        "seed",
        "mode_rank",
        "band_id",
        "support_size",
        "actual_support_size",
        "stroke_count",
        "rejection_attempts",
        "stroke_id",
        "point_index",
    }
    float_columns = {
        "q_rank",
        "epsilon",
        "energy",
        "energy_error",
        "rayleigh_quotient",
        "target_mode_purity",
        "matched_target_rayleigh",
        "matched_rayleigh_error",
    }
    for column in string_columns:
        frame[column] = frame[column].astype("string")
    for column in integer_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in float_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["valid"] = frame["valid"].astype("boolean")
    return frame


def iter_intervention_records(
    samples: Sequence[CalligraphySample],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    energies: Sequence[float] = ENERGIES,
    include_exact_modes: bool = True,
) -> Iterable[dict[str, Any]]:
    """Yield the complete intervention set in deterministic canonical order."""
    if not seeds:
        raise ValueError("at least one intervention seed is required")
    for sample in samples:
        yield _record_intervention(
            sample,
            seed=0,
            epsilon=0.0,
            kind="identity",
            delta=np.zeros_like(sample.points),
            target_modes=None,
            support_type="none",
            requested_support=np.asarray([], dtype=int),
            coalition_id=None,
            attempts=1,
        )
        for seed in seeds:
            for epsilon in energies:
                direction_rng = np.random.default_rng(
                    stable_int(sample.character, "shared_direction", seed, epsilon)
                )
                shared_direction = _random_orientation(direction_rng)
                for kind in ["common", "stroke_rigid", "endpoint_local", "interior_local"]:
                    yield _make_one_intervention(
                        sample, int(seed), float(epsilon), kind, direction=shared_direction
                    )
                for component_id in sorted(sample.component_supports):
                    for kind in ["semantic_component", "matched_random_component"]:
                        yield _make_one_intervention(
                            sample,
                            int(seed),
                            float(epsilon),
                            kind,
                            component_id=component_id,
                            direction=shared_direction,
                        )
                for band_id in range(N_BANDS):
                    yield _make_one_intervention(
                        sample, int(seed), float(epsilon), f"band_{band_id}"
                    )
            if include_exact_modes and int(seed) == int(seeds[0]):
                for mode in range(1, len(sample.eigenvalues)):
                    yield _make_one_intervention(
                        sample,
                        int(seed),
                        PRIMARY_ENERGY,
                        f"exact_mode_{mode}",
                    )


def generate_interventions(
    samples: Sequence[CalligraphySample],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    energies: Sequence[float] = ENERGIES,
    include_exact_modes: bool = True,
) -> pd.DataFrame:
    """Generate a complete table; retained for small tests and diagnostics.

    The formal path uses :func:`stream_intervention_manifest` so this
    compatibility API is never used at pilot scale.
    """
    rows = list(iter_intervention_records(samples, seeds, energies, include_exact_modes))
    delta_arrays = [row.pop("_delta_array", None) for row in rows]
    frame = pd.DataFrame(rows)
    frame.attrs["delta_arrays"] = delta_arrays
    if frame["intervention_id"].duplicated().any():
        raise RuntimeError("intervention_id collision")
    return frame


def _public_intervention_frame(interventions: pd.DataFrame) -> pd.DataFrame:
    """Drop runtime-only ndarray attrs before serializing the audit table."""
    public = interventions.drop(columns=["_delta_array"], errors="ignore").copy(deep=False)
    public.attrs = {}
    return public


def validate_interventions(
    samples: Sequence[CalligraphySample], interventions: pd.DataFrame
) -> dict[str, Any]:
    sample_map = {sample.sample_id: sample for sample in samples}
    errors: list[str] = []
    if interventions["intervention_id"].duplicated().any():
        errors.append("duplicate intervention_id")
    unknown = set(interventions["source_sample_id"]) - set(sample_map)
    if unknown:
        errors.append(f"unknown samples: {sorted(unknown)[:3]}")
    energy_failures = 0
    boundary_failures = 0
    max_energy_error = 0.0
    for row in interventions.itertuples(index=False):
        if not bool(row.valid):
            continue
        sample = sample_map[str(row.source_sample_id)]
        delta = np.asarray(json.loads(row.delta), dtype=np.float64)
        energy_error = abs(float(np.linalg.norm(delta)) - float(row.epsilon))
        max_energy_error = max(max_energy_error, energy_error)
        if energy_error > 1e-7:
            energy_failures += 1
        if not np.all(np.isfinite(delta)) or not np.all(np.abs(sample.points + delta) <= BOUND + 1e-9):
            boundary_failures += 1
    if energy_failures:
        errors.append(f"{energy_failures} valid rows exceed energy tolerance")
    if boundary_failures:
        errors.append(f"{boundary_failures} valid rows exceed boundary")
    valid = interventions["valid"].astype(bool)
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "n_rows": int(len(interventions)),
        "n_valid": int(valid.sum()),
        "n_invalid": int((~valid).sum()),
        "n_identity": int((interventions["kind"] == "identity").sum()),
        "max_energy_error": float(max_energy_error),
    }


def stream_intervention_manifest(
    samples: Sequence[CalligraphySample],
    output_path: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    energies: Sequence[float] = ENERGIES,
    include_exact_modes: bool = True,
    chunk_size: int = INTERVENTION_CHUNK_SIZE,
) -> dict[str, Any]:
    """Generate, validate, and persist interventions with a hard row bound.

    The intervention definition and RNG order are identical to
    :func:`generate_interventions`.  Only the lifetime of the Python row and
    delta caches changes: at most ``chunk_size`` records are materialized at a
    time, and the public audit manifest is written incrementally.
    """
    if chunk_size <= 0:
        raise ValueError("intervention chunk size must be positive")
    sample_map = {sample.sample_id: sample for sample in samples}
    errors: list[str] = []
    seen_ids: set[str] = set()
    unknown_examples: list[str] = []
    duplicate_count = 0
    energy_failures = 0
    boundary_failures = 0
    max_energy_error = 0.0
    n_rows = 0
    n_valid = 0
    n_identity = 0
    chunk: list[dict[str, Any]] = []

    def inspect(row: dict[str, Any]) -> None:
        nonlocal duplicate_count, energy_failures, boundary_failures
        nonlocal max_energy_error, n_rows, n_valid, n_identity
        n_rows += 1
        intervention_id = str(row["intervention_id"])
        if intervention_id in seen_ids:
            duplicate_count += 1
        else:
            seen_ids.add(intervention_id)
        sample_id = str(row["source_sample_id"])
        sample = sample_map.get(sample_id)
        if sample is None:
            if sample_id not in unknown_examples and len(unknown_examples) < 3:
                unknown_examples.append(sample_id)
            return
        if str(row["kind"]) == "identity":
            n_identity += 1
        if not bool(row["valid"]):
            return
        n_valid += 1
        # Validate the serialized audit representation, not only the transient
        # float32 runtime cache.  This keeps the acceptance criterion identical
        # to validate_interventions after a parquet round trip.
        delta = np.asarray(json.loads(str(row["delta"])), dtype=np.float64)
        energy_error = abs(float(np.linalg.norm(delta)) - float(row["epsilon"]))
        max_energy_error = max(max_energy_error, energy_error)
        if energy_error > 1e-7:
            energy_failures += 1
        if not np.all(np.isfinite(delta)) or not np.all(
            np.abs(sample.points + delta) <= BOUND + 1e-9
        ):
            boundary_failures += 1

    with _ParquetChunkWriter(output_path) as writer:
        for row in iter_intervention_records(samples, seeds, energies, include_exact_modes):
            inspect(row)
            chunk.append(row)
            if len(chunk) < chunk_size:
                continue
            for item in chunk:
                item.pop("_delta_array", None)
            writer.write(_public_intervention_chunk(chunk))
            chunk.clear()
        if chunk:
            for item in chunk:
                item.pop("_delta_array", None)
            writer.write(_public_intervention_chunk(chunk))
            chunk.clear()
    # The set is useful only until duplicate detection has completed.  Release
    # it before model training starts on the 32 GB host.
    seen_ids.clear()
    if duplicate_count:
        errors.append("duplicate intervention_id")
    if unknown_examples:
        errors.append(f"unknown samples: {unknown_examples}")
    if energy_failures:
        errors.append(f"{energy_failures} valid rows exceed energy tolerance")
    if boundary_failures:
        errors.append(f"{boundary_failures} valid rows exceed boundary")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "n_rows": int(n_rows),
        "n_valid": int(n_valid),
        "n_invalid": int(n_rows - n_valid),
        "n_identity": int(n_identity),
        "max_energy_error": float(max_energy_error),
        "streamed": True,
        "chunk_size": int(chunk_size),
    }


def build_execution_contract(
    n_chars: int = 800,
    epochs: int = 25,
    smoke: bool = False,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    energies: Sequence[float] = ENERGIES,
) -> dict[str, Any]:
    return {
        "phase": "CALLIGRAPHY_CONTROLLED_REPLICATION",
        "checkpoint": "P1_HANZI_PILOT",
        "study_mode": "exploratory_discovery",
        "data_domain": "chinese_calligraphy_vector_controlled",
        "data_source": "Make Me a Hanzi dictionary.txt + graphics.txt",
        "data_root": "data/raw/calligraphy/makemeahanzi",
        "n_chars": int(n_chars),
        "split": {"train": 0.70, "dev": 0.15, "heldout": 0.15},
        "character_level_isolation": True,
        "stroke_representation": {
            "object": "ordered_stroke_polyline",
            "nodes_per_stroke": STROKE_POINTS,
            "sampling": "endpoints plus three equal_arc interior points",
            "direction": "ordered point features: tangent, arc_position, start/end flags, curvature, stroke_order",
        },
        "component_definition": "top_level_IDS_component_from_dictionary_matches; structural annotation, not etymological semantics",
        "graph": {
            "edges": "within_stroke_chain plus cross_stroke_geometric_knn plus connectivity edges",
            "laplacian": "symmetric_normalized",
            "graph_fixed_under_intervention": True,
        },
        "mainline_models": ["deepsets_ae", "gat_ae"],
        "seeds": [int(seed) for seed in seeds],
        "epochs": int(epochs),
        "energies": [float(energy) for energy in energies],
        "primary_energy": PRIMARY_ENERGY,
        "n_spectral_bands": N_BANDS,
        "full_spectrum": "all_nonzero_laplacian_modes",
        "common_operator_is_separate_from_laplacian_zero_mode": True,
        "auxiliary_model": "resnet18",
        "auxiliary_policy": "local_pretrained_checkpoint_only; untrained_smoke_is_not_evidence",
        "natural_image_data": False,
        "bioinformatics_isolation": True,
        "core_computation_status": "smoke_only" if smoke else "requested_pilot",
    }


def pad_samples(
    samples: Sequence[CalligraphySample],
    point_overrides: Sequence[np.ndarray] | None = None,
    include_adjacency: bool = True,
) -> PaddedBatch:
    """Pad variable-length characters while retaining a hard node mask."""
    import torch

    if not samples:
        raise ValueError("pad_samples requires at least one sample")
    if point_overrides is not None and len(point_overrides) != len(samples):
        raise ValueError("point_overrides length mismatch")
    max_nodes = max(len(sample.points) for sample in samples)
    features = np.zeros((len(samples), max_nodes, NODE_FEATURE_DIM), dtype=np.float32)
    adjacency = (
        np.zeros((len(samples), max_nodes, max_nodes), dtype=np.float32)
        if include_adjacency
        else None
    )
    mask = np.zeros((len(samples), max_nodes), dtype=bool)
    targets = np.zeros((len(samples), max_nodes, 2), dtype=np.float32)
    point_list = [
        sample.points if point_overrides is None else np.asarray(point_overrides[index])
        for index, sample in enumerate(samples)
    ]
    by_node_count: dict[int, list[int]] = {}
    for index, points in enumerate(point_list):
        by_node_count.setdefault(len(points), []).append(index)
    feature_cache: dict[int, np.ndarray] = {}
    for n_nodes, indices in by_node_count.items():
        n_strokes = n_nodes // STROKE_POINTS
        feature_cache[n_nodes] = _node_features_batch(
            np.stack([point_list[index] for index in indices]), n_strokes
        )[0]
        for local_index, batch_index in enumerate(indices):
            current_features = feature_cache[n_nodes][local_index]
            points = point_list[batch_index]
            sample = samples[batch_index]
            n_nodes = len(points)
            features[batch_index, :n_nodes] = current_features.astype(np.float32)
            if adjacency is not None:
                adjacency[batch_index, :n_nodes, :n_nodes] = sample.adjacency.astype(np.float32)
            mask[batch_index, :n_nodes] = True
            targets[batch_index, :n_nodes] = points.astype(np.float32)
    return PaddedBatch(
        features=torch.from_numpy(features),
        adjacency=None if adjacency is None else torch.from_numpy(adjacency),
        mask=torch.from_numpy(mask),
        targets=torch.from_numpy(targets),
    )


def build_point_models(latent_dim: int = LATENT_DIM) -> dict[str, Any]:
    """Return the two point-set autoencoders used by the mainline."""
    import torch
    from torch import nn

    class DeepSetsAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.node = nn.Sequential(
                nn.Linear(NODE_FEATURE_DIM, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            self.project = nn.Sequential(nn.Linear(256, latent_dim), nn.ReLU())
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim + 128, 128), nn.ReLU(), nn.Linear(128, 2)
            )

        @staticmethod
        def _pool(h: Any, mask: Any) -> Any:
            valid = mask.unsqueeze(-1)
            mean = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            maximum = h.masked_fill(~valid, -1e4).max(dim=1).values
            return torch.cat([mean, maximum], dim=-1)

        def forward(self, features: Any, mask: Any) -> tuple[Any, Any]:
            h = self.node(features)
            z = self.project(self._pool(h, mask))
            recon = self.decoder(torch.cat([h, z.unsqueeze(1).expand(-1, h.shape[1], -1)], dim=-1))
            return z, recon

        def encode(self, features: Any, mask: Any) -> Any:
            """Return the latent without constructing the reconstruction."""
            h = self.node(features)
            return self.project(self._pool(h, mask))

    class GraphAttentionLayer(nn.Module):
        def __init__(self, hidden_dim: int = 128, head_dim: int = 32) -> None:
            super().__init__()
            self.query = nn.Linear(hidden_dim, head_dim, bias=False)
            self.key = nn.Linear(hidden_dim, head_dim, bias=False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.self_projection = nn.Linear(hidden_dim, hidden_dim)
            self.neighbor_projection = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, h: Any, adjacency: Any, mask: Any) -> Any:
            batch, n_nodes, _ = h.shape
            eye = torch.eye(n_nodes, device=h.device, dtype=adjacency.dtype).unsqueeze(0)
            graph = adjacency + eye * mask.unsqueeze(-1).to(adjacency.dtype)
            valid_pairs = (graph > 0) & mask.unsqueeze(1) & mask.unsqueeze(2)
            query = self.query(h)
            key = self.key(h)
            # Keep attention logits, masking and softmax in FP32 under V100
            # autocast. Linear projections and the value matmul can still use
            # FP16 tensor cores without turning -1e4 masks into NaNs.
            scores = torch.bmm(query.float(), key.float().transpose(1, 2)) / math.sqrt(query.shape[-1])
            scores = scores.masked_fill(~valid_pairs, -1e4)
            scores = scores + torch.log(graph.float().clamp_min(1e-8))
            attention = torch.softmax(scores, dim=-1)
            values = self.value(h)
            neighbors = torch.bmm(attention.to(values.dtype), values)
            output = torch.relu(self.self_projection(h) + self.neighbor_projection(neighbors))
            return output * mask.unsqueeze(-1)

    class GATAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_projection = nn.Linear(NODE_FEATURE_DIM, 128)
            self.layers = nn.ModuleList([GraphAttentionLayer(), GraphAttentionLayer()])
            self.project = nn.Sequential(nn.Linear(256, latent_dim), nn.ReLU())
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim + 128, 128), nn.ReLU(), nn.Linear(128, 2)
            )

        @staticmethod
        def _pool(h: Any, mask: Any) -> Any:
            valid = mask.unsqueeze(-1)
            mean = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            maximum = h.masked_fill(~valid, -1e4).max(dim=1).values
            return torch.cat([mean, maximum], dim=-1)

        def forward(self, features: Any, adjacency: Any, mask: Any) -> tuple[Any, Any]:
            h = torch.relu(self.in_projection(features))
            for layer in self.layers:
                h = layer(h, adjacency, mask)
            z = self.project(self._pool(h, mask))
            recon = self.decoder(torch.cat([h, z.unsqueeze(1).expand(-1, h.shape[1], -1)], dim=-1))
            return z, recon

        def encode(self, features: Any, adjacency: Any, mask: Any) -> Any:
            """Return the latent without running the decoder."""
            h = torch.relu(self.in_projection(features))
            for layer in self.layers:
                h = layer(h, adjacency, mask)
            return self.project(self._pool(h, mask))

    return {"deepsets_ae": DeepSetsAE(), "gat_ae": GATAE()}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def resolve_runtime(
    device_request: str = "auto",
    amp_request: str = "auto",
    gpu_profile: str = "none",
) -> tuple[Any, bool, dict[str, Any]]:
    """Resolve execution without silently moving a requested CUDA job to CPU."""
    import torch

    if device_request not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported device request: {device_request}")
    if amp_request not in {"auto", "on", "off"}:
        raise ValueError(f"unsupported amp request: {amp_request}")
    cuda_available = bool(torch.cuda.is_available())
    if device_request == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device("cuda" if device_request == "cuda" or (device_request == "auto" and cuda_available) else "cpu")
    if gpu_profile not in {"none", "v100"}:
        raise ValueError(f"unsupported gpu profile: {gpu_profile}")
    if gpu_profile == "v100" and device.type != "cuda":
        raise RuntimeError("gpu_profile=v100 requires a CUDA device")
    amp_enabled = device.type == "cuda" and (amp_request == "on" or amp_request == "auto")
    if amp_request == "on" and device.type != "cuda":
        raise RuntimeError("AMP was requested on a non-CUDA device")
    metadata: dict[str, Any] = {
        "requested_device": device_request,
        "resolved_device": str(device),
        "amp_requested": amp_request,
        "amp_enabled": bool(amp_enabled),
        "gpu_profile": gpu_profile,
        "cuda_available": cuda_available,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "gpu_name": str(properties.name),
                "gpu_total_memory_bytes": int(properties.total_memory),
                "cuda_capability": f"{properties.major}.{properties.minor}",
            }
        )
    else:
        metadata["gpu_name"] = None
    return device, bool(amp_enabled), metadata


def resolve_batch_sizes(
    device: Any,
    batch_size: int | None = None,
    train_batch_size: int | None = None,
) -> tuple[int, int]:
    """Choose larger inference batches while preserving the training default."""
    device_type = getattr(device, "type", str(device))
    default_inference = 256 if device_type == "cuda" else 64
    # Keep the default logical training batch at 16 on every device: changing
    # it changes Adam's update sequence and therefore the scientific model
    # condition.  A larger value remains available as an explicit option.
    default_train = 16
    effective_inference = default_inference if batch_size is None else int(batch_size)
    effective_train = default_train if train_batch_size is None else int(train_batch_size)
    if effective_inference <= 0 or effective_train <= 0:
        raise ValueError("batch sizes must be positive")
    return effective_inference, effective_train


def _autocast_context(device: Any, amp_enabled: bool):
    import torch

    if amp_enabled and getattr(device, "type", None) == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _move_batch(batch: PaddedBatch, device: Any) -> PaddedBatch:
    def move(tensor: Any) -> Any:
        if getattr(device, "type", None) == "cuda":
            tensor = tensor.pin_memory()
        return tensor.to(device, non_blocking=True)

    return PaddedBatch(
        features=move(batch.features),
        adjacency=None if batch.adjacency is None else move(batch.adjacency),
        mask=move(batch.mask),
        targets=move(batch.targets),
    )


def _model_forward(model: Any, family: str, batch: PaddedBatch) -> tuple[Any, Any]:
    if family == "deepsets_ae":
        return model(batch.features, batch.mask)
    if family == "gat_ae":
        return model(batch.features, batch.adjacency, batch.mask)
    raise ValueError(family)


def _model_encode(model: Any, family: str, batch: PaddedBatch) -> Any:
    if family == "deepsets_ae":
        return model.encode(batch.features, batch.mask)
    if family == "gat_ae":
        if batch.adjacency is None:
            raise ValueError("GAT inference requires adjacency")
        return model.encode(batch.features, batch.adjacency, batch.mask)
    raise ValueError(family)


def train_point_models(
    samples: Sequence[CalligraphySample],
    output_dir: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    epochs: int = 25,
    batch_size: int = 16,
    device: Any = "cpu",
    amp_enabled: bool = False,
) -> list[dict[str, Any]]:
    """Fit reconstruction encoders on train characters only."""
    import torch

    train_samples = [sample for sample in samples if sample.split == "train"]
    if not train_samples:
        raise ValueError("no train characters")
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    specs: list[dict[str, Any]] = []
    device = torch.device(device)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    for family in ["deepsets_ae", "gat_ae"]:
        for seed in seeds:
            _set_seed(int(seed))
            model = build_point_models()[family]
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            try:
                scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
            last_loss = float("nan")
            for _epoch in range(int(epochs)):
                order = np.random.default_rng(stable_int(family, seed, _epoch)).permutation(len(train_samples))
                model.train()
                losses: list[float] = []
                for start in range(0, len(order), batch_size):
                    batch_samples = [train_samples[int(index)] for index in order[start : start + batch_size]]
                    batch = _move_batch(
                        pad_samples(batch_samples, include_adjacency=family == "gat_ae"),
                        device,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast_context(device, amp_enabled):
                        _, reconstruction = _model_forward(model, family, batch)
                        valid = batch.mask.unsqueeze(-1).expand_as(reconstruction)
                        residual = reconstruction.float() - batch.targets.float()
                        loss = (residual.square()[valid]).mean()
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                last_loss = float(np.mean(losses))
            model_path = model_dir / f"{family}_seed{int(seed)}.pt"
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, model_path)
            specs.append(
                {
                    "model_id": f"{family}_seed{int(seed)}",
                    "family": family,
                    "seed": int(seed),
                    "path": str(model_path),
                    "latent_dim": LATENT_DIM,
                    "input_dim": NODE_FEATURE_DIM,
                    "train_characters": len(train_samples),
                    "epochs": int(epochs),
                    "final_train_reconstruction_mse": last_loss,
                    "train_device": str(device),
                    "train_amp": bool(amp_enabled),
                    "status": "complete",
                }
            )
    return specs


def _load_point_model(spec: dict[str, Any]) -> Any:
    import torch

    model = build_point_models()[str(spec["family"])]
    state = torch.load(spec["path"], map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def render_vector(
    sample: CalligraphySample,
    points: np.ndarray,
    size: int = 224,
    line_width: int = 5,
) -> np.ndarray:
    """Render controlled five-point polylines with fixed stroke width/caps."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    def pixel(point: np.ndarray) -> tuple[int, int]:
        x = int(np.clip((float(point[0]) + 1.0) * 0.5 * (size - 1), 0, size - 1))
        y = int(np.clip((1.0 - (float(point[1]) + 1.0) * 0.5) * (size - 1), 0, size - 1))
        return x, y

    for stroke in range(sample.n_strokes):
        indices = np.flatnonzero(sample.stroke_ids == stroke)
        pixels = [pixel(points[index]) for index in indices]
        draw.line(pixels, fill=(0, 0, 0), width=line_width, joint="curve")
        radius = line_width / 2.0
        for x, y in [pixels[0], pixels[-1]]:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 0, 0))
    return np.asarray(image, dtype=np.uint8)


def build_resnet18_encoder(
    checkpoint: Path | None = None,
    allow_untrained: bool = False,
) -> Any:
    """Build ResNet-18 without downloading weights implicitly."""
    import torch.nn as nn
    from torchvision.models import resnet18

    if checkpoint is None and not allow_untrained:
        raise FileNotFoundError("no local ResNet-18 checkpoint supplied")
    model = resnet18(weights=None)
    if checkpoint is not None:
        import torch

        payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if isinstance(state, dict):
            state = {str(key).removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=False)
    encoder = nn.Sequential(*list(model.children())[:-1])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def resnet18_embeddings(model: Any, images: np.ndarray) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(np.asarray(images, dtype=np.uint8)).float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        embedding = model((tensor - mean) / std).flatten(1)
    return embedding.cpu().numpy().astype(np.float32)


def resnet_interface_smoke(samples: Sequence[CalligraphySample]) -> dict[str, Any]:
    """Check the auxiliary interface, explicitly without scientific status."""
    try:
        model = build_resnet18_encoder(allow_untrained=True)
        images = np.stack([render_vector(sample, sample.points) for sample in samples[:2]])
        embeddings = resnet18_embeddings(model, images)
        return {
            "model_id": "resnet18",
            "status": "interface_only_untrained",
            "n_images": int(len(images)),
            "embedding_shape": list(embeddings.shape),
            "scientific_evidence": False,
        }
    except Exception as exc:  # interface status belongs in the artifact, not a false green gate
        return {"model_id": "resnet18", "status": "blocked", "reason": repr(exc), "scientific_evidence": False}


def _iter_intervention_frames(
    interventions: pd.DataFrame | Path,
    chunk_size: int,
) -> Iterable[pd.DataFrame]:
    """Read a manifest in bounded record batches, or pass through a small frame."""
    if isinstance(interventions, pd.DataFrame):
        yield interventions
        return
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(interventions)
    for record_batch in parquet_file.iter_batches(batch_size=chunk_size):
        yield record_batch.to_pandas()


def _public_response_chunk(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).reindex(columns=RESPONSE_COLUMNS)
    string_columns = {
        "model_id",
        "intervention_id",
        "pair_id",
        "source_sample_id",
        "character",
        "split",
        "kind",
        "coalition_id",
    }
    for column in string_columns:
        frame[column] = frame[column].astype("string")
    for column in {"seed", "mode_rank", "band_id"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in {
        "q_rank",
        "epsilon",
        "energy_error",
        "rayleigh_quotient",
        "target_mode_purity",
        "response_distance",
        "normalized_response",
    }:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame


def run_point_embeddings(
    samples: Sequence[CalligraphySample],
    interventions: pd.DataFrame | Path,
    model_specs: Sequence[dict[str, Any]],
    output_dir: Path,
    batch_size: int = 32,
    device: Any = "cpu",
    amp_enabled: bool = False,
    intervention_chunk_size: int = INTERVENTION_CHUNK_SIZE,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Measure responses while keeping manifest and response rows bounded.

    Grouping by node count is performed inside each manifest chunk.  This keeps
    the fixed graph semantics and CUDA batching while avoiding a process-wide
    in-memory delta bank and response-row list.
    """
    import torch

    if intervention_chunk_size <= 0:
        raise ValueError("intervention chunk size must be positive")
    device = torch.device(device)
    sample_map = {sample.sample_id: sample for sample in samples}
    sample_groups: dict[int, list[CalligraphySample]] = {}
    for sample in samples:
        sample_groups.setdefault(len(sample.points), []).append(sample)

    def encode_batch(
        model: Any,
        family: str,
        batch_samples: Sequence[CalligraphySample],
        point_overrides: Sequence[np.ndarray] | None,
        stats: dict[str, int],
    ) -> np.ndarray:
        if not batch_samples:
            return np.empty((0, LATENT_DIM), dtype=np.float32)
        try:
            batch = _move_batch(
                pad_samples(
                    batch_samples,
                    point_overrides,
                    include_adjacency=family == "gat_ae",
                ),
                device,
            )
            with torch.inference_mode(), _autocast_context(device, amp_enabled):
                z = _model_encode(model, family, batch)
            return z.float().cpu().numpy().astype(np.float32)
        except RuntimeError as exc:
            is_cuda_oom = device.type == "cuda" and "out of memory" in str(exc).lower()
            if not is_cuda_oom or len(batch_samples) <= 1:
                raise
            stats["oom_batch_splits"] = stats.get("oom_batch_splits", 0) + 1
            torch.cuda.empty_cache()
            midpoint = len(batch_samples) // 2
            left = encode_batch(
                model,
                family,
                batch_samples[:midpoint],
                None if point_overrides is None else point_overrides[:midpoint],
                stats,
            )
            right = encode_batch(
                model,
                family,
                batch_samples[midpoint:],
                None if point_overrides is None else point_overrides[midpoint:],
                stats,
            )
            return np.concatenate([left, right], axis=0)

    response_frames: list[pd.DataFrame] = []
    statuses: list[dict[str, Any]] = []
    embedding_dir = output_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    for spec in model_specs:
        model_id = str(spec["model_id"])
        family = str(spec["family"])
        model = _load_point_model(spec)
        model.to(device)
        model.eval()
        baseline_map: dict[str, np.ndarray] = {}
        runtime_stats: dict[str, int] = {"oom_batch_splits": 0}
        for n_nodes in sorted(sample_groups):
            group_samples = sample_groups[n_nodes]
            for start in range(0, len(group_samples), batch_size):
                chunk_samples = group_samples[start : start + batch_size]
                vectors = encode_batch(model, family, chunk_samples, None, runtime_stats)
                for sample, vector in zip(chunk_samples, vectors):
                    baseline_map[sample.sample_id] = vector

        response_path = embedding_dir / f"{model_id}_responses.parquet"
        n_responses = 0
        with _ParquetChunkWriter(response_path) as response_writer:
            for intervention_frame in _iter_intervention_frames(
                interventions, intervention_chunk_size
            ):
                valid_interventions = intervention_frame[
                    intervention_frame["valid"].astype(bool)
                ]
                delta_bank = intervention_frame.attrs.get("delta_arrays")
                if delta_bank is not None and len(delta_bank) != len(intervention_frame):
                    delta_bank = None
                intervention_groups: dict[
                    int, list[tuple[CalligraphySample, Any, np.ndarray]]
                ] = {}
                for row_index, row in valid_interventions.iterrows():
                    sample = sample_map[str(row.source_sample_id)]
                    delta = None
                    if delta_bank is not None:
                        delta = delta_bank[int(row_index)]
                    if delta is None:
                        delta = np.asarray(json.loads(row.delta), dtype=np.float64)
                    intervention_groups.setdefault(len(sample.points), []).append(
                        (sample, row, np.asarray(delta, dtype=np.float64))
                    )
                for n_nodes in sorted(intervention_groups):
                    group = intervention_groups[n_nodes]
                    for start in range(0, len(group), batch_size):
                        batch_items = group[start : start + batch_size]
                        chunk_samples = [item[0] for item in batch_items]
                        chunk_points = [item[0].points + item[2] for item in batch_items]
                        vectors = encode_batch(
                            model, family, chunk_samples, chunk_points, runtime_stats
                        )
                        model_rows: list[dict[str, Any]] = []
                        for (_, row, _), vector in zip(batch_items, vectors):
                            baseline = baseline_map[str(row.source_sample_id)]
                            distance = float(np.linalg.norm(vector - baseline))
                            model_rows.append(
                                {
                                    "model_id": model_id,
                                    "intervention_id": row.intervention_id,
                                    "pair_id": row.pair_id,
                                    "source_sample_id": row.source_sample_id,
                                    "character": row.character,
                                    "split": row.split,
                                    "seed": row.seed,
                                    "kind": row.kind,
                                    "coalition_id": row.coalition_id,
                                    "mode_rank": row.mode_rank,
                                    "q_rank": row.q_rank,
                                    "band_id": row.band_id,
                                    "epsilon": row.epsilon,
                                    "energy_error": row.energy_error,
                                    "rayleigh_quotient": row.rayleigh_quotient,
                                    "target_mode_purity": row.target_mode_purity,
                                    "response_distance": distance,
                                    "normalized_response": distance
                                    / max(float(np.linalg.norm(baseline)), 1e-8),
                                }
                            )
                        response_writer.write(_public_response_chunk(model_rows))
                        n_responses += len(model_rows)
                del intervention_groups, valid_interventions, intervention_frame

        # The processing order is grouped by node count for throughput.  Save
        # the baseline matrix in canonical sample order for downstream joins.
        ordered_baselines = [baseline_map[sample.sample_id] for sample in samples]
        np.save(
            embedding_dir / f"{model_id}_baseline.npy",
            np.stack(ordered_baselines).astype(np.float32),
        )
        response_frames.append(pd.read_parquet(response_path))
        statuses.append(
            {
                "model_id": model_id,
                "status": "complete",
                "n_responses": int(n_responses),
                "device": str(device),
                "amp_enabled": bool(amp_enabled),
                "intervention_chunk_size": int(intervention_chunk_size),
                **runtime_stats,
            }
        )
        del baseline_map, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    responses = (
        pd.concat(response_frames, ignore_index=True)
        if response_frames
        else pd.DataFrame(columns=RESPONSE_COLUMNS)
    )
    responses.to_parquet(output_dir / "spectrum_response.parquet", index=False)
    return responses, statuses


def _bootstrap_group(
    group: pd.DataFrame,
    value_column: str = "normalized_response",
    n_boot: int = 200,
) -> tuple[float, float, float, int]:
    by_character = group.groupby("character")[value_column].mean().to_numpy(dtype=float)
    if len(by_character) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(stable_int(*(group.iloc[0][key] for key in ["model_id", "kind", "epsilon"])))
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        draws[index] = rng.choice(by_character, size=len(by_character), replace=True).mean()
    return float(by_character.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), int(len(by_character))


def summarize_responses(responses: pd.DataFrame, n_boot: int = 200) -> pd.DataFrame:
    if responses.empty:
        return pd.DataFrame()
    group_columns = ["model_id", "kind", "epsilon", "band_id", "mode_rank"]
    rows: list[dict[str, Any]] = []
    for keys, group in responses.groupby(group_columns, dropna=False):
        mean, lower, upper, n_characters = _bootstrap_group(group, n_boot=n_boot)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "mean_response": mean,
                "ci95_low": lower,
                "ci95_high": upper,
                "n_characters": n_characters,
                "n_rows": int(len(group)),
                "mean_rayleigh": float(group["rayleigh_quotient"].mean()),
                "mean_mode_purity": float(group["target_mode_purity"].mean()) if group["target_mode_purity"].notna().any() else None,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_errr(responses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if responses.empty:
        return pd.DataFrame()
    for model_id, group in responses.groupby("model_id"):
        common = (
            group[group["kind"] == "common"]
            .groupby(["source_sample_id", "seed", "epsilon"])["response_distance"]
            .mean()
            .rename("common")
        )
        semantic = (
            group[group["kind"] == "semantic_component"]
            .groupby(["source_sample_id", "seed", "epsilon"])["response_distance"]
            .mean()
            .rename("semantic")
        )
        paired = pd.concat([common, semantic], axis=1).dropna()
        rows.append(
            {
                "model_id": model_id,
                "n_pairs": int(len(paired)),
                "errr_common_gt_semantic": float((paired["common"] > paired["semantic"]).mean()) if len(paired) else None,
                "mean_common_minus_semantic": float((paired["common"] - paired["semantic"]).mean()) if len(paired) else None,
            }
        )
    return pd.DataFrame(rows)


def compute_accessibility(responses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if responses.empty:
        return pd.DataFrame()
    exact = responses[(responses["kind"].str.startswith("exact_mode_")) & (responses["split"] == "heldout")]
    for model_id, group in exact.groupby("model_id"):
        for split in ["heldout", "dev", "train"]:
            subset = responses[(responses["model_id"] == model_id) & (responses["kind"].str.startswith("exact_mode_")) & (responses["split"] == split)]
            if len(subset) < 3:
                continue
            correlation = float(subset["mode_rank"].corr(subset["response_distance"]))
            rows.append({"model_id": model_id, "split": split, "n_rows": int(len(subset)), "mode_rank_response_correlation": correlation})
    return pd.DataFrame(rows)


def make_figures(
    responses: pd.DataFrame,
    summary: pd.DataFrame,
    errr: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if not summary.empty:
        figure, axis = plt.subplots(figsize=(8, 5))
        bands = summary[(summary["kind"].str.startswith("band_")) & summary["epsilon"].eq(PRIMARY_ENERGY)]
        if not bands.empty:
            for model_id, group in bands.groupby("model_id"):
                group = group.sort_values("band_id")
                band_values = group["band_id"].to_numpy(dtype=float)
                mean_values = group["mean_response"].to_numpy(dtype=float)
                lower_values = group["ci95_low"].to_numpy(dtype=float)
                upper_values = group["ci95_high"].to_numpy(dtype=float)
                axis.plot(band_values, mean_values, marker="o", label=model_id)
                axis.fill_between(band_values, lower_values, upper_values, alpha=0.12)
        axis.set_xlabel("spectral band (equal rank bins)")
        axis.set_ylabel("normalized embedding response")
        axis.set_title("Hanzi pilot: spectral transfer at primary energy")
        axis.legend(fontsize=7)
        figure.tight_layout()
        path = figure_dir / "spectral_transfer.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path))
    if not errr.empty:
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.bar(errr["model_id"], errr["errr_common_gt_semantic"].fillna(0.0))
        axis.set_ylim(0, 1)
        axis.set_ylabel("ERRR: common > semantic")
        axis.tick_params(axis="x", rotation=30)
        figure.tight_layout()
        path = figure_dir / "errr.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path))
    return paths


def save_sample_artifacts(samples: Sequence[CalligraphySample], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray([sample.points for sample in samples], dtype=object)
    np.savez_compressed(output_dir / "canonical_points.npz", points=points)
    with (output_dir / "sample_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            handle.write(
                json.dumps(
                    {
                        "sample_index": index,
                        "sample_id": sample.sample_id,
                        "character": sample.character,
                        "split": sample.split,
                        "n_strokes": sample.n_strokes,
                        "n_nodes": int(len(sample.points)),
                        "decomposition": sample.decomposition,
                        "component_supports": {key: value.tolist() for key, value in sample.component_supports.items()},
                        "eigenvalue_min": float(sample.eigenvalues.min()),
                        "eigenvalue_max": float(sample.eigenvalues.max()),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_report(
    output_dir: Path,
    samples: Sequence[CalligraphySample],
    interventions: pd.DataFrame | None,
    validation: dict[str, Any],
    model_specs: Sequence[dict[str, Any]],
    statuses: Sequence[dict[str, Any]],
    errr: pd.DataFrame,
    accessibility: pd.DataFrame,
    figure_paths: Sequence[str],
    elapsed_seconds: float,
    smoke: bool,
    runtime: dict[str, Any] | None = None,
    batch_size: int | None = None,
    train_batch_size: int | None = None,
) -> Path:
    report_path = REPORT_ROOT / ("CALLIGRAPHY_PILOT_SMOKE_REPORT.md" if smoke else "CALLIGRAPHY_PILOT_REPORT.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    intervention_count = validation.get("n_rows") if interventions is None else len(interventions)
    lines = [
        "# Make Me a Hanzi 受控复现报告",
        "",
        f"状态：{'smoke，仅接口和小样本核验' if smoke else '探索性 pilot 已执行'}",
        "",
        "## 技术路线",
        "",
        "- 数据：Make Me a Hanzi 的 `medians`、`dictionary.matches`；不使用自然书法图像。",
        "- 部件：`matches` 的顶层 IDS 结构标注；这里的 `semantic_component` 不是字源学语义标签。",
        "- 节点：每个笔画是 5 个有序中线点，首尾点保留，内部点按等弧长采样。",
        "- 特征：坐标、切向、弧位置、首尾标记、局部曲率、笔画顺序；部件标签只用于干预。",
        "- 图：笔画链边 + 跨笔画几何 KNN + 连通性补边；使用对称归一化 Laplacian，方向进入特征而非有向谱。",
        "- 主模型：DeepSets-AE、GAT-AE；ResNet-18 只作为需要本地权重的辅助矢量渲染接口。",
        "",
        "## 执行证据",
        "",
        f"- 字符数：{len(samples)}；节点数范围：{min(len(s.points) for s in samples)}–{max(len(s.points) for s in samples)}。",
        f"- 干预行数：{intervention_count}；有效：{validation.get('n_valid')}；边界无效：{validation.get('n_invalid')}。",
        f"- 干预校验：`{validation.get('status')}`；最大能量误差：`{validation.get('max_energy_error')}`。",
        f"- 点集模型状态：{json.dumps(list(statuses), ensure_ascii=False)}",
        f"- 运行时：{json.dumps(runtime or {}, ensure_ascii=False)}；推理 batch：{batch_size}；训练 batch：{train_batch_size}。",
        "- 优化路径：干预 manifest 与响应均按固定小块流式写盘；按节点数分组的 dense batch、`inference_mode`、可选 CUDA FP16；CUDA 未请求或不可用时保持 CPU 路径。",
        f"- 用时：{elapsed_seconds:.2f} 秒。",
        "",
        "## 结果与边界",
        "",
        "本报告不把 smoke 或候选曲线写成跨域确认。只有在完整 pilot、字符级 bootstrap 和严格同频对照完成后，才解释 ERRR、频段曲线或 notch 候选。",
        "",
        "### ERRR",
        "",
        errr.to_markdown(index=False) if not errr.empty else "尚无点集响应结果。",
        "",
        "### 模式可访问性",
        "",
        accessibility.to_markdown(index=False) if not accessibility.empty else "暂无足够 exact-mode 行。",
        "",
        "### 图表",
        "",
        *[f"- `{path}`" for path in figure_paths],
        "",
        "### 未执行/不确定项",
        "",
        "- ResNet-18 若没有本地预训练权重，只能做 `interface_only_untrained` smoke，不能作为科学证据。",
        "- 当前 pilot 使用受控中线渲染，不等同于自然书法风格；局部干预的渲染边界已单独记录。",
        "- 所有结论仍属于发现性候选，需后续严格控制和任务验证。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    n_chars = 24 if args.smoke else int(args.n_chars)
    epochs = 2 if args.smoke else int(args.epochs)
    seeds = [11] if args.smoke else [int(seed) for seed in args.seeds]
    output_dir = Path(args.out).resolve()
    if args.smoke and output_dir == DEFAULT_OUT.resolve():
        output_dir = output_dir / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    device, amp_enabled, runtime = resolve_runtime(
        device_request=str(args.device),
        amp_request=str(args.amp),
        gpu_profile=str(args.gpu_profile),
    )
    batch_size, train_batch_size = resolve_batch_sizes(
        device,
        batch_size=args.batch_size,
        train_batch_size=args.train_batch_size,
    )
    contract = build_execution_contract(n_chars, epochs, args.smoke, seeds=seeds)
    contract["runtime"] = runtime
    contract["optimization"] = {
        "group_by_node_count": True,
        "inference_mode": True,
        "cuda_amp": bool(amp_enabled),
        "batch_size": batch_size,
        "train_batch_size": train_batch_size,
        "stream_interventions": True,
        "intervention_chunk_size": int(args.intervention_chunk_size),
    }
    (output_dir / "execution_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    records = load_makemeahanzi_records(Path(args.data_root))
    splits = split_records(records, n_chars=n_chars, seed=int(args.split_seed))
    samples = [
        build_calligraphy_sample(record, split)
        for split in ["train", "dev", "heldout"]
        for record in splits[split]
    ]
    save_sample_artifacts(samples, output_dir)
    intervention_path = output_dir / "intervention_manifest.parquet"
    validation = stream_intervention_manifest(
        samples,
        intervention_path,
        seeds=seeds,
        energies=ENERGIES,
        chunk_size=int(args.intervention_chunk_size),
    )
    (output_dir / "intervention_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if validation["status"] != "pass":
        raise RuntimeError(f"intervention validation failed: {validation['errors']}")
    model_specs = train_point_models(
        samples,
        output_dir,
        seeds=seeds,
        epochs=epochs,
        batch_size=train_batch_size,
        device=device,
        amp_enabled=amp_enabled,
    )
    (output_dir / "model_manifest.json").write_text(
        json.dumps(model_specs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    responses, statuses = run_point_embeddings(
        samples,
        intervention_path,
        model_specs,
        output_dir,
        batch_size=batch_size,
        device=device,
        amp_enabled=amp_enabled,
        intervention_chunk_size=int(args.intervention_chunk_size),
    )
    summary = summarize_responses(responses, n_boot=50 if args.smoke else 200)
    summary.to_parquet(output_dir / "spectrum_summary.parquet", index=False)
    errr = compute_errr(responses)
    errr.to_csv(output_dir / "errr.csv", index=False)
    accessibility = compute_accessibility(responses)
    accessibility.to_csv(output_dir / "mode_accessibility.csv", index=False)
    resnet_status = resnet_interface_smoke(samples) if args.smoke else {
        "model_id": "resnet18",
        "status": "not_run_missing_local_checkpoint" if args.resnet_checkpoint is None else "not_run_auxiliary_by_default",
        "scientific_evidence": False,
    }
    (output_dir / "auxiliary_resnet18_status.json").write_text(
        json.dumps(resnet_status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    figure_paths = make_figures(responses, summary, errr, output_dir)
    elapsed = time.time() - started
    report_path = write_report(
        output_dir,
        samples,
        None,
        validation,
        model_specs,
        statuses,
        errr,
        accessibility,
        figure_paths,
        elapsed,
        args.smoke,
        runtime=runtime,
        batch_size=batch_size,
        train_batch_size=train_batch_size,
    )
    preflight = {
        "run_type": "smoke" if args.smoke else "pilot",
        "elapsed_seconds": elapsed,
        "n_chars": n_chars,
        "epochs": epochs,
        "seeds": seeds,
        "runtime": runtime,
        "batch_size": batch_size,
        "train_batch_size": train_batch_size,
        "optimization": contract["optimization"],
        "intervention_rows": int(validation["n_rows"]),
        "response_rows": int(len(responses)),
        "formal_estimate_seconds_from_linear_extrapolation": None if not args.smoke else elapsed * (800 / n_chars) * (25 / epochs) * (3 / len(seeds)),
        "formal_estimate_is_not_a_commitment": True,
        "resnet_status": resnet_status,
    }
    (output_dir / "preflight_cost.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pipeline_summary = {
        "status": "smoke_only" if args.smoke else "pilot_complete",
        "output_dir": str(output_dir),
        "report": str(report_path),
        "samples": {"n": len(samples), "splits": {key: len(value) for key, value in splits.items()}},
        "interventions": validation,
        "models": model_specs,
        "response_rows": int(len(responses)),
        "errr": errr.to_dict(orient="records"),
        "mode_accessibility": accessibility.to_dict(orient="records"),
        "resnet18": resnet_status,
        "runtime": runtime,
        "batch_size": batch_size,
        "train_batch_size": train_batch_size,
        "optimization": contract["optimization"],
        "elapsed_seconds": elapsed,
        "formal_estimate_seconds_from_linear_extrapolation": preflight["formal_estimate_seconds_from_linear_extrapolation"],
        "formal_run_started": not args.smoke,
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(pipeline_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"output_dir": str(output_dir), "report": str(report_path), **preflight}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="24 chars, one seed, two epochs, no scientific ResNet evidence")
    parser.add_argument("--n-chars", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--split-seed", type=int, default=20260828)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resnet-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--gpu-profile", choices=["none", "v100"], default="none")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument(
        "--intervention-chunk-size",
        type=int,
        default=INTERVENTION_CHUNK_SIZE,
        help="bounded manifest/inference row chunk; does not change the intervention set",
    )
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
