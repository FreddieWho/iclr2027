#!/usr/bin/env python3
"""Run the P2-A/B matching-only stage.

This stage deliberately stops before model inference.  It constructs the P2
intervention arms on the existing P1 point-set samples, records all invalid or
unmatched cases, and checks that the matched controls have the requested
energy/frequency properties.  The formal response run is a later stage.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from p2_matched_controls import (  # noqa: E402
    GraphInvalid,
    TopologyFeatures,
    boundary_slack,
    boundary_valid,
    candidate_id,
    delaunay_adjacency,
    enumerate_same_team_supports,
    frequency_ratio,
    mode_power,
    normalized_band_power,
    normalized_laplacian,
    normalized_rayleigh,
    randomized_sign_spectral_delta_with_retries,
    rigid_subset_translation,
    stable_int,
    topology_features,
    topology_match_score,
)
from p2_matching_diagnostics import assert_response_blind_schema, build_coverage_curve  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "phase2.yaml"


@dataclass(frozen=True)
class Sample:
    sample_index: int
    sample_id: str
    match_id: str
    split: str
    frame: int
    positions: np.ndarray
    adjacency: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    team_slots: np.ndarray
    role_groups: tuple[str, ...]


@dataclass(frozen=True)
class GraphView:
    graph_id: str
    adjacency: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"P2 config must be a mapping: {path}")
    return payload


def load_samples(root: Path, max_samples: int | None = None) -> list[Sample]:
    source_dir = root / "artifacts" / "phase1"
    npz_path = source_dir / "canonical_samples.npz"
    jsonl_path = source_dir / "canonical_samples.jsonl"
    if not npz_path.exists() or not jsonl_path.exists():
        raise FileNotFoundError("P1 canonical sample artifacts are required for P2-A")
    metadata: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                metadata.append(json.loads(line))
    if max_samples is not None:
        metadata = metadata[: int(max_samples)]
    if not metadata:
        raise ValueError("no canonical samples selected")
    with np.load(npz_path) as archive:
        positions = np.asarray(archive["positions"][: len(metadata)], dtype=np.float64)
        adjacency = np.asarray(archive["adjacency"][: len(metadata)], dtype=np.float64)
        eigenvalues = np.asarray(archive["eigenvalues"][: len(metadata)], dtype=np.float64)
        eigenvectors = np.asarray(archive["eigenvectors"][: len(metadata)], dtype=np.float64)
        team_slots = np.asarray(archive["team_slots"][: len(metadata)], dtype=np.int64)
    if len(metadata) != len(positions):
        raise ValueError("metadata and canonical arrays have different lengths")
    samples: list[Sample] = []
    for index, item in enumerate(metadata):
        samples.append(
            Sample(
                sample_index=index,
                sample_id=str(item["sample_id"]),
                match_id=str(item["match_id"]),
                split=str(item["split"]),
                frame=int(item["frame"]),
                positions=positions[index],
                adjacency=adjacency[index],
                eigenvalues=eigenvalues[index],
                eigenvectors=eigenvectors[index],
                team_slots=team_slots[index],
                role_groups=tuple(str(value) for value in item["role_groups"]),
            )
        )
    return samples


def select_samples_per_match(samples: list[Any], samples_per_match: int) -> list[Any]:
    """Select deterministic, time-spread samples from every represented match."""
    if samples_per_match < 1:
        raise ValueError("samples_per_match must be positive")
    grouped: dict[str, list[Any]] = {}
    for sample in samples:
        grouped.setdefault(str(sample.match_id), []).append(sample)
    selected: list[Any] = []
    for match_id in sorted(grouped):
        group = sorted(grouped[match_id], key=lambda sample: (int(sample.frame), str(sample.sample_id)))
        if len(group) < samples_per_match:
            raise ValueError(f"match {match_id} has only {len(group)} samples; need {samples_per_match}")
        indexes = np.linspace(0, len(group) - 1, samples_per_match, dtype=int)
        selected.extend(group[int(index)] for index in indexes)
    return selected


def graph_views(sample: Sample, graph_ids: Iterable[str]) -> tuple[list[GraphView], list[dict[str, Any]]]:
    views: list[GraphView] = []
    statuses: list[dict[str, Any]] = []
    for graph_id in graph_ids:
        try:
            if graph_id == "knn4":
                adjacency = sample.adjacency
                eigenvalues = sample.eigenvalues
                eigenvectors = sample.eigenvectors
            elif graph_id == "delaunay":
                adjacency = delaunay_adjacency(sample.positions)
                eigenvalues, eigenvectors = np.linalg.eigh(normalized_laplacian(adjacency))
            else:
                raise GraphInvalid(f"unknown graph: {graph_id}")
            views.append(GraphView(graph_id, adjacency, eigenvalues, eigenvectors))
            statuses.append({"sample_id": sample.sample_id, "graph_id": graph_id, "status": "valid", "reason": None})
        except (GraphInvalid, ValueError) as exc:
            statuses.append({"sample_id": sample.sample_id, "graph_id": graph_id, "status": "GRAPH_INVALID", "reason": str(exc)})
    return views, statuses


def role_coalitions(sample: Sample) -> list[tuple[str, int, str, np.ndarray]]:
    coalitions: list[tuple[str, int, str, np.ndarray]] = []
    teams = sorted(int(value) for value in np.unique(sample.team_slots))
    roles = sorted(set(sample.role_groups))
    for team in teams:
        for role in roles:
            support = np.asarray(
                [index for index, (slot, group) in enumerate(zip(sample.team_slots, sample.role_groups)) if int(slot) == team and group == role],
                dtype=int,
            )
            if len(support) >= 2:
                coalitions.append((f"team{team}_{role}", team, role, support))
    return coalitions


def json_array(values: np.ndarray | Iterable[float] | Iterable[int]) -> str:
    array = np.asarray(values)
    return json.dumps(np.round(array, 10).tolist() if array.dtype.kind == "f" else array.tolist(), separators=(",", ":"))


def empty_or_value(value: Any) -> Any:
    return None if value is None else value


def intervention_properties(
    support: np.ndarray | None,
    delta: np.ndarray | None,
    graph: GraphView,
    n_bands: int,
) -> dict[str, Any]:
    if delta is None:
        return {
            "energy": None,
            "energy_error": None,
            "support_indices": None if support is None else json_array(support),
            "support_size": None if support is None else int(len(support)),
            "mode_power": None,
            "band_power": None,
            "rayleigh_quotient_normalized": None,
            "component_count": None,
            "induced_density": None,
            "cut_weight": None,
        }
    power = mode_power(graph.eigenvectors, delta)
    features = None
    if support is not None:
        features = topology_features(support, graph.adjacency, graph.eigenvalues, graph.eigenvectors, delta, n_bands)
    else:
        active = np.flatnonzero(np.linalg.norm(delta, axis=1) > 1e-12)
        features = topology_features(active, graph.adjacency, graph.eigenvalues, graph.eigenvectors, delta, n_bands)
        support = active
    return {
        "energy": float(np.linalg.norm(delta)),
        "energy_error": None,
        "support_indices": json_array(support),
        "support_size": int(len(support)),
        "mode_power": json_array(power),
        "band_power": json_array(np.asarray(features.band_power)),
        "rayleigh_quotient_normalized": float(features.rq_normalized),
        "component_count": int(features.component_count),
        "induced_density": float(features.induced_density),
        "cut_weight": float(features.cut_weight),
    }


def invalid_arm(
    sample: Sample,
    graph: GraphView,
    matched_set_id: str,
    coalition_id: str,
    team: int,
    role: str,
    epsilon: float,
    direction: np.ndarray,
    arm_kind: str,
    control_family: str,
    support: np.ndarray | None,
    reason: str,
    control_seed: int | None,
) -> dict[str, Any]:
    properties = intervention_properties(support, None, graph, n_bands=6)
    return {
        "matched_set_id": matched_set_id,
        "arm_kind": arm_kind,
        "control_family": control_family,
        "status": "UNMATCHED",
        "failure_code": "UNMATCHED_OTHER",
        "reason": reason,
        "source_sample_id": sample.sample_id,
        "sample_index": sample.sample_index,
        "source_match_id": sample.match_id,
        "split": sample.split,
        "frame": sample.frame,
        "graph_id": graph.graph_id,
        "coalition_id": coalition_id,
        "team_slot": team,
        "role_group": role,
        "epsilon": float(epsilon),
        "direction": json_array(direction),
        "control_seed": control_seed,
        "valid": False,
        "boundary_status": "not_constructed",
        **properties,
    }


def valid_arm(
    sample: Sample,
    graph: GraphView,
    matched_set_id: str,
    coalition_id: str,
    team: int,
    role: str,
    epsilon: float,
    direction: np.ndarray,
    arm_kind: str,
    control_family: str,
    support: np.ndarray | None,
    delta: np.ndarray,
    control_seed: int | None,
    n_bands: int,
) -> dict[str, Any]:
    properties = intervention_properties(support, delta, graph, n_bands=n_bands)
    properties["energy_error"] = float(abs(float(properties["energy"]) - float(epsilon)))
    return {
        "matched_set_id": matched_set_id,
        "arm_kind": arm_kind,
        "control_family": control_family,
        "status": "VALID",
        "failure_code": None,
        "reason": None,
        "source_sample_id": sample.sample_id,
        "sample_index": sample.sample_index,
        "source_match_id": sample.match_id,
        "split": sample.split,
        "frame": sample.frame,
        "graph_id": graph.graph_id,
        "coalition_id": coalition_id,
        "team_slot": team,
        "role_group": role,
        "epsilon": float(epsilon),
        "direction": json_array(direction),
        "control_seed": control_seed,
        "valid": True,
        "boundary_status": "valid",
        "delta": json_array(delta),
        **properties,
    }


def matching_calipers(config: dict[str, Any]) -> dict[str, Any]:
    matching = config.get("matching", {})
    return {
        "rq_normalized_caliper": float(matching.get("rq_normalized_caliper", 0.05)),
        "band_power_l1_caliper": float(matching.get("band_power_l1_caliper", 0.20)),
        "rq_reference": float(matching.get("frequency_reference_rq", 0.05)),
        "band_reference": float(matching.get("frequency_reference_band", 0.20)),
        "density_scale": float(matching.get("topology_score_density_scale", 0.10)),
        "cut_scale": float(matching.get("topology_score_cut_scale", 1.0)),
        "require_same_component_count": bool(matching.get("require_same_component_count", True)),
    }


def build_candidate_geometry(
    sample: Sample,
    graph: GraphView,
    coalition_id: str,
    team: int,
    role: str,
    anchor_support: np.ndarray,
    candidates: list[np.ndarray],
    n_bands: int,
    calipers: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute energy/direction-invariant candidate diagnostics once."""
    unit_direction = np.array([1.0, 0.0], dtype=np.float64)
    anchor_delta = rigid_subset_translation(len(sample.positions), anchor_support, 1.0, unit_direction)
    anchor_features = topology_features(
        anchor_support,
        graph.adjacency,
        graph.eigenvalues,
        graph.eigenvectors,
        anchor_delta,
        n_bands,
    )
    anchor_set = set(int(value) for value in anchor_support)
    rows: list[dict[str, Any]] = []
    for support in candidates:
        candidate_delta = rigid_subset_translation(len(sample.positions), support, 1.0, unit_direction)
        features = topology_features(
            support,
            graph.adjacency,
            graph.eigenvalues,
            graph.eigenvectors,
            candidate_delta,
            n_bands,
        )
        rq_difference = float(abs(features.rq_normalized - anchor_features.rq_normalized))
        band_l1_difference = float(
            np.abs(np.asarray(features.band_power) - np.asarray(anchor_features.band_power)).sum()
        )
        component_match = features.component_count == anchor_features.component_count
        overlap_count = len(anchor_set & set(int(value) for value in support))
        rows.append(
            {
                "candidate_id": candidate_id(
                    sample.sample_id,
                    graph.graph_id,
                    coalition_id,
                    anchor_support,
                    support,
                ),
                "source_sample_id": sample.sample_id,
                "sample_index": sample.sample_index,
                "source_match_id": sample.match_id,
                "split": sample.split,
                "frame": sample.frame,
                "graph_id": graph.graph_id,
                "coalition_id": coalition_id,
                "team_slot": int(team),
                "role_group": role,
                "anchor_support": json_array(anchor_support),
                "candidate_support": json_array(support),
                "support_size": int(len(support)),
                "semantic_overlap_count": int(overlap_count),
                "semantic_overlap_fraction": float(overlap_count / len(anchor_support)),
                "anchor_component_count": int(anchor_features.component_count),
                "candidate_component_count": int(features.component_count),
                "component_match": bool(component_match),
                "component_match_required": bool(calipers["require_same_component_count"]),
                "anchor_induced_density": float(anchor_features.induced_density),
                "candidate_induced_density": float(features.induced_density),
                "density_difference": float(abs(features.induced_density - anchor_features.induced_density)),
                "anchor_cut_weight": float(anchor_features.cut_weight),
                "candidate_cut_weight": float(features.cut_weight),
                "cut_weight_difference": float(abs(features.cut_weight - anchor_features.cut_weight)),
                "anchor_rq_normalized": float(anchor_features.rq_normalized),
                "candidate_rq_normalized": float(features.rq_normalized),
                "rq_difference": rq_difference,
                "band_power_l1_difference": band_l1_difference,
                "frequency_ratio": frequency_ratio(
                    rq_difference,
                    band_l1_difference,
                    rq_reference=calipers.get("rq_reference", calipers["rq_normalized_caliper"]),
                    band_reference=calipers.get("band_reference", calipers["band_power_l1_caliper"]),
                ),
                "topology_match_score": float(
                    topology_match_score(
                        anchor_features,
                        features,
                        density_scale=calipers["density_scale"],
                        cut_scale=calipers["cut_scale"],
                    )
                ),
            }
        )
    return rows


def build_candidate_feasibility(
    sample: Sample,
    geometry: list[dict[str, Any]],
    matched_set_id: str,
    epsilon: float,
    direction: np.ndarray,
) -> list[dict[str, Any]]:
    """Add energy/direction boundary feasibility without model information."""
    rows: list[dict[str, Any]] = []
    for item in geometry:
        anchor_support = np.asarray(json.loads(str(item["anchor_support"])), dtype=int)
        support = np.asarray(json.loads(str(item["candidate_support"])), dtype=int)
        anchor_delta = rigid_subset_translation(len(sample.positions), anchor_support, epsilon, direction)
        candidate_delta = rigid_subset_translation(len(sample.positions), support, epsilon, direction)
        anchor_slack = boundary_slack(sample.positions, anchor_delta)
        candidate_slack = boundary_slack(sample.positions, candidate_delta)
        within_boundary = anchor_slack >= -1e-9 and candidate_slack >= -1e-9
        component_eligible = bool(item["component_match"]) or not bool(item["component_match_required"])
        eligible = within_boundary and component_eligible
        rows.append(
            {
                "matched_set_id": matched_set_id,
                "candidate_id": str(item["candidate_id"]),
                "source_sample_id": sample.sample_id,
                "source_match_id": sample.match_id,
                "graph_id": str(item["graph_id"]),
                "coalition_id": str(item["coalition_id"]),
                "epsilon": float(epsilon),
                "direction": json_array(direction),
                "anchor_boundary_slack": float(anchor_slack),
                "boundary_slack": float(candidate_slack),
                "anchor_boundary_valid": bool(anchor_slack >= -1e-9),
                "candidate_boundary_valid": bool(candidate_slack >= -1e-9),
                "boundary_valid": bool(within_boundary),
                "component_match": bool(item["component_match"]),
                "eligible": bool(eligible),
                "frequency_ratio": float(item["frequency_ratio"]),
                "passes_reference_calipers": bool(eligible and float(item["frequency_ratio"]) <= 1.0),
                "topology_match_score": float(item["topology_match_score"]),
            }
        )
    return rows


def choose_topology_frequency_from_diagnostics(
    sample: Sample,
    geometry: list[dict[str, Any]],
    feasibility: list[dict[str, Any]],
    epsilon: float,
    direction: np.ndarray,
    calipers: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, float | None, str | None, str | None]:
    """Select a control using only geometry, frequency, and boundary diagnostics."""
    feasibility_by_id = {str(item["candidate_id"]): item for item in feasibility}
    matches: list[tuple[float, float, tuple[int, ...], str, np.ndarray]] = []
    component_eligible = 0
    boundary_eligible = 0
    for item in geometry:
        component_ok = bool(item["component_match"]) or not bool(item["component_match_required"])
        if component_ok:
            component_eligible += 1
        feasible = feasibility_by_id[str(item["candidate_id"])]
        if bool(feasible["eligible"]):
            boundary_eligible += 1
        if not bool(feasible["eligible"]):
            continue
        if float(item["rq_difference"]) > calipers["rq_normalized_caliper"]:
            continue
        if float(item["band_power_l1_difference"]) > calipers["band_power_l1_caliper"]:
            continue
        support = np.asarray(json.loads(str(item["candidate_support"])), dtype=int)
        matches.append(
            (
                float(item["frequency_ratio"]),
                float(item["topology_match_score"]),
                tuple(int(value) for value in support),
                str(item["candidate_id"]),
                support,
            )
        )
    if not matches:
        if not geometry:
            code = "UNMATCHED_NO_SUPPORT"
            reason = "no eligible same-team support"
        elif component_eligible == 0:
            code = "UNMATCHED_TOPOLOGY"
            reason = "no same-team support matched the required component count"
        elif boundary_eligible == 0:
            code = "UNMATCHED_BOUNDARY"
            reason = "all component-eligible same-team supports exceeded the coordinate boundary"
        else:
            code = "UNMATCHED_FREQUENCY"
            reason = "no boundary-feasible same-team support passed frequency calipers"
        return None, None, reason, None, None, code
    matches.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    _, score, _, selected_id, support = matches[0]
    delta = rigid_subset_translation(len(sample.positions), support, epsilon, direction)
    return support, delta, None, score, selected_id, None


def choose_arbitrary_control(
    sample: Sample,
    graph: GraphView,
    candidates: list[np.ndarray],
    epsilon: float,
    direction: np.ndarray,
    key: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    ordered = sorted(candidates, key=lambda support: stable_int(key, tuple(int(x) for x in support)))
    for support in ordered:
        delta = rigid_subset_translation(len(sample.positions), support, epsilon, direction)
        if boundary_valid(sample.positions, delta):
            return support, delta, None
    if ordered:
        return ordered[0], None, "all same-team supports exceeded the coordinate boundary"
    return None, None, "no eligible same-team support"


def choose_topology_frequency_control(
    sample: Sample,
    graph: GraphView,
    anchor_support: np.ndarray,
    anchor_delta: np.ndarray,
    candidates: list[np.ndarray],
    epsilon: float,
    direction: np.ndarray,
    n_bands: int,
    calipers: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, float | None]:
    anchor_features = topology_features(anchor_support, graph.adjacency, graph.eigenvalues, graph.eigenvectors, anchor_delta, n_bands)
    matches: list[tuple[float, tuple[int, ...], np.ndarray, np.ndarray]] = []
    for support in candidates:
        delta = rigid_subset_translation(len(sample.positions), support, epsilon, direction)
        if not boundary_valid(sample.positions, delta):
            continue
        features = topology_features(support, graph.adjacency, graph.eigenvalues, graph.eigenvectors, delta, n_bands)
        if calipers["require_same_component_count"] and features.component_count != anchor_features.component_count:
            continue
        if abs(features.rq_normalized - anchor_features.rq_normalized) > calipers["rq_normalized_caliper"]:
            continue
        band_l1 = float(np.abs(np.asarray(features.band_power) - np.asarray(anchor_features.band_power)).sum())
        if band_l1 > calipers["band_power_l1_caliper"]:
            continue
        score = topology_match_score(
            anchor_features,
            features,
            density_scale=calipers["density_scale"],
            cut_scale=calipers["cut_scale"],
        )
        matches.append((score, tuple(int(x) for x in support), support, delta))
    if not matches:
        return None, None, "no same-team support passed topology/frequency calipers", None
    matches.sort(key=lambda item: (item[0], item[1]))
    score, _, support, delta = matches[0]
    return support, delta, None, float(score)


def build_rows_for_set(
    sample: Sample,
    graph: GraphView,
    coalition_id: str,
    team: int,
    role: str,
    anchor_support: np.ndarray,
    epsilon: float,
    direction: np.ndarray,
    config: dict[str, Any],
    candidate_geometry: list[dict[str, Any]] | None = None,
    candidate_feasibility_sink: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    n_bands = int(config.get("n_bands", 6))
    matched_set_id = f"{sample.sample_id}:{graph.graph_id}:{coalition_id}:eps{epsilon:g}:dir{int(stable_int(tuple(direction))):08x}"
    anchor_delta = rigid_subset_translation(len(sample.positions), anchor_support, epsilon, direction)
    anchor_valid = boundary_valid(sample.positions, anchor_delta)
    rows: list[dict[str, Any]] = []
    if anchor_valid:
        rows.append(valid_arm(sample, graph, matched_set_id, coalition_id, team, role, epsilon, direction, "semantic_rigid", "anchor", anchor_support, anchor_delta, None, n_bands))
    else:
        row = invalid_arm(sample, graph, matched_set_id, coalition_id, team, role, epsilon, direction, "semantic_rigid", "anchor", anchor_support, "semantic rigid translation exceeded the coordinate boundary", None)
        row["failure_code"] = "UNMATCHED_BOUNDARY_ANCHOR"
        rows.append(row)

    overlap_fraction = float(config.get("same_team_max_overlap_fraction", 0.5))
    max_overlap = int(math.floor(len(anchor_support) * overlap_fraction))
    candidates = enumerate_same_team_supports(sample.team_slots, team, len(anchor_support), exclude=anchor_support, max_overlap=max_overlap)
    calipers = matching_calipers(config)
    if candidate_geometry is None:
        candidate_geometry = build_candidate_geometry(
            sample,
            graph,
            coalition_id,
            team,
            role,
            anchor_support,
            candidates,
            n_bands,
            calipers,
        )
    candidate_feasibility = build_candidate_feasibility(
        sample,
        candidate_geometry,
        matched_set_id,
        epsilon,
        direction,
    )
    if candidate_feasibility_sink is not None:
        candidate_feasibility_sink.extend(candidate_feasibility)
    requested_controls = [str(value) for value in config.get("controls", [])]
    for family in requested_controls:
        if family == "arbitrary_same_team":
            support, delta, reason = choose_arbitrary_control(
                sample, graph, candidates, epsilon, direction, f"{matched_set_id}:{family}"
            )
            if delta is None:
                row = invalid_arm(sample, graph, matched_set_id, coalition_id, team, role, epsilon, direction, "control", family, support, reason or "control construction failed", stable_int(matched_set_id, family))
                row["failure_code"] = "UNMATCHED_BOUNDARY_OR_SUPPORT"
                rows.append(row)
            else:
                rows.append(valid_arm(sample, graph, matched_set_id, coalition_id, team, role, epsilon, direction, "control", family, support, delta, stable_int(matched_set_id, family), n_bands))
        elif family == "topology_frequency_matched":
            support, delta, reason, score, selected_id, failure_code = choose_topology_frequency_from_diagnostics(
                sample,
                candidate_geometry,
                candidate_feasibility,
                epsilon,
                direction,
                calipers,
            )
            if delta is None:
                row = invalid_arm(sample, graph, matched_set_id, coalition_id, team, role, epsilon, direction, "control", family, support, reason or "control construction failed", stable_int(matched_set_id, family))
                row["topology_match_score"] = score
                row["selected_candidate_id"] = selected_id
                row["failure_code"] = failure_code or "UNMATCHED_OTHER"
                rows.append(row)
            else:
                row = valid_arm(sample, graph, matched_set_id, coalition_id, team, role, epsilon, direction, "control", family, support, delta, stable_int(matched_set_id, family), n_bands)
                row["topology_match_score"] = score
                row["selected_candidate_id"] = selected_id
                rows.append(row)
        elif family == "spectrum_exact_sign_randomized":
            control_seed = stable_int(matched_set_id, family)
            max_attempts = int(config.get("exact_spectrum_max_attempts", 1))
            retry = randomized_sign_spectral_delta_with_retries(
                graph.eigenvectors,
                anchor_delta,
                sample.positions,
                seed=control_seed,
                max_attempts=max_attempts,
            )
            if retry.success:
                row = valid_arm(sample, graph, matched_set_id, coalition_id, team, role, epsilon, direction, "control", family, None, retry.delta, control_seed, n_bands)
                row["modal_signs"] = json_array(retry.signs)
                row["exact_attempt_count"] = retry.attempt_count
                row["exact_attempt_slacks"] = json_array(np.asarray(retry.boundary_slacks))
                rows.append(row)
            else:
                row = invalid_arm(sample, graph, matched_set_id, coalition_id, team, role, epsilon, direction, "control", family, None, "sign-randomized spectrum exhausted boundary retries", control_seed)
                row["failure_code"] = "UNMATCHED_BOUNDARY_EXACT_RETRY"
                row["modal_signs"] = None
                row["exact_attempt_count"] = retry.attempt_count
                row["exact_attempt_slacks"] = json_array(np.asarray(retry.boundary_slacks))
                rows.append(row)
        else:
            row = invalid_arm(sample, graph, matched_set_id, coalition_id, team, role, epsilon, direction, "control", family, None, "unknown control family", stable_int(matched_set_id, family))
            row["failure_code"] = "UNMATCHED_UNKNOWN_CONTROL"
            rows.append(row)
    return rows


def add_set_status(rows: list[dict[str, Any]]) -> None:
    by_set: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_set.setdefault(str(row["matched_set_id"]), []).append(row)
    for group in by_set.values():
        expected = 1 + len({row["control_family"] for row in group if row["control_family"] != "anchor"})
        complete = len(group) == expected and all(bool(row["valid"]) for row in group)
        status = "COMPLETE" if complete else "INCOMPLETE"
        for row in group:
            row["matched_set_status"] = status


def run_matching(
    samples: list[Sample],
    config: dict[str, Any],
    collect_diagnostics: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | tuple[
    pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame
]:
    rows: list[dict[str, Any]] = []
    graph_statuses: list[dict[str, Any]] = []
    candidate_geometry_rows: list[dict[str, Any]] = []
    candidate_feasibility_rows: list[dict[str, Any]] = []
    directions = [np.asarray(value, dtype=np.float64) for value in config.get("directions", [])]
    energies = [float(value) for value in config.get("energies", [])]
    graph_ids = [str(value) for value in config.get("graphs", [])]
    n_bands = int(config.get("n_bands", 6))
    calipers = matching_calipers(config)
    overlap_fraction = float(config.get("same_team_max_overlap_fraction", 0.5))
    for sample in samples:
        coalitions = role_coalitions(sample)
        views, statuses = graph_views(sample, graph_ids)
        graph_statuses.extend(statuses)
        for view in views:
            for coalition_id, team, role, support in coalitions:
                max_overlap = int(math.floor(len(support) * overlap_fraction))
                candidates = enumerate_same_team_supports(
                    sample.team_slots,
                    team,
                    len(support),
                    exclude=support,
                    max_overlap=max_overlap,
                )
                geometry = build_candidate_geometry(
                    sample,
                    view,
                    coalition_id,
                    team,
                    role,
                    support,
                    candidates,
                    n_bands,
                    calipers,
                )
                if collect_diagnostics:
                    candidate_geometry_rows.extend(geometry)
                for epsilon in energies:
                    for direction in directions:
                        rows.extend(
                            build_rows_for_set(
                                sample,
                                view,
                                coalition_id,
                                team,
                                role,
                                support,
                                epsilon,
                                direction,
                                config,
                                candidate_geometry=geometry,
                                candidate_feasibility_sink=(candidate_feasibility_rows if collect_diagnostics else None),
                            )
                        )
    add_set_status(rows)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("P2-A produced no matching rows")
    set_frame = frame.drop_duplicates("matched_set_id").copy()
    set_frame = set_frame[["matched_set_id", "source_sample_id", "sample_index", "source_match_id", "graph_id", "coalition_id", "team_slot", "role_group", "epsilon", "direction", "matched_set_status"]]
    manifest = {"graphs": graph_statuses, "n_samples": len(samples), "n_rows": len(frame), "n_sets": len(set_frame)}
    if collect_diagnostics:
        return (
            frame,
            set_frame,
            manifest,
            pd.DataFrame(candidate_geometry_rows),
            pd.DataFrame(candidate_feasibility_rows),
        )
    return frame, set_frame, manifest


def validate_matching(frame: pd.DataFrame, set_frame: pd.DataFrame, graph_manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if frame["matched_set_id"].isna().any():
        errors.append("missing matched_set_id")
    # Multiple arms per set are expected; only arm IDs must be unique.
    if frame.duplicated(["matched_set_id", "arm_kind", "control_family"]).any():
        errors.append("duplicate arm within matched set")
    valid = frame[frame["valid"].astype(bool)]
    max_energy_error = float(valid["energy_error"].max()) if not valid.empty else None
    if max_energy_error is not None and max_energy_error > 1e-7:
        errors.append(f"energy error exceeds tolerance: {max_energy_error}")
    complete = set_frame["matched_set_status"].eq("COMPLETE")
    exact_power_errors: list[float] = []
    for _, group in frame.groupby("matched_set_id", sort=False):
        anchor = group[group["control_family"] == "anchor"]
        exact = group[group["control_family"] == "spectrum_exact_sign_randomized"]
        if len(anchor) == 1 and len(exact) == 1 and bool(anchor.iloc[0]["valid"]) and bool(exact.iloc[0]["valid"]):
            a = np.asarray(json.loads(anchor.iloc[0]["mode_power"]), dtype=float)
            b = np.asarray(json.loads(exact.iloc[0]["mode_power"]), dtype=float)
            exact_power_errors.append(float(np.max(np.abs(a - b))))
    max_exact_power_error = max(exact_power_errors) if exact_power_errors else None
    if max_exact_power_error is not None and max_exact_power_error > 1e-8:
        errors.append(f"exact-spectrum control changed modal power: {max_exact_power_error}")
    graph_invalid = [item for item in graph_manifest["graphs"] if item["status"] != "valid"]
    n_incomplete_sets = int((~complete).sum())
    incomplete_reasons = []
    if n_incomplete_sets:
        incomplete_reasons.append("one or more matched sets contain an invalid or unmatched arm")
    if graph_invalid:
        incomplete_reasons.append("one or more requested probe graphs are GRAPH_INVALID")
    status = "fail" if errors else ("incomplete" if incomplete_reasons else "pass")
    return {
        "status": status,
        "errors": errors,
        "incomplete_reasons": incomplete_reasons,
        "n_rows": int(len(frame)),
        "n_sets": int(len(set_frame)),
        "n_complete_sets": int(complete.sum()),
        "n_incomplete_sets": n_incomplete_sets,
        "n_valid_rows": int(valid.shape[0]),
        "n_invalid_rows": int((~frame["valid"].astype(bool)).sum()),
        "max_energy_error": max_energy_error,
        "max_exact_modal_power_error": max_exact_power_error,
        "n_graph_invalid": len(graph_invalid),
        "core_model_inference": "NOT_RUN",
        "scientific_interpretation": "NOT_RUN",
    }


def write_outputs(
    root: Path,
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    samples: list[Sample],
    frame: pd.DataFrame,
    set_frame: pd.DataFrame,
    graph_manifest: dict[str, Any],
    validation: dict[str, Any],
    candidate_geometry: pd.DataFrame | None = None,
    candidate_feasibility: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_geometry = pd.DataFrame() if candidate_geometry is None else candidate_geometry.copy()
    candidate_feasibility = pd.DataFrame() if candidate_feasibility is None else candidate_feasibility.copy()
    if not candidate_geometry.empty:
        assert_response_blind_schema(candidate_geometry)
    if not candidate_feasibility.empty:
        assert_response_blind_schema(candidate_feasibility)
    validation["n_candidate_geometry_rows"] = int(len(candidate_geometry))
    validation["n_candidate_feasibility_rows"] = int(len(candidate_feasibility))
    source_dir = root / "artifacts" / "phase1"
    contract = {
        "phase": "P2_MATCHING_ONLY",
        "checkpoint": "P2-B",
        "study_mode": "exploratory_discovery",
        "data_domain": "football_tracking_point_set",
        "model_inference": "NOT_RUN",
        "scientific_claim": "NOT_RUN",
        "candidate_diagnostics": "RUN_RESPONSE_BLIND" if not candidate_feasibility.empty else "NOT_RUN",
        "configuration_path": str(config_path.relative_to(root)),
        "configuration_sha256": sha256(config_path),
        "configuration": config,
        "n_samples": len(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "source_artifacts": {
            "canonical_samples_npz": {"path": str((source_dir / "canonical_samples.npz").relative_to(root)), "sha256": sha256(source_dir / "canonical_samples.npz")},
            "canonical_samples_jsonl": {"path": str((source_dir / "canonical_samples.jsonl").relative_to(root)), "sha256": sha256(source_dir / "canonical_samples.jsonl")},
        },
        "bioinformatics_isolation": "No bioinformatics index or data was read, updated, or rebuilt.",
    }
    (output_dir / "execution_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    frame.to_parquet(output_dir / "matched_intervention_manifest.parquet", index=False)
    # Keep the contractual matched-set manifest at arm level so semantic and
    # every control identity/failure reason are directly auditable.  The
    # one-row-per-set view is a separate convenience summary.
    frame.to_parquet(output_dir / "matched_set_manifest.parquet", index=False)
    set_frame.to_parquet(output_dir / "matched_set_summary.parquet", index=False)
    balance = (
        frame.groupby(["graph_id", "control_family", "matched_set_status"], dropna=False)
        .agg(n_rows=("matched_set_id", "size"), n_valid=("valid", "sum"), mean_energy_error=("energy_error", "mean"))
        .reset_index()
    )
    balance.to_parquet(output_dir / "matching_balance.parquet", index=False)
    sample_manifest = pd.DataFrame(
        [
            {
                "sample_index": sample.sample_index,
                "sample_id": sample.sample_id,
                "match_id": sample.match_id,
                "split": sample.split,
                "frame": sample.frame,
            }
            for sample in samples
        ]
    )
    sample_manifest.to_parquet(output_dir / "sample_manifest.parquet", index=False)
    set_balance = set_frame.assign(
        complete=set_frame["matched_set_status"].eq("COMPLETE")
    ).groupby(
        ["source_match_id", "graph_id", "role_group", "epsilon", "direction"],
        dropna=False,
    ).agg(
        n_sets=("matched_set_id", "size"),
        n_complete=("complete", "sum"),
    ).reset_index()
    set_balance["complete_fraction"] = set_balance["n_complete"] / set_balance["n_sets"]
    set_balance.to_parquet(output_dir / "matching_balance_by_stratum.parquet", index=False)
    failures = frame.loc[~frame["valid"].astype(bool)].copy()
    failure_taxonomy = (
        failures.groupby(
            ["failure_code", "control_family", "graph_id", "epsilon"],
            dropna=False,
        )
        .size()
        .rename("n_rows")
        .reset_index()
    )
    failure_taxonomy.to_parquet(output_dir / "failure_taxonomy.parquet", index=False)
    exact_attempts = frame.loc[
        frame["control_family"].eq("spectrum_exact_sign_randomized"),
        [
            "matched_set_id",
            "source_sample_id",
            "source_match_id",
            "graph_id",
            "coalition_id",
            "epsilon",
            "direction",
            "valid",
            "failure_code",
            "exact_attempt_count",
            "exact_attempt_slacks",
        ],
    ].copy()
    exact_attempts.to_parquet(output_dir / "exact_spectrum_attempts.parquet", index=False)

    if not candidate_geometry.empty and not candidate_feasibility.empty:
        candidate_geometry.to_parquet(output_dir / "candidate_geometry.parquet", index=False)
        candidate_feasibility.to_parquet(output_dir / "candidate_feasibility.parquet", index=False)
        diagnostic_columns = [
            "candidate_id",
            "rq_difference",
            "band_power_l1_difference",
            "density_difference",
            "cut_weight_difference",
        ]
        curve_input = candidate_feasibility.merge(
            candidate_geometry[diagnostic_columns],
            on="candidate_id",
            how="left",
            validate="many_to_one",
        )
        assert_response_blind_schema(curve_input)
        levels = config.get("diagnostics", {}).get(
            "coverage_levels",
            [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
        )
        coverage_curve = build_coverage_curve(
            curve_input,
            expected_set_ids=set_frame["matched_set_id"].astype(str),
            levels=[float(value) for value in levels],
        )
        coverage_curve.to_parquet(output_dir / "coverage_balance_curve.parquet", index=False)
        coverage_curve.to_csv(output_dir / "coverage_balance_curve.csv", index=False)
        selected = frame.loc[
            frame["control_family"].eq("topology_frequency_matched") & frame["valid"].astype(bool)
        ].copy()
        selected_balance = selected.merge(
            candidate_geometry[diagnostic_columns],
            left_on="selected_candidate_id",
            right_on="candidate_id",
            how="left",
            validate="many_to_one",
        )
        selected_balance.to_parquet(output_dir / "selected_control_balance.parquet", index=False)
        reuse = (
            selected.groupby("selected_candidate_id", dropna=False)
            .agg(
                n_uses=("matched_set_id", "size"),
                n_samples=("source_sample_id", "nunique"),
                n_energy_direction_sets=("matched_set_id", "nunique"),
            )
            .reset_index()
            .sort_values(["n_uses", "selected_candidate_id"], ascending=[False, True])
        )
        reuse.to_parquet(output_dir / "candidate_reuse.parquet", index=False)
    (output_dir / "graph_manifest.json").write_text(json.dumps(graph_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "matching_validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": {
            "pass": "PASS_MATCHING_ONLY",
            "incomplete": "INCOMPLETE_MATCHING_ONLY",
            "fail": "FAIL_MATCHING_ONLY",
        }[validation["status"]],
        "checkpoint": "P2-B",
        "matcher_decision": (
            "MATCHER_VERSION_SELECTED" if config.get("selection_basis") else "DIAGNOSTIC_NOT_SELECTED"
        ),
        "matching_validation": validation,
        "outputs": [
            "execution_contract.json",
            "matched_intervention_manifest.parquet",
            "matched_set_manifest.parquet",
            "matched_set_summary.parquet",
            "matching_balance.parquet",
            "sample_manifest.parquet",
            "matching_balance_by_stratum.parquet",
            "failure_taxonomy.parquet",
            "exact_spectrum_attempts.parquet",
            "candidate_geometry.parquet",
            "candidate_feasibility.parquet",
            "coverage_balance_curve.parquet",
            "coverage_balance_curve.csv",
            "selected_control_balance.parquet",
            "candidate_reuse.parquet",
            "graph_manifest.json",
            "matching_validation.json",
            "pipeline_summary.json",
        ],
        "next_stage": "P2-C one-model response smoke; P2-D formal response run remains NOT_RUN.",
    }
    (output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--samples-per-match", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else (root / args.config)
    config = load_config(config_path.resolve())
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = root / str(config.get("smoke", {}).get("output_dir", "artifacts/phase2/smoke_matching"))
    elif not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty; use --overwrite explicitly: {output_dir}")
    configured_samples_per_match = config.get("selection", {}).get("samples_per_match")
    samples_per_match = args.samples_per_match
    if samples_per_match is None and configured_samples_per_match is not None:
        samples_per_match = int(configured_samples_per_match)
    if samples_per_match is not None:
        if samples_per_match <= 0:
            raise ValueError("--samples-per-match must be positive")
        samples = select_samples_per_match(load_samples(root), samples_per_match)
    else:
        max_samples = args.max_samples
        if max_samples is None:
            max_samples = int(config.get("smoke", {}).get("max_samples", 3))
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        samples = load_samples(root, max_samples=max_samples)
    frame, set_frame, graph_manifest, candidate_geometry, candidate_feasibility = run_matching(
        samples,
        config,
        collect_diagnostics=True,
    )
    validation = validate_matching(frame, set_frame, graph_manifest, config)
    write_outputs(
        root,
        output_dir,
        config_path.resolve(),
        config,
        samples,
        frame,
        set_frame,
        graph_manifest,
        validation,
        candidate_geometry,
        candidate_feasibility,
    )
    print(json.dumps({"output_dir": str(output_dir), "validation": validation}, ensure_ascii=False, indent=2))
    # An incomplete smoke is an expected, auditable research state (for
    # example, boundary-invalid high-energy arms), not a Python crash.  Its
    # artifact status remains INCOMPLETE and cannot advance the P2 gate.
    return 0 if validation["status"] in {"pass", "incomplete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
