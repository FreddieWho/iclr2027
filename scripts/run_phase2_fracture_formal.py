#!/usr/bin/env python3
"""Run the provenance-locked P2 fracture-continuity diagnostic.

The run has four explicit stages: response-blind matching, frozen-model
response, match-level statistics, and an independent report.  It uses a new
operator and output tree; the P2 rigid artifacts are never appended to.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from p2_fracture_controls import (  # noqa: E402
    FractureAnchor,
    boundary_valid,
    deterministic_tie_break,
    enumerate_target_supports,
    enumerate_vector_bijections,
    make_fracture_anchor,
    nonzero_vector_multiset,
    reassigned_vector_field,
    scale_displacement_field,
    spectral_topology_features_2d,
    stable_seed,
    unit_base_displacement_field,
)
from p2_fracture_statistics import (  # noqa: E402
    aggregate_architecture_matches,
    aggregate_match_effects,
    build_pair_effects,
    leave_one_match_out,
    residual_association,
    summarize_architecture_matches,
)
from p2_point_model_adapter import file_sha256, load_point_adapter  # noqa: E402
from run_phase2_pipeline import graph_views, load_samples, role_coalitions, select_samples_per_match  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "phase2_fracture_continuity_v1.yaml"
DEFAULT_RUN_DIR = ROOT / "artifacts" / "phase2" / "p2_fracture_continuity_v1"
MATCHING_COMPLETE = "FRACTURE_MATCHING_COMPLETE_WITH_EXPLICIT_UNMATCHED"
RESPONSE_COMPLETE = "FRACTURE_RESPONSE_COMPLETE"
STATISTICS_COMPLETE = "FRACTURE_STATISTICS_COMPLETE"
REPORT_COMPLETE = "P2_FRACTURE_C2_COMPLETE"
POINT_MODEL_IDS = [
    f"{family}_seed{seed}"
    for family in ("deepsets_ae", "gat_ae", "phase_gat")
    for seed in (11, 23, 47)
]


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def relative_path(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def artifact_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(root, path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def write_receipt(root: Path, path: Path, core: dict[str, Any], outputs: dict[str, Path]) -> dict[str, Any]:
    wall_seconds = float(core.pop("_wall_seconds", 0.0))
    payload = {
        **core,
        "outputs": {name: artifact_record(root, output) for name, output in outputs.items()},
        "runtime": {
            "wall_seconds": wall_seconds,
            "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        },
    }
    atomic_write_json(path, payload)
    return payload


def read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"fracture config must be a mapping: {path}")
    return value


def current_environment() -> dict[str, Any]:
    names = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "torch": "torch", "pyarrow": "pyarrow", "PyYAML": "PyYAML"}
    return {
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "packages": {key: importlib.metadata.version(package) for key, package in names.items()},
    }


def validate_locked_files(root: Path, records: dict[str, Any], label: str) -> None:
    for name, record in records.items():
        if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
            raise ValueError(f"{label} record is malformed: {name}")
        path = (root / str(record["path"])).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"{label} path escapes project root: {name}") from exc
        observed = file_sha256(path)
        if observed != str(record["sha256"]):
            raise ValueError(f"{label} SHA256 mismatch for {name}: expected {record['sha256']}, observed {observed}")


def validate_config(root: Path, config_path: Path, config: dict[str, Any]) -> None:
    if config.get("phase") != "P2_FRACTURE_CONTINUITY":
        raise ValueError("wrong fracture phase")
    if config.get("study_mode") != "exploratory_discovery":
        raise ValueError("fracture run must remain exploratory")
    if config.get("bioinformatics_isolation") is not True:
        raise ValueError("bioinformatics isolation must be explicit")
    validate_locked_files(root, config.get("source_code", {}), "source_code")
    validate_locked_files(root, config.get("input_files", {}), "input")
    expected_environment = config.get("environment_lock", {})
    observed_environment = current_environment()
    if observed_environment != expected_environment:
        raise ValueError(f"environment lock mismatch: expected {expected_environment}, observed {observed_environment}")
    if config.get("models") != POINT_MODEL_IDS:
        raise ValueError("fracture formal run must use the frozen nine point models")
    fracture = config.get("fracture", {})
    if fracture.get("operator_id") != "p2_fracture_endpoint_reallocation_v1":
        raise ValueError("unexpected fracture operator")
    if list(fracture.get("graphs", [])) != ["knn4", "delaunay"]:
        raise ValueError("fracture run must use both probe graphs")
    if list(fracture.get("energies", [])) != [0.25, 0.5]:
        raise ValueError("fracture formal scope must remain the two low energies")


def selected_samples(root: Path, config: dict[str, Any], smoke: bool) -> list[Any]:
    samples = load_samples(root)
    if not smoke:
        return samples
    smoke_config = config.get("smoke", {})
    match_ids = {str(value) for value in smoke_config.get("match_ids", [])}
    filtered = [sample for sample in samples if str(sample.match_id) in match_ids]
    if not filtered:
        raise ValueError("smoke match selection is empty")
    return select_samples_per_match(filtered, int(smoke_config.get("samples_per_match", 2)))


def full_samples(root: Path) -> list[Any]:
    return load_samples(root)


def vector_id(field: np.ndarray) -> str:
    values = nonzero_vector_multiset(field)
    payload = np.asarray(values, dtype="<f8").tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def array_json(value: np.ndarray | Iterable[Any]) -> str:
    return json.dumps(np.asarray(value).tolist(), separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def feature_payload(features: Any) -> dict[str, Any]:
    return {
        "power_x": array_json(features.power_x),
        "power_y": array_json(features.power_y),
        "power_total": array_json(features.power_total),
        "band_power_x": array_json(features.band_power_x),
        "band_power_y": array_json(features.band_power_y),
        "band_power": array_json(features.band_power_total),
        "rq_x": float(features.rq_x),
        "rq_y": float(features.rq_y),
        "rayleigh_quotient_normalized": float(features.rq_total),
        "component_count": int(features.component_count),
        "induced_density": float(features.induced_density),
        "cut_weight": float(features.cut_weight),
    }


def pair_residual(anchor_features: Any, candidate_features: Any, config: dict[str, Any]) -> tuple[float, float, float]:
    fracture = config["fracture"]
    frequency = abs(float(anchor_features.rq_total) - float(candidate_features.rq_total))
    frequency += float(np.abs(np.asarray(anchor_features.band_power_total) - np.asarray(candidate_features.band_power_total)).sum())
    topology = abs(float(anchor_features.induced_density) - float(candidate_features.induced_density)) / max(float(fracture.get("density_scale", 0.10)), 1e-12)
    topology += abs(float(anchor_features.cut_weight) - float(candidate_features.cut_weight)) / max(float(fracture.get("cut_scale", 1.0)), 1e-12)
    return frequency + topology, frequency, topology


def common_row(
    sample: Any,
    graph_id: str,
    coalition_id: str,
    team: int,
    role: str,
    source_support: tuple[int, ...],
    target_support: Iterable[int] | None,
    delta: np.ndarray | None,
    epsilon: float,
    draw: int,
    matched_set_id: str,
    arm_kind: str,
    control_family: str,
    vector_multiset_id: str,
    endpoint_assignment_id: str,
    feature: Any | None,
    frequency_residual: float,
    topology_residual: float,
    match_score: float | None,
    candidate_count: int,
    eligible_count: int,
    failure_code: str | None,
    operator_id: str,
) -> dict[str, Any]:
    valid = delta is not None and boundary_valid(sample.positions, delta)
    return {
        "matched_set_id": matched_set_id,
        "intervention_id": f"{matched_set_id}:{arm_kind}:{control_family}",
        "operator_id": operator_id,
        "arm_kind": arm_kind,
        "control_family": control_family,
        "source_sample_id": str(sample.sample_id),
        "sample_index": int(sample.sample_index),
        "source_match_id": str(sample.match_id),
        "split": str(sample.split),
        "frame": int(sample.frame),
        "graph_id": graph_id,
        "coalition_id": coalition_id,
        "team_slot": int(team),
        "role_group": role,
        "epsilon": float(epsilon),
        "fracture_draw": int(draw),
        "anchor_seed": int(stable_seed(operator_id, sample.sample_id, coalition_id, draw)),
        "direction": None,
        "source_support_indices": array_json(np.asarray(source_support, dtype=int)),
        "target_support_indices": array_json(np.asarray([], dtype=int) if target_support is None else np.asarray(tuple(target_support), dtype=int)),
        "support_size": int(len(source_support)),
        "endpoint_assignment_id": endpoint_assignment_id,
        "vector_multiset_id": vector_multiset_id,
        "delta": None if delta is None else array_json(delta),
        "energy": None if delta is None else float(np.linalg.norm(delta)),
        "energy_error": None if delta is None else float(abs(np.linalg.norm(delta) - epsilon)),
        "band_power": None if feature is None else array_json(feature.band_power_total),
        "rayleigh_quotient_normalized": None if feature is None else float(feature.rq_total),
        "component_count": None if feature is None else int(feature.component_count),
        "induced_density": None if feature is None else float(feature.induced_density),
        "cut_weight": None if feature is None else float(feature.cut_weight),
        "frequency_residual": float(frequency_residual),
        "topology_residual": float(topology_residual),
        "topology_match_score": None if match_score is None else float(match_score),
        "candidate_count": int(candidate_count),
        "eligible_count": int(eligible_count),
        "boundary_status": "valid" if valid else "invalid",
        "valid": bool(valid),
        "failure_code": failure_code,
    }


def build_sets_for_sample(
    sample: Any,
    graph_id: str,
    graph: Any,
    config: dict[str, Any],
    draw: int,
    epsilon: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    operator_id = str(config["fracture"]["operator_id"])
    n_bands = int(config["fracture"].get("n_bands", 6))
    for coalition_id, team, role, support_array in role_coalitions(sample):
        source_support = tuple(int(value) for value in support_array)
        matched_set_id = f"{sample.sample_id}:{graph_id}:{coalition_id}:draw{draw}:eps{epsilon:g}"
        base = unit_base_displacement_field(
            len(sample.positions), source_support,
            (str(config["fracture"].get("seed_root", 2026083101)), sample.sample_id, coalition_id, draw),
        )
        anchor = scale_displacement_field(base, epsilon)
        vector_multiset_id = vector_id(anchor)
        anchor_valid = boundary_valid(sample.positions, anchor)
        anchor_features = spectral_topology_features_2d(
            graph.eigenvalues, graph.eigenvectors, anchor, source_support, graph.adjacency, n_bands
        )
        candidates = enumerate_target_supports(sample.team_slots, source_support, team=team)
        candidate_count = 0
        eligible_count = 0
        ranked: list[tuple[float, float, float, tuple[int, ...], tuple[tuple[int, int], ...], np.ndarray, Any]] = []
        for target_array in candidates:
            target = tuple(int(value) for value in target_array)
            for bijection in enumerate_vector_bijections(source_support, target):
                candidate_count += 1
                candidate_delta = reassigned_vector_field(base, source_support, target, bijection, epsilon)
                if not boundary_valid(sample.positions, candidate_delta):
                    continue
                candidate_features = spectral_topology_features_2d(
                    graph.eigenvalues, graph.eigenvectors, candidate_delta, target, graph.adjacency, n_bands
                )
                if candidate_features.component_count != anchor_features.component_count:
                    continue
                eligible_count += 1
                score, frequency, topology = pair_residual(anchor_features, candidate_features, config)
                assignment = tuple(sorted((int(left), int(right)) for left, right in bijection.items()))
                ranked.append((score, frequency, topology, target, assignment, candidate_delta, candidate_features))
        if ranked:
            ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
            selected = ranked[0]
            _, frequency, topology, target, assignment, control_delta, control_features = selected
            failure = None
            score = float(selected[0])
        else:
            target = None
            assignment = ()
            control_delta = None
            control_features = None
            frequency = float("nan")
            topology = float("nan")
            score = None
            failure = "UNMATCHED_NO_BOUNDARY_COMPONENT_COMPATIBLE_BIJECTION"
        status = "COMPLETE" if anchor_valid and control_delta is not None else "INCOMPLETE"
        anchor_failure = None if anchor_valid else "UNMATCHED_BOUNDARY_ANCHOR"
        anchor_row = common_row(
            sample, graph_id, coalition_id, team, role, source_support, None, anchor, epsilon, draw,
            matched_set_id, "anchor", "anchor", vector_multiset_id, "source_support", anchor_features,
            0.0, 0.0, None, candidate_count, eligible_count, anchor_failure, operator_id,
        )
        control_row = common_row(
            sample, graph_id, coalition_id, team, role, source_support, target, control_delta, epsilon, draw,
            matched_set_id, "control", "topology_frequency_same_vector_reassignment",
            vector_multiset_id, json.dumps(assignment, separators=(",", ":")), control_features,
            frequency, topology, score, candidate_count, eligible_count, failure, operator_id,
        )
        anchor_row["matched_set_status"] = status
        control_row["matched_set_status"] = status
        rows.extend([anchor_row, control_row])
        diagnostics.append({
            "matched_set_id": matched_set_id,
            "source_sample_id": str(sample.sample_id),
            "source_match_id": str(sample.match_id),
            "graph_id": graph_id,
            "coalition_id": coalition_id,
            "team_slot": int(team),
            "role_group": role,
            "epsilon": float(epsilon),
            "fracture_draw": int(draw),
            "candidate_count": int(candidate_count),
            "eligible_count": int(eligible_count),
            "complete": bool(status == "COMPLETE"),
            "anchor_boundary_valid": bool(anchor_valid),
            "frequency_residual": float(frequency),
            "topology_residual": float(topology),
            "match_score": score,
            "target_support_indices": None if target is None else array_json(target),
            "endpoint_assignment_id": json.dumps(assignment, separators=(",", ":")),
            "failure_code": failure or anchor_failure,
        })
    return rows, {"diagnostics": diagnostics}


def matching_shard_paths(run_dir: Path, match_id: str) -> tuple[Path, Path, Path]:
    directory = run_dir / "matching" / "shards" / f"match_id={match_id}"
    return directory / "arms.parquet", directory / "diagnostics.parquet", directory / "receipt.json"


def validate_output_record(root: Path, record: dict[str, Any]) -> None:
    path = (root / str(record["path"])).resolve()
    if not path.exists() or int(path.stat().st_size) != int(record["bytes"]):
        raise ValueError(f"output size mismatch: {path}")
    if file_sha256(path) != str(record["sha256"]):
        raise ValueError(f"output hash mismatch: {path}")


def validate_matching_manifest(
    root: Path, manifest_path: Path, expected_config_sha256: str | None = None
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != MATCHING_COMPLETE:
        raise ValueError("matching manifest is not complete")
    if expected_config_sha256 is not None and manifest.get("config_sha256") != expected_config_sha256:
        raise ValueError(
            "matching manifest config hash mismatch: "
            f"expected {expected_config_sha256}, observed {manifest.get('config_sha256')}"
        )
    for shard in manifest.get("shards", []):
        receipt_path = root / str(shard["receipt_path"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if expected_config_sha256 is not None and receipt.get("config_sha256") != expected_config_sha256:
            raise ValueError(f"matching receipt config hash mismatch: {receipt_path}")
        validate_output_record(root, receipt["outputs"]["arms"])
        validate_output_record(root, receipt["outputs"]["diagnostics"])
        if file_sha256(receipt_path) != str(shard["receipt_sha256"]):
            raise ValueError(f"matching receipt hash mismatch: {receipt_path}")
    return manifest


def run_matching_stage(root: Path, config_path: Path, config: dict[str, Any], run_dir: Path, smoke: bool, resume: bool) -> dict[str, Any]:
    manifest_path = run_dir / "matching" / "matching_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise RuntimeError("matching manifest exists; use --resume for immutable verification")
        return validate_matching_manifest(root, manifest_path, file_sha256(config_path))
    started = time.perf_counter()
    samples = selected_samples(root, config, smoke)
    by_match: dict[str, list[Any]] = {}
    for sample in samples:
        by_match.setdefault(str(sample.match_id), []).append(sample)
    shards: list[dict[str, Any]] = []
    total_sets = total_arms = complete_sets = valid_arms = 0
    for match_id in sorted(by_match):
        arms_path, diagnostics_path, receipt_path = matching_shard_paths(run_dir, match_id)
        arms: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        graph_status: list[dict[str, Any]] = []
        for sample in sorted(by_match[match_id], key=lambda item: (int(item.frame), str(item.sample_id))):
            views, statuses = graph_views(sample, config["fracture"]["graphs"])
            graph_status.extend(statuses)
            for graph in views:
                for draw in config["fracture"]["draws"]:
                    for epsilon in config["fracture"]["energies"]:
                        generated, diagnostic_payload = build_sets_for_sample(sample, graph.graph_id, graph, config, int(draw), float(epsilon))
                        arms.extend(generated)
                        diagnostics.extend(diagnostic_payload["diagnostics"])
        arm_frame = pd.DataFrame(arms)
        diagnostic_frame = pd.DataFrame(diagnostics)
        atomic_write_parquet(arms_path, arm_frame)
        atomic_write_parquet(diagnostics_path, diagnostic_frame)
        atomic_write_json(run_dir / "matching" / f"graph_status_match={match_id}.json", graph_status)
        n_sets = int(len(diagnostic_frame))
        n_complete = int(diagnostic_frame["complete"].sum()) if n_sets else 0
        n_valid = int(arm_frame["valid"].sum()) if len(arm_frame) else 0
        total_sets += n_sets
        total_arms += int(len(arm_frame))
        complete_sets += n_complete
        valid_arms += n_valid
        receipt_core = {
            "stage": "matching",
            "shard_id": f"match={match_id}",
            "status": "complete",
            "config_sha256": file_sha256(config_path),
            "operator_id": config["fracture"]["operator_id"],
            "n_samples": int(len(by_match[match_id])),
            "n_sets": n_sets,
            "n_complete_sets": n_complete,
            "n_arms": int(len(arm_frame)),
            "n_valid_arms": n_valid,
            "_wall_seconds": time.perf_counter() - started,
        }
        receipt = write_receipt(root, receipt_path, receipt_core, {"arms": arms_path, "diagnostics": diagnostics_path})
        shards.append({
            "match_id": match_id,
            "receipt_path": relative_path(root, receipt_path),
            "receipt_sha256": file_sha256(receipt_path),
            "arms_path": relative_path(root, arms_path),
            "arms_sha256": file_sha256(arms_path),
            "diagnostics_path": relative_path(root, diagnostics_path),
            "diagnostics_sha256": file_sha256(diagnostics_path),
            "n_sets": n_sets,
            "n_complete_sets": n_complete,
            "n_arms": int(len(arm_frame)),
            "n_valid_arms": n_valid,
        })
    manifest = {
        "status": MATCHING_COMPLETE,
        "checkpoint": "P2-FC-B",
        "run_id": config["run_id"],
        "operator_id": config["fracture"]["operator_id"],
        "config_sha256": file_sha256(config_path),
        "n_samples": int(len(samples)),
        "n_matches": int(len(by_match)),
        "n_sets": total_sets,
        "n_complete_sets": complete_sets,
        "n_arms": total_arms,
        "n_valid_arms": valid_arms,
        "shards": shards,
        "wall_seconds": float(time.perf_counter() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def load_model_specs(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "artifacts" / "phase1" / "point_mainline" / "model_manifest.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["model_id"]): row for row in rows}


def load_baseline(root: Path, samples: list[Any], model_id: str, batch_size: int) -> tuple[Any, dict[str, Any], np.ndarray, dict[str, Any]]:
    adapter, model_receipt = load_point_adapter(root, model_id=model_id, batch_size=batch_size)
    specs = load_model_specs(root)
    spec = specs[model_id]
    embedding_manifest = pd.read_parquet(root / "artifacts" / "phase1" / "point_mainline" / "embedding_manifest.parquet")
    rows = embedding_manifest[
        embedding_manifest["model_id"].eq(model_id) & embedding_manifest["embedding_kind"].eq("baseline")
    ].sort_values("row_index")
    if len(rows) != len(samples) or rows["sample_id"].astype(str).tolist() != [str(x.sample_id) for x in samples]:
        raise ValueError(f"baseline sample order mismatch: {model_id}")
    baseline_path = (root / str(rows.iloc[0]["path"])).resolve()
    baseline = np.asarray(np.load(baseline_path), dtype=np.float32)
    positions = np.stack([np.asarray(sample.positions, dtype=np.float32) for sample in samples])
    teams = np.stack([np.asarray(sample.team_slots, dtype=np.int64) for sample in samples])
    adjacency = np.stack([np.asarray(sample.adjacency, dtype=np.float32) for sample in samples])
    recomputed = adapter.encode(positions, teams, adjacency=adjacency if adapter.uses_adjacency else None)
    difference = np.abs(recomputed.astype(np.float64) - baseline.astype(np.float64))
    parity = {
        "status": "PASS" if np.allclose(recomputed, baseline, rtol=1e-6, atol=5e-7) else "BLOCKED_BASELINE_PARITY",
        "model_id": model_id,
        "n_samples_checked": int(len(samples)),
        "shape": list(recomputed.shape),
        "max_abs_difference": float(difference.max()) if difference.size else 0.0,
        "mean_abs_difference": float(difference.mean()) if difference.size else 0.0,
        "absolute_tolerance": 5e-7,
        "relative_tolerance": 1e-6,
    }
    if parity["status"] != "PASS":
        raise RuntimeError(f"baseline parity failed: {parity}")
    return adapter, {**model_receipt, "checkpoint_sha256": spec["checkpoint_sha256"]}, baseline, parity


def response_shard_paths(run_dir: Path, model_id: str, match_id: str) -> tuple[Path, Path]:
    directory = run_dir / "response" / "shards" / f"model_id={model_id}" / f"match_id={match_id}"
    return directory / "response.parquet", directory / "receipt.json"


def run_response_stage(root: Path, config_path: Path, config: dict[str, Any], run_dir: Path, smoke: bool, resume: bool) -> dict[str, Any]:
    manifest_path = run_dir / "response" / "response_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise RuntimeError("response manifest exists; use --resume for immutable verification")
        return validate_response_manifest(root, run_dir, file_sha256(config_path))
    matching = validate_matching_manifest(root, run_dir / "matching" / "matching_manifest.json", file_sha256(config_path))
    started = time.perf_counter()
    samples = full_samples(root)
    by_index = {int(sample.sample_index): sample for sample in samples}
    specs = load_model_specs(root)
    model_ids = [str(config["models"][0])] if smoke else [str(x) for x in config["models"]]
    match_ids = [str(item["match_id"]) for item in matching["shards"]]
    all_parity: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    total_rows = 0
    for model_id in model_ids:
        adapter, model_receipt, baseline, parity = load_baseline(root, samples, model_id, int(config["runtime"]["batch_size"]))
        all_parity.append(parity)
        atomic_write_json(run_dir / "response" / "models" / f"model_id={model_id}" / "model_receipt.json", model_receipt)
        atomic_write_json(run_dir / "response" / "models" / f"model_id={model_id}" / "baseline_parity.json", parity)
        for match_id in match_ids:
            arms_path = root / str(next(item for item in matching["shards"] if str(item["match_id"]) == match_id)["arms_path"])
            arms = pd.read_parquet(arms_path)
            valid = arms[arms["valid"].astype(bool)].copy()
            if valid.empty:
                response = valid.copy()
            else:
                positions: list[np.ndarray] = []
                sample_indices: list[int] = []
                for encoded in valid["delta"]:
                    sample_index = int(valid.iloc[len(sample_indices)]["sample_index"])
                    sample = by_index[sample_index]
                    delta = np.asarray(json.loads(str(encoded)), dtype=np.float32)
                    if delta.shape != sample.positions.shape or not np.isfinite(delta).all():
                        raise ValueError("matching delta is invalid")
                    positions.append(np.asarray(sample.positions, dtype=np.float32) + delta)
                    sample_indices.append(sample_index)
                moved = np.stack(positions)
                teams = np.stack([np.asarray(by_index[index].team_slots, dtype=np.int64) for index in sample_indices])
                adjacency = np.stack([np.asarray(by_index[index].adjacency, dtype=np.float32) for index in sample_indices])
                after = adapter.encode(moved, teams, adjacency=adjacency if adapter.uses_adjacency else None)
                before = baseline[np.asarray(sample_indices, dtype=int)]
                before_norm = before / np.maximum(np.linalg.norm(before, axis=1, keepdims=True), 1e-12)
                after_norm = after / np.maximum(np.linalg.norm(after, axis=1, keepdims=True), 1e-12)
                distances = 1.0 - np.clip(np.sum(before_norm * after_norm, axis=1), -1.0, 1.0)
                response = valid.reset_index(drop=True)
                response["model_id"] = model_id
                response["architecture"] = str(specs[model_id]["architecture"])
                response["model_seed"] = int(specs[model_id]["seed"])
                response["response_distance"] = distances.astype(float)
                response["normalized_response"] = response["response_distance"] / response["epsilon"].astype(float)
            output_path, receipt_path = response_shard_paths(run_dir, model_id, match_id)
            atomic_write_parquet(output_path, response)
            receipt_core = {
                "stage": "response",
                "shard_id": f"model={model_id}:match={match_id}",
                "status": "complete",
                "config_sha256": file_sha256(config_path),
                "matching_manifest_sha256": file_sha256(run_dir / "matching" / "matching_manifest.json"),
                "matching_arms_sha256": str(next(item for item in matching["shards"] if str(item["match_id"]) == match_id)["arms_sha256"]),
                "model_id": model_id,
                "checkpoint_sha256": str(specs[model_id]["checkpoint_sha256"]),
                "n_response_rows": int(len(response)),
                "_wall_seconds": time.perf_counter() - started,
            }
            receipt = write_receipt(root, receipt_path, receipt_core, {"response": output_path})
            total_rows += int(len(response))
            shards.append({
                "model_id": model_id,
                "match_id": match_id,
                "response_path": relative_path(root, output_path),
                "response_sha256": file_sha256(output_path),
                "receipt_path": relative_path(root, receipt_path),
                "receipt_sha256": file_sha256(receipt_path),
                "n_response_rows": int(len(response)),
            })
    manifest = {
        "status": RESPONSE_COMPLETE,
        "checkpoint": "P2-FC-C",
        "run_id": config["run_id"],
        "config_sha256": file_sha256(config_path),
        "matching_manifest_sha256": file_sha256(run_dir / "matching" / "matching_manifest.json"),
        "n_models": len(model_ids),
        "n_shards": len(shards),
        "n_response_rows": total_rows,
        "baseline_parity": all_parity,
        "shards": shards,
        "wall_seconds": float(time.perf_counter() - started),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def statistics_paths(run_dir: Path) -> dict[str, Path]:
    directory = run_dir / "statistics"
    return {
        "pairs": directory / "fracture_pair_effects.parquet",
        "unit": directory / "source_unit_effects.parquet",
        "model_match": directory / "model_match_effects.parquet",
        "architecture_match": directory / "architecture_match_effects.parquet",
        "summary": directory / "fracture_summary.parquet",
        "loo": directory / "leave_one_match_out.parquet",
        "residual": directory / "residual_association.parquet",
    }


def validate_response_manifest(
    root: Path, run_dir: Path, expected_config_sha256: str | None = None
) -> dict[str, Any]:
    path = run_dir / "response" / "response_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != RESPONSE_COMPLETE:
        raise ValueError("response manifest is not complete")
    if expected_config_sha256 is not None and manifest.get("config_sha256") != expected_config_sha256:
        raise ValueError(
            "response manifest config hash mismatch: "
            f"expected {expected_config_sha256}, observed {manifest.get('config_sha256')}"
        )
    for shard in manifest["shards"]:
        receipt_path = root / str(shard["receipt_path"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if expected_config_sha256 is not None and receipt.get("config_sha256") != expected_config_sha256:
            raise ValueError(f"response receipt config hash mismatch: {receipt_path}")
        validate_output_record(root, receipt["outputs"]["response"])
        if file_sha256(receipt_path) != str(shard["receipt_sha256"]):
            raise ValueError(f"response receipt hash mismatch: {receipt_path}")
    return manifest


def run_statistics_stage(root: Path, config_path: Path, config: dict[str, Any], run_dir: Path, resume: bool) -> dict[str, Any]:
    manifest_path = run_dir / "statistics" / "statistics_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise RuntimeError("statistics manifest exists; use --resume for immutable verification")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_config_sha256 = file_sha256(config_path)
    matching = validate_matching_manifest(root, run_dir / "matching" / "matching_manifest.json", expected_config_sha256)
    response_manifest = validate_response_manifest(root, run_dir, expected_config_sha256)
    frames: list[pd.DataFrame] = []
    for shard in response_manifest["shards"]:
        frames.append(pd.read_parquet(root / str(shard["response_path"])))
    response = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    pairs = build_pair_effects(response)
    unit, model_match = aggregate_match_effects(pairs)
    architecture_match = aggregate_architecture_matches(model_match)
    summary = summarize_architecture_matches(
        architecture_match,
        bootstrap_replicates=int(config["statistics"]["bootstrap_replicates"]),
        bootstrap_seed=int(config["statistics"]["bootstrap_seed"]),
    )
    loo = leave_one_match_out(architecture_match)
    residual = residual_association(pairs)
    outputs = statistics_paths(run_dir)
    for name, frame in {
        "pairs": pairs,
        "unit": unit,
        "model_match": model_match,
        "architecture_match": architecture_match,
        "summary": summary,
        "loo": loo,
        "residual": residual,
    }.items():
        atomic_write_parquet(outputs[name], frame)
    receipt_path = run_dir / "statistics" / "receipt.json"
    receipt_core = {
        "stage": "statistics",
        "status": "complete",
        "config_sha256": file_sha256(config_path),
        "matching_manifest_sha256": file_sha256(run_dir / "matching" / "matching_manifest.json"),
        "response_manifest_sha256": file_sha256(run_dir / "response" / "response_manifest.json"),
        "n_pair_rows": int(len(pairs)),
        "n_source_unit_rows": int(len(unit)),
        "n_model_match_rows": int(len(model_match)),
        "n_architecture_match_rows": int(len(architecture_match)),
        "n_summary_rows": int(len(summary)),
        "_wall_seconds": 0.0,
    }
    receipt = write_receipt(root, receipt_path, receipt_core, outputs)
    manifest = {
        "status": STATISTICS_COMPLETE,
        "checkpoint": "P2-FC-E",
        "run_id": config["run_id"],
        "config_sha256": file_sha256(config_path),
        "matching_manifest_sha256": file_sha256(run_dir / "matching" / "matching_manifest.json"),
        "response_manifest_sha256": file_sha256(run_dir / "response" / "response_manifest.json"),
        "receipt_sha256": file_sha256(receipt_path),
        "n_pair_rows": int(len(pairs)),
        "n_summary_rows": int(len(summary)),
        "outputs": {name: artifact_record(root, path) for name, path in outputs.items()},
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def make_report_figures(run_dir: Path, summary: pd.DataFrame) -> list[Path]:
    figures: list[Path] = []
    if summary.empty:
        return figures
    forest = run_dir / "figures" / "fracture_effect_forest.png"
    plot = summary.copy()
    plot["label"] = plot["architecture"] + " / " + plot["graph_id"] + " / " + plot["role_group"] + " / eps=" + plot["epsilon"].astype(str)
    plot = plot.sort_values("raw_scg_mean")
    height = max(4.0, 0.22 * len(plot) + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))
    y = np.arange(len(plot))
    ax.errorbar(plot["raw_scg_mean"], y, xerr=[plot["raw_scg_mean"] - plot["raw_scg_ci_low"], plot["raw_scg_ci_high"] - plot["raw_scg_mean"]], fmt="o", ms=3)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"], fontsize=7)
    ax.set_xlabel("fracture gap: anchor - reassigned control")
    ax.set_title("P2 fracture continuity: response-blind topology/frequency reassignment")
    fig.tight_layout()
    forest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(forest, dpi=160)
    plt.close(fig)
    figures.append(forest)
    return figures


def write_report_stage(root: Path, config_path: Path, config: dict[str, Any], run_dir: Path, smoke: bool, resume: bool) -> dict[str, Any]:
    manifest_path = run_dir / "report_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise RuntimeError("report manifest exists; use --resume for immutable verification")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    matching = validate_matching_manifest(root, run_dir / "matching" / "matching_manifest.json")
    stats = json.loads((run_dir / "statistics" / "statistics_manifest.json").read_text(encoding="utf-8"))
    summary_path = root / str(stats["outputs"]["summary"]["path"])
    summary = pd.read_parquet(summary_path)
    figures = make_report_figures(run_dir, summary)
    report_path = run_dir / ("report_draft.md" if smoke else "report.md")
    interpretation = "WIRING_ONLY_NOT_SCIENTIFIC" if smoke else "SCIENTIFIC_REVIEW_REQUIRED"
    lines = [
        "# P2 fracture continuity report",
        "",
        f"- stage: `P2_NOVELTY_DISCRIMINATOR`",
        f"- task: `fracture continuity`",
        f"- operator: `{config['fracture']['operator_id']}`",
        f"- status: `{interpretation}`",
        "- this report is independent of P1 and P2 rigid response tables.",
        "",
        "## Scope",
        "",
        f"- matching sets planned/complete: {matching['n_sets']} / {matching['n_complete_sets']}",
        f"- valid arms: {matching['n_valid_arms']} / {matching['n_arms']}",
        f"- response/statistics summary rows: {stats['n_summary_rows']}",
        "- anchor and control share the exact non-zero displacement-vector multiset; control support is same-team, same-size and disjoint.",
        "",
        "## Interpretation boundary",
        "",
        "The gap is a representation response contrast, not downstream task performance. Frequency/topology residuals, graph dependence, role dependence and match-level uncertainty must be read together. No fixed numerical threshold is used to force a scientific label.",
        "",
        "## Summary table",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No valid pair summary was produced.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_manifest = {
        "status": "P2_FC_REPORT_READY_FOR_REVIEW" if not smoke else "WIRING_ONLY_NOT_SCIENTIFIC",
        "checkpoint": "P2-FC-E",
        "run_id": config["run_id"],
        "config_sha256": file_sha256(config_path),
        "statistics_manifest_sha256": file_sha256(run_dir / "statistics" / "statistics_manifest.json"),
        "scientific_interpretation": interpretation,
        "report_draft": artifact_record(root, report_path),
        "figures": [artifact_record(root, path) for path in figures],
    }
    atomic_write_json(manifest_path, report_manifest)
    if not smoke:
        target_report = root / "reports" / "EXPLORATION_CHECKPOINT_2_FRACTURE.md"
        target_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    return report_manifest


def write_pipeline_summary(root: Path, run_dir: Path, config: dict[str, Any], stage: str, payload: dict[str, Any], smoke: bool) -> None:
    path = run_dir / "pipeline_summary.json"
    rank = {"matching": 1, "response": 2, "statistics": 3, "report": 4}
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    if int(existing.get("stage_rank", 0)) > rank[stage]:
        return
    summary = {
        **existing,
        "status": payload.get("status", "IN_PROGRESS"),
        "run_id": config["run_id"],
        "checkpoint": payload.get("checkpoint", stage),
        "last_completed_stage": stage,
        "stage_rank": rank[stage],
        "smoke": bool(smoke),
        "bioinformatics_isolation": True,
    }
    atomic_write_json(path, summary)


def write_final_manifest(root: Path, config_path: Path, run_dir: Path) -> None:
    paths = [
        config_path,
        run_dir / "matching" / "matching_manifest.json",
        run_dir / "response" / "response_manifest.json",
        run_dir / "statistics" / "receipt.json",
        run_dir / "statistics" / "statistics_manifest.json",
        run_dir / "report_manifest.json",
        run_dir / "pipeline_summary.json",
    ]
    report_manifest = json.loads((run_dir / "report_manifest.json").read_text(encoding="utf-8"))
    paths.append(root / str(report_manifest["report_draft"]["path"]))
    for figure in report_manifest.get("figures", []):
        paths.append(root / str(figure["path"]))
    manifest_path = run_dir / "MANIFEST_SHA256.txt"
    lines = [f"{file_sha256(path)}  {relative_path(root, path)}" for path in paths]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--stage", choices=["matching", "response", "statistics", "report", "all"], default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = read_config(config_path)
    validate_config(root, config_path, config)
    run_dir = (root / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    stages = ["matching", "response", "statistics", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "matching":
            result = run_matching_stage(root, config_path, config, run_dir, args.smoke, args.resume)
        elif stage == "response":
            result = run_response_stage(root, config_path, config, run_dir, args.smoke, args.resume)
        elif stage == "statistics":
            result = run_statistics_stage(root, config_path, config, run_dir, args.resume)
        else:
            result = write_report_stage(root, config_path, config, run_dir, args.smoke, args.resume)
        write_pipeline_summary(root, run_dir, config, stage, result, args.smoke)
    if args.stage == "all":
        write_final_manifest(root, config_path, run_dir)
    print(json.dumps(json.loads((run_dir / "pipeline_summary.json").read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
