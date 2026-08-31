#!/usr/bin/env python3
"""Run the formal, sharded P2 rigid-intervention evidence pipeline."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import p2_statistics as stats  # noqa: E402
from p2_matching_diagnostics import assert_response_blind_schema  # noqa: E402
from p2_point_model_adapter import file_sha256, load_point_adapter  # noqa: E402
from run_phase2_pipeline import (  # noqa: E402
    Sample,
    load_samples,
    role_coalitions,
    run_matching,
    validate_matching,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "phase2_rigid_formal_v2.yaml"
EXPECTED_CONTROLS = (
    "arbitrary_same_team",
    "topology_frequency_matched",
    "spectrum_exact_sign_randomized",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(encoded)


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def validate_locked_files(
    root: Path,
    locked: dict[str, Any],
    label: str,
) -> dict[str, dict[str, str]]:
    if not isinstance(locked, dict) or not locked:
        raise ValueError(f"{label} lock must be a non-empty mapping")
    observed: dict[str, dict[str, str]] = {}
    for name, item in sorted(locked.items()):
        if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
            raise ValueError(f"invalid {label} lock entry: {name}")
        path = _resolve(root, str(item["path"]))
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"{label} path escapes project root: {path}") from exc
        expected = str(item["sha256"])
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"{label} SHA256 mismatch for {name}: expected {expected}, observed {actual}"
            )
        observed[str(name)] = {"path": str(item["path"]), "sha256": actual}
    return observed


def current_environment_lock(root: Path) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in ["numpy", "pandas", "scipy", "torch", "pyarrow", "PyYAML"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "NOT_INSTALLED"
    requirement_paths = ["requirements-core.txt", "requirements-models.txt"]
    return {
        "python": platform.python_version(),
        "packages": packages,
        "requirements": {
            path: file_sha256(root / path)
            for path in requirement_paths
        },
    }


def provenance_fields(formal: dict[str, Any]) -> dict[str, str]:
    return {
        "source_code_manifest_sha256": json_sha256(formal["source_code"]),
        "environment_lock_sha256": json_sha256(formal["environment_lock"]),
    }


def load_formal_config(root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    config_path = config_path.resolve()
    formal = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(formal, dict):
        raise ValueError("formal P2 config must be a mapping")
    parent = formal.get("parent_matching_config", {})
    if not isinstance(parent, dict) or "path" not in parent or "sha256" not in parent:
        raise ValueError("formal config must lock parent_matching_config path and sha256")
    parent_path = _resolve(root, str(parent["path"]))
    observed = file_sha256(parent_path)
    expected = str(parent["sha256"])
    if observed != expected:
        raise ValueError(
            f"parent matching config SHA256 mismatch: expected {expected}, observed {observed}"
        )
    matching = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    if not isinstance(matching, dict):
        raise ValueError("parent matching config must be a mapping")
    if formal.get("study_mode") != "exploratory_discovery":
        raise ValueError("formal run must preserve exploratory_discovery study mode")
    if formal.get("bioinformatics_isolation") is not True:
        raise ValueError("formal run must explicitly preserve bioinformatics isolation")
    if tuple(str(value) for value in matching.get("controls", [])) != EXPECTED_CONTROLS:
        raise ValueError("parent matcher must provide the frozen three-control design")
    observed_sources = validate_locked_files(root, formal.get("source_code", {}), label="source code")
    if observed_sources != formal["source_code"]:
        raise ValueError("source code lock must use normalized path/SHA256 records")
    observed_environment = current_environment_lock(root)
    if observed_environment != formal.get("environment_lock"):
        raise ValueError(
            f"environment lock mismatch: expected {formal.get('environment_lock')}, observed {observed_environment}"
        )
    return formal, matching


def group_samples_by_match(samples: Iterable[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for sample in samples:
        grouped.setdefault(str(sample.match_id), []).append(sample)
    return {
        match_id: sorted(grouped[match_id], key=lambda item: (int(item.frame), str(item.sample_id)))
        for match_id in sorted(grouped)
    }


def _output_items(receipt: dict[str, Any]) -> list[tuple[Path, str]]:
    if isinstance(receipt.get("outputs"), dict):
        return [
            (Path(str(item["path"])), str(item["sha256"]))
            for item in receipt["outputs"].values()
        ]
    if "output_path" in receipt and "output_sha256" in receipt:
        return [(Path(str(receipt["output_path"])), str(receipt["output_sha256"]))]
    return []


def receipt_can_resume(receipt: dict[str, Any], expected: dict[str, Any]) -> bool:
    if receipt.get("status") != "completed":
        return False
    if any(receipt.get(key) != value for key, value in expected.items()):
        return False
    try:
        validate_receipt_outputs(receipt)
    except ValueError:
        return False
    return True


def validate_receipt_outputs(receipt: dict[str, Any]) -> None:
    if receipt.get("status") != "completed":
        raise ValueError("receipt status is not completed")
    outputs = receipt.get("outputs")
    if isinstance(outputs, dict) and outputs:
        for name, item in outputs.items():
            path = Path(str(item.get("path", "")))
            if not path.is_file():
                raise ValueError(f"receipt output is missing: {name}: {path}")
            expected_bytes = int(item.get("bytes", -1))
            if expected_bytes < 0 or path.stat().st_size != expected_bytes:
                raise ValueError(f"receipt output size mismatch: {name}: {path}")
            observed = file_sha256(path)
            if observed != str(item.get("sha256")):
                raise ValueError(f"receipt output hash mismatch: {name}: {path}")
        return
    legacy = _output_items(receipt)
    if not legacy:
        raise ValueError("receipt has no output records")
    for path, digest in legacy:
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"receipt output hash mismatch: {path}")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path.resolve()), "bytes": path.stat().st_size}


def _write_receipt(path: Path, core: dict[str, Any], outputs: dict[str, Path], runtime: dict[str, Any]) -> dict[str, Any]:
    payload = {
        **core,
        "status": "completed",
        "outputs": {name: _artifact_record(output) for name, output in outputs.items()},
        "runtime": runtime,
    }
    atomic_write_json(path, payload)
    return payload


def _canonical_hashes(root: Path) -> dict[str, str]:
    source = root / "artifacts" / "phase1"
    return {
        "canonical_npz_sha256": file_sha256(source / "canonical_samples.npz"),
        "canonical_jsonl_sha256": file_sha256(source / "canonical_samples.jsonl"),
    }


def _validate_full_selection(
    samples: list[Sample],
    formal: dict[str, Any],
    matching: dict[str, Any],
) -> tuple[dict[str, list[Sample]], int, int]:
    expected = formal["expected"]
    grouped = group_samples_by_match(samples)
    if len(samples) != int(expected["samples"]):
        raise ValueError(f"expected {expected['samples']} canonical samples, found {len(samples)}")
    if len(grouped) != int(expected["matches"]):
        raise ValueError(f"expected {expected['matches']} matches, found {len(grouped)}")
    wrong = {key: len(value) for key, value in grouped.items() if len(value) != int(expected["samples_per_match"])}
    if wrong:
        raise ValueError(f"unexpected samples per match: {wrong}")
    n_graphs = len(matching.get("graphs", []))
    n_energies = len(matching.get("energies", []))
    n_directions = len(matching.get("directions", []))
    planned_sets = sum(
        len(role_coalitions(sample)) * n_graphs * n_energies * n_directions
        for sample in samples
    )
    planned_arms = planned_sets * (1 + len(matching.get("controls", [])))
    if planned_sets != int(expected["matched_sets"]) or planned_arms != int(expected["arm_slots"]):
        raise ValueError(
            f"formal schedule drift: sets={planned_sets}, arms={planned_arms}, expected={expected}"
        )
    return grouped, planned_sets, planned_arms


def _matching_shard_paths(run_dir: Path, match_id: str) -> tuple[Path, dict[str, Path]]:
    shard_dir = run_dir / "matching" / "shards" / f"match_id={match_id}"
    outputs = {
        "arms": shard_dir / "matched_set_manifest.parquet",
        "sets": shard_dir / "matched_set_summary.parquet",
        "candidate_geometry": shard_dir / "candidate_geometry.parquet",
        "candidate_feasibility": shard_dir / "candidate_feasibility.parquet",
        "graph_manifest": shard_dir / "graph_manifest.json",
        "validation": shard_dir / "matching_validation.json",
    }
    return shard_dir / "receipt.json", outputs


def _require_inside(path: Path, parent: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(parent.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes expected directory: {resolved}") from exc
    return resolved


def validate_matching_chain(
    root: Path,
    config_path: Path,
    formal: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    manifest_path = run_dir / "matching" / "matching_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != formal["status_contract"]["matching_complete"]:
        raise ValueError("matching manifest status is not complete")
    expected_config_hash = file_sha256(config_path)
    if manifest.get("formal_config_sha256") != expected_config_hash:
        raise ValueError("matching manifest formal config hash mismatch")
    if manifest.get("parent_matching_config") != formal["parent_matching_config"]:
        raise ValueError("matching manifest parent config lock mismatch")
    if manifest.get("source_hashes") != _canonical_hashes(root):
        raise ValueError("matching canonical source hash mismatch")
    provenance = provenance_fields(formal)
    if any(manifest.get(key) != value for key, value in provenance.items()):
        raise ValueError("matching code/environment provenance mismatch")
    shards = manifest.get("shards", [])
    if len(shards) != int(formal["expected"]["matches"]):
        raise ValueError("matching manifest shard count mismatch")
    total_sets = 0
    total_arms = 0
    complete_sets = 0
    for item in shards:
        match_id = str(item["match_id"])
        expected_receipt_path, expected_outputs = _matching_shard_paths(run_dir, match_id)
        receipt_path = _require_inside(Path(str(item["receipt_path"])), run_dir, "matching receipt")
        if receipt_path != expected_receipt_path.resolve():
            raise ValueError(f"matching receipt path mismatch for {match_id}")
        if file_sha256(receipt_path) != str(item["receipt_sha256"]):
            raise ValueError(f"matching receipt index hash mismatch for {match_id}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_receipt_outputs(receipt)
        required = {
            "stage": "matching",
            "shard_id": f"match={match_id}",
            "match_id": match_id,
            "formal_config_sha256": expected_config_hash,
            "parent_matching_config_sha256": str(formal["parent_matching_config"]["sha256"]),
            **_canonical_hashes(root),
            **provenance,
        }
        if any(receipt.get(key) != value for key, value in required.items()):
            raise ValueError(f"matching receipt frozen-input mismatch for {match_id}")
        for name, path in expected_outputs.items():
            record = receipt["outputs"].get(name)
            if not record or Path(str(record["path"])).resolve() != path.resolve():
                raise ValueError(f"matching receipt output path mismatch: {match_id}/{name}")
        total_sets += int(receipt["runtime"]["n_sets"])
        total_arms += int(receipt["runtime"]["n_arms"])
        complete_sets += int(receipt["runtime"]["n_complete_sets"])
    if total_sets != int(formal["expected"]["matched_sets"]):
        raise ValueError("matching set count does not cover the formal schedule")
    if total_arms != int(formal["expected"]["arm_slots"]):
        raise ValueError("matching arm count does not cover the formal schedule")
    if (
        int(manifest.get("n_sets", -1)) != total_sets
        or int(manifest.get("n_arms", -1)) != total_arms
        or int(manifest.get("n_complete_sets", -1)) != complete_sets
    ):
        raise ValueError("matching manifest aggregate counts differ from receipts")
    return manifest


def run_matching_stage(
    root: Path,
    config_path: Path,
    formal: dict[str, Any],
    matching: dict[str, Any],
    run_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_path = run_dir / "matching" / "matching_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise RuntimeError("completed matching manifest exists; use --resume for immutable verification")
        return validate_matching_chain(root, config_path, formal, run_dir)
    samples = load_samples(root)
    grouped, planned_sets, planned_arms = _validate_full_selection(samples, formal, matching)
    config_hash = file_sha256(config_path)
    source_hashes = _canonical_hashes(root)
    parent_hash = str(formal["parent_matching_config"]["sha256"])
    provenance = provenance_fields(formal)
    receipts: list[dict[str, Any]] = []
    match_ids = list(grouped)
    for order, match_id in enumerate(match_ids):
        receipt_path, outputs = _matching_shard_paths(run_dir, match_id)
        shard_sets = sum(
            len(role_coalitions(sample))
            * len(matching["graphs"])
            * len(matching["energies"])
            * len(matching["directions"])
            for sample in grouped[match_id]
        )
        core = {
            "stage": "matching",
            "shard_id": f"match={match_id}",
            "match_id": match_id,
            "formal_config_sha256": config_hash,
            "parent_matching_config_sha256": parent_hash,
            **source_hashes,
            **provenance,
            "sample_ids_sha256": json_sha256([sample.sample_id for sample in grouped[match_id]]),
            "planned_sets": shard_sets,
            "planned_arm_slots": shard_sets * 4,
            "canary": order == 0,
        }
        if receipt_path.exists():
            observed = json.loads(receipt_path.read_text(encoding="utf-8"))
            if resume and receipt_can_resume(observed, core):
                receipts.append(observed)
                print(
                    json.dumps(
                        {"stage": "matching", "match_id": match_id, "status": "resumed"},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            if observed.get("status") == "completed":
                raise RuntimeError(f"completed matching shard cannot be overwritten: {match_id}")
        shard_started = time.perf_counter()
        result = run_matching(grouped[match_id], matching, collect_diagnostics=True)
        frame, set_frame, graph_manifest, candidate_geometry, candidate_feasibility = result
        assert_response_blind_schema(frame)
        assert_response_blind_schema(candidate_geometry)
        assert_response_blind_schema(candidate_feasibility)
        validation = validate_matching(frame, set_frame, graph_manifest, matching)
        if validation["status"] == "fail":
            raise RuntimeError(f"matching correctness failed for match {match_id}: {validation['errors']}")
        if len(set_frame) != shard_sets or len(frame) != shard_sets * 4:
            raise RuntimeError(
                f"matching schedule incomplete for match {match_id}: sets={len(set_frame)}/{shard_sets}, arms={len(frame)}/{shard_sets * 4}"
            )
        atomic_write_parquet(outputs["arms"], frame)
        atomic_write_parquet(outputs["sets"], set_frame)
        atomic_write_parquet(outputs["candidate_geometry"], candidate_geometry)
        atomic_write_parquet(outputs["candidate_feasibility"], candidate_feasibility)
        atomic_write_json(outputs["graph_manifest"], graph_manifest)
        atomic_write_json(outputs["validation"], validation)
        runtime = {
            "wall_seconds": float(time.perf_counter() - shard_started),
            "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
            "n_samples": len(grouped[match_id]),
            "n_sets": len(set_frame),
            "n_arms": len(frame),
            "n_complete_sets": int(set_frame["matched_set_status"].eq("COMPLETE").sum()),
        }
        receipts.append(_write_receipt(receipt_path, core, outputs, runtime))
        print(
            json.dumps(
                {
                    "stage": "matching",
                    "match_id": match_id,
                    "status": "completed",
                    "canary": order == 0,
                    "wall_seconds": runtime["wall_seconds"],
                    "peak_rss_mib": runtime["peak_rss_mib"],
                    "n_complete_sets": runtime["n_complete_sets"],
                    "n_sets": runtime["n_sets"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    n_sets = sum(int(item["runtime"]["n_sets"]) for item in receipts)
    n_arms = sum(int(item["runtime"]["n_arms"]) for item in receipts)
    if n_sets != planned_sets or n_arms != planned_arms:
        raise RuntimeError("matching shard receipts do not cover the complete formal schedule")
    shard_index = [
        {
            "match_id": item["match_id"],
            "receipt_path": str(_matching_shard_paths(run_dir, item["match_id"])[0].resolve()),
            "receipt_sha256": file_sha256(_matching_shard_paths(run_dir, item["match_id"])[0]),
            "n_sets": item["runtime"]["n_sets"],
            "n_arms": item["runtime"]["n_arms"],
            "n_complete_sets": item["runtime"]["n_complete_sets"],
        }
        for item in receipts
    ]
    manifest = {
        "status": formal["status_contract"]["matching_complete"],
        "checkpoint": "FORMAL_MATCHING_READY",
        "run_id": formal["run_id"],
        "formal_config_path": str(config_path.relative_to(root)),
        "formal_config_sha256": config_hash,
        "parent_matching_config": formal["parent_matching_config"],
        "source_hashes": source_hashes,
        "source_code": formal["source_code"],
        "environment_lock": formal["environment_lock"],
        **provenance,
        "n_samples": len(samples),
        "n_matches": len(grouped),
        "n_sets": n_sets,
        "n_arms": n_arms,
        "n_complete_sets": sum(int(item["runtime"]["n_complete_sets"]) for item in receipts),
        "shards": shard_index,
        "matching_manifest_hash_basis": json_sha256(shard_index),
        "model_inference": "NOT_RUN_IN_MATCHING_STAGE",
        "bioinformatics_isolation": "No bioinformatics index or data was read, updated, or rebuilt.",
        "wall_seconds": float(time.perf_counter() - started),
    }
    atomic_write_json(manifest_path, manifest)
    return validate_matching_chain(root, config_path, formal, run_dir)


def normalized_cosine_distance(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    first = np.asarray(before, dtype=np.float64)
    second = np.asarray(after, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("before and after embeddings must have equal 2D shape")
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    if np.any(first_norm <= 1e-12) or np.any(second_norm <= 1e-12):
        raise ValueError("cosine distance is undefined for zero-norm embeddings")
    similarity = np.sum(first * second, axis=1) / (first_norm * second_norm)
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def _load_canonical_arrays(root: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    source = root / "artifacts" / "phase1"
    metadata = [
        json.loads(line)
        for line in (source / "canonical_samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    with np.load(source / "canonical_samples.npz") as archive:
        positions = np.asarray(archive["positions"], dtype=np.float32)
        team_slots = np.asarray(archive["team_slots"], dtype=np.int64)
        adjacency = np.asarray(archive["adjacency"], dtype=np.float32)
    if len(metadata) != len(positions) or positions.shape[:2] != team_slots.shape:
        raise ValueError("canonical metadata and arrays are inconsistent")
    if adjacency.shape != (len(positions), positions.shape[1], positions.shape[1]):
        raise ValueError("canonical adjacency shape is inconsistent")
    return metadata, positions, team_slots, adjacency


def validate_model_baseline(
    root: Path,
    model_id: str,
    adapter: Any,
    model_receipt: dict[str, Any],
    embedding_manifest: pd.DataFrame,
    metadata: list[dict[str, Any]],
    positions: np.ndarray,
    team_slots: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, str]]:
    rows = embedding_manifest[
        embedding_manifest["model_id"].eq(model_id)
        & embedding_manifest["embedding_kind"].eq("baseline")
    ].sort_values("row_index")
    if len(rows) != len(metadata) or rows["sample_id"].duplicated().any():
        raise ValueError(f"baseline manifest is not one-to-one for {model_id}")
    expected_ids = [str(item["sample_id"]) for item in metadata]
    if rows["sample_id"].astype(str).tolist() != expected_ids:
        raise ValueError(f"baseline sample order mismatch for {model_id}")
    if rows["row_index"].astype(int).tolist() != list(range(len(metadata))):
        raise ValueError(f"baseline row indexes are not canonical for {model_id}")
    if set(rows["checkpoint_sha256"].astype(str)) != {str(model_receipt["checkpoint_sha256"])}:
        raise ValueError(f"baseline checkpoint hash mismatch for {model_id}")
    expected_contract = {
        "embedding_dim": 128,
        "dtype": "float32",
        "preprocess": "coordinate_normalized_xy_v1",
        "pooling": "team_mean",
    }
    for column, expected in expected_contract.items():
        if set(rows[column].tolist()) != {expected}:
            raise ValueError(f"baseline {column} mismatch for {model_id}")
    paths = set(rows["path"].astype(str))
    if len(paths) != 1:
        raise ValueError(f"baseline path is not unique for {model_id}")
    baseline_path = _resolve(root, next(iter(paths)))
    baseline = np.asarray(np.load(baseline_path), dtype=np.float32)
    if baseline.shape != (len(metadata), 128) or not np.isfinite(baseline).all():
        raise ValueError(f"invalid saved baseline array for {model_id}")
    recomputed = adapter.encode(
        positions,
        team_slots,
        adjacency=adjacency if adapter.uses_adjacency else None,
    )
    difference = np.abs(recomputed.astype(np.float64) - baseline.astype(np.float64))
    equivalent = bool(np.allclose(recomputed, baseline, rtol=1e-6, atol=5e-7))
    report = {
        "status": "PASS" if equivalent else "BLOCKED_BASELINE_PARITY",
        "model_id": model_id,
        "n_samples_checked": len(metadata),
        "shape": list(recomputed.shape),
        "max_abs_difference": float(difference.max()) if difference.size else 0.0,
        "mean_abs_difference": float(difference.mean()) if difference.size else 0.0,
        "absolute_tolerance": 5e-7,
        "relative_tolerance": 1e-6,
    }
    if not equivalent:
        raise RuntimeError(f"baseline parity failed for {model_id}: {report}")
    return baseline, report, {
        "baseline_embedding_path": str(baseline_path),
        "baseline_embedding_sha256": file_sha256(baseline_path),
    }


def build_model_response_rows(
    matching: pd.DataFrame,
    model_id: str,
    architecture: str,
    model_seed: int,
    baseline: np.ndarray,
    positions: np.ndarray,
    team_slots: np.ndarray,
    adjacency: np.ndarray,
    adapter: Any,
) -> pd.DataFrame:
    selected = matching[matching["valid"].astype(bool)].copy()
    if selected.empty:
        raise ValueError("matching shard has no valid arms")
    records: list[dict[str, Any]] = []
    batch_size = int(adapter.batch_size)
    retained = [
        "matched_set_id",
        "matched_set_status",
        "arm_kind",
        "control_family",
        "source_sample_id",
        "sample_index",
        "source_match_id",
        "split",
        "frame",
        "graph_id",
        "coalition_id",
        "team_slot",
        "role_group",
        "epsilon",
        "direction",
        "control_seed",
        "energy",
        "energy_error",
        "support_size",
        "band_power",
        "rayleigh_quotient_normalized",
        "component_count",
        "induced_density",
        "cut_weight",
        "topology_match_score",
        "selected_candidate_id",
    ]
    retained = [column for column in retained if column in selected.columns]
    for start in range(0, len(selected), batch_size):
        batch = selected.iloc[start : start + batch_size]
        sample_indices = batch["sample_index"].astype(int).to_numpy()
        moved: list[np.ndarray] = []
        for sample_index, encoded_delta in zip(sample_indices, batch["delta"]):
            delta = np.asarray(json.loads(str(encoded_delta)), dtype=np.float32)
            if delta.shape != positions[sample_index].shape or not np.isfinite(delta).all():
                raise ValueError("matching manifest contains an invalid delta")
            moved.append(positions[sample_index] + delta)
        after = adapter.encode(
            np.stack(moved),
            team_slots[sample_indices],
            adjacency=adjacency[sample_indices] if adapter.uses_adjacency else None,
        )
        distances = normalized_cosine_distance(baseline[sample_indices], after)
        for offset, (_, row) in enumerate(batch.iterrows()):
            epsilon = float(row["epsilon"])
            if epsilon <= 0:
                raise ValueError("epsilon must be positive")
            record = {column: row[column] for column in retained}
            record.update(
                {
                    "model_id": model_id,
                    "architecture": architecture,
                    "model_seed": int(model_seed),
                    "baseline_row_index": int(row["sample_index"]),
                    "response_distance": float(distances[offset]),
                    "normalized_response": float(distances[offset] / epsilon),
                }
            )
            records.append(record)
    return pd.DataFrame(records)


def validate_response_shard(response: pd.DataFrame, matching: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    selected = matching[matching["valid"].astype(bool)]
    key = ["matched_set_id", "arm_kind", "control_family"]
    if len(response) != len(selected):
        errors.append("response row count differs from valid matching arm row count")
    if response.duplicated(key).any():
        errors.append("duplicate response arm within matched set")
    if selected.duplicated(key).any():
        errors.append("duplicate valid matching arm within matched set")
    if not response.empty and not selected.empty:
        observed = set(map(tuple, response[key].astype(str).to_numpy()))
        expected = set(map(tuple, selected[key].astype(str).to_numpy()))
        if observed != expected:
            errors.append("response arm identities differ from valid matching arms")
    required_values = [column for column in ["response_distance", "normalized_response"] if column in response]
    if len(required_values) != 2:
        errors.append("response metric columns are missing")
        finite = False
    else:
        values = response[required_values].to_numpy(dtype=float)
        finite = bool(np.isfinite(values).all())
        if not finite:
            errors.append("response contains non-finite values")
        if len(response) and (
            (response["response_distance"] < -1e-7).any()
            or (response["response_distance"] > 2.0 + 1e-7).any()
        ):
            errors.append("cosine response is outside [0,2]")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "n_valid_matching_arms": int(len(selected)),
        "n_response_rows": int(len(response)),
        "all_finite": finite,
    }


def _response_shard_paths(run_dir: Path, model_id: str, match_id: str) -> tuple[Path, Path]:
    shard_dir = run_dir / "response" / "shards" / f"model_id={model_id}" / f"match_id={match_id}"
    return shard_dir / "receipt.json", shard_dir / "response.parquet"


def _load_model_specs(root: Path, requested: list[str]) -> dict[str, dict[str, Any]]:
    manifest_path = root / "artifacts" / "phase1" / "point_mainline" / "model_manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {str(row["model_id"]): row for row in rows}
    if set(requested) != set(by_id) or len(requested) != len(by_id):
        raise ValueError("formal model list must equal the frozen nine-model P1 manifest")
    return by_id


def validate_response_chain(
    root: Path,
    config_path: Path,
    formal: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    matching_manifest_path = run_dir / "matching" / "matching_manifest.json"
    matching_manifest = validate_matching_chain(root, config_path, formal, run_dir)
    response_manifest_path = run_dir / "response" / "response_manifest.json"
    manifest = json.loads(response_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != formal["status_contract"]["response_complete"]:
        raise ValueError("response manifest status is not complete")
    config_hash = file_sha256(config_path)
    matching_hash = file_sha256(matching_manifest_path)
    provenance = provenance_fields(formal)
    required_manifest = {
        "formal_config_sha256": config_hash,
        "matching_manifest_sha256": matching_hash,
        **provenance,
    }
    if any(manifest.get(key) != value for key, value in required_manifest.items()):
        raise ValueError("response manifest frozen-input/provenance mismatch")
    model_ids = [str(value) for value in formal["models"]]
    specs = _load_model_specs(root, model_ids)
    shards = manifest.get("shards", [])
    expected_shards = len(model_ids) * int(formal["expected"]["matches"])
    if len(shards) != expected_shards:
        raise ValueError("response manifest shard count mismatch")
    matching_by_id = {str(item["match_id"]): item for item in matching_manifest["shards"]}
    total_rows = 0
    seen: set[tuple[str, str]] = set()
    for item in shards:
        model_id = str(item["model_id"])
        match_id = str(item["match_id"])
        identity = (model_id, match_id)
        if identity in seen:
            raise ValueError(f"duplicate response shard index: {identity}")
        seen.add(identity)
        if model_id not in specs or match_id not in matching_by_id:
            raise ValueError(f"unknown response shard identity: {identity}")
        expected_receipt_path, expected_output_path = _response_shard_paths(run_dir, model_id, match_id)
        receipt_path = _require_inside(Path(str(item["receipt_path"])), run_dir, "response receipt")
        response_path = _require_inside(Path(str(item["response_path"])), run_dir, "response output")
        if receipt_path != expected_receipt_path.resolve() or response_path != expected_output_path.resolve():
            raise ValueError(f"response shard path mismatch: {identity}")
        if file_sha256(receipt_path) != str(item["receipt_sha256"]):
            raise ValueError(f"response receipt index hash mismatch: {identity}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_receipt_outputs(receipt)
        matching_receipt_path, matching_outputs = _matching_shard_paths(run_dir, match_id)
        spec = specs[model_id]
        required_receipt = {
            "stage": "response",
            "shard_id": f"model={model_id}:match={match_id}",
            "model_id": model_id,
            "architecture": str(spec["architecture"]),
            "model_seed": int(spec["seed"]),
            "match_id": match_id,
            "formal_config_sha256": config_hash,
            "matching_manifest_sha256": matching_hash,
            "matching_receipt_sha256": file_sha256(matching_receipt_path),
            "matching_arms_sha256": file_sha256(matching_outputs["arms"]),
            "checkpoint_sha256": str(spec["checkpoint_sha256"]),
            **provenance,
        }
        if any(receipt.get(key) != value for key, value in required_receipt.items()):
            raise ValueError(f"response receipt frozen-input mismatch: {identity}")
        output_record = receipt["outputs"].get("response")
        if not output_record or Path(str(output_record["path"])).resolve() != response_path:
            raise ValueError(f"response receipt output path mismatch: {identity}")
        if str(output_record["sha256"]) != str(item["response_sha256"]):
            raise ValueError(f"response manifest output hash mismatch: {identity}")
        checkpoint = _resolve(root, str(spec["path"]))
        if file_sha256(checkpoint) != str(spec["checkpoint_sha256"]):
            raise ValueError(f"response checkpoint changed: {model_id}")
        baseline_path = Path(str(receipt["baseline_embedding_path"]))
        if file_sha256(baseline_path) != str(receipt["baseline_embedding_sha256"]):
            raise ValueError(f"response baseline embedding changed: {model_id}")
        total_rows += int(receipt["runtime"]["n_response_rows"])
    if len(seen) != expected_shards or total_rows != int(manifest.get("n_response_rows", -1)):
        raise ValueError("response manifest coverage/count mismatch")
    return manifest


def run_response_stage(
    root: Path,
    config_path: Path,
    formal: dict[str, Any],
    run_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    matching_manifest_path = run_dir / "matching" / "matching_manifest.json"
    response_manifest_path = run_dir / "response" / "response_manifest.json"
    if response_manifest_path.exists():
        if not resume:
            raise RuntimeError("completed response manifest exists; use --resume for immutable verification")
        return validate_response_chain(root, config_path, formal, run_dir)
    matching_manifest = validate_matching_chain(root, config_path, formal, run_dir)
    matching_manifest_sha = file_sha256(matching_manifest_path)
    metadata, positions, team_slots, adjacency = _load_canonical_arrays(root)
    embedding_manifest = pd.read_parquet(
        root / "artifacts" / "phase1" / "point_mainline" / "embedding_manifest.parquet"
    )
    model_ids = [str(value) for value in formal["models"]]
    specs = _load_model_specs(root, model_ids)
    config_hash = file_sha256(config_path)
    provenance = provenance_fields(formal)
    batch_size = int(formal["runtime"]["batch_size"])
    receipts: list[dict[str, Any]] = []
    parity_reports: list[dict[str, Any]] = []
    for model_id in model_ids:
        spec = specs[model_id]
        adapter, model_receipt = load_point_adapter(root, model_id=model_id, batch_size=batch_size)
        baseline, parity, baseline_provenance = validate_model_baseline(
            root,
            model_id,
            adapter,
            model_receipt,
            embedding_manifest,
            metadata,
            positions,
            team_slots,
            adjacency,
        )
        parity_reports.append(parity)
        model_dir = run_dir / "response" / "models" / f"model_id={model_id}"
        atomic_write_json(model_dir / "model_receipt.json", {**model_receipt, **baseline_provenance})
        atomic_write_json(model_dir / "baseline_parity.json", parity)
        for shard in matching_manifest["shards"]:
            match_id = str(shard["match_id"])
            matching_receipt_path, matching_outputs = _matching_shard_paths(run_dir, match_id)
            matching_frame_path = matching_outputs["arms"]
            receipt_path, output_path = _response_shard_paths(run_dir, model_id, match_id)
            core = {
                "stage": "response",
                "shard_id": f"model={model_id}:match={match_id}",
                "model_id": model_id,
                "architecture": str(spec["architecture"]),
                "model_seed": int(spec["seed"]),
                "match_id": match_id,
                "formal_config_sha256": config_hash,
                "matching_manifest_sha256": matching_manifest_sha,
                "matching_receipt_sha256": file_sha256(matching_receipt_path),
                "matching_arms_sha256": file_sha256(matching_frame_path),
                "checkpoint_sha256": str(spec["checkpoint_sha256"]),
                "baseline_embedding_path": baseline_provenance["baseline_embedding_path"],
                "baseline_embedding_sha256": baseline_provenance["baseline_embedding_sha256"],
                **provenance,
            }
            if receipt_path.exists():
                observed = json.loads(receipt_path.read_text(encoding="utf-8"))
                if resume and receipt_can_resume(observed, core):
                    receipts.append(observed)
                    continue
                if observed.get("status") == "completed":
                    raise RuntimeError(f"completed response shard cannot be overwritten: {core['shard_id']}")
            shard_started = time.perf_counter()
            matching_frame = pd.read_parquet(matching_frame_path)
            response = build_model_response_rows(
                matching_frame,
                model_id=model_id,
                architecture=str(spec["architecture"]),
                model_seed=int(spec["seed"]),
                baseline=baseline,
                positions=positions,
                team_slots=team_slots,
                adjacency=adjacency,
                adapter=adapter,
            )
            validation = validate_response_shard(response, matching_frame)
            if validation["status"] != "pass":
                raise RuntimeError(f"response validation failed for {core['shard_id']}: {validation}")
            atomic_write_parquet(output_path, response)
            runtime = {
                "wall_seconds": float(time.perf_counter() - shard_started),
                "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
                "n_response_rows": len(response),
                "n_matched_sets": int(response["matched_set_id"].nunique()),
                "validation": validation,
            }
            receipts.append(_write_receipt(receipt_path, core, {"response": output_path}, runtime))
        print(
            json.dumps(
                {
                    "stage": "response",
                    "model_id": model_id,
                    "status": "completed_or_resumed",
                    "completed_shards": sum(item["model_id"] == model_id for item in receipts),
                    "baseline_parity": parity["status"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    expected_shards = len(model_ids) * int(formal["expected"]["matches"])
    if len(receipts) != expected_shards:
        raise RuntimeError(f"response receipts incomplete: {len(receipts)}/{expected_shards}")
    shard_index = [
        {
            "model_id": item["model_id"],
            "architecture": item["architecture"],
            "model_seed": item["model_seed"],
            "match_id": item["match_id"],
            "receipt_path": str(_response_shard_paths(run_dir, item["model_id"], item["match_id"])[0].resolve()),
            "receipt_sha256": file_sha256(_response_shard_paths(run_dir, item["model_id"], item["match_id"])[0]),
            "response_path": str(_response_shard_paths(run_dir, item["model_id"], item["match_id"])[1].resolve()),
            "response_sha256": item["outputs"]["response"]["sha256"],
            "n_response_rows": item["runtime"]["n_response_rows"],
        }
        for item in receipts
    ]
    manifest = {
        "status": formal["status_contract"]["response_complete"],
        "checkpoint": "RIGID_RESPONSE_COMPLETE",
        "run_id": formal["run_id"],
        "formal_config_sha256": config_hash,
        "matching_manifest_sha256": matching_manifest_sha,
        "source_code": formal["source_code"],
        "environment_lock": formal["environment_lock"],
        **provenance,
        "n_models": len(model_ids),
        "n_shards": len(shard_index),
        "n_response_rows": sum(int(item["n_response_rows"]) for item in shard_index),
        "baseline_parity": parity_reports,
        "shards": shard_index,
        "training": "NOT_RUN_FROZEN_MODEL_INFERENCE_ONLY",
        "wall_seconds": float(time.perf_counter() - started),
    }
    atomic_write_json(response_manifest_path, manifest)
    return validate_response_chain(root, config_path, formal, run_dir)


def _matching_coverage(run_dir: Path, matching_manifest: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage_parts: list[pd.DataFrame] = []
    failure_parts: list[pd.DataFrame] = []
    for shard in matching_manifest["shards"]:
        _, outputs = _matching_shard_paths(run_dir, str(shard["match_id"]))
        frame = pd.read_parquet(outputs["arms"])
        sets = frame.drop_duplicates("matched_set_id").copy()
        sets["complete"] = sets["matched_set_status"].eq("COMPLETE")
        coverage_parts.append(
            sets.groupby(
                ["source_match_id", "split", "graph_id", "role_group", "team_slot", "epsilon"],
                dropna=False,
            )
            .agg(n_sets=("matched_set_id", "size"), n_complete=("complete", "sum"))
            .reset_index()
        )
        invalid = frame[~frame["valid"].astype(bool)]
        if not invalid.empty:
            failure_parts.append(
                invalid.groupby(
                    ["source_match_id", "split", "graph_id", "role_group", "team_slot", "epsilon", "control_family", "failure_code"],
                    dropna=False,
                )
                .size()
                .rename("n_rows")
                .reset_index()
            )
    coverage = pd.concat(coverage_parts, ignore_index=True)
    coverage["complete_fraction"] = coverage["n_complete"] / coverage["n_sets"]
    failures = pd.concat(failure_parts, ignore_index=True) if failure_parts else pd.DataFrame()
    return coverage, failures


def validate_statistics_chain(
    root: Path,
    config_path: Path,
    formal: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    matching_manifest_path = run_dir / "matching" / "matching_manifest.json"
    response_manifest_path = run_dir / "response" / "response_manifest.json"
    validate_response_chain(root, config_path, formal, run_dir)
    statistics_dir = run_dir / "statistics"
    manifest_path = statistics_dir / "statistics_manifest.json"
    receipt_path = statistics_dir / "receipt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "RIGID_STATISTICS_COMPLETE":
        raise ValueError("statistics manifest status is not complete")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_receipt_outputs(receipt)
    provenance = provenance_fields(formal)
    required = {
        "stage": "statistics",
        "shard_id": "aggregate",
        "formal_config_sha256": file_sha256(config_path),
        "matching_manifest_sha256": file_sha256(matching_manifest_path),
        "response_manifest_sha256": file_sha256(response_manifest_path),
        "bootstrap_unit": "source_match_id",
        "bootstrap_replicates": int(formal["statistics"]["bootstrap_replicates"]),
        "bootstrap_seed": int(formal["statistics"]["bootstrap_seed"]),
        **provenance,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("statistics receipt frozen-input/provenance mismatch")
    if file_sha256(receipt_path) != str(manifest.get("receipt_sha256")):
        raise ValueError("statistics manifest receipt hash mismatch")
    if any(manifest.get(key) != value for key, value in provenance.items()):
        raise ValueError("statistics manifest code/environment provenance mismatch")
    pair_index_path = Path(str(receipt["outputs"]["pair_shard_index"]["path"]))
    pair_index = json.loads(pair_index_path.read_text(encoding="utf-8"))
    expected_pairs = len(formal["models"]) * int(formal["expected"]["matches"])
    if len(pair_index) != expected_pairs:
        raise ValueError("statistics paired shard index is incomplete")
    seen: set[tuple[str, str]] = set()
    for item in pair_index:
        identity = (str(item["model_id"]), str(item["match_id"]))
        if identity in seen:
            raise ValueError(f"duplicate paired shard index: {identity}")
        seen.add(identity)
        path = _require_inside(Path(str(item["path"])), run_dir, "paired effect shard")
        if file_sha256(path) != str(item["sha256"]):
            raise ValueError(f"paired effect shard hash mismatch: {identity}")
        if int(pq.ParquetFile(path).metadata.num_rows) != int(item["n_rows"]):
            raise ValueError(f"paired effect shard row count mismatch: {identity}")
    if sum(int(item["n_rows"]) for item in pair_index) != int(receipt["runtime"]["n_pair_rows"]):
        raise ValueError("statistics paired row total differs from receipt")
    return manifest


def run_statistics_stage(
    root: Path,
    config_path: Path,
    formal: dict[str, Any],
    run_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    response_manifest_path = run_dir / "response" / "response_manifest.json"
    matching_manifest_path = run_dir / "matching" / "matching_manifest.json"
    statistics_dir = run_dir / "statistics"
    statistics_manifest_path = statistics_dir / "statistics_manifest.json"
    statistics_receipt_path = statistics_dir / "receipt.json"
    if statistics_manifest_path.exists() or statistics_receipt_path.exists():
        if not resume:
            raise RuntimeError("completed statistics outputs exist; use --resume for immutable verification")
        return validate_statistics_chain(root, config_path, formal, run_dir)
    response_manifest = validate_response_chain(root, config_path, formal, run_dir)
    matching_manifest = validate_matching_chain(root, config_path, formal, run_dir)
    strict_l1 = float(formal["statistics"]["strict_band_power_l1"])
    model_match_parts: list[pd.DataFrame] = []
    pair_index: list[dict[str, Any]] = []
    for shard in response_manifest["shards"]:
        response_path = Path(str(shard["response_path"]))
        response = pd.read_parquet(response_path)
        pairs = stats.build_paired_effects(response, strict_band_power_l1=strict_l1)
        model_match, _ = stats.aggregate_match_effects(pairs)
        model_match_parts.append(model_match)
        pair_path = (
            statistics_dir
            / "paired_shards"
            / f"model_id={shard['model_id']}"
            / f"match_id={shard['match_id']}"
            / "paired_effects.parquet"
        )
        atomic_write_parquet(pair_path, pairs)
        pair_index.append(
            {
                "model_id": shard["model_id"],
                "match_id": shard["match_id"],
                "path": str(pair_path.resolve()),
                "sha256": file_sha256(pair_path),
                "n_rows": len(pairs),
            }
        )
    model_match = pd.concat(model_match_parts, ignore_index=True)
    architecture_match = stats.aggregate_architecture_matches(model_match)
    summary = stats.summarize_match_effects(
        architecture_match,
        n_bootstrap=int(formal["statistics"]["bootstrap_replicates"]),
        seed=int(formal["statistics"]["bootstrap_seed"]),
    )
    loo = stats.leave_one_match_out(architecture_match)
    splits = stats.split_descriptive(architecture_match)
    coverage, failures = _matching_coverage(run_dir, matching_manifest)
    outputs = {
        "model_seed_match_effects": statistics_dir / "model_seed_match_effects.parquet",
        "match_level_effects": statistics_dir / "match_level_effects.parquet",
        "scg_summary": statistics_dir / "scg_summary.parquet",
        "leave_one_match_out": statistics_dir / "leave_one_match_out.parquet",
        "split_descriptive": statistics_dir / "split_descriptive.parquet",
        "matching_coverage": statistics_dir / "matching_coverage.parquet",
        "failure_taxonomy": statistics_dir / "failure_taxonomy.parquet",
        "pair_shard_index": statistics_dir / "pair_shard_index.json",
    }
    atomic_write_parquet(outputs["model_seed_match_effects"], model_match)
    atomic_write_parquet(outputs["match_level_effects"], architecture_match)
    atomic_write_parquet(outputs["scg_summary"], summary)
    atomic_write_parquet(outputs["leave_one_match_out"], loo)
    atomic_write_parquet(outputs["split_descriptive"], splits)
    atomic_write_parquet(outputs["matching_coverage"], coverage)
    atomic_write_parquet(outputs["failure_taxonomy"], failures)
    atomic_write_json(outputs["pair_shard_index"], pair_index)
    receipt_core = {
        "stage": "statistics",
        "shard_id": "aggregate",
        "formal_config_sha256": file_sha256(config_path),
        "matching_manifest_sha256": file_sha256(matching_manifest_path),
        "response_manifest_sha256": file_sha256(response_manifest_path),
        "bootstrap_unit": "source_match_id",
        "bootstrap_replicates": int(formal["statistics"]["bootstrap_replicates"]),
        "bootstrap_seed": int(formal["statistics"]["bootstrap_seed"]),
        **provenance_fields(formal),
    }
    runtime = {
        "wall_seconds": float(time.perf_counter() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "n_pair_rows": sum(int(item["n_rows"]) for item in pair_index),
        "n_model_match_rows": len(model_match),
        "n_architecture_match_rows": len(architecture_match),
        "n_summary_rows": len(summary),
    }
    receipt = _write_receipt(statistics_dir / "receipt.json", receipt_core, outputs, runtime)
    manifest = {
        "status": "RIGID_STATISTICS_COMPLETE",
        "checkpoint": "RIGID_RESULTS_READY_FOR_REPORT",
        "run_id": formal["run_id"],
        "estimand": "normalized_scg=(semantic_response-control_response)/epsilon",
        "raw_sensitivity": "raw_scg=semantic_response-control_response",
        "controls_pooled": False,
        "architectures_pooled": False,
        "model_seeds_are_independent_matches": False,
        "support_modes": ["common_four", "pairwise", "strict_band_l1_020"],
        "energy_roles": formal["statistics"],
        "source_code": formal["source_code"],
        "environment_lock": formal["environment_lock"],
        **provenance_fields(formal),
        "receipt_sha256": file_sha256(statistics_dir / "receipt.json"),
        "runtime": runtime,
    }
    atomic_write_json(statistics_manifest_path, manifest)
    return validate_statistics_chain(root, config_path, formal, run_dir)


def _write_figures(run_dir: Path, summary: pd.DataFrame, architecture_match: pd.DataFrame, coverage: pd.DataFrame) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    main = summary[
        summary["support_mode"].eq("common_four")
        & summary["role_stratum"].eq("__overall__")
        & summary["epsilon"].astype(float).isin([0.25, 0.5])
    ].copy()
    for energy in [0.25, 0.5]:
        subset = main[np.isclose(main["epsilon"].astype(float), energy)]
        labels = [f"{row.architecture}\n{row.control_family}\n{row.graph_id}" for row in subset.itertuples()]
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.32), 5))
        ax.bar(np.arange(len(subset)), subset["mean_normalized_scg"].to_numpy(), color="#3569a8")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(subset)), labels, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("normalized SCG")
        ax.set_title(f"P2 rigid common-four response, energy={energy:g}")
        fig.tight_layout()
        path = figure_dir / f"scg_primary_energy_{energy:g}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    forest = architecture_match[
        architecture_match["support_mode"].eq("common_four")
        & architecture_match["role_stratum"].eq("__overall__")
        & architecture_match["control_family"].eq("topology_frequency_matched")
        & np.isclose(architecture_match["epsilon"].astype(float), 0.25)
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, (label, group) in enumerate(forest.groupby(["architecture", "graph_id"], sort=True)):
        x = np.arange(len(group)) + (index - 2.5) * 0.05
        ax.scatter(x, group["normalized_scg"], label=" / ".join(label), s=22)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(forest["source_match_id"].nunique()), sorted(forest["source_match_id"].astype(str).unique()), rotation=45)
    ax.set_ylabel("match-level normalized SCG")
    ax.set_title("Topology-frequency control: match-level effects at energy 0.25")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = figure_dir / "match_effect_forest.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    coverage_plot = coverage.groupby(["graph_id", "epsilon"], as_index=False).agg(n_sets=("n_sets", "sum"), n_complete=("n_complete", "sum"))
    coverage_plot["complete_fraction"] = coverage_plot["n_complete"] / coverage_plot["n_sets"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for graph_id, group in coverage_plot.groupby("graph_id", sort=True):
        ax.plot(group["epsilon"], group["complete_fraction"], marker="o", label=graph_id)
    ax.set_ylim(0, 1)
    ax.set_xlabel("energy")
    ax.set_ylabel("complete four-arm fraction")
    ax.set_title("Matching completeness")
    ax.legend()
    fig.tight_layout()
    path = figure_dir / "matching_coverage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def validate_report_chain(
    root: Path,
    config_path: Path,
    formal: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    statistics_manifest_path = run_dir / "statistics" / "statistics_manifest.json"
    validate_statistics_chain(root, config_path, formal, run_dir)
    manifest_path = run_dir / "report_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = {"RIGID_C2_RESULTS_READY_FOR_SCIENTIFIC_SYNTHESIS", "P2_RIGID_C2_COMPLETE"}
    if manifest.get("status") not in allowed:
        raise ValueError("report manifest status is not a recognized completed state")
    required = {
        "formal_config_sha256": file_sha256(config_path),
        "statistics_manifest_sha256": file_sha256(statistics_manifest_path),
        **provenance_fields(formal),
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("report manifest frozen-input/provenance mismatch")
    for item in [*manifest.get("figures", []), manifest.get("report_draft", {})]:
        path = _require_inside(Path(str(item.get("path", ""))), run_dir, "report artifact")
        if path.stat().st_size != int(item.get("bytes", -1)):
            raise ValueError(f"report artifact size mismatch: {path}")
        if file_sha256(path) != str(item.get("sha256")):
            raise ValueError(f"report artifact hash mismatch: {path}")
    return manifest


def run_report_stage(
    root: Path,
    config_path: Path,
    formal: dict[str, Any],
    run_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    statistics_manifest_path = run_dir / "statistics" / "statistics_manifest.json"
    report_manifest_path = run_dir / "report_manifest.json"
    if report_manifest_path.exists():
        if not resume:
            raise RuntimeError("completed report outputs exist; use --resume for immutable verification")
        return validate_report_chain(root, config_path, formal, run_dir)
    validate_statistics_chain(root, config_path, formal, run_dir)
    summary = pd.read_parquet(run_dir / "statistics" / "scg_summary.parquet")
    architecture_match = pd.read_parquet(run_dir / "statistics" / "match_level_effects.parquet")
    coverage = pd.read_parquet(run_dir / "statistics" / "matching_coverage.parquet")
    figures = _write_figures(run_dir, summary, architecture_match, coverage)
    primary = summary[
        summary["support_mode"].eq("common_four")
        & summary["role_stratum"].eq("__overall__")
        & summary["epsilon"].astype(float).isin([0.25, 0.5])
    ].copy()
    table = primary[
        [
            "architecture",
            "control_family",
            "graph_id",
            "epsilon",
            "n_matches",
            "mean_normalized_scg",
            "normalized_ci_low",
            "normalized_ci_high",
            "positive_match_fraction",
        ]
    ].to_string(index=False)
    report = f"""# P2 rigid formal report draft

Status: `RIGID_C2_RESULTS_READY_FOR_SCIENTIFIC_SYNTHESIS`

This run completed rigid matching, frozen-model response, and match-level statistics. The fracture operator remains `{formal['status_contract']['fracture']}` and is not part of these results.

## Primary low-energy common-support summary

```text
{table}
```

Controls, energies, probe graphs, and architectures remain separate. Bootstrap resampled the 10 matches; model seeds were averaged within architecture and were not treated as independent matches. Heldout split summaries are descriptive because it contains two matches.

## Interpretation boundary

Choose one evidence description after inspecting the complete tables and figures: `organization_signal_survives`, `frequency_bias_only`, `mixed_or_graph_specific`, or `matching_inadequate`. No numerical threshold is applied automatically.

## Figures

""" + "\n".join(f"- `{path.relative_to(run_dir)}`" for path in figures) + "\n"
    draft = run_dir / "report_draft.md"
    draft.parent.mkdir(parents=True, exist_ok=True)
    temporary = draft.with_name(f".{draft.name}.tmp.{os.getpid()}")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, draft)
    atomic_write_json(
        report_manifest_path,
        {
            "status": "RIGID_C2_RESULTS_READY_FOR_SCIENTIFIC_SYNTHESIS",
            "scientific_interpretation": "PENDING_HUMAN_AGENT_SYNTHESIS",
            "fracture": formal["status_contract"]["fracture"],
            "formal_config_sha256": file_sha256(config_path),
            "statistics_manifest_sha256": file_sha256(statistics_manifest_path),
            **provenance_fields(formal),
            "figures": [_artifact_record(path) for path in figures],
            "report_draft": _artifact_record(draft),
        },
    )
    return validate_report_chain(root, config_path, formal, run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=["matching", "response", "statistics", "report", "all"],
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    config_path = _resolve(root, args.config)
    formal, matching = load_formal_config(root, config_path)
    run_dir = root / "artifacts" / "phase2" / str(formal["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    stages = ["matching", "response", "statistics", "report"] if args.stage == "all" else [args.stage]
    results: dict[str, Any] = {}
    current_stage = "initialization"
    try:
        for current_stage in stages:
            if current_stage == "matching":
                results[current_stage] = run_matching_stage(
                    root, config_path, formal, matching, run_dir, args.resume
                )
            elif current_stage == "response":
                results[current_stage] = run_response_stage(
                    root, config_path, formal, run_dir, args.resume
                )
            elif current_stage == "statistics":
                results[current_stage] = run_statistics_stage(
                    root, config_path, formal, run_dir, args.resume
                )
            elif current_stage == "report":
                results[current_stage] = run_report_stage(
                    root, config_path, formal, run_dir, args.resume
                )
        atomic_write_json(
            run_dir / "pipeline_summary.json",
            {
                "status": results[stages[-1]]["status"],
                "last_completed_stage": stages[-1],
                "requested_stage": args.stage,
                "fracture": formal["status_contract"]["fracture"],
                "scientific_interpretation": (
                    "PENDING_HUMAN_AGENT_SYNTHESIS" if "report" in stages else "NOT_RUN"
                ),
                "bioinformatics_isolation": "No bioinformatics index or data was read, updated, or rebuilt.",
            },
        )
    except Exception as exc:
        attempt_path = run_dir / "attempts" / f"failed_{time.time_ns()}_{current_stage}.json"
        atomic_write_json(
            attempt_path,
            {
                "status": "INCOMPLETE",
                "failed_stage": current_stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scientific_interpretation": "NOT_RUN",
                "fracture": formal["status_contract"]["fracture"],
            },
        )
        raise
    print(json.dumps({"run_dir": str(run_dir), "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
