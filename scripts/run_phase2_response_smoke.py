#!/usr/bin/env python3
"""Run the bounded P2-C DeepSets response wiring smoke on matched low-energy sets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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

from p2_point_model_adapter import file_sha256, load_deepsets_adapter  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCHING_DIR = ROOT / "artifacts" / "phase2" / "matching_v2_selected_smoke"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "phase2" / "p2c_deepsets_seed11_smoke"
MODEL_ID = "deepsets_ae_seed11"
EXPECTED_ARMS = {
    ("semantic_rigid", "anchor"),
    ("control", "arbitrary_same_team"),
    ("control", "topology_frequency_matched"),
    ("control", "spectrum_exact_sign_randomized"),
}


class BaselineParityError(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        super().__init__("saved P1 baseline embedding does not match recomputed baseline")
        self.report = report


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def select_complete_low_energy_sets(
    frame: pd.DataFrame,
    energies: Iterable[float] = (0.25, 0.5),
) -> pd.DataFrame:
    required = {
        "matched_set_id",
        "matched_set_status",
        "epsilon",
        "valid",
        "arm_kind",
        "control_family",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"matching manifest missing columns: {missing}")
    selected = frame[
        frame["matched_set_status"].eq("COMPLETE")
        & frame["epsilon"].astype(float).isin([float(value) for value in energies])
    ].copy()
    if selected.empty:
        raise ValueError("no complete low-energy matched sets")
    for matched_set_id, group in selected.groupby("matched_set_id", sort=False):
        arm_pairs = set(zip(group["arm_kind"].astype(str), group["control_family"].astype(str)))
        if len(group) != 4 or arm_pairs != EXPECTED_ARMS or not group["valid"].astype(bool).all():
            raise ValueError(f"complete set is not an exact valid four-arm set: {matched_set_id}")
    family_order = {
        "anchor": 0,
        "arbitrary_same_team": 1,
        "topology_frequency_matched": 2,
        "spectrum_exact_sign_randomized": 3,
    }
    selected["_family_order"] = selected["control_family"].map(family_order)
    if selected["_family_order"].isna().any():
        raise ValueError("four-arm selection contains an unknown control family")
    return selected.sort_values(["matched_set_id", "_family_order"]).drop(columns="_family_order").reset_index(drop=True)


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


def build_pipeline_summary(validation: dict[str, Any]) -> dict[str, Any]:
    passed = validation.get("status") == "pass"
    return {
        "status": "WIRING_ONLY_NOT_SCIENTIFIC" if passed else "BLOCKED_RESPONSE_VALIDATION",
        "checkpoint": "P2-C",
        "response_validation": validation,
        "scientific_interpretation": "NOT_RUN",
        "p2_completion": "NOT_CLAIMED",
        "next_stage": "P2-D resource/scientific gate; formal multi-model response remains NOT_RUN.",
    }


def _load_canonical(root: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    source = root / "artifacts" / "phase1"
    metadata = [json.loads(line) for line in (source / "canonical_samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    with np.load(source / "canonical_samples.npz") as archive:
        positions = np.asarray(archive["positions"], dtype=np.float32)
        team_slots = np.asarray(archive["team_slots"], dtype=np.int64)
    if len(metadata) != len(positions) or positions.shape[:2] != team_slots.shape:
        raise ValueError("P1 canonical metadata/array shape mismatch")
    return metadata, positions, team_slots


def validate_baseline_parity(
    root: Path,
    selected: pd.DataFrame,
    adapter,
    model_receipt: dict[str, Any],
    metadata: list[dict[str, Any]],
    positions: np.ndarray,
    team_slots: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    manifest_path = root / "artifacts" / "phase1" / "point_mainline" / "embedding_manifest.parquet"
    embedding_manifest = pd.read_parquet(manifest_path)
    rows = embedding_manifest[
        embedding_manifest["model_id"].eq(MODEL_ID)
        & embedding_manifest["embedding_kind"].eq("baseline")
    ].copy()
    if len(rows) != len(metadata) or rows["sample_id"].duplicated().any():
        raise ValueError("P1 baseline embedding manifest is not one-to-one with canonical samples")
    expected_contract = {
        "embedding_dim": 128,
        "dtype": "float32",
        "preprocess": "coordinate_normalized_xy_v1",
        "pooling": "team_mean",
        "checkpoint_sha256": model_receipt["checkpoint_sha256"],
    }
    for column, expected in expected_contract.items():
        values = set(rows[column].tolist())
        if values != {expected}:
            raise ValueError(f"baseline manifest {column} mismatch: {values}")
    paths = set(rows["path"].astype(str))
    if len(paths) != 1:
        raise ValueError(f"baseline manifest has multiple paths: {sorted(paths)}")
    baseline_path = (root / next(iter(paths))).resolve()
    baseline_path.relative_to(root)
    baseline = np.load(baseline_path)
    if baseline.shape != (len(metadata), 128) or baseline.dtype != np.float32 or not np.isfinite(baseline).all():
        raise ValueError(f"invalid saved baseline array: shape={baseline.shape}, dtype={baseline.dtype}")

    sample_rows = selected[["source_sample_id", "sample_index"]].drop_duplicates().sort_values("sample_index")
    sample_indices = sample_rows["sample_index"].astype(int).to_numpy()
    if np.any(sample_indices < 0) or np.any(sample_indices >= len(metadata)):
        raise ValueError("selected sample index is outside canonical arrays")
    by_sample = rows.set_index("sample_id")
    for source_sample_id, sample_index in sample_rows.itertuples(index=False):
        sample_index = int(sample_index)
        if str(metadata[sample_index]["sample_id"]) != str(source_sample_id):
            raise ValueError(f"canonical sample index mismatch for {source_sample_id}")
        manifest_row = by_sample.loc[str(source_sample_id)]
        if int(manifest_row["sample_index"]) != sample_index or int(manifest_row["row_index"]) != sample_index:
            raise ValueError(f"baseline row mapping mismatch for {source_sample_id}")

    recomputed = adapter.encode(positions[sample_indices], team_slots[sample_indices])
    saved = baseline[sample_indices]
    difference = np.abs(recomputed.astype(np.float64) - saved.astype(np.float64))
    exact = bool(np.array_equal(recomputed, saved))
    absolute_tolerance = 5e-7
    relative_tolerance = 1e-6
    numerically_equivalent = bool(
        np.allclose(
            recomputed,
            saved,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    )
    report = {
        "status": "PASS" if numerically_equivalent else "BLOCKED_BASELINE_PARITY",
        "n_samples_checked": int(len(sample_indices)),
        "sample_indices": sample_indices.tolist(),
        "shape": list(recomputed.shape),
        "exact_array_equal": exact,
        "float32_numerically_equivalent": numerically_equivalent,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "max_abs_difference": float(difference.max()) if difference.size else 0.0,
        "mean_abs_difference": float(difference.mean()) if difference.size else 0.0,
    }
    provenance = {
        "embedding_manifest_path": str(manifest_path.relative_to(root)),
        "embedding_manifest_sha256": file_sha256(manifest_path),
        "baseline_embedding_path": str(baseline_path.relative_to(root)),
        "baseline_embedding_sha256": file_sha256(baseline_path),
    }
    if not numerically_equivalent:
        raise BaselineParityError(report)
    return baseline, report, provenance


def build_response_rows(
    selected: pd.DataFrame,
    baseline: np.ndarray,
    positions: np.ndarray,
    team_slots: np.ndarray,
    adapter,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    batch_size = int(adapter.batch_size)
    for start in range(0, len(selected), batch_size):
        batch = selected.iloc[start : start + batch_size]
        sample_indices = batch["sample_index"].astype(int).to_numpy()
        moved: list[np.ndarray] = []
        for sample_index, encoded_delta in zip(sample_indices, batch["delta"]):
            delta = np.asarray(json.loads(str(encoded_delta)), dtype=np.float32)
            if delta.shape != positions[int(sample_index)].shape or not np.isfinite(delta).all():
                raise ValueError("matching manifest contains an invalid delta")
            moved.append(positions[int(sample_index)] + delta)
        after = adapter.encode(np.stack(moved), team_slots[sample_indices])
        before = baseline[sample_indices]
        distances = normalized_cosine_distance(before, after)
        for offset, (_, source_row) in enumerate(batch.iterrows()):
            row = source_row.to_dict()
            epsilon = float(row["epsilon"])
            records.append(
                {
                    "model_id": MODEL_ID,
                    "matched_set_id": str(row["matched_set_id"]),
                    "arm_kind": str(row["arm_kind"]),
                    "control_family": str(row["control_family"]),
                    "source_sample_id": str(row["source_sample_id"]),
                    "sample_index": int(row["sample_index"]),
                    "baseline_row_index": int(row["sample_index"]),
                    "source_match_id": str(row["source_match_id"]),
                    "graph_id": str(row["graph_id"]),
                    "coalition_id": str(row["coalition_id"]),
                    "team_slot": int(row["team_slot"]),
                    "role_group": str(row["role_group"]),
                    "epsilon": epsilon,
                    "direction": str(row["direction"]),
                    "response_distance": float(distances[offset]),
                    "normalized_response": float(distances[offset] / epsilon),
                }
            )
    return pd.DataFrame(records)


def validate_response(response: pd.DataFrame, selected: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    key = ["matched_set_id", "arm_kind", "control_family"]
    if len(response) != len(selected):
        errors.append("response row count differs from selected intervention rows")
    if response.duplicated(key).any():
        errors.append("duplicate response arm within matched set")
    values = response[["response_distance", "normalized_response"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        errors.append("response contains non-finite values")
    if len(response) and ((response["response_distance"] < -1e-7).any() or (response["response_distance"] > 2.0 + 1e-7).any()):
        errors.append("cosine response is outside [0,2]")
    sizes = response.groupby("matched_set_id").size()
    if not sizes.eq(4).all():
        errors.append("response does not preserve four-arm matched sets")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "n_response_rows": int(len(response)),
        "n_matched_sets": int(response["matched_set_id"].nunique()),
        "n_source_samples": int(response["source_sample_id"].nunique()),
        "energies": sorted(float(value) for value in response["epsilon"].unique()),
        "models": sorted(str(value) for value in response["model_id"].unique()),
        "all_finite": bool(np.isfinite(values).all()),
        "scientific_interpretation": "NOT_RUN",
    }


def validate_matching_source(root: Path, matching_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract_path = matching_dir / "execution_contract.json"
    summary_path = matching_dir / "pipeline_summary.json"
    manifest_path = matching_dir / "matched_set_manifest.parquet"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("matcher_decision") != "MATCHER_VERSION_SELECTED":
        raise ValueError("matching source is not marked MATCHER_VERSION_SELECTED")
    config_path = (root / str(contract["configuration_path"])).resolve()
    config_path.relative_to(root)
    observed_config_hash = file_sha256(config_path)
    if contract.get("configuration_sha256") != observed_config_hash:
        raise ValueError("matching configuration SHA256 mismatch")
    if yaml.safe_load(config_path.read_text(encoding="utf-8")) != contract.get("configuration"):
        raise ValueError("matching execution contract configuration differs from source YAML")
    frame = pd.read_parquet(manifest_path)
    provenance = {
        "matching_directory": str(matching_dir.relative_to(root)),
        "matching_execution_contract_path": str(contract_path.relative_to(root)),
        "matching_execution_contract_sha256": file_sha256(contract_path),
        "matching_pipeline_summary_path": str(summary_path.relative_to(root)),
        "matching_pipeline_summary_sha256": file_sha256(summary_path),
        "matching_manifest_path": str(manifest_path.relative_to(root)),
        "matching_manifest_sha256": file_sha256(manifest_path),
        "matching_configuration_path": str(config_path.relative_to(root)),
        "matching_configuration_sha256": observed_config_hash,
        "matcher_decision": summary["matcher_decision"],
    }
    return frame, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--matching-dir", type=Path, default=DEFAULT_MATCHING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    root = args.root.resolve()
    matching_dir = args.matching_dir if args.matching_dir.is_absolute() else root / args.matching_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    matching_dir = matching_dir.resolve()
    output_dir = output_dir.resolve()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty; use --overwrite explicitly: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        matching, source_provenance = validate_matching_source(root, matching_dir)
        selected = select_complete_low_energy_sets(matching, energies=(0.25, 0.5))
        metadata, positions, team_slots = _load_canonical(root)
        adapter, model_receipt = load_deepsets_adapter(root, batch_size=args.batch_size)
        baseline, parity, baseline_provenance = validate_baseline_parity(
            root,
            selected,
            adapter,
            model_receipt,
            metadata,
            positions,
            team_slots,
        )
        source_provenance.update(baseline_provenance)
        source_dir = root / "artifacts" / "phase1"
        source_provenance.update(
            {
                "canonical_samples_npz_sha256": file_sha256(source_dir / "canonical_samples.npz"),
                "canonical_samples_jsonl_sha256": file_sha256(source_dir / "canonical_samples.jsonl"),
            }
        )
        response = build_response_rows(selected, baseline, positions, team_slots, adapter)
        validation = validate_response(response, selected)
        summary = build_pipeline_summary(validation)
        contract = {
            "phase": "P2_RESPONSE_WIRING_SMOKE",
            "checkpoint": "P2-C",
            "study_mode": "exploratory_discovery",
            "data_domain": "football_tracking_point_set",
            "model_id": MODEL_ID,
            "energies": [0.25, 0.5],
            "n_selected_sets": int(selected["matched_set_id"].nunique()),
            "n_selected_arms": int(len(selected)),
            "training": "NOT_RUN",
            "model_comparison": "NOT_RUN",
            "scientific_interpretation": "NOT_RUN",
            "matcher_response_separation": "Matcher was selected before and without model outputs.",
            "bioinformatics_isolation": "No bioinformatics index or data was read, updated, or rebuilt.",
        }
        response.to_parquet(output_dir / "response.parquet", index=False)
        write_json(output_dir / "execution_contract.json", contract)
        write_json(output_dir / "source_provenance.json", source_provenance)
        write_json(output_dir / "model_receipt.json", model_receipt)
        write_json(output_dir / "baseline_parity.json", parity)
        write_json(output_dir / "response_validation.json", validation)
        write_json(output_dir / "pipeline_summary.json", summary)
        runtime = {
            "wall_seconds": float(time.perf_counter() - started),
            "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
            "batch_size": int(args.batch_size),
            "n_response_rows": int(len(response)),
            "rows_per_second": float(len(response) / max(time.perf_counter() - started, 1e-12)),
        }
        write_json(output_dir / "runtime.json", runtime)
        print(json.dumps({"output_dir": str(output_dir), "summary": summary, "runtime": runtime}, ensure_ascii=False, indent=2))
        return 0 if validation["status"] == "pass" else 1
    except BaselineParityError as exc:
        write_json(output_dir / "baseline_parity.json", exc.report)
        write_json(
            output_dir / "pipeline_summary.json",
            {
                "status": "BLOCKED_BASELINE_PARITY",
                "checkpoint": "P2-C",
                "scientific_interpretation": "NOT_RUN",
                "error": str(exc),
            },
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
