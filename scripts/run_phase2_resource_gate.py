#!/usr/bin/env python3
"""Bounded P2-D CPU benchmark and resource projection; no formal response run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from p2_point_model_adapter import file_sha256, load_point_adapter  # noqa: E402
from run_phase2_response_smoke import select_complete_low_energy_sets  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCHING_DIR = ROOT / "artifacts" / "phase2" / "matching_v2_selected_smoke"
DEFAULT_P2C_DIR = ROOT / "artifacts" / "phase2" / "p2c_deepsets_seed11_smoke"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "phase2" / "p2d_resource_gate"
REPRESENTATIVE_MODELS = ("deepsets_ae_seed11", "gat_ae_seed11", "phase_gat_seed11")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def project_formal_resources(
    benchmarks: list[dict[str, Any]],
    source_samples: int,
    source_sets: int,
    target_samples: int,
    seeds_per_architecture: int,
    source_response_rows: int,
    source_response_bytes: int,
    observed_peak_rss_mib: float,
    source_end_to_end_seconds: float | None = None,
) -> dict[str, Any]:
    if min(source_samples, source_sets, target_samples, seeds_per_architecture, source_response_rows) <= 0:
        raise ValueError("resource projection counts must be positive")
    if len(benchmarks) != 3 or any(float(item["rows_per_second"]) <= 0 for item in benchmarks):
        raise ValueError("three positive architecture benchmarks are required")
    projected_sets = int(round(source_sets / source_samples * target_samples))
    rows_per_model = int(projected_sets * 4)
    total_models = int(len(benchmarks) * seeds_per_architecture)
    total_rows = int(rows_per_model * total_models)
    inference_seconds = float(
        sum(rows_per_model * seeds_per_architecture / float(item["rows_per_second"]) for item in benchmarks)
    )
    load_seconds = float(sum(float(item["load_seconds"]) * seeds_per_architecture for item in benchmarks))
    total_seconds = inference_seconds + load_seconds
    conservative_seconds = total_seconds
    if source_end_to_end_seconds is not None:
        if source_end_to_end_seconds <= 0:
            raise ValueError("source_end_to_end_seconds must be positive")
        conservative_seconds = max(total_seconds, float(source_end_to_end_seconds) * total_models)
    bytes_per_row = float(source_response_bytes / source_response_rows)
    projected_bytes = int(round(bytes_per_row * total_rows))
    return {
        "basis": "20-sample low-energy complete-set density plus bounded seed11 architecture throughput",
        "source_samples": int(source_samples),
        "source_sets": int(source_sets),
        "target_samples": int(target_samples),
        "architectures": len(benchmarks),
        "seeds_per_architecture": int(seeds_per_architecture),
        "projected_models": total_models,
        "projected_sets": projected_sets,
        "projected_rows_per_model": rows_per_model,
        "projected_total_response_rows": total_rows,
        "projected_inference_seconds": inference_seconds,
        "projected_model_load_seconds": load_seconds,
        "projected_total_seconds": total_seconds,
        "projected_total_minutes": total_seconds / 60.0,
        "projected_conservative_seconds": conservative_seconds,
        "projected_conservative_minutes": conservative_seconds / 60.0,
        "projected_response_bytes": projected_bytes,
        "projected_response_mib": projected_bytes / (1024.0 * 1024.0),
        "observed_peak_rss_mib": float(observed_peak_rss_mib),
        "execution_strategy": "sequential_models_local_cpu",
        "compute_decision": "LOCAL_CPU_SUFFICIENT_GPU_NOT_JUSTIFIED",
        "formal_run": "NOT_RUN",
        "risk_notes": [
            "The set-count projection assumes the 20-sample complete-set density generalizes to all 250 samples.",
            "Throughput uses one seed per architecture; other seeds should have similar shapes but were not benchmarked.",
            "Full 250-sample matching cost is not included in the response-inference projection.",
            "The conservative runtime floor multiplies the complete P2-C end-to-end wall time by nine models.",
            "Sequential model execution is required; multiplying peak RAM by nine would be incorrect.",
        ],
    }


def build_benchmark_inputs(
    selected: pd.DataFrame,
    positions: np.ndarray,
    team_slots: np.ndarray,
    adjacency: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    count = min(int(n_rows), len(selected))
    indexes = np.linspace(0, len(selected) - 1, count, dtype=int)
    subset = selected.iloc[indexes]
    moved: list[np.ndarray] = []
    sample_indices = subset["sample_index"].astype(int).to_numpy()
    for sample_index, encoded_delta in zip(sample_indices, subset["delta"]):
        delta = np.asarray(json.loads(str(encoded_delta)), dtype=np.float32)
        moved.append(positions[int(sample_index)] + delta)
    return (
        np.stack(moved),
        team_slots[sample_indices],
        adjacency[sample_indices],
        subset["matched_set_id"].astype(str).tolist(),
    )


def benchmark_models(
    root: Path,
    positions: np.ndarray,
    team_slots: np.ndarray,
    adjacency: np.ndarray,
    repeats: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    benchmarks: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for model_id in REPRESENTATIVE_MODELS:
        load_started = time.perf_counter()
        adapter, receipt = load_point_adapter(root, model_id=model_id, batch_size=batch_size)
        load_seconds = time.perf_counter() - load_started
        adapter.encode(positions[: min(64, len(positions))], team_slots[: min(64, len(positions))], adjacency[: min(64, len(positions))])
        timings: list[float] = []
        checksum = 0.0
        for _ in range(repeats):
            started = time.perf_counter()
            encoded = adapter.encode(positions, team_slots, adjacency)
            timings.append(time.perf_counter() - started)
            checksum += float(np.sum(encoded, dtype=np.float64))
        median_seconds = float(np.median(timings))
        benchmarks.append(
            {
                "model_id": model_id,
                "architecture": receipt["architecture"],
                "n_rows": int(len(positions)),
                "repeats": int(repeats),
                "batch_size": int(batch_size),
                "timings_seconds": timings,
                "median_seconds": median_seconds,
                "rows_per_second": float(len(positions) / median_seconds),
                "load_seconds": float(load_seconds),
                "embedding_shape": [int(len(positions)), 128],
                "finite": bool(np.isfinite(encoded).all()),
                "deterministic_checksum": checksum,
                "adjacency_source": "P1 canonical baseline_fixed_weighted_knn4",
            }
        )
        receipts.append(receipt)
    return benchmarks, receipts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--matching-dir", type=Path, default=DEFAULT_MATCHING_DIR)
    parser.add_argument("--p2c-dir", type=Path, default=DEFAULT_P2C_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--benchmark-rows", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    matching_dir = args.matching_dir if args.matching_dir.is_absolute() else root / args.matching_dir
    p2c_dir = args.p2c_dir if args.p2c_dir.is_absolute() else root / args.p2c_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    matching_dir = matching_dir.resolve()
    p2c_dir = p2c_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty; use --overwrite explicitly: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    p2c_summary = json.loads((p2c_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
    if p2c_summary.get("status") != "WIRING_ONLY_NOT_SCIENTIFIC":
        raise ValueError("P2-C source is not a completed wiring-only smoke")
    matching = pd.read_parquet(matching_dir / "matched_set_manifest.parquet")
    selected = select_complete_low_energy_sets(matching, energies=(0.25, 0.5))
    with np.load(root / "artifacts" / "phase1" / "canonical_samples.npz") as archive:
        canonical_positions = np.asarray(archive["positions"], dtype=np.float32)
        canonical_team_slots = np.asarray(archive["team_slots"], dtype=np.int64)
        canonical_adjacency = np.asarray(archive["adjacency"], dtype=np.float32)
    positions, team_slots, adjacency, set_ids = build_benchmark_inputs(
        selected,
        canonical_positions,
        canonical_team_slots,
        canonical_adjacency,
        n_rows=args.benchmark_rows,
    )
    benchmarks, receipts = benchmark_models(
        root,
        positions,
        team_slots,
        adjacency,
        repeats=args.repeats,
        batch_size=args.batch_size,
    )
    p2c_runtime = json.loads((p2c_dir / "runtime.json").read_text(encoding="utf-8"))
    p2c_response_path = p2c_dir / "response.parquet"
    projection = project_formal_resources(
        benchmarks=benchmarks,
        source_samples=int(p2c_summary["response_validation"]["n_source_samples"]),
        source_sets=int(p2c_summary["response_validation"]["n_matched_sets"]),
        target_samples=250,
        seeds_per_architecture=3,
        source_response_rows=int(p2c_summary["response_validation"]["n_response_rows"]),
        source_response_bytes=int(p2c_response_path.stat().st_size),
        observed_peak_rss_mib=max(
            float(p2c_runtime["peak_rss_mib"]),
            float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        ),
        source_end_to_end_seconds=float(p2c_runtime["wall_seconds"]),
    )
    benchmark_receipt = {
        "status": "PASS",
        "scope": "bounded_engineering_benchmark",
        "n_input_rows": int(len(positions)),
        "n_unique_matched_sets": len(set(set_ids)),
        "representative_models": list(REPRESENTATIVE_MODELS),
        "fixed_adjacency": "P1 canonical baseline_fixed_weighted_knn4",
        "benchmarks": benchmarks,
        "model_receipts": receipts,
        "scientific_response": "NOT_COMPUTED",
    }
    contract = {
        "phase": "P2_RESOURCE_GATE",
        "checkpoint": "P2-D_PRE_RUN_GATE",
        "study_mode": "exploratory_discovery",
        "execution_target": "local_cpu",
        "formal_run": "NOT_RUN",
        "benchmark_rows": int(len(positions)),
        "bioinformatics_isolation": "No bioinformatics index or data was read, updated, or rebuilt.",
    }
    provenance = {
        "matching_manifest_sha256": file_sha256(matching_dir / "matched_set_manifest.parquet"),
        "p2c_summary_sha256": file_sha256(p2c_dir / "pipeline_summary.json"),
        "p2c_runtime_sha256": file_sha256(p2c_dir / "runtime.json"),
        "p2c_response_sha256": file_sha256(p2c_response_path),
        "canonical_samples_npz_sha256": file_sha256(root / "artifacts" / "phase1" / "canonical_samples.npz"),
    }
    summary = {
        "status": "RESOURCE_GATE_CPU_SUFFICIENT_FORMAL_NOT_RUN",
        "checkpoint": "P2-D_PRE_RUN_GATE",
        "compute_decision": projection["compute_decision"],
        "formal_run": "NOT_RUN",
        "scientific_interpretation": "NOT_RUN",
        "scientific_gate": "REQUIRES_EXPLORATORY_ANALYSIS_DECISION",
        "next_decision": "Choose the first formal multi-model analysis scope and match-level aggregation before expanding computation.",
    }
    write_json(output_dir / "execution_contract.json", contract)
    write_json(output_dir / "source_provenance.json", provenance)
    write_json(output_dir / "benchmark_receipt.json", benchmark_receipt)
    write_json(output_dir / "resource_estimate.json", projection)
    write_json(output_dir / "pipeline_summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary, "projection": projection}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
