#!/usr/bin/env python3
"""Read-only audit for the completed P3-T5R3 sanity run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1" / "t5r3_sanity_v3"
LOCK_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
EXPECTED_VARIANTS = {
    "raw_single_channel_phase_gat_team_mean",
    "centered_single_channel_phase_gat_team_mean",
    "raw_relational_pooling_phase_gat",
    "fixed_dual_channel_shared_phase_gat",
}
EXPECTED_SEEDS = {11, 23, 47}
EXPECTED_MATCHES = {
    "train": {"J03WMX", "J03WOH", "J03WPY", "J03WR9"},
    "valid": {"J03WN1", "J03WOY"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def finite_metric(value: Any) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    failures: list[str] = []
    manifest_path = run / "manifest.json"
    protocol_path = run / "protocol.json"
    results_path = run / "model_results.json"
    summary_path = run / "summary.json"
    analytic_path = run / "analytic_control.json"
    for path in (manifest_path, protocol_path, results_path, summary_path, analytic_path):
        require(path.is_file(), f"missing T5R3 artifact: {path}", failures)
    if failures:
        print("T5R3_AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    records = json.loads(results_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    analytic = json.loads(analytic_path.read_text(encoding="utf-8"))
    baseline_lock_path = LOCK_ROOT / "baseline_and_metric_lock.json"
    baseline_lock = json.loads(baseline_lock_path.read_text(encoding="utf-8"))

    require(manifest.get("status") == "T5R3_FIXED_DUAL_CHANNEL_SANITY_COMPLETE", "manifest status mismatch", failures)
    require(protocol.get("status") == manifest.get("status"), "protocol/manifest status mismatch", failures)
    require(manifest.get("task_lane") == "P3-T5R3", "task lane mismatch", failures)
    require(manifest.get("model_count") == 12, "expected 12 neural model results", failures)
    require(manifest.get("variant_count") == 4, "expected 4 variants", failures)
    require(manifest.get("seed_count") == 3, "expected 3 seeds", failures)
    require(manifest.get("reserved_holdout_loaded") is False, "reserved holdout was loaded", failures)
    require(manifest.get("external_results_used") is False, "external results were used", failures)
    require(manifest.get("old_heldout_used") is False, "old heldout was used", failures)
    require(manifest.get("candidate_selection_performed") is False, "candidate selection was performed", failures)
    require(protocol.get("reserved_holdout", {}).get("loaded") is False, "protocol says reserved holdout loaded", failures)
    require(protocol.get("reserved_holdout", {}).get("task_arrays_written") is False, "protocol says reserved arrays written", failures)
    require(protocol.get("model_results_used_for_selection") is False, "model results were used for selection", failures)
    require(protocol.get("seeds") == sorted(EXPECTED_SEEDS), "seed lock mismatch", failures)
    require(protocol.get("t5r2_lock_status") == "T5R2_CLOSED_BASELINE_AND_METRIC_LOCK", "T5R2 lock status mismatch", failures)
    require(protocol.get("t5r2_lock_sha256") == sha256_file(baseline_lock_path), "T5R2 lock hash mismatch", failures)
    require(baseline_lock.get("firewall", {}).get("model_results_used") is False, "T5R2 firewall was changed", failures)

    require(len(records) == 12, f"model result length is {len(records)}", failures)
    observed_pairs: set[tuple[str, int]] = set()
    parameter_counts: set[int] = set()
    for record in records:
        variant = str(record.get("variant"))
        seed = int(record.get("seed", -1))
        observed_pairs.add((variant, seed))
        parameter_counts.add(int(record.get("parameter_count", -1)))
        require(variant in EXPECTED_VARIANTS, f"unexpected variant: {variant}", failures)
        require(seed in EXPECTED_SEEDS, f"unexpected seed: {seed}", failures)
        require(record.get("architecture") == "fixed-graph-message-passing-equivalent", f"architecture mismatch: {variant}/{seed}", failures)
        checkpoint_path = ROOT / record["checkpoint"]["path"]
        require(checkpoint_path.is_file(), f"checkpoint is missing: {variant}/{seed}", failures)
        if checkpoint_path.is_file():
            require(record.get("checkpoint", {}).get("sha256") == sha256_file(checkpoint_path), f"checkpoint hash mismatch: {variant}/{seed}", failures)
        for split in ("train", "valid"):
            context = record[split]["context"]
            intrinsic = record[split]["intrinsic"]
            translation = record[split]["translation"]
            for key in (
                "phase_match_grouped_macro_f1",
                "field_zone_match_grouped_macro_f1",
                "centroid_match_grouped_mae",
            ):
                require(finite_metric(context[key]), f"non-finite context metric {variant}/{seed}/{split}/{key}", failures)
            for key in ("match_grouped_ranking_accuracy", "match_grouped_mrr_at_2", "geometry_response_spearman"):
                require(finite_metric(intrinsic[key]), f"non-finite intrinsic metric {variant}/{seed}/{split}/{key}", failures)
            require(finite_metric(translation["z_mode_global_translation_response"]), f"non-finite translation metric {variant}/{seed}/{split}", failures)
            require(context["match_count"] == len(EXPECTED_MATCHES[split]), f"match count mismatch {variant}/{seed}/{split}", failures)
            require(context["sample_count"] == (23052 if split == "train" else 6127), f"snapshot count mismatch {variant}/{seed}/{split}", failures)
            require(intrinsic["pair_count"] == (3750 if split == "train" else 980), f"pair count mismatch {variant}/{seed}/{split}", failures)
    require(observed_pairs == {(variant, seed) for variant in EXPECTED_VARIANTS for seed in EXPECTED_SEEDS}, "variant/seed grid is incomplete or duplicated", failures)
    require(len(parameter_counts) == 1, f"neural parameter counts are not matched: {sorted(parameter_counts)}", failures)

    summary_variants = {row.get("variant") for row in summary}
    require(summary_variants == EXPECTED_VARIANTS, "summary variant set mismatch", failures)
    dual = next((row for row in summary if row.get("variant") == "fixed_dual_channel_shared_phase_gat"), None)
    raw = next((row for row in summary if row.get("variant") == "raw_single_channel_phase_gat_team_mean"), None)
    require(dual is not None and raw is not None, "raw or dual summary missing", failures)
    if dual is not None and raw is not None:
        require(dual["valid_phase_f1_mean"] - raw["valid_phase_f1_mean"] >= -0.05, "dual phase F1 exceeds non-inferiority drop margin", failures)
        require(dual["valid_centroid_mae_mean"] - raw["valid_centroid_mae_mean"] <= 0.05, "dual centroid MAE exceeds non-inferiority margin", failures)
        require(dual["valid_pair_accuracy_mean"] > raw["valid_pair_accuracy_mean"], "dual intrinsic ranking did not improve over raw", failures)
        require(dual["valid_z_mode_translation_response_mean"] <= 1e-5, "dual z_mode translation response is not near invariant", failures)
        require(dual["valid_geometry_relation_spearman_mean"] > 0.0, "dual natural geometry relation collapsed or reversed", failures)

    require(analytic.get("variant") == "raw_coordinate_procrustes", "analytic control identity mismatch", failures)
    require(analytic.get("valid", {}).get("procrustes_pair_count") == 980, "analytic valid control pair count mismatch", failures)

    if failures:
        print("T5R3_AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("T5R3_AUDIT: PASS")
    print("- model_grid=4_variants_x_3_seeds")
    print("- parameter_counts=matched")
    print("- reserved_holdout=J03WQQ_not_loaded")
    print("- candidate_selection=false")
    print("- dual_context_noninferiority=true")
    print("- dual_intrinsic_improvement=true")
    print("- dual_z_mode_translation_invariant=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
