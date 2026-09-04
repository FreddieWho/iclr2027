#!/usr/bin/env python3
"""Structural audit for the bounded P3-T5R4 Round 2 output (§18)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1" / "t5r4_round2_v1"
DEFAULT_CONFIG = ROOT / "configs" / "t5r4_round2.yaml"
EXPECTED_SEEDS = {11, 23, 47}
EXPECTED_QUOTAS = {"update_ratio_3to1": (315, 105), "update_ratio_2to1": (280, 140)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run = args.run.resolve()
    failures: list[str] = []
    paths = {
        name: run / name
        for name in (
            "manifest.json",
            "protocol.json",
            "model_results.json",
            "summary.json",
            "comparison_vs_t5r3.json",
            "semantic_alias_snapshot.json",
            "SHA256SUMS",
        )
    }
    for name, path in paths.items():
        require(path.is_file(), f"missing artifact: {name}", failures)
    if failures:
        print("T5R4_ROUND2_AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    protocol = json.loads(paths["protocol.json"].read_text(encoding="utf-8"))
    records = json.loads(paths["model_results.json"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
    comparison = json.loads(paths["comparison_vs_t5r3.json"].read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))

    # 1-2: exactly the two authorized candidates with the exact ratios.
    candidate_ids = {candidate["id"] for candidate in protocol.get("candidates", [])}
    require(candidate_ids == set(EXPECTED_QUOTAS), f"candidate set mismatch: {sorted(candidate_ids)}", failures)
    for candidate in protocol.get("candidates", []):
        expected_ctx, expected_int = EXPECTED_QUOTAS[candidate["id"]]
        require(candidate.get("context_updates_per_epoch") == expected_ctx, f"context quota mismatch: {candidate['id']}", failures)
        require(candidate.get("intrinsic_updates_per_epoch") == expected_int, f"intrinsic quota mismatch: {candidate['id']}", failures)
    # 3-5: matched total budget and exact per-candidate splits.
    require(protocol.get("axis", {}).get("total_optimizer_steps_per_epoch") == 420, "total step budget is not 420", failures)
    for candidate_id, (expected_ctx, expected_int) in EXPECTED_QUOTAS.items():
        require(expected_ctx + expected_int == 420, f"quota arithmetic mismatch: {candidate_id}", failures)
    require(manifest.get("status") == "T5R4_ROUND2_COMPLETE", "manifest status mismatch", failures)
    require(protocol.get("status") == manifest.get("status"), "protocol/manifest status mismatch", failures)
    # 6-8: loss weights, routing, normalization frozen.
    for record in records:
        training = record.get("training", {})
        require(training.get("context_weight") == 1.0 and training.get("intrinsic_weight") == 1.0, f"loss weights not 1.0: {record.get('candidate_id')}/{record.get('seed')}", failures)
        require(training.get("gradient_routing") == "shared", f"routing not shared: {record.get('candidate_id')}/{record.get('seed')}", failures)
        require(training.get("head_readout_normalization") == "none", f"readout norm not none: {record.get('candidate_id')}/{record.get('seed')}", failures)
        quotas = EXPECTED_QUOTAS[str(record.get("candidate_id"))]
        require(training.get("context_updates_per_epoch") == quotas[0], "recorded context quota mismatch", failures)
        require(training.get("intrinsic_updates_per_epoch") == quotas[1], "recorded intrinsic quota mismatch", failures)
        require(training.get("total_updates_per_epoch") == 420, "recorded total step mismatch", failures)
        require(training.get("context_optimizer_steps_total") == quotas[0] * training.get("epochs", 0), "context step total mismatch", failures)
        require(training.get("intrinsic_optimizer_steps_total") == quotas[1] * training.get("epochs", 0), "intrinsic step total mismatch", failures)
    # 9: architecture/capacity matches the T5R3 reference.
    parameter_counts: set[int] = set()
    for record in records:
        parameter_counts.add(int(record.get("parameter_count", -1)))
        require(record.get("architecture") == "fixed-graph-message-passing-equivalent", "architecture mismatch", failures)
        require(record.get("pooling") == "team_mean", "pooling mismatch", failures)
    require(parameter_counts == {141769}, f"parameter counts not matched: {sorted(parameter_counts)}", failures)
    # 10-11: seeds and train/valid-only scope.
    require(protocol.get("seeds") == sorted(EXPECTED_SEEDS), "seed lock mismatch", failures)
    require(protocol.get("visible_splits") == ["train", "valid"], "visible split mismatch", failures)
    # 12-14: firewall flags.
    require(manifest.get("reserved_holdout_loaded") is False, "reserved holdout was loaded", failures)
    require(manifest.get("old_heldout_used") is False, "old heldout was used", failures)
    require(manifest.get("external_results_used") is False, "external results were used", failures)
    require(protocol.get("reserved_holdout", {}).get("loaded") is False, "protocol reserved firewall mismatch", failures)
    # 15: config/protocol/reference hash alignment.
    require(protocol.get("config_sha256") == sha256_file(args.config.resolve()), "config hash mismatch", failures)
    reference_summary = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1" / "t5r3_sanity_v3" / "summary.json"
    require(protocol.get("t5r3_reference_summary_sha256") == sha256_file(reference_summary), "reference summary hash mismatch", failures)
    t5r2_lock = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1" / "baseline_and_metric_lock.json"
    require(protocol.get("t5r2_lock_sha256") == sha256_file(t5r2_lock), "t5r2 lock hash mismatch", failures)
    require(protocol.get("config") == "configs/t5r4_round2.yaml", "config path mismatch", failures)
    # 16: output candidate IDs match the config.
    config_ids = {candidate["id"] for candidate in config.get("candidates", [])}
    require(candidate_ids == config_ids, "output/config candidate ID mismatch", failures)
    require({row.get("candidate_id") for row in summary} == config_ids, "summary candidate set mismatch", failures)
    require(len(records) == len(config_ids) * len(EXPECTED_SEEDS), "model result count mismatch", failures)
    observed = {(str(record.get("candidate_id")), int(record.get("seed", -1))) for record in records}
    require(observed == {(candidate_id, seed) for candidate_id in config_ids for seed in EXPECTED_SEEDS}, "candidate/seed grid mismatch", failures)
    # 17: translation evaluation keys unchanged from the T5R3 contract.
    for record in records:
        translation = record["valid"]["translation"]
        require("z_mode_global_translation_response" in translation, "z_mode translation key missing", failures)
        require(finite(translation["z_mode_global_translation_response"]), "non-finite translation response", failures)
        for value in (
            record["valid"]["context"]["phase_match_grouped_macro_f1"],
            record["valid"]["context"]["field_zone_match_grouped_macro_f1"],
            record["valid"]["context"]["centroid_match_grouped_mae"],
            record["valid"]["intrinsic"]["match_grouped_ranking_accuracy"],
            record["valid"]["intrinsic"]["natural_geometry_latent_distance_spearman"],
        ):
            require(finite(value), f"non-finite metric: {record.get('candidate_id')}/{record.get('seed')}", failures)
    # 18: metric alias and semantic addendum consistency.
    addendum = json.loads(paths["semantic_alias_snapshot.json"].read_text(encoding="utf-8"))
    require(addendum.get("status") == "ACTIVE_SEMANTIC_ADDENDUM", "semantic snapshot status mismatch", failures)
    for record in records:
        intrinsic = record["valid"]["intrinsic"]
        require("natural_geometry_latent_distance_spearman" in intrinsic, "preferred geometry key missing", failures)
        require("geometry_response_spearman" in intrinsic, "legacy geometry alias key missing", failures)
        require(intrinsic["natural_geometry_latent_distance_spearman"] == intrinsic["geometry_response_spearman"], "geometry alias value mismatch", failures)
    # 19-20: no reserved read and no P4 artifact.
    require(manifest.get("candidate_lock_created") is False, "candidate lock was created", failures)
    require(manifest.get("P4_release") is False, "P4 was released", failures)
    require(protocol.get("candidate_lock_created") is False, "protocol candidate-lock flag mismatch", failures)
    require(protocol.get("P4_release") is False, "protocol P4 flag mismatch", failures)
    require(not (ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1" / "candidate_lock.json").exists(), "candidate lock file exists", failures)
    for record in records:
        checkpoint = ROOT / record["checkpoint"]["path"]
        require(checkpoint.is_file(), f"missing checkpoint: {record.get('candidate_id')}/{record.get('seed')}", failures)
        if checkpoint.is_file():
            require(record["checkpoint"].get("sha256") == sha256_file(checkpoint), "checkpoint hash mismatch", failures)
    require(comparison.get("reference_variant") == "fixed_dual_channel_shared_phase_gat", "comparison reference mismatch", failures)

    if failures:
        print("T5R4_ROUND2_AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("T5R4_ROUND2_AUDIT: PASS")
    print(f"- candidates={len(candidate_ids)}")
    print(f"- models={len(records)}")
    print("- total_updates_per_epoch=420")
    print("- train_valid_only=true")
    print("- reserved_holdout=J03WQQ_not_loaded")
    print("- parameter_counts=matched")
    print("- p4_released=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
