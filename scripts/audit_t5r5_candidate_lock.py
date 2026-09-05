#!/usr/bin/env python3
"""Audit the T5R5 candidate lock before any reserved read (§12).

Asserts: selected candidate exactly 2:1; all 9 frozen checkpoint hashes;
config/report/protocol hashes; split IDs; no reserved read (reserved output
absent, no reserved task arrays, T5R2 firewall flags still false); no
external/old-heldout use; no Round 3; semantics addendum active; hidden task
rules equal T5R2; intervention protocol frozen; T5R6 shadow contract frozen;
P4 blocked.

Use --allow-reserved for a post-hoc lock-integrity recheck after the single
authorized hidden read (skips only the reserved-absence assertion).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
LOCK_PATH = T5R_ROOT / "candidate_lock.json"
HIDDEN_CONFIG = ROOT / "configs" / "t5r5_hidden_confirmation.yaml"
INTERVENTION_LOCK = T5R_ROOT / "t5r5_intervention_lock.json"
SHADOW_LOCK = T5R_ROOT / "t5r6_shadow_confirmation_lock.json"
RESERVED_DIR = T5R_ROOT / "t5r5_reserved_j03wqq_v1"
T5R2_THRESHOLDS = {
    "positive_internal_max": 0.28,
    "positive_centroid_min": 0.35,
    "hard_negative_centroid_max": 0.12,
    "hard_negative_internal_min": 0.35,
    "minimum_time_separation_s": 4.0,
    "pair_anchor_stride": 5,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-reserved", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    require(LOCK_PATH.is_file(), "candidate_lock.json missing", failures)
    require(HIDDEN_CONFIG.is_file(), "t5r5_hidden_confirmation.yaml missing", failures)
    require(INTERVENTION_LOCK.is_file(), "t5r5_intervention_lock.json missing", failures)
    require(SHADOW_LOCK.is_file(), "t5r6_shadow_confirmation_lock.json missing", failures)
    if failures:
        print("T5R5_LOCK_AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = yaml.safe_load(HIDDEN_CONFIG.read_text(encoding="utf-8"))
    intervention = json.loads(INTERVENTION_LOCK.read_text(encoding="utf-8"))
    shadow = json.loads(SHADOW_LOCK.read_text(encoding="utf-8"))

    # Selected candidate exactly 2:1.
    require(lock.get("status") == "T5R5_CANDIDATE_LOCKED", "lock status mismatch", failures)
    require(lock.get("identity", {}).get("selected_candidate") == "update_ratio_2to1", "selected candidate is not 2:1", failures)
    require(config.get("selected_candidate") == "update_ratio_2to1", "hidden config candidate mismatch", failures)
    schedule = lock.get("model", {}).get("update_schedule_per_epoch", {})
    require(schedule == {"context": 280, "intrinsic": 140, "total": 420}, "update schedule mismatch", failures)
    require(lock.get("model", {}).get("parameter_count") == 141769, "parameter count mismatch", failures)

    # Checkpoint hashes match live files (all 9 frozen checkpoints).
    checked = 0
    for seed, record in lock.get("model", {}).get("checkpoints", {}).items():
        path = ROOT / record["path"]
        require(path.is_file(), f"missing selected checkpoint seed {seed}", failures)
        if path.is_file():
            require(sha256_file(path) == record["sha256"], f"selected checkpoint hash mismatch seed {seed}", failures)
            checked += 1
    for group in ("reference", "raw_baseline"):
        for seed, record in lock.get("frozen_comparison_set", {}).get(group, {}).get("checkpoints", {}).items():
            path = ROOT / record["path"]
            require(path.is_file(), f"missing {group} checkpoint seed {seed}", failures)
            if path.is_file():
                require(sha256_file(path) == record["sha256"], f"{group} checkpoint hash mismatch seed {seed}", failures)
                checked += 1
    require(checked == 9, f"expected 9 checkpoints, checked {checked}", failures)

    # Config/report/protocol hashes match.
    for key, field in (
        ("t5r4_config_sha256", "configs/t5r4_round2.yaml"),
        ("t5r4_report_sha256", "reports/P3_T5R4_ROUND2_REPORT.md"),
        ("t5r4_protocol_sha256", "artifacts/phase3/task_semantic_repair_v1/t5r4_round2_v1/protocol.json"),
    ):
        live = sha256_file(ROOT / field)
        require(lock.get("identity", {}).get(key) == live, f"lock hash mismatch for {field}", failures)
    require(config.get("candidate_lock_sha256") == sha256_file(LOCK_PATH), "hidden config lock-hash mismatch", failures)
    require(config.get("intervention_lock_sha256") == sha256_file(INTERVENTION_LOCK), "hidden config intervention-hash mismatch", failures)
    require(config.get("shadow_lock_sha256") == sha256_file(SHADOW_LOCK), "hidden config shadow-hash mismatch", failures)

    # Split IDs match.
    split_lock = json.loads((T5R_ROOT / "split_lock.json").read_text(encoding="utf-8"))
    require(split_lock.get("split_map", {}).get("J03WQQ") == "reserved_holdout", "split map reserved mismatch", failures)
    require(lock.get("data", {}).get("reserved_match_ids") == ["J03WQQ"], "lock reserved IDs mismatch", failures)
    require(lock.get("data", {}).get("split_lock_sha256") == sha256_file(T5R_ROOT / "split_lock.json"), "split lock hash mismatch", failures)

    # No reserved read before lock.
    if not args.allow_reserved:
        require(not RESERVED_DIR.exists(), "reserved output already exists before lock", failures)
        require(not list((T5R_ROOT / "data_views").glob("*reserved*")), "reserved task arrays already written", failures)
    t5r2_lock = json.loads((T5R_ROOT / "baseline_and_metric_lock.json").read_text(encoding="utf-8"))
    require(t5r2_lock.get("firewall", {}).get("reserved_holdout_task_arrays_written") is False, "T5R2 firewall flag changed", failures)
    task_manifest = json.loads((T5R_ROOT / "task_manifest.json").read_text(encoding="utf-8"))
    require(task_manifest.get("holdout_firewall", {}).get("reserved_match_loaded") is False, "task manifest firewall flag changed", failures)

    # No external/old-heldout use; no Round 3.
    firewall = lock.get("firewall", {})
    require(firewall.get("old_heldout_for_selection") == "forbidden", "old-heldout firewall missing", failures)
    require(firewall.get("external_results_for_selection") == "forbidden", "external firewall missing", failures)
    require(firewall.get("P4_AMR") == "blocked", "P4 not blocked in lock", failures)
    require(firewall.get("round_3") == "forbidden", "round-3 firewall missing", failures)
    require(not (ROOT / "configs" / "t5r4_round3.yaml").exists(), "round-3 config exists", failures)
    require(not list((T5R_ROOT).glob("t5r4_round3*")), "round-3 artifacts exist", failures)
    require(not list(ROOT.glob("artifacts/phase3/*external*")), "external artifacts exist", failures)

    # Semantics addendum active.
    addendum = json.loads((T5R_ROOT / "metric_semantics_addendum.json").read_text(encoding="utf-8"))
    require(addendum.get("status") == "ACTIVE_SEMANTIC_ADDENDUM", "addendum not active", failures)
    require(lock.get("task_semantics", {}).get("addendum_sha256") == sha256_file(T5R_ROOT / "metric_semantics_addendum.json"), "addendum hash mismatch", failures)

    # Hidden task rules equal T5R2.
    rules = lock.get("hidden_task_generation", {})
    for key, expected in T5R2_THRESHOLDS.items():
        require(rules.get("natural_pair_thresholds", {}).get(key) == expected, f"hidden rule differs from T5R2: {key}", failures)
    require(t5r2_lock.get("intrinsic_task", {}).get("minimum_time_separation_seconds") == 4.0, "T5R2 separation mismatch", failures)

    # Intervention protocol frozen.
    require(intervention.get("status") == "T5R5_INTERVENTION_LOCKED", "intervention lock status mismatch", failures)
    require(intervention.get("perturbation_magnitude", {}).get("epsilon") == 0.25, "epsilon is not 0.25", failures)
    require(intervention.get("matched_pair_construction", {}).get("support_size") == 4, "support size is not 4", failures)
    require(intervention.get("snapshot_sampling", {}).get("max_snapshots") == 250, "max snapshots is not 250", failures)
    require(config.get("intervention", {}).get("epsilon") == 0.25, "hidden config epsilon mismatch", failures)

    # T5R6 shadow contract frozen.
    require(shadow.get("status") == "T5R6_SHADOW_LOCKED", "shadow lock status mismatch", failures)

    if failures:
        print("T5R5_LOCK_AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("T5R5_LOCK_AUDIT: PASS")
    print("- selected=update_ratio_2to1")
    print(f"- checkpoints={checked}/9 hash-verified")
    print("- reserved_unread_before_lock=" + str(not RESERVED_DIR.exists()))
    print("- round_3_absent=true")
    print("- p4_blocked=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
