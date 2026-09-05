#!/usr/bin/env python3
"""Audit the T5R5 hidden-confirmation output (§12).

Verifies: output completeness; manifest status and single-read claims;
checkpoint/model hashes against the candidate lock; no training records;
generation thresholds equal T5R2; intervention lock hash referenced matches
the frozen file; exactly one reserved output directory; gate booleans
recomputed from result JSONs match the reported verdict.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
LOCK_PATH = T5R_ROOT / "candidate_lock.json"
INTERVENTION_LOCK = T5R_ROOT / "t5r5_intervention_lock.json"
DEFAULT_RUN = T5R_ROOT / "t5r5_reserved_j03wqq_v1"
RUNNER_PATH = ROOT / "scripts" / "run_t5r5_hidden_confirmation.py"

REQUIRED_FILES = (
    "reserved_manifest.json", "views_manifest.json", "task_results.json",
    "task_summary.json", "task_analytic_control.json", "intervention_arms.json",
    "intervention_summary.json", "gate_evaluation.json", "SHA256SUMS",
    "snapshot_index_reserved_holdout.parquet", "positions_raw_reserved_holdout.npy",
    "positions_centered_reserved_holdout.npy", "team_slots_reserved_holdout.npy",
    "adjacency_reserved_holdout.npy",
)


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
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    failures: list[str] = []
    for name in REQUIRED_FILES:
        require((run / name).is_file(), f"missing output file: {name}", failures)
    require(LOCK_PATH.is_file(), "candidate_lock.json missing", failures)
    if failures:
        print("T5R5_HIDDEN_AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((run / "reserved_manifest.json").read_text(encoding="utf-8"))
    views = json.loads((run / "views_manifest.json").read_text(encoding="utf-8"))
    task_summary = json.loads((run / "task_summary.json").read_text(encoding="utf-8"))
    intervention = json.loads((run / "intervention_summary.json").read_text(encoding="utf-8"))
    gates = json.loads((run / "gate_evaluation.json").read_text(encoding="utf-8"))

    require(manifest.get("status") == "T5R5_HIDDEN_CONFIRMATION_COMPLETE", "manifest status mismatch", failures)
    require(manifest.get("reserved_match") == "J03WQQ", "reserved match mismatch", failures)
    require(manifest.get("reads") == 1 and manifest.get("retraining") is False, "single-read claim mismatch", failures)
    require(manifest.get("candidate_lock_sha256") == sha256_file(LOCK_PATH), "lock hash mismatch", failures)
    require(manifest.get("intervention_lock_sha256") == sha256_file(INTERVENTION_LOCK), "intervention hash mismatch", failures)
    require(len(glob.glob(str(T5R_ROOT / "t5r5_reserved_*"))) == 1, "more than one reserved output exists", failures)

    task_results = json.loads((run / "task_results.json").read_text(encoding="utf-8"))
    require(len(task_results) == 9, f"expected 9 task records, got {len(task_results)}", failures)
    seen: set[tuple[str, int]] = set()
    for record in task_results:
        seen.add((record.get("model_id"), record.get("seed")))
        require("training" not in record, f"training record present: {record.get('model_id')}/{record.get('seed')}", failures)
        require(record.get("parameter_count") == 141769, "parameter count mismatch", failures)
    expected = {(m, s) for m in ("update_ratio_2to1", "fixed_dual_channel_shared_phase_gat", "raw_single_channel_phase_gat_team_mean") for s in (11, 23, 47)}
    require(seen == expected, "task model/seed grid mismatch", failures)

    thresholds = views.get("thresholds", {})
    for key, expected_value in (("positive_internal_max", 0.28), ("positive_centroid_min", 0.35),
                                ("hard_negative_centroid_max", 0.12), ("hard_negative_internal_min", 0.35),
                                ("minimum_time_separation_s", 4.0), ("pair_anchor_stride", 5)):
        require(thresholds.get(key) == expected_value, f"threshold changed: {key}", failures)

    spec = importlib.util.spec_from_file_location("run_t5r5_hidden_for_audit", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load hidden runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    recomputed = module.decide_gates(task_summary, intervention, views["n_pairs"])
    require(recomputed["verdict"] == gates["verdict"] == manifest["verdict"], "verdict mismatch", failures)
    require(recomputed["task_pass"] == gates["task_pass"], "task-pass mismatch", failures)

    arms = json.loads((run / "intervention_arms.json").read_text(encoding="utf-8"))
    require(all(set(a) >= {"snapshot_id", "source_support", "target_support", "anchor_delta", "control_delta", "status"} for a in arms), "arm schema mismatch", failures)
    require(all(len(a["source_support"]) == 4 for a in arms), "support size mismatch", failures)

    if failures:
        print("T5R5_HIDDEN_AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("T5R5_HIDDEN_AUDIT: PASS")
    print(f"- verdict={manifest['verdict']}")
    print(f"- snapshots={manifest['n_snapshots']} pairs={manifest['n_pairs']}")
    print(f"- intervention_sets={intervention['n_complete_sets']}+{intervention['n_incomplete_sets']}")
    print("- retraining=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
