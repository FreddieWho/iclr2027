#!/usr/bin/env python3
"""Fail-closed audit for the P3-T5R current-state protocol.

The audit is read-only: it parses project/config state and checks required
route markers and paths. It does not inspect external data or write reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def load_yaml(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        failures.append(f"YAML unreadable: {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        failures.append(f"YAML root is not a mapping: {path}")
        return {}
    return value


def load_json(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        failures.append(f"JSON unreadable: {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        failures.append(f"JSON root is not a mapping: {path}")
        return {}
    return value


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    failures: list[str] = []

    required_paths = [
        "README.md",
        "MASTER_AGENT_PROMPT.md",
        "STATUS.md",
        "DECISIONS.md",
        "CLAIM_LEDGER.md",
        "TODO.md",
        "PROJECT_PACKAGE_CONSOLIDATED.md",
        "QA.md",
        "docs/01_SCIENTIFIC_BLUEPRINT.md",
        "docs/02_METHOD_SPEC_AMR.md",
        "docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md",
        "docs/04_DATA_AND_ENVIRONMENT_GUIDE.md",
        "docs/05_AGENT_EXECUTION_MANUAL.md",
        "reports/P3_TASK_SEMANTIC_REPAIR_PLAN.md",
        "configs/project.yaml",
        "configs/experiment_matrix.yaml",
        "configs/data_manifest.yaml",
        "configs/phase3_task_semantic_repair_v1.yaml",
        "reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md",
        "scripts/audit_idsse_t5r2.py",
        "artifacts/data_v2/idsse/source_file_manifest.json",
        "artifacts/data_v2/idsse/RAW_SHA256SUMS",
        "artifacts/data_v2/idsse/canonical_manifest.json",
        "artifacts/data_v2/idsse/source_provenance.json",
        "artifacts/phase3/task_semantic_repair_v1/split_lock.json",
        "artifacts/phase3/task_semantic_repair_v1/task_manifest.json",
        "artifacts/phase3/task_semantic_repair_v1/baseline_and_metric_lock.json",
        "reports/P3_T5R3_SANITY_REPORT.md",
        "reports/P3_T5R3_CLOSURE_REVIEW.md",
        "scripts/audit_t5r3_sanity.py",
        "artifacts/phase3/task_semantic_repair_v1/t5r3_sanity_v3/manifest.json",
        "configs/t5r4_round1.yaml",
        "scripts/run_t5r4_round1.py",
        "scripts/audit_t5r4_round1.py",
        "artifacts/phase3/task_semantic_repair_v1/t5r4_round1_v1/manifest.json",
        "reports/P3_T5R4_ROUND1_REPORT.md",
        "artifacts/phase3/task_semantic_repair_v1/metric_semantics_addendum.json",
        "reports/P3_T5R4_ROUND1_REVIEW_AND_ROUND2_AUTHORIZATION.md",
        "configs/t5r4_round2.yaml",
        "scripts/run_t5r4_round2.py",
        "scripts/audit_t5r4_round2.py",
        "tests/test_t5r4_round2.py",
        "artifacts/phase3/task_semantic_repair_v1/t5r4_round2_v1/manifest.json",
        "reports/P3_T5R4_ROUND2_REPORT.md",
        "reports/P3_T5R4_TO_T5R5_CLOSURE_AUDIT.md",
        "scripts/compute_t5r5_prelock_audit.py",
        "artifacts/phase3/task_semantic_repair_v1/t5r5_prelock_audit_v1/manifest.json",
        "artifacts/phase3/task_semantic_repair_v1/candidate_lock.json",
        "artifacts/phase3/task_semantic_repair_v1/t5r5_intervention_lock.json",
        "artifacts/phase3/task_semantic_repair_v1/t5r6_shadow_confirmation_lock.json",
        "configs/t5r5_hidden_confirmation.yaml",
        "reports/P3_T5R5_CANDIDATE_LOCK.md",
        "scripts/audit_t5r5_candidate_lock.py",
        "scripts/run_t5r5_hidden_confirmation.py",
        "scripts/audit_t5r5_hidden_confirmation.py",
        "tests/test_t5r5_candidate_lock.py",
        "tests/test_t5r5_hidden_confirmation.py",
    ]
    for relative in required_paths:
        require((root / relative).is_file(), f"missing required path: {relative}", failures)

    project = load_yaml(root / "configs/project.yaml", failures)
    matrix = load_yaml(root / "configs/experiment_matrix.yaml", failures)
    data_manifest = load_yaml(root / "configs/data_manifest.yaml", failures)
    repair = load_yaml(root / "configs/phase3_task_semantic_repair_v1.yaml", failures)
    t5r2_lock = load_json(root / "artifacts/phase3/task_semantic_repair_v1/baseline_and_metric_lock.json", failures)
    split_lock = load_json(root / "artifacts/phase3/task_semantic_repair_v1/split_lock.json", failures)
    t5r3_manifest = load_json(root / "artifacts/phase3/task_semantic_repair_v1/t5r3_sanity_v3/manifest.json", failures)
    t5r4_manifest = load_json(root / "artifacts/phase3/task_semantic_repair_v1/t5r4_round1_v1/manifest.json", failures)
    t5r4r2_manifest = load_json(root / "artifacts/phase3/task_semantic_repair_v1/t5r4_round2_v1/manifest.json", failures)

    project_state = project.get("project", {})
    require(project_state.get("active_phase") == "P3_CAUSAL_MECHANISM", "active phase is not P3_CAUSAL_MECHANISM", failures)
    require(project_state.get("current_checkpoint") == "P3_T5R5_LOCKED", "project checkpoint mismatch", failures)
    require(project_state.get("current_route") == "support_conditioned_geometry_with_task_semantic_repair", "project route mismatch", failures)
    require(project_state.get("current_phase_status") == "P3_T5R5_CANDIDATE_LOCKED_SINGLE_READ_AUTHORIZED", "project phase status mismatch", failures)
    require(project_state.get("current_next_action") == "run_t5r5_single_hidden_read", "project next action mismatch", failures)
    require(project_state.get("p4_status") == "blocked_pending_p3_t5r_gate", "P4 is not blocked pending T5R gate", failures)
    require(project_state.get("p5_status") == "blocked_pending_sports_prediction_lock", "P5 status mismatch", failures)
    require(project_state.get("legacy_heldout_status") == "EXPOSED_DURING_CANDIDATE_SEARCH", "legacy heldout status mismatch", failures)
    repair_state = project.get("exploration", {}).get("p3_task_semantic_repair", {})
    require(repair_state.get("status") == "t5r5_locked_single_read_authorized", "T5R project status is not t5r5_locked_single_read_authorized", failures)
    require(repair_state.get("temporary_sngar_substitute") == "IDSSE", "temporary IDSSE substitution is missing", failures)

    require(matrix.get("p3_task_semantic_repair", {}).get("phase") == "P3_CAUSAL_MECHANISM", "T5R matrix is not inside P3", failures)
    require(matrix.get("p3_task_semantic_repair", {}).get("task_lane") == "P3-T5R", "T5R task lane missing", failures)
    require("C2.5" not in str(matrix), "forbidden C2.5 marker found", failures)

    policy = data_manifest.get("policy", {})
    require(policy.get("legacy_heldout_status") == "EXPOSED_DURING_CANDIDATE_SEARCH", "data manifest heldout policy mismatch", failures)
    require(policy.get("final_test_requires_candidate_lock") is True, "data manifest lacks candidate-lock test firewall", failures)
    datasets = data_manifest.get("datasets", {})
    sngar = datasets.get("sngar_tracking", {})
    require(sngar.get("status") == "deferred_while_idsse_substitute_active", "SNGAR deferral status mismatch", failures)
    require(sngar.get("access_probe_status") == "blocked_external_network_2026-09-02", "SNGAR access probe status mismatch", failures)
    require(sngar.get("local_data_present") is False, "SNGAR local data presence is not explicitly false", failures)
    idsse = datasets.get("idsse", {})
    require(idsse.get("status") == "canonical_conversion_complete_t5r2_lock_complete", "IDSSE status mismatch", failures)
    require(idsse.get("path") == "sports/idsse-data", "IDSSE path mismatch", failures)
    require(idsse.get("role") == "temporary_development_substitute_for_sngar", "IDSSE replacement role mismatch", failures)
    require(idsse.get("external_confirmation_allowed_in_same_run") is False, "IDSSE external-confirmation reuse boundary missing", failures)
    require(idsse.get("official_revision") == "a715a38dfbaf5f58e431727c2b78d174101a703c", "IDSSE official revision mismatch", failures)
    require(idsse.get("official_file_tree_matches_local") is True, "IDSSE official file-tree match not recorded", failures)

    require(matrix.get("p3_task_semantic_repair", {}).get("status") == "t5r5_locked_single_read_authorized", "T5R matrix status mismatch", failures)
    require(repair.get("status") == "t5r5_locked_single_read_authorized", "repair config status mismatch", failures)
    require(repair.get("autoresearch", {}).get("status") == "round_2_complete_autoresearch_closed", "autoresearch Round 2 status mismatch", failures)
    require(split_lock.get("status") == "MATCH_SPLIT_FROZEN_BEFORE_T5R2_TASK_CONSTRUCTION", "T5R2 split lock status mismatch", failures)
    require(t5r2_lock.get("status") == "T5R2_CLOSED_BASELINE_AND_METRIC_LOCK", "T5R2 baseline lock status mismatch", failures)
    require(t5r2_lock.get("firewall", {}).get("model_results_used") is False, "T5R2 lock used model results", failures)
    require(t5r2_lock.get("firewall", {}).get("reserved_holdout_task_arrays_written") is False, "reserved task arrays were written", failures)
    require(t5r3_manifest.get("status") == "T5R3_FIXED_DUAL_CHANNEL_SANITY_COMPLETE", "T5R3 manifest status mismatch", failures)
    require(t5r3_manifest.get("reserved_holdout_loaded") is False, "T5R3 manifest reserved holdout mismatch", failures)
    require(t5r3_manifest.get("candidate_selection_performed") is False, "T5R3 manifest candidate selection mismatch", failures)
    require(t5r4r2_manifest.get("status") == "T5R4_ROUND2_COMPLETE", "T5R4 Round 2 manifest status mismatch", failures)
    require(t5r4r2_manifest.get("reserved_holdout_loaded") is False, "T5R4 Round 2 manifest reserved holdout mismatch", failures)
    require(t5r4r2_manifest.get("candidate_selection_performed") is False, "T5R4 Round 2 manifest candidate selection mismatch", failures)
    require(t5r4r2_manifest.get("candidate_lock_created") is False, "T5R4 Round 2 manifest candidate-lock mismatch", failures)
    require(t5r4r2_manifest.get("P4_release") is False, "T5R4 Round 2 manifest P4 mismatch", failures)

    require(repair.get("phase") == "P3_CAUSAL_MECHANISM", "repair config phase mismatch", failures)
    require(repair.get("task_lane") == "P3-T5R", "repair config task lane missing", failures)
    firewall = repair.get("firewall", {})
    for key in (
        "sngar_test_files_present_during_search",
        "sngar_test_labels_read_during_search",
        "idsse_results_read_during_search",
        "soccertrack_final_results_read_during_search",
    ):
        require(firewall.get(key) is False, f"firewall flag is not false: {key}", failures)
    require(repair.get("candidate_lock", {}).get("required_before_test") is True, "candidate lock requirement missing", failures)

    text_checks = {
        "README.md": ["P3-T5R", "EXPOSED_DURING_CANDIDATE_SEARCH", "P4 AMR 继续 blocked"],
        "MASTER_AGENT_PROMPT.md": ["P3-T5R0", "candidate lock", "禁止先训练 AMR"],
        "STATUS.md": ["P3_T5R5_LOCKED", "T5R3", "T5R4", "p4_status: blocked_pending_p3_t5r_gate"],
        "CLAIM_LEDGER.md": ["P3-R1", "P3-R2", "P3-R3", "P3-R4", "P3-R5", "DATA-C1"],
        "DECISIONS.md": ["D-20260902-P3-011", "D-20260902-P3-012", "D-20260902-P3-013", "D-20260902-P3-014", "D-20260902-P3-015", "D-20260904-P3-017", "D-20260904-P3-018", "D-20260904-P3-019", "D-20260904-P3-020", "D-20260904-P3-021", "D-20260904-P3-022", "D-20260905-P3-025", "D-20260905-P3-026"],
        "docs/01_SCIENTIFIC_BLUEPRINT.md": ["P3-T5R", "context", "intrinsic"],
        "docs/02_METHOD_SPEC_AMR.md": ["z_ctx", "z_mode", "P3-T5R"],
        "docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md": ["P3-T5R", "candidate lock"],
        "docs/04_DATA_AND_ENVIRONMENT_GUIDE.md": ["SNGAR", "IDSSE", "candidate lock"],
        "docs/05_AGENT_EXECUTION_MANUAL.md": ["Task worker", "Autoresearch worker", "candidate lock"],
        "reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md": ["heldout", "T5R"],
        "TODO.md": ["当前固定决定", "IDSSE", "分支记录", "变更记录", "T5R4 Round 2", "T5R5"],
    }
    for relative, markers in text_checks.items():
        path = root / relative
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        for marker in markers:
            require(marker in content, f"{relative} missing marker: {marker}", failures)

    test_root = root / "data/raw/sports/sngar_tracking"
    candidate_lock = root / "artifacts/phase3/task_semantic_repair_v1/candidate_lock.json"
    require(candidate_lock.is_file(), "candidate lock missing after T5R5 lock", failures)
    if candidate_lock.is_file():
        locked = load_json(candidate_lock, failures)
        require(locked.get("status") == "T5R5_CANDIDATE_LOCKED", "candidate lock status mismatch", failures)
        require(locked.get("identity", {}).get("selected_candidate") == "update_ratio_2to1", "locked candidate mismatch", failures)
    if test_root.exists():
        test_markers = list(test_root.glob("test/**")) + list(test_root.glob("*annotations_test*"))
        require(not test_markers, "SNGAR test files are present before candidate lock", failures)

    if failures:
        print("CURRENT_STATE_AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("CURRENT_STATE_AUDIT: PASS")
    print("- active_phase=P3_CAUSAL_MECHANISM")
    print("- task_lane=P3-T5R")
    print("- legacy_heldout=EXPOSED_DURING_CANDIDATE_SEARCH")
    print("- p4_status=blocked_pending_p3_t5r_gate")
    print("- t5r2_baseline_lock=T5R2_CLOSED_BASELINE_AND_METRIC_LOCK")
    print("- t5r3_status=T5R3_FIXED_DUAL_CHANNEL_SANITY_COMPLETE")
    print("- t5r4_status=T5R4_ROUND2_COMPLETE_AUTORESEARCH_CLOSED")
    print("- t5r5_status=CANDIDATE_LOCKED_SINGLE_READ_AUTHORIZED")
    print("- final_test_requires_candidate_lock=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
