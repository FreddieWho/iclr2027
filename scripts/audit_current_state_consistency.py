#!/usr/bin/env python3
"""Fail-closed audit for the P3-T5R current-state protocol.

The audit is read-only: it parses project/config state and checks required
route markers and paths. It does not inspect external data or write reports.
"""
from __future__ import annotations

import argparse
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
    ]
    for relative in required_paths:
        require((root / relative).is_file(), f"missing required path: {relative}", failures)

    project = load_yaml(root / "configs/project.yaml", failures)
    matrix = load_yaml(root / "configs/experiment_matrix.yaml", failures)
    data_manifest = load_yaml(root / "configs/data_manifest.yaml", failures)
    repair = load_yaml(root / "configs/phase3_task_semantic_repair_v1.yaml", failures)

    project_state = project.get("project", {})
    require(project_state.get("active_phase") == "P3_CAUSAL_MECHANISM", "active phase is not P3_CAUSAL_MECHANISM", failures)
    require(project_state.get("current_checkpoint") == "P3_T5R0_DOCUMENT_MIGRATION_COMPLETE", "project checkpoint mismatch", failures)
    require(project_state.get("current_route") == "support_conditioned_geometry_with_task_semantic_repair", "project route mismatch", failures)
    require(project_state.get("p4_status") == "blocked_pending_p3_t5r_gate", "P4 is not blocked pending T5R gate", failures)
    require(project_state.get("p5_status") == "blocked_pending_sports_prediction_lock", "P5 status mismatch", failures)
    require(project_state.get("legacy_heldout_status") == "EXPOSED_DURING_CANDIDATE_SEARCH", "legacy heldout status mismatch", failures)
    repair_state = project.get("exploration", {}).get("p3_task_semantic_repair", {})
    require(repair_state.get("status") == "t5r0_complete_data_access_pending", "T5R project status is not t5r0_complete_data_access_pending", failures)

    require(matrix.get("p3_task_semantic_repair", {}).get("phase") == "P3_CAUSAL_MECHANISM", "T5R matrix is not inside P3", failures)
    require(matrix.get("p3_task_semantic_repair", {}).get("task_lane") == "P3-T5R", "T5R task lane missing", failures)
    require("C2.5" not in str(matrix), "forbidden C2.5 marker found", failures)

    policy = data_manifest.get("policy", {})
    require(policy.get("legacy_heldout_status") == "EXPOSED_DURING_CANDIDATE_SEARCH", "data manifest heldout policy mismatch", failures)
    require(policy.get("final_test_requires_candidate_lock") is True, "data manifest lacks candidate-lock test firewall", failures)
    datasets = data_manifest.get("datasets", {})
    sngar = datasets.get("sngar_tracking", {})
    require(sngar.get("status") == "access_not_assumed", "SNGAR access was silently marked available", failures)
    require(datasets.get("idsse", {}).get("status") == "not_downloaded_by_this_package", "IDSSE status mismatch", failures)

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
        "STATUS.md": ["P3_T5R0_DOCUMENT_MIGRATION_COMPLETE", "p4_status: blocked_pending_p3_t5r_gate"],
        "CLAIM_LEDGER.md": ["P3-R1", "P3-R2", "P3-R3", "P3-R4", "P3-R5", "DATA-C1"],
        "DECISIONS.md": ["D-20260902-P3-011", "D-20260902-P3-012", "D-20260902-P3-013", "D-20260902-P3-014", "D-20260902-P3-015"],
        "docs/01_SCIENTIFIC_BLUEPRINT.md": ["P3-T5R", "context", "intrinsic"],
        "docs/02_METHOD_SPEC_AMR.md": ["z_ctx", "z_mode", "P3-T5R"],
        "docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md": ["P3-T5R", "candidate lock"],
        "docs/04_DATA_AND_ENVIRONMENT_GUIDE.md": ["SNGAR", "IDSSE", "candidate lock"],
        "docs/05_AGENT_EXECUTION_MANUAL.md": ["Task worker", "Autoresearch worker", "candidate lock"],
        "reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md": ["heldout", "T5R"],
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
    if not candidate_lock.exists() and test_root.exists():
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
    print("- final_test_requires_candidate_lock=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
