#!/usr/bin/env python3
"""Audit the P4 AMR M1 config lock (P4-N2): lock integrity, code hash match,
routing assignment shape, dev-only scope, and that no M1 training record
predates the lock."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "artifacts" / "phase4_amr"
LOCK = P4 / "m1_config_lock.json"
FAILURES: list[str] = []


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, label: str) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        FAILURES.append(label)


def main() -> int:
    if not LOCK.is_file():
        print("REFUSED: lock missing")
        return 2
    lock = json.loads(LOCK.read_text(encoding="utf-8"))

    check(lock["lock_id"] == "amr_m1_config_lock", "lock id")
    check(lock["frozen_before"] == "any AMR M1 training or result inspection", "freeze boundary declared")
    alpha = lock["routing_assignment"]["alpha"]
    check(len(alpha) == lock["model"]["n_bands"] == 6, "alpha length matches n_bands")
    check(all(a in (0.0, 1.0) for a in alpha), "alpha is binary (M1 fixed)")
    check(all(a == 0.0 for a in alpha) or (alpha[0] == 1.0 and all(a == 0.0 for a in alpha[1:])),
          "routing assignment is one of the two pre-registered evidence-consistent forms")
    for rel, digest in lock["code_hashes"].items():
        path = ROOT / rel
        check(path.is_file() and sha256_file(path) == digest, f"code hash: {rel}")
    budget = lock["training_protocol"]["update_budget_per_epoch"]
    check(budget["context"] + budget["intrinsic_pair"] + budget["route"] == budget["total"] == 420,
          "update budget sums to frozen 420/epoch")
    check(lock["training_protocol"]["seeds"] == [11, 23, 47], "seeds")
    scope = lock["evidence_scope"]
    check("dev only" in scope and "J03WQQ" in scope and "SoccerTrack" in scope,
          "dev-only scope with consumed-asset re-authorization noted")
    closed_runs = {"m1_v1", "m1_v2"}
    trainings = [p for p in P4.rglob("*") if p.name.startswith(("record_", "summary")) and p.suffix == ".json"
                 and not (closed_runs & set(p.parts))]
    check(not trainings, "no M1 training records outside closed documented runs (pre-training gate)")
    criteria = lock["evaluation_plan"]["success_criteria_descriptive"]
    check("H1" in criteria and "H2" in criteria and "no m2" in criteria.lower(),
          "descriptive success criteria with contraction rule")

    if FAILURES:
        print(f"AMR_M1_LOCK_AUDIT: FAIL ({len(FAILURES)} checks)")
        return 1
    print("AMR_M1_LOCK_AUDIT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
