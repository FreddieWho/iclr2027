#!/usr/bin/env python3
"""Audit the P4 R2 v5cap control lock (research-driven redesign).

Checks: lock identity, 3-band density-equalized basis, all-recoverable
routing, code hashes, constants sync markers, spectral cache hashes,
dev-only scope, calibration record, and no v4 training records yet
(pre-training gate; closed runs m1_v1/v2/v3/v4/v4b are acknowledged history).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P4 = ROOT / "artifacts" / "phase4_amr"
LOCK = P4 / "m1_v5cap_config_lock.json"
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
    check(lock["lock_id"] == "amr_m1_v5cap_config_lock", "lock id")
    check(lock["model"]["n_bands"] == 3, "3-band basis")
    check(lock["model"]["band_edges"] == [0.0, 1.0, 1.3, 2.0], "frozen density-equalized edges")
    check("Chebyshev" in lock["model"]["spectral_basis"] or "EXACT" in lock["model"]["spectral_basis"],
          "exact-basis design recorded")
    ra = lock["routing_assignment"]
    check("CapLoss" in ra and "Slepian" in ra.get("rule", ""),
          "CAP faithful rule recorded (identical Slepian sampling)")
    check("Slepian" in lock["routing_assignment"].get("rule", ""), "Slepian sampling recorded")
    check("stop-grad" not in lock["routing_assignment"].get("rule", "").lower(),
          "CAP has no stop-grad by design (faithful M0; both branches need grads)")
    for rel, digest in lock["code_hashes"].items():
        path = ROOT / rel
        check(path.is_file() and sha256_file(path) == digest, f"code hash: {rel}")
    consts = lock["constants"]
    check(consts == {"W_CAP": 2.0, "W_VICVAR": 1.0, "W_VICCOV": 0.04, "VIC_GAMMA": 0.02},
          "constants block frozen")
    for split in ("train", "valid"):
        path = P4 / f"spectral_cache_{split}.npz"
        check(path.is_file() and sha256_file(path) == lock[f"spectral_cache_{split}_sha256"],
              f"spectral cache: {split}")
    budget = lock["training_protocol"]["update_budget_per_epoch"]
    check(budget["context"] + budget["intrinsic_pair"] + budget["route"] == budget["total"] == 420,
          "update budget 420/epoch")
    check(lock["training_protocol"]["seeds"] == [11, 23, 47], "seeds")
    check("COLLAPSED" in lock["training_protocol"]["early_stop"], "early-stop rule pre-registered")
    check("J03WQQ" in lock["evidence_scope"] and "SoccerTrack" in lock["evidence_scope"],
          "dev-only scope with consumed-asset note")
    check("W_ROUTE_rationale" in lock["calibration"] and "VIC_GAMMA_rationale" in lock["calibration"],
          "calibration rationale recorded")
    v5dir = P4 / "m1_v5cap"
    existing = [p for p in v5dir.rglob("*") if p.name.startswith(("record_", "summary")) and p.suffix == ".json"] \
        if v5dir.is_dir() else []
    check(not existing, "no v5cap training records predate the lock")
    if FAILURES:
        print(f"AMR_V5CAP_LOCK_AUDIT: FAIL ({len(FAILURES)} checks)")
        return 1
    print("AMR_V5CAP_LOCK_AUDIT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
