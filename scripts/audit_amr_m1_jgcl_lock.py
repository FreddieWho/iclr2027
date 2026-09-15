#!/usr/bin/env python3
"""Audit the goal R2 JGCL design lock (single-geometry InfoNCE).

Checks: lock identity, reused-unmodified v6 architecture (hash equality
against the v6 lock proves zero arch change), single-loss rule, code
hashes, constants, spectral cache, 420 budget, seeds, guards, dev-only
scope, and no R2 records yet (pre-training gate).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "artifacts" / "phase4_amr"
LOCK = P4 / "m1_jgcl_config_lock.json"
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
    check(lock["lock_id"] == "amr_m1_jgcl_config_lock", "lock id")
    check("InfoNCE" in lock["loss"]["rule"], "single-geometry InfoNCE rule recorded")
    check("triplet" in lock["loss"].get("dropped", []), "triplet documented as dropped")
    # architecture reuse proof: same file hash as frozen in the v6 lock
    v6lock = json.loads((P4 / "m1_v6_config_lock.json").read_text(encoding="utf-8"))
    check(lock["code_hashes"].get("scripts/amr_model_v6.py")
          == v6lock["code_hashes"].get("scripts/amr_model_v6.py"),
          "v6 architecture reused unmodified")
    for rel, digest in lock["code_hashes"].items():
        path = ROOT / rel
        check(path.is_file() and sha256_file(path) == digest, f"code hash: {rel}")
    check(lock["constants"] == {"W_CTX": 1.0, "W_INFO": 1.0, "TAU": 0.1},
          "constants block frozen")
    for split in ("train", "valid"):
        path = P4 / f"spectral_cache_{split}.npz"
        check(path.is_file() and sha256_file(path) == lock[f"spectral_cache_{split}_sha256"],
              f"spectral cache: {split}")
    budget = lock["schedule"]["budget_per_epoch"]
    check(budget["context"] + budget["infonce"] == budget["total"] == 420,
          "update budget 420/epoch")
    check(lock["schedule"]["seeds"] == [11, 23, 47], "seeds")
    check("COLLAPSED" in lock["guards"]["collapse"], "collapse guard pre-registered")
    check("J03WQQ" in lock["evidence_scope"] and "SoccerTrack" in lock["evidence_scope"],
          "dev-only scope with consumed-asset note")
    outdir = P4 / "m1_jgcl"
    existing = [p for p in outdir.rglob("*") if p.name.startswith(("record_", "summary")) and p.suffix == ".json"] \
        if outdir.is_dir() else []
    check(not existing, "no R2 training records predate the lock")
    if FAILURES:
        print(f"AMR_JGCL_LOCK_AUDIT: FAIL ({len(FAILURES)} checks)")
        return 1
    print("AMR_JGCL_LOCK_AUDIT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
