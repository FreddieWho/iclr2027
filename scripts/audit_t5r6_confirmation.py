#!/usr/bin/env python3
"""Audit T5R6 external confirmation output (proportionate, read-only)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
RUN_ROOT = T5R_ROOT / "t5r6_soccertrack_v1"
VIEW_ROOT = T5R_ROOT / "data_views_soccertrack"
T5R6_LOCK = T5R_ROOT / "t5r6_confirmation_lock.json"
CANDIDATE_LOCK = T5R_ROOT / "candidate_lock.json"

MATCHES = ["118575", "118576", "118577", "118578", "128057", "128058", "132831", "132877"]
SMOKE = {"117092", "117093"}


def fail(msg: str, failures: list) -> None:
    failures.append(msg)


def main() -> int:
    failures: list[str] = []
    for name in ("manifest.json", "summary.json", "task_records.json",
                 "task_analytic_control.json", "intervention_per_match.json", "SHA256SUMS"):
        if not (RUN_ROOT / name).is_file():
            fail(f"missing {name}", failures)
    if failures:
        print("T5R6_AUDIT: FAIL", *failures, sep="\n- ")
        return 1
    manifest = json.loads((RUN_ROOT / "manifest.json").read_text())
    summary = json.loads((RUN_ROOT / "summary.json").read_text())
    records = json.loads((RUN_ROOT / "task_records.json").read_text())
    intervention = json.loads((RUN_ROOT / "intervention_per_match.json").read_text())

    if manifest.get("status") != "T5R6_EXTERNAL_CONFIRMATION_COMPLETE":
        fail("manifest status", failures)
    if manifest.get("matches") != MATCHES:
        fail("match list differs from lock", failures)
    lock = json.loads(T5R6_LOCK.read_text())
    if manifest.get("t5r6_lock_sha256") != hashlib.sha256(T5R6_LOCK.read_bytes()).hexdigest():
        fail("t5r6 lock hash mismatch", failures)
    if manifest.get("candidate_lock_sha256") != hashlib.sha256(CANDIDATE_LOCK.read_bytes()).hexdigest():
        fail("candidate lock hash mismatch", failures)

    # no smoke matches anywhere in views
    import pandas as pd
    for match in MATCHES:
        idx = pd.read_parquet(VIEW_ROOT / f"snapshot_index_{match}.parquet")
        found = set(idx["source_match_id"].astype(str)) | set(
            pd.read_parquet(VIEW_ROOT / f"natural_pair_ranking_{match}.parquet")["source_match_id"].astype(str))
        if found != {match}:
            fail(f"{match}: foreign match ids {found}", failures)
        if found & SMOKE:
            fail(f"{match}: smoke contamination", failures)

    # grid completeness: 8 matches x 3 models x 3 seeds = 72 task records
    if len(records) != 72:
        fail(f"task records {len(records)} != 72", failures)
    got = {(r["match"], r["model_id"], r["seed"]) for r in records}
    want = {(m, mid, s) for m in MATCHES for mid in
            ("update_ratio_2to1", "fixed_dual_channel_shared_phase_gat", "raw_single_channel_phase_gat_team_mean")
            for s in (11, 23, 47)}
    if got != want:
        fail("task grid incomplete", failures)
    if set(intervention) != set(MATCHES):
        fail("intervention matches incomplete", failures)
    for match in MATCHES:
        for model_id in ("update_ratio_2to1", "fixed_dual_channel_shared_phase_gat"):
            seeds = [intervention[match][model_id][s] for s in ("11", "23", "47")]
            if not all(s["status"] == "OK" for s in seeds):
                fail(f"{match}/{model_id}: seed status not all OK", failures)

    # verdict recompute from frozen rule
    full = summary["intervention_per_match_full"]
    base = summary["intervention_per_match_baseline"]
    direction = summary["intervention_per_match_direction"]
    n_ok = sum(1 for v in full if v == v)
    n_above = sum(1 for f, b in zip(full, base) if f == f and b == b and f > b)
    expected = ("T5R6_CONFIRMED" if (n_ok == 8 and n_above == 8 and all(d > 0.5 for d in direction if d == d))
                else "T5R6_MIXED" if n_above >= 5 else "T5R6_NOT_CONFIRMED")
    if summary["verdict"] != expected or manifest["verdict"] != expected:
        fail(f"verdict mismatch: files say {summary['verdict']}/{manifest['verdict']}, rule says {expected}", failures)

    if failures:
        print("T5R6_AUDIT: FAIL", *failures, sep="\n- ")
        return 1
    print(f"T5R6_AUDIT: PASS verdict={expected} matches_full_above_baseline={n_above}/8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
