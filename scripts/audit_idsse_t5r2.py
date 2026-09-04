#!/usr/bin/env python3
"""Fail-closed audit for the generated IDSSE T5R2 task package.

This audit is intentionally independent of scipy/sklearn and never loads the
reserved match's canonical rows.  It checks the generated artifacts rather
than trusting their status strings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IDSSE_ROOT = ROOT / "artifacts" / "data_v2" / "idsse"
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def check(fact: bool, message: str, failures: list[str]) -> None:
    if not fact:
        fail(message, failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idsse-root", type=Path, default=IDSSE_ROOT)
    parser.add_argument("--t5r-root", type=Path, default=T5R_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    idsse_root = args.idsse_root.resolve()
    t5r_root = args.t5r_root.resolve()
    failures: list[str] = []
    source_manifest_path = idsse_root / "source_file_manifest.json"
    canonical_manifest_path = idsse_root / "canonical_manifest.json"
    provenance_path = idsse_root / "source_provenance.json"
    split_path = t5r_root / "split_lock.json"
    task_path = t5r_root / "task_manifest.json"
    baseline_path = t5r_root / "baseline_and_metric_lock.json"
    required = [source_manifest_path, canonical_manifest_path, provenance_path, split_path, task_path, baseline_path]
    for path in required:
        check(path.is_file(), f"missing artifact: {path}", failures)
    if failures:
        print("T5R2_AUDIT: FAIL")
        for message in failures:
            print(f"- {message}")
        return 1

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    canonical_manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    split_lock = json.loads(split_path.read_text(encoding="utf-8"))
    task_manifest = json.loads(task_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    expected_split = {
        "J03WMX": "train",
        "J03WOH": "train",
        "J03WPY": "train",
        "J03WR9": "train",
        "J03WN1": "valid",
        "J03WOY": "valid",
        "J03WQQ": "reserved_holdout",
    }
    check(provenance.get("status") == "OFFICIAL_PAGE_VERIFIED_RAW_MANIFEST_AND_CANONICAL_QC_COMPLETE", "source provenance status mismatch", failures)
    check(canonical_manifest.get("status") == "IDSSE_T5R1_CANONICAL_CONVERSION_COMPLETE", "canonical manifest status mismatch", failures)
    check(canonical_manifest.get("official_revision") == "a715a38dfbaf5f58e431727c2b78d174101a703c", "official revision mismatch", failures)
    check(split_lock.get("status") == "MATCH_SPLIT_FROZEN_BEFORE_T5R2_TASK_CONSTRUCTION", "split was not frozen before tasks", failures)
    check(split_lock.get("split_map") == expected_split, "split map mismatch", failures)
    check(task_manifest.get("status") == "T5R2_TASK_CONSTRUCTION_COMPLETE", "task manifest status mismatch", failures)
    check(baseline.get("status") == "T5R2_CLOSED_BASELINE_AND_METRIC_LOCK", "baseline lock status mismatch", failures)
    check(baseline.get("split_lock", {}).get("split_map") == expected_split, "baseline lock split mismatch", failures)

    raw_files = source_manifest.get("files", [])
    check(source_manifest.get("file_count") == 23, "raw manifest does not contain 23 files", failures)
    check(len(raw_files) == 23, "raw manifest file rows mismatch", failures)
    actual_bytes = 0
    for row in raw_files:
        path = ROOT / "data" / "raw" / "sports" / "idsse-data" / str(row["relative_path"])
        check(path.is_file(), f"raw file missing: {row['relative_path']}", failures)
        if path.is_file():
            actual_bytes += path.stat().st_size
            check(path.stat().st_size == int(row["bytes"]), f"raw size changed: {row['relative_path']}", failures)
            check(len(str(row.get("sha256", ""))) == 64, f"raw SHA-256 missing: {row['relative_path']}", failures)
    check(actual_bytes == int(source_manifest.get("total_bytes", -1)), "raw manifest total bytes mismatch", failures)
    check(sha256_file(source_manifest_path) == canonical_manifest.get("raw_manifest_sha256"), "canonical raw manifest hash reference mismatch", failures)
    check(sha256_file(source_manifest_path) == baseline.get("dataset", {}).get("source_manifest_sha256"), "baseline source manifest hash reference mismatch", failures)

    check(canonical_manifest.get("match_ids") == sorted(expected_split), "canonical match ID list mismatch", failures)
    receipts_root = idsse_root / "canonical" / "conversion_receipts"
    receipt_paths = sorted(receipts_root.glob("*.json"))
    check([path.stem for path in receipt_paths] == sorted(expected_split), "conversion receipt list mismatch", failures)
    for path in receipt_paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        qc = receipt.get("qc", {})
        check(receipt.get("status") == "CANONICAL_CONVERSION_COMPLETE", f"receipt status mismatch: {path.name}", failures)
        check(qc.get("frame_rows", 0) > 0, f"no canonical frames: {path.name}", failures)
        check(qc.get("event_count", 0) > 0, f"no canonical events: {path.name}", failures)
        check(qc.get("missing_xy_rows") == 0, f"missing coordinates: {path.name}", failures)
        check(qc.get("frame_number_gap_count") == 0, f"frame number gap: {path.name}", failures)
        check(qc.get("max_within_frameset_time_gap_ms") == 40, f"unexpected frame interval: {path.name}", failures)
        check(qc.get("sampled_exact_22_source_player_fraction") == 1.0 or path.stem in {"J03WN1", "J03WQQ"}, f"source player QC mismatch: {path.name}", failures)
        check(qc.get("sampled_exact_20_outfield_fraction") is not None, f"outfield QC missing: {path.name}", failures)

    views_root = t5r_root / "data_views"
    check(not (views_root / "snapshot_index_reserved_holdout.parquet").exists(), "reserved holdout snapshot index was written", failures)
    check(not list(views_root.glob("*reserved_holdout*")), "reserved holdout task artifact was written", failures)
    indexes: dict[str, pd.DataFrame] = {}
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    pairs: dict[str, pd.DataFrame] = {}
    for split in ("train", "valid"):
        index_path = views_root / f"snapshot_index_{split}.parquet"
        pair_path = views_root / f"natural_pair_ranking_{split}.parquet"
        raw_path = views_root / f"positions_raw_{split}.npy"
        centered_path = views_root / f"positions_centered_{split}.npy"
        team_path = views_root / f"team_slots_{split}.npy"
        for path in (index_path, pair_path, raw_path, centered_path, team_path):
            check(path.is_file(), f"missing task view: {path.name}", failures)
        index = pd.read_parquet(index_path)
        raw = np.load(raw_path)
        centered = np.load(centered_path)
        teams = np.load(team_path)
        pair = pd.read_parquet(pair_path)
        indexes[split] = index
        arrays[split] = (raw, centered, teams)
        pairs[split] = pair
        check(len(index) > 0 and len(pair) > 0, f"empty {split} task view", failures)
        check(raw.shape == (len(index), 20, 2), f"raw array shape mismatch: {split}", failures)
        check(centered.shape == raw.shape, f"centered array shape mismatch: {split}", failures)
        check(teams.shape == (len(index), 20), f"team array shape mismatch: {split}", failures)
        check(np.isfinite(raw).all() and np.isfinite(centered).all(), f"non-finite task arrays: {split}", failures)
        check(np.max(np.abs(centered.mean(axis=1))) < 1e-5, f"z_mode is not globally centered: {split}", failures)
        check(np.all((teams[:, :10] == 0) & (teams[:, 10:] == 1)), f"team slot layout mismatch: {split}", failures)
        check(set(index["source_match_id"]) == {key for key, value in expected_split.items() if value == split}, f"match set mismatch: {split}", failures)
        check(set(index["split"]) == {split}, f"split column mismatch: {split}", failures)
        check(not set(pair.get("source_match_id", [])) & {key for key, value in expected_split.items() if value == "reserved_holdout"}, f"holdout pair leakage: {split}", failures)
        lookup = index.set_index("snapshot_id")
        threshold = baseline["intrinsic_task"]
        for row in pair.itertuples(index=False):
            for field in ("query_snapshot_id", "positive_snapshot_id", "negative_snapshot_id"):
                check(getattr(row, field) in lookup.index, f"unknown pair snapshot {getattr(row, field)}", failures)
            if all(getattr(row, field) in lookup.index for field in ("query_snapshot_id", "positive_snapshot_id", "negative_snapshot_id")):
                q = lookup.loc[row.query_snapshot_id]
                p = lookup.loc[row.positive_snapshot_id]
                n = lookup.loc[row.negative_snapshot_id]
                check(q.source_match_id == p.source_match_id == n.source_match_id == row.source_match_id, "pair crosses match", failures)
                check(q.game_section == p.game_section == n.game_section == row.game_section, "pair crosses half", failures)
                check(q.split == p.split == n.split == row.split, "pair split mismatch", failures)
            check(float(row.positive_internal_geometry_distance) <= 0.28 + 1e-7, "positive geometry threshold violated", failures)
            check(float(row.positive_centroid_distance) >= 0.35 - 1e-7, "positive centroid threshold violated", failures)
            check(float(row.negative_internal_geometry_distance) >= 0.35 - 1e-7, "hard-negative geometry threshold violated", failures)
            check(float(row.negative_centroid_distance) <= 0.12 + 1e-7, "hard-negative centroid threshold violated", failures)
            check(float(row.positive_time_gap_s) >= 4.0 and float(row.negative_time_gap_s) >= 4.0, "pair time separation violated", failures)

    check(task_manifest.get("holdout_firewall", {}).get("reserved_match_loaded") is False, "task manifest says holdout loaded", failures)
    check(task_manifest.get("holdout_firewall", {}).get("reserved_match_labels_used") is False, "task manifest says holdout labels used", failures)
    check(baseline.get("firewall", {}).get("reserved_holdout_task_arrays_written") is False, "baseline firewall mismatch", failures)
    check(baseline.get("firewall", {}).get("model_results_used") is False, "model results entered baseline lock", failures)
    check(baseline.get("firewall", {}).get("external_results_used") is False, "external results entered baseline lock", failures)
    check(baseline.get("candidate_budget", {}).get("candidate_lock_status") == "NOT_CREATED", "candidate budget status drifted", failures)

    if failures:
        print("T5R2_AUDIT: FAIL")
        for message in failures:
            print(f"- {message}")
        return 1
    print("T5R2_AUDIT: PASS")
    print(f"- train_snapshots={len(indexes['train'])} train_pairs={len(pairs['train'])}")
    print(f"- valid_snapshots={len(indexes['valid'])} valid_pairs={len(pairs['valid'])}")
    print("- reserved_holdout=J03WQQ not loaded into task views")
    print("- baseline_and_metric_lock=T5R2_CLOSED_BASELINE_AND_METRIC_LOCK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
