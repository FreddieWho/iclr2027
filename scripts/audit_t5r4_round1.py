#!/usr/bin/env python3
"""Small structural audit for the bounded T5R4 Round 1 output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1" / "t5r4_round1_v1"
EXPECTED_SEEDS = {11, 23, 47}


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
    args = parser.parse_args()
    run = args.run.resolve()
    failures: list[str] = []
    paths = {name: run / name for name in ("manifest.json", "protocol.json", "model_results.json", "summary.json")}
    for name, path in paths.items():
        require(path.is_file(), f"missing artifact: {name}", failures)
    if failures:
        print("T5R4_ROUND1_AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    protocol = json.loads(paths["protocol.json"].read_text(encoding="utf-8"))
    records = json.loads(paths["model_results.json"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
    require(manifest.get("status") == "T5R4_ROUND1_COMPLETE", "manifest status mismatch", failures)
    require(protocol.get("status") == manifest.get("status"), "protocol/manifest status mismatch", failures)
    require(manifest.get("task_lane") == "P3-T5R4", "task lane mismatch", failures)
    require(manifest.get("reserved_holdout_loaded") is False, "reserved holdout was loaded", failures)
    require(manifest.get("external_results_used") is False, "external results were used", failures)
    require(manifest.get("old_heldout_used") is False, "old heldout was used", failures)
    require(manifest.get("candidate_selection_performed") is False, "candidate selection was performed", failures)
    require(protocol.get("visible_splits") == ["train", "valid"], "visible split mismatch", failures)
    require(protocol.get("reserved_holdout", {}).get("loaded") is False, "protocol reserved firewall mismatch", failures)
    require(protocol.get("candidate_selection_performed") is False, "protocol candidate selection mismatch", failures)
    require(protocol.get("seeds") == sorted(EXPECTED_SEEDS), "seed lock mismatch", failures)

    candidate_ids = {candidate["id"] for candidate in protocol.get("candidates", [])}
    require(len(candidate_ids) == manifest.get("candidate_count"), "candidate count mismatch", failures)
    require(len(records) == len(candidate_ids) * len(EXPECTED_SEEDS), "model result count mismatch", failures)
    observed: set[tuple[str, int]] = set()
    parameter_counts: set[int] = set()
    for record in records:
        candidate_id = str(record.get("candidate_id"))
        seed = int(record.get("seed", -1))
        observed.add((candidate_id, seed))
        parameter_counts.add(int(record.get("parameter_count", -1)))
        require(candidate_id in candidate_ids, f"unknown candidate: {candidate_id}", failures)
        require(seed in EXPECTED_SEEDS, f"unexpected seed: {seed}", failures)
        require(record.get("architecture") == "fixed-graph-message-passing-equivalent", f"architecture mismatch: {candidate_id}/{seed}", failures)
        checkpoint = ROOT / record["checkpoint"]["path"]
        require(checkpoint.is_file(), f"missing checkpoint: {candidate_id}/{seed}", failures)
        if checkpoint.is_file():
            require(record["checkpoint"].get("sha256") == sha256_file(checkpoint), f"checkpoint hash mismatch: {candidate_id}/{seed}", failures)
        for split in ("train", "valid"):
            context = record[split]["context"]
            intrinsic = record[split]["intrinsic"]
            translation = record[split]["translation"]
            for value in (
                context["phase_match_grouped_macro_f1"],
                context["field_zone_match_grouped_macro_f1"],
                context["centroid_match_grouped_mae"],
                intrinsic["match_grouped_ranking_accuracy"],
                intrinsic["match_grouped_mrr_at_2"],
                translation["z_mode_global_translation_response"],
            ):
                require(finite(value), f"non-finite metric: {candidate_id}/{seed}/{split}", failures)
    require(observed == {(candidate_id, seed) for candidate_id in candidate_ids for seed in EXPECTED_SEEDS}, "candidate/seed grid mismatch", failures)
    require(len(parameter_counts) == 1, f"parameter counts not matched: {sorted(parameter_counts)}", failures)
    require({row.get("candidate_id") for row in summary} == candidate_ids, "summary candidate set mismatch", failures)

    if failures:
        print("T5R4_ROUND1_AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("T5R4_ROUND1_AUDIT: PASS")
    print(f"- candidates={len(candidate_ids)}")
    print(f"- models={len(records)}")
    print("- train_valid_only=true")
    print("- reserved_holdout=J03WQQ_not_loaded")
    print("- automatic_promotion=false")
    print("- parameter_counts=matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
