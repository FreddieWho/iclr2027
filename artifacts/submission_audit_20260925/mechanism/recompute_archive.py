#!/usr/bin/env python3
"""CPU-only recomputation of archived M1/M2 evidence; never trains or writes legacy results."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "mechanism_transfer_v3"))
sys.path.insert(0, str(ROOT / "experiments" / "mechanism_transfer_v3" / "m1" / "training"))
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))

from common import geom_features, metrics  # noqa: E402
from features import batch_orbit_representative, orbit_tie_fraction  # noqa: E402
from u10_oracles import tri_oracle  # noqa: E402

BASELINE = "3fc9757"
M1_ART = ROOT / "artifacts" / "mechanism_transfer_v3" / "m1"
M1_TRAIN = M1_ART / "training"
M1_BANK = ROOT / "artifacts" / "e832_focus" / "structure" / "bank_quartets.npz"
M1_TRAIN_SCENES = ROOT / "artifacts" / "f095_campaign" / "D01" / "scenes_N" / "scenes.npz"
M2_ART = ROOT / "artifacts" / "mechanism_transfer_v3" / "m2"
M2_BANK = M2_ART / "bank"
M2_OLD_OUT = M2_ART / "gpu_run_remote" / "out"
OUTPUT_DIR = ROOT / "artifacts" / "submission_audit_20260925" / "mechanism"
EXPECTED_M1_BANK_SHA = "15f5bf181b37dc9b4309582490f7fc11dfca8b03491251c3724fcb2e1bdd804c"
EXPECTED_M2_BANK_SHA = "69c4bb5c0986701ee1ab7a20e6b50a3a8ae22e2aa959fb57906947fbe6184342"
EXPECTED_M2_HEAD_SHA = "8b9c43f805f1d48acd8a2a4f1644d1ce41ff2804c4e8fd0b91865db189e3dde3"
BOOT_SEED = 924
BOOT_N = 2000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_at_baseline(path: str) -> tuple[int, str]:
    blob = subprocess.run(
        ["git", "show", f"{BASELINE}:{path}"], check=True, stdout=subprocess.PIPE
    ).stdout
    return len(blob), hashlib.sha256(blob).hexdigest()


def ci_pair(values, parents) -> dict:
    values = np.asarray(values, dtype=np.float64).ravel()
    parents = np.asarray(parents).ravel()
    return {
        "row_mean": float(values.mean()),
        "row_weighted_cluster_ci95": metrics.row_weighted_cluster_ci(
            values, parents, seed=BOOT_SEED, n_boot=BOOT_N
        )["ci95"],
        "equal_parent_mean": float(
            np.asarray([values[parents == p].mean() for p in np.unique(parents)]).mean()
        ),
        "equal_parent_mean_ci95": metrics.parent_cluster_ci(
            values, parents, seed=BOOT_SEED, n_boot=BOOT_N
        )["ci95"],
        "n_quartets": int(len(values)),
        "n_parents": int(len(np.unique(parents))),
    }


def interval_sign(ci) -> str:
    if ci[0] > 0:
        return "positive"
    if ci[1] < 0:
        return "negative"
    return "crosses_zero"


def audit_t1_witness() -> dict:
    pair_path = M1_ART / "collision_pair.npz"
    archive = np.load(pair_path)
    expected_inside, expected_outside = geom_features.t1_collision_pair()
    inside, outside = archive["inside"], archive["outside"]
    records = {}
    for dtype in (np.float64, np.float32):
        a = geom_features.legacy_t1_features(inside.astype(dtype), sorted_features=True)
        b = geom_features.legacy_t1_features(outside.astype(dtype), sorted_features=True)
        au = geom_features.legacy_t1_features(inside.astype(dtype), sorted_features=False)
        bu = geom_features.legacy_t1_features(outside.astype(dtype), sorted_features=False)
        records[np.dtype(dtype).name] = {
            "legacy_sorted_gap": float(np.max(np.abs(a - b))),
            "legacy_sorted_bitwise_equal": bool(np.array_equal(a, b)),
            "legacy_unsorted_gap": float(np.max(np.abs(au - bu))),
        }
    oracle_inside = tri_oracle(inside, min_margin=0.02)
    oracle_outside = tri_oracle(outside, min_margin=0.02)
    search = json.loads((M1_ART / "source_search.json").read_text())
    return {
        "archive_pair_matches_current_formal_witness": bool(
            np.array_equal(inside, expected_inside) and np.array_equal(outside, expected_outside)
        ),
        "inside_oracle": oracle_inside,
        "outside_oracle": oracle_outside,
        "feature_gaps": records,
        "whole_orbit_gap": float(
            np.max(
                np.abs(
                    geom_features.t1_orbit_representative(inside)
                    - geom_features.t1_orbit_representative(outside)
                )
            )
        ),
        "finite_bank_search": {
            "n_bank_scenes": search["n_bank_scenes"],
            "n_positive_margin_scenes": search["n_positive_margin_scenes"],
            "nearest_opposite_label_observed_distance": search["nearest_opposite_label"]["g8_feature_distance"],
            "local_search_steps": search["bounded_local_search"]["steps"],
            "local_search_best_observed_distance": search["bounded_local_search"]["best_distance"],
            "exact_merge_sampled": search["bounded_local_search"]["exact_merge_found"],
            "interpretation": (
                "The finite bank value is an exact minimum for that enumerated bank. "
                "The random local search reports only its best sampled candidate; "
                "a positive value is not a lower bound and does not exclude an exact collision."
            ),
        },
    }


def audit_m1() -> dict:
    bank_sha = sha256(M1_BANK)
    if bank_sha != EXPECTED_M1_BANK_SHA:
        raise RuntimeError(f"M1 bank hash mismatch: {bank_sha}")
    bank = np.load(M1_BANK, allow_pickle=True)
    parents = np.asarray(bank["parent_index"])
    y = np.stack([bank["y0"], bank["y0"], bank["y0"], bank["yab"]], axis=1)
    old_metrics = json.loads((M1_TRAIN / "metrics_3seed.json").read_text())
    old_arms = {
        (r["arm"], int(r["seed"]), r["supervision"]): r
        for r in old_metrics["arms"]
    }
    selection = json.loads((M1_TRAIN / "selection.json").read_text())
    correct_map = {}
    cells = []
    prediction_checks = 0
    for row in selection:
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        prediction_path = checkpoint.parent / "quartet_correct.npz"
        pred = np.load(prediction_path)
        logits = np.asarray(pred["logits"])
        saved_correct = np.asarray(pred["correct"], dtype=bool)
        correct = (logits > 0) == (y > 0.5)
        matches = bool(np.array_equal(saved_correct, correct))
        if not matches:
            raise RuntimeError(f"archived M1 correctness mismatch: {prediction_path}")
        prediction_checks += 1
        key = (row["arm"], int(row["seed"]), row["supervision"])
        correct_map[key] = correct
        j3 = correct[:, 1:].all(axis=1).astype(float)
        j4 = correct.all(axis=1).astype(float)
        old = old_arms[key]
        cells.append(
            {
                "arm": key[0],
                "seed": key[1],
                "supervision": key[2],
                "selected_lr": row["lr"],
                "selection_rule": row.get("selection"),
                "J3_row_mean_recomputed": float(j3.mean()),
                "J3_row_mean_stored": old["J3"]["row_mean"],
                "J3_row_mean_matches": bool(abs(float(j3.mean()) - old["J3"]["row_mean"]) < 1e-12),
                "J4_row_mean_recomputed": float(j4.mean()),
                "J4_row_mean_stored": old["J4"]["row_mean"],
                "J4_row_mean_matches": bool(abs(float(j4.mean()) - old["J4"]["row_mean"]) < 1e-12),
            }
        )
    if len(cells) != 36:
        raise RuntimeError(f"expected 36 selected M1 cells, found {len(cells)}")

    pairs = [
        ("flip_orbit_minus_sorted", "orbit_distance", "rich8_sorted", "flip"),
        ("flip_orbit_minus_unsorted", "orbit_distance", "rich8_unsorted", "flip"),
        ("clean_orbit_minus_sorted", "orbit_distance", "rich8_sorted", "clean"),
        ("clean_orbit_minus_unsorted", "orbit_distance", "rich8_unsorted", "clean"),
        ("clean_orbit_minus_raw", "orbit_distance", "raw", "clean"),
        ("flip_segment_moment_minus_sorted", "segment_moment", "rich8_sorted", "flip"),
        ("flip_segment_moment_minus_orbit", "segment_moment", "orbit_distance", "flip"),
    ]
    contrasts = []
    for name, left, right, supervision in pairs:
        for seed in (11, 23, 47):
            left_j3 = correct_map[(left, seed, supervision)][:, 1:].all(axis=1).astype(float)
            right_j3 = correct_map[(right, seed, supervision)][:, 1:].all(axis=1).astype(float)
            summary = ci_pair(left_j3 - right_j3, parents)
            summary.update(
                {
                    "contrast": name,
                    "seed": seed,
                    "supervision": supervision,
                    "row_weighted_direction": interval_sign(summary["row_weighted_cluster_ci95"]),
                    "equal_parent_direction": interval_sign(summary["equal_parent_mean_ci95"]),
                }
            )
            contrasts.append(summary)

    # Whole-orbit parity is checked on every archived P/A/B/AB scene.
    scenes = np.concatenate([np.asarray(bank["states"][i]) for i in range(4)], axis=0)
    base = batch_orbit_representative(scenes)
    max_group_deviation = 0.0
    for perm in geom_features.SOURCE_PERMS:
        moved = batch_orbit_representative(scenes[:, list(perm), :])
        max_group_deviation = max(max_group_deviation, float(np.max(np.abs(moved - base))))

    seam_path = M1_TRAIN / "orbit_seam.json"
    old_seam = json.loads(seam_path.read_text())
    training = np.load(M1_TRAIN_SCENES)["positions"]
    populations = {
        "train_scenes": training,
        "bank_A_states": np.asarray(bank["states"][1]),
        "bank_P_states": np.asarray(bank["states"][0]),
    }
    seam = {}
    for name, values in populations.items():
        previous = old_seam["populations"][name]
        current = orbit_tie_fraction(values, atol=1e-9)
        seam[name] = {
            "n_scenes": int(len(values)),
            "old_reported_tie_fraction": previous["tie_fraction"],
            "corrected_tie_fraction": current,
            "old_equals_corrected": bool(abs(previous["tie_fraction"] - current) < 1e-15),
        }

    baseline_files = {}
    for filename in ("features.py", "evaluate.py"):
        current_path = f"experiments/mechanism_transfer_v3/m1/training/{filename}"
        length, digest = sha_at_baseline(current_path)
        baseline_files[filename] = {
            "baseline_bytes": length,
            "baseline_sha256": digest,
            "m1_run_manifest_sha256": json.loads((M1_TRAIN / "MANIFEST.json").read_text())[
                "code_sha256"
            ][filename],
            "manifest_matches_baseline": digest
            == json.loads((M1_TRAIN / "MANIFEST.json").read_text())["code_sha256"][filename],
        }
    return {
        "bank": {
            "path": str(M1_BANK.relative_to(ROOT)),
            "sha256": bank_sha,
            "n_quartets": int(len(parents)),
            "n_parents": int(len(np.unique(parents))),
        },
        "archived_prediction_validation": {
            "selected_cells": prediction_checks,
            "correctness_matches_logits_and_frozen_labels": prediction_checks == 36,
            "all_stored_J3_J4_row_means_match": all(
                r["J3_row_mean_matches"] and r["J4_row_mean_matches"] for r in cells
            ),
        },
        "cells": cells,
        "pre_registered_contrasts_recomputed": contrasts,
        "orbit_parity": {
            "scenes_checked": int(len(scenes)),
            "legal_group_elements": len(geom_features.SOURCE_PERMS),
            "maximum_absolute_feature_deviation": max_group_deviation,
        },
        "orbit_tie_diagnostic": seam,
        "baseline_source_fingerprints": baseline_files,
    }


def audit_m2(out_root: Path) -> dict:
    archive_path = M2_BANK / "data.npz"
    manifest_path = M2_BANK / "data_manifest.json"
    bank_sha = sha256(archive_path)
    manifest_sha = sha256(manifest_path)
    if bank_sha != EXPECTED_M2_BANK_SHA:
        raise RuntimeError(f"M2 bank hash mismatch: {bank_sha}")
    data = dict(np.load(archive_path))
    parents = np.asarray(data["quartet_parents"], dtype=np.int64)
    expected_labels = np.asarray(data["quartet_labels"], dtype=np.float64)
    cells = []
    for result_path in sorted(out_root.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        prediction_path = result_path.parent / "predictions.npz"
        pred = np.load(prediction_path)
        logits = np.asarray(pred["logits"], dtype=np.float64)
        labels = np.asarray(pred["labels"], dtype=np.float64)
        pred_parents = np.asarray(pred["parents"], dtype=np.int64) if "parents" in pred.files else parents
        if logits.size != labels.size:
            raise RuntimeError(f"M2 logit/label count mismatch: {prediction_path}")
        logits = logits.reshape(labels.shape)
        if not np.array_equal(labels, expected_labels):
            raise RuntimeError(f"M2 archived labels differ from frozen bank: {prediction_path}")
        if not np.array_equal(pred_parents, parents):
            raise RuntimeError(f"M2 prediction parents differ from frozen bank: {prediction_path}")
        metric = metrics.joint_metrics(logits, labels, p_available=True)
        correct_j3 = ((logits[:, 1:] > 0) == (labels[:, 1:] > 0.5)).all(axis=1).astype(float)
        row_ci = metrics.row_weighted_cluster_ci(correct_j3, parents, seed=int(result["seed"]))
        parent_ci = metrics.parent_cluster_ci(correct_j3, parents, seed=int(result["seed"]))
        saved = result["quartet_table"]
        cells.append(
            {
                "cell": result_path.parent.name,
                "mode": result["mode"],
                "seed": result["seed"],
                "n_quartets": int(len(labels)),
                "n_parents": int(len(np.unique(parents))),
                "prediction_parent_ids_saved": "parents" in pred.files,
                "J3_row_mean_recomputed": metric["J3"],
                "J3_row_mean_matches_result": bool(abs(metric["J3"] - saved["J3"]) < 1e-12),
                "J4_recomputed": metric["J4"],
                "J4_matches_result": bool(abs(metric["J4"] - saved["J4"]) < 1e-12),
                "row_weighted_cluster_ci95_for_J3": row_ci["ci95"],
                "row_weighted_ci_point_matches_J3": bool(abs(row_ci["estimate"] - metric["J3"]) < 1e-12),
                "equal_parent_mean": parent_ci["estimate"],
                "equal_parent_cluster_ci95": parent_ci["ci95"],
                "saved_legacy_parent_cluster_estimate": saved["J3_parent_cluster_ci"]["estimate"],
                "legacy_point_CI_estimand_mismatch": bool(
                    abs(metric["J3"] - saved["J3_parent_cluster_ci"]["estimate"]) > 1e-12
                ),
                "dev_selection_best_loss": result.get("train", {}).get("best_dev_loss"),
                "dev_geometry_mse": result.get("dev_geometry_mse"),
                "seconds": result.get("seconds"),
                "predictions_sha256": sha256(prediction_path),
            }
        )
    if len(cells) != 16:
        raise RuntimeError(f"expected 16 archived M2 cells, found {len(cells)} in {out_root}")

    clean_count = int(np.asarray(data["train_clean"], dtype=bool).sum())
    full_count = int(len(data["train_labels"]))
    direct = [r for r in cells if r["mode"] == "direct_full"]
    clean = [r for r in cells if r["mode"] == "direct_clean"]
    true_geometry = [r for r in cells if r["mode"] == "true_geometry"]
    random = [r for r in cells if r["mode"] == "random_head"]
    front = [r for r in cells if r["mode"] == "geom_front"]
    result = {
        "bank": {
            "archive_path": str(archive_path.relative_to(ROOT)),
            "archive_sha256": bank_sha,
            "manifest_sha256": manifest_sha,
            "n_quartets": int(len(expected_labels)),
            "n_test_parents": int(len(np.unique(parents))),
        },
        "archived_cells": cells,
        "all_16_J3_J4_recomputations_match": all(
            c["J3_row_mean_matches_result"] and c["J4_matches_result"] for c in cells
        ),
        "direct_full_J3_row_means": [c["J3_row_mean_recomputed"] for c in direct],
        "direct_clean_J3_row_means": [c["J3_row_mean_recomputed"] for c in clean],
        "true_geometry_J3_row_mean": true_geometry[0]["J3_row_mean_recomputed"] if true_geometry else None,
        "random_head_J3_row_means": [c["J3_row_mean_recomputed"] for c in random],
        "geom_front_old_J3_row_means": [c["J3_row_mean_recomputed"] for c in front],
        "old_direct_clean_exposure": {
            "unique_clean_rows": clean_count,
            "full_singleton_rows": full_count,
            "per_epoch_clean_exposures_old": clean_count,
            "per_epoch_full_exposures": full_count,
            "old_clean_to_full_exposure_ratio": clean_count / full_count,
            "epochs": 20,
            "old_total_clean_exposures": clean_count * 20,
            "matched_total_full_exposures": full_count * 20,
        },
        "input_and_reference_checks": {
            "expected_frozen_head_sha256": EXPECTED_M2_HEAD_SHA,
            "frozen_head_checkpoint_sha256": sha256(
                M1_TRAIN / "runs" / "orbit_distance_s11_flip_lr0.001" / "model.pt"
            ),
            "frozen_head_sha_matches": sha256(
                M1_TRAIN / "runs" / "orbit_distance_s11_flip_lr0.001" / "model.pt"
            )
            == EXPECTED_M2_HEAD_SHA,
            "analytic_baseline_path": str((M2_ART / "analytic_baseline.json").relative_to(ROOT)),
            "analytic_baseline_sha256": sha256(M2_ART / "analytic_baseline.json"),
            "analytic_baseline_interpretation": "separate same-bank image baseline; not an upper ceiling",
        },
        "selection": {
            "train_loop_selects_best_dev_loss": True,
            "test_J_is_used_for_model_or_epoch_selection": False,
            "evidence_path": "experiments/mechanism_transfer_v3/m2/gpu_transfer.py",
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-out-root", type=Path, default=M2_OLD_OUT)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "archive_recomputation.json")
    args = parser.parse_args()
    record = {
        "schema": "iclr2027.submission_mechanism_audit_archive_recomputation/1",
        "status": "ARCHIVED_OUTPUTS_RECOMPUTED; NEW_CORRECTED_GPU_RUN_DEFERRED_PENDING_ACCEPTANCE",
        "method": "CPU-only; reads frozen bank, receipts, selected logits and saved predictions; no training or sealed pools",
        "m1": audit_m1(),
        "m2": audit_m2(args.m2_out_root),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "m1_cells": len(record["m1"]["cells"]), "m2_cells": len(record["m2"]["archived_cells"]), "m2_recomputations_match": record["m2"]["all_16_J3_J4_recomputations_match"]}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
