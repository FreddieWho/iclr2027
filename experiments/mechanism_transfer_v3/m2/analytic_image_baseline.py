#!/usr/bin/env python3
"""Analytic image baseline: native RGB -> color masks -> per-color line fit -> oracle rule.

This is an oracle-free, learning-free reference for the M2 visual front. It
reads only RGB pixels; it never receives coordinates, masks from the oracle,
parent ids, edit answers or combination ids. It exists to establish the
*ceiling* a simple segmentation-plus-geometry pipeline reaches on this task
family, so that a later learned front can be compared against something
stronger than chance.

Endpoint recovery. The frozen renderer stamps a square of half-width
``stamp_half_width`` at every sampled pixel of a segment, so the extreme
projection of the mask along the fitted principal direction overshoots the true
endpoint centre by ``stamp_half_width * (|v_x| + |v_y|)``. The estimator
subtracts that known renderer offset. The offset is renderer geometry, not
oracle information.

Reports per-image accuracy, per-quartet J3/J4 marginals, parent-cluster
intervals, an endpoint-recovery error, and an explicit failure decomposition
(empty mask, too few pixels, zero-length fit, border contact). Trains nothing,
uses no GPU, opens no sealed pool.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _v3common import metrics, visual_pool  # noqa: E402
from bank_contract import FROZEN_STAMP_HALF_WIDTH  # noqa: E402
from scene_primitives import IMAGE_SIZE, oracle  # noqa: E402

ROOT = HERE.parents[2]
DEFAULT_BANK = ROOT / "artifacts" / "mechanism_transfer_v3" / "m2" / "bank"
PILOT_BANK = ROOT / "artifacts" / "e832_focus" / "route2" / "data" / "data.npz"
MIN_MASK_PIXELS = 25
MIN_EIGEN_RATIO = 1e-6
SCENE_SCALE = 2.0 / (IMAGE_SIZE - 1)
BOOTSTRAP_SEED = 832831


def _to_float_nchw(image) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"expected a (3,H,W) image, got {array.shape}")
    return array[None]


def fit_segment_from_mask(mask, stamp_half_width: int = FROZEN_STAMP_HALF_WIDTH) -> tuple:
    """Principal-axis fit with the known stamp offset removed. Returns (endpair, info)."""
    m = np.asarray(mask, dtype=bool)
    info = {"n_pixels": int(m.sum()), "degenerate": False, "border_contact": False}
    rows, cols = np.nonzero(m)
    if len(rows) < MIN_MASK_PIXELS:
        info["degenerate"] = True
        info["reason"] = "too_few_pixels"
        return None, info
    info["border_contact"] = bool(
        (rows == 0).any() or (rows == IMAGE_SIZE - 1).any() or (cols == 0).any() or (cols == IMAGE_SIZE - 1).any()
    )
    pixels = np.stack([cols, rows], axis=1).astype(np.float64)  # x = column, y = row
    centre = pixels.mean(axis=0)
    centred = pixels - centre
    covariance = centred.T @ centred / max(len(pixels) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[-1] <= MIN_EIGEN_RATIO:
        info["degenerate"] = True
        info["reason"] = "zero_variance"
        return None, info
    direction = eigenvectors[:, -1]
    info["eigen_ratio"] = float(eigenvalues[0] / eigenvalues[-1])
    projection = centred @ direction
    cap = float(stamp_half_width) * float(abs(direction[0]) + abs(direction[1]))
    low = float(projection.min()) + cap
    high = float(projection.max()) - cap
    info["cap"] = cap
    info["length_after_cap"] = high - low
    if high <= low + 1e-6:
        info["degenerate"] = True
        info["reason"] = "zero_length_after_cap"
        return None, info
    endpoints_px = np.stack([centre + low * direction, centre + high * direction])
    endpoints = endpoints_px * SCENE_SCALE - 1.0
    return endpoints, info


def predict_image(image) -> dict:
    """RGB image -> (label, 4 endpoints, diagnostics). No oracle input."""
    nchw = _to_float_nchw(image)
    masks = visual_pool.color_masks(nchw)
    red, red_info = fit_segment_from_mask(masks["red"][0].numpy())
    blue, blue_info = fit_segment_from_mask(masks["blue"][0].numpy())
    result = {
        "red": red_info,
        "blue": blue_info,
        "valid": bool(red is not None and blue is not None),
    }
    if not result["valid"]:
        result["label"] = None
        result["endpoints"] = None
        return result
    scene = np.concatenate([red, blue], axis=0)
    label, margin, ambiguous = oracle(scene)
    result.update({"label": int(label), "margin": float(margin), "ambiguous": bool(ambiguous),
                   "endpoints": scene})
    return result


def _failure_decomposition(records) -> dict:
    return {
        "n": len(records),
        "both_segments_fit": int(sum(record["valid"] for record in records)),
        "red_degenerate": int(sum(record["red"]["degenerate"] for record in records)),
        "blue_degenerate": int(sum(record["blue"]["degenerate"] for record in records)),
        "border_contact": int(
            sum(record["red"]["border_contact"] or record["blue"]["border_contact"] for record in records)
        ),
        "red_reasons": _reason_counts(records, "red"),
        "blue_reasons": _reason_counts(records, "blue"),
    }


def _reason_counts(records, key: str) -> dict:
    counts: dict[str, int] = {}
    for record in records:
        reason = record[key].get("reason")
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def evaluate_singletons(images, labels) -> dict:
    records = [predict_image(image) for image in images]
    labels = np.asarray(labels).astype(int).ravel()
    predicted = np.asarray([record["label"] if record["valid"] else -1 for record in records])
    valid = predicted >= 0
    return {
        "n_images": int(len(records)),
        "accuracy_all_images": float((predicted == labels).mean()),
        "accuracy_on_fitted": float((predicted[valid] == labels[valid]).mean()) if valid.any() else None,
        "n_fitted": int(valid.sum()),
        "failure_decomposition": _failure_decomposition(records),
        "predictive_accuracy_note": (
            "an unfittable image is counted as a miss in accuracy_all_images; the fitted-only "
            "number is reported separately so the two are never conflated"
        ),
    }


def flow_counts(correct: np.ndarray) -> dict:
    """Repair-flow counts on one quartet-state correctness matrix (P,A,B,AB)."""
    correct = np.asarray(correct, dtype=bool).reshape(-1, 4)
    a, b, ab = correct[:, 1], correct[:, 2], correct[:, 3]
    return {
        "n_quartets": int(len(correct)),
        "n_P_correct": int(correct[:, 0].sum()),
        "n_A_B_correct": int((a & b).sum()),
        "n_A_B_correct_AB_wrong": int((a & b & ~ab).sum()),
        "n_full_repair_J3": int((a & b & ab).sum()),
    }


def evaluate_quartets(images4, labels4, parents=None) -> dict:
    images4 = np.asarray(images4)
    labels4 = np.asarray(labels4).astype(float).reshape(-1, 4)
    predictions = np.full_like(labels4, -1.0)
    records = []
    for index in range(len(images4)):
        row = []
        for state in range(4):
            record = predict_image(images4[index, state])
            records.append(record)
            row.append(record["label"] if record["valid"] else -1)
        predictions[index] = row
    valid = (predictions >= 0).all(axis=1)
    logits = np.where(predictions > 0.5, 1.0, -1.0).astype(np.float64)
    evaluated = logits[valid]
    evaluation_labels = labels4[valid]
    payload = {
        "n_quartets": int(len(images4)),
        "n_quartets_all_states_fitted": int(valid.sum()),
        "failure_decomposition": _failure_decomposition(records),
    }
    if len(evaluated) == 0:
        payload["metrics"] = None
        return payload
    result = metrics.joint_metrics(evaluated, evaluation_labels)
    correct = (evaluated > 0) == (evaluation_labels > 0.5)
    joint3 = correct[:, 1] & correct[:, 2] & correct[:, 3]
    payload["metrics"] = result
    payload["flow"] = flow_counts(correct)
    payload["baseline110"] = payload["flow"]
    payload["baseline110_note"] = (
        "n_A_B_correct_AB_wrong is the paired baseline-110 denominator; the same object is exposed "
        "under flow and baseline110 so no caller has to recompute it"
    )
    if parents is not None:
        parent_ids = np.asarray(parents)[valid]
        payload["J3_parent_cluster_ci"] = metrics.parent_cluster_ci(
            joint3.astype(float), parent_ids, seed=BOOTSTRAP_SEED
        )
        payload["n_parents"] = int(len(set(parent_ids.tolist())))
    return payload


def endpoint_error(predicted_scene, truth_scene) -> float:
    """Mean per-segment endpoint error, tolerating the arbitrary within-segment order."""
    predicted = np.asarray(predicted_scene, dtype=np.float64).reshape(4, 2)
    truth = np.asarray(truth_scene, dtype=np.float64).reshape(4, 2)
    errors = []
    for first, second in ((0, 1), (2, 3)):
        direct = (
            np.linalg.norm(predicted[first] - truth[first])
            + np.linalg.norm(predicted[second] - truth[second])
        )
        swapped = (
            np.linalg.norm(predicted[first] - truth[second])
            + np.linalg.norm(predicted[second] - truth[first])
        )
        errors.append(min(direct, swapped) / 2.0)
    return float(np.mean(errors))


def singleton_state_coords(data, split: str) -> np.ndarray:
    parents = data[f"{split}_parent_coords"][data[f"{split}_parents"]]
    return parents + data[f"{split}_edit_vectors"].astype(np.float64)


def quartet_state_coords(data) -> np.ndarray:
    base = data["test_parent_coords"][data["quartet_parents"]]
    return np.stack(
        [
            base,
            base + data["quartet_edits_a"],
            base + data["quartet_edits_b"],
            base + data["quartet_edits_a"] + data["quartet_edits_b"],
        ],
        axis=1,
    )


def evaluate_endpoint_recovery(images, coords) -> dict:
    records = [predict_image(image) for image in images]
    errors, n_fit, n_border = [], 0, 0
    for record, truth in zip(records, coords):
        if not record["valid"]:
            continue
        n_fit += 1
        n_border += int(record["red"]["border_contact"] or record["blue"]["border_contact"])
        errors.append(endpoint_error(record["endpoints"], truth))
    if not errors:
        return {"n": 0}
    errors = np.asarray(errors)
    return {
        "n": int(n_fit),
        "n_border_contact": int(n_border),
        "mean_endpoint_error": float(errors.mean()),
        "median_endpoint_error": float(np.median(errors)),
        "p90_endpoint_error": float(np.quantile(errors, 0.9)),
        "note": "per-segment mean endpoint distance in scene units; within-segment order is free",
    }


def run(bank_dir: Path = DEFAULT_BANK, pilot_bank: Path = PILOT_BANK) -> dict:
    payload = {
        "schema": "mechanism_transfer_v3.m2.analytic_baseline/1",
        "method": (
            "native RGB -> frozen color masks -> per-color principal-axis line fit with the known "
            "stamp offset removed -> oracle crossing rule on the four recovered endpoints"
        ),
        "mask_thresholds": {"red": dict(visual_pool.RED), "blue": dict(visual_pool.BLUE)},
        "stamp_half_width": FROZEN_STAMP_HALF_WIDTH,
        "min_mask_pixels": MIN_MASK_PIXELS,
        "reads_oracle": False,
        "trains_nothing": True,
        "claim": "none: analytic reference ceiling, not a method contribution",
    }
    payload["pilot"] = _evaluate_pilot(pilot_bank)
    payload["expanded"] = _evaluate_expanded(bank_dir)
    return payload


def _evaluate_pilot(pilot_bank: Path) -> dict:
    with np.load(pilot_bank, allow_pickle=False) as data:
        singletons = evaluate_singletons(data["test_images"], data["test_labels"])
        quartets = evaluate_quartets(
            data["quartet_images"], data["quartet_labels"], parents=data["quartet_parents"]
        )
    return {
        "bank": str(pilot_bank),
        "note": "archived 28-quartet / 19-parent Route-2 pilot bank, development evidence only",
        "test_singletons": singletons,
        "test_quartets": quartets,
    }


def _evaluate_expanded(bank_dir: Path) -> dict:
    bank_dir = Path(bank_dir)
    with np.load(bank_dir / "data.npz", allow_pickle=False) as data:
        singletons = evaluate_singletons(data["test_images"], data["test_labels"])
        quartets = evaluate_quartets(
            data["quartet_images"], data["quartet_labels"], parents=data["quartet_parents"]
        )
        train_singletons = evaluate_singletons(data["train_images"], data["train_labels"])
        coords = singleton_state_coords(data, "test")
        recovery = evaluate_endpoint_recovery(data["test_images"], coords)
        quartet_coords = quartet_state_coords(data)
        quartet_recovery = evaluate_endpoint_recovery(
            data["quartet_images"].reshape(-1, 3, IMAGE_SIZE, IMAGE_SIZE),
            quartet_coords.reshape(-1, 4, 2),
        )
    manifest = json.loads((bank_dir / "data_manifest.json").read_text())
    return {
        "bank": str(bank_dir),
        "bank_archive_sha256": manifest["archive"]["sha256"],
        "train_singletons": train_singletons,
        "test_singletons": singletons,
        "test_quartets": quartets,
        "test_singleton_endpoint_recovery": recovery,
        "quartet_state_endpoint_recovery": quartet_recovery,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--pilot", type=Path, default=PILOT_BANK)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "mechanism_transfer_v3" / "m2" / "analytic_baseline.json",
    )
    args = parser.parse_args()
    payload = run(args.bank, args.pilot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    pilot_metrics = payload["pilot"]["test_quartets"].get("metrics") or {}
    expanded_metrics = payload["expanded"]["test_quartets"].get("metrics") or {}
    print(json.dumps({
        "out": str(args.out),
        "pilot_singleton_accuracy": payload["pilot"]["test_singletons"]["accuracy_all_images"],
        "pilot_quartet_J3": pilot_metrics.get("J3"),
        "pilot_quartet_J4": pilot_metrics.get("J4"),
        "expanded_singleton_accuracy": payload["expanded"]["test_singletons"]["accuracy_all_images"],
        "expanded_quartet_J3": expanded_metrics.get("J3"),
        "expanded_endpoint_error": payload["expanded"]["test_singleton_endpoint_recovery"].get(
            "mean_endpoint_error"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
