#!/usr/bin/env python3
"""Mask-fidelity audit for the M2 visual banks (repair R4).

Measures, per split and per color, the rate at which a *natively visible*
segment is erased by mask downsampling, under the legacy nearest rule and the
fixed area rule, at the grids the ResNet-18 front actually produces. Missing
color (no native pixels at all) is counted separately from downsampling loss;
the two are never merged.

The last section is an explicitly descriptive error-correlation audit on the
archived 28-quartet Route-2 pilot bank, using the archived contract-corrected
v2 predictions. It is reported with its n and never as evidence.

Reads bank images and archived predictions; trains nothing, uses no GPU, opens
no sealed pool.
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
import bank_contract  # noqa: E402
from scene_primitives import changed_pixels  # noqa: E402

ROOT = HERE.parents[2]
DEFAULT_BANK = ROOT / "artifacts" / "mechanism_transfer_v3" / "m2" / "bank"
PILOT_BANK = ROOT / "artifacts" / "e832_focus" / "route2" / "data" / "data.npz"
V2_RUNS = ROOT / "artifacts" / "e832_focus" / "route2" / "gpu_run_v2_remote_20260925/e832_focus/route2"
MODES = ("nearest", "area")
COLORS = ("red", "blue")


def as_float01(images) -> np.ndarray:
    """uint8 bank images -> float32 in [0,1]; already-float pilot images pass through."""
    array = np.asarray(images)
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    return np.asarray(array, dtype=np.float32)


def _mask_stats_for(images_uint8: np.ndarray, output_size: int, chunk: int = 256) -> dict:
    """Aggregate pooled-zero rates for one output grid over one image set."""
    totals = {
        mode: {
            color: {
                "n_images": 0,
                "native_visible": 0,
                "missing_color": 0,
                "erased_native_visible": 0,
                "pooled_cells_sum": 0.0,
                "pooled_cells_list": [],
                "native_pixels_sum": 0.0,
                "pooled_mass_sum": 0.0,
            }
            for color in COLORS
        }
        for mode in MODES
    }
    for start in range(0, len(images_uint8), chunk):
        block = as_float01(images_uint8[start : start + chunk])
        masks = visual_pool.color_masks(block)
        for mode in MODES:
            for color in COLORS:
                stats = visual_pool.mask_mass_stats(
                    masks[color], (output_size, output_size), mode=mode
                )
                weights = visual_pool.pooled_mask_weights(
                    masks[color], (output_size, output_size), mode=mode
                )
                cells = (weights.reshape(len(weights), -1) > 0).sum(dim=1).numpy()
                native = stats["native_pixels"].numpy()
                pooled = stats["pooled_mass"].numpy()
                erased = stats["native_visible_but_pooled_zero"].numpy()
                record = totals[mode][color]
                record["n_images"] += int(len(native))
                record["native_visible"] += int((native > 0).sum())
                record["missing_color"] += int((native == 0).sum())
                record["erased_native_visible"] += int(erased.sum())
                record["pooled_cells_sum"] += float(cells.sum())
                record["pooled_cells_list"].extend(cells.tolist())
                record["native_pixels_sum"] += float(native.sum())
                record["pooled_mass_sum"] += float(pooled.sum())
    for mode in MODES:
        for color in COLORS:
            record = totals[mode][color]
            cells = np.asarray(record.pop("pooled_cells_list"), dtype=float)
            visible = max(record["native_visible"], 1)
            record["erased_rate_of_native_visible"] = record["erased_native_visible"] / visible
            record["n_pooled_cells"] = int(output_size * output_size)
            record["pooled_cells_mean"] = float(cells.mean()) if len(cells) else 0.0
            record["pooled_cells_median"] = float(np.median(cells)) if len(cells) else 0.0
            record["pooled_cells_min"] = float(cells.min()) if len(cells) else 0.0
            record["mean_native_pixels"] = record["native_pixels_sum"] / max(record["n_images"], 1)
            record["mean_pooled_mass"] = record["pooled_mass_sum"] / max(record["n_images"], 1)
    return totals


def audit_bank(bank_dir: Path = DEFAULT_BANK, output_sizes=(7, 14)) -> dict:
    bank_dir = Path(bank_dir)
    manifest = json.loads((bank_dir / "data_manifest.json").read_text())
    payload = {
        "schema": "mechanism_transfer_v3.m2.mask_audit/1",
        "bank": str(bank_dir),
        "bank_archive_sha256": manifest["archive"]["sha256"],
        "output_sizes": list(output_sizes),
        "modes": list(MODES),
        "purpose": (
            "measure mask downsampling loss before any model is trained; missing color and "
            "downsampling loss are separate columns"
        ),
        "splits": {},
    }
    with np.load(bank_dir / "data.npz", allow_pickle=False) as data:
        for split in ("train", "dev", "test"):
            payload["splits"][split] = {
                size: _mask_stats_for(data[f"{split}_images"], size) for size in output_sizes
            }
        quartet_images = data["quartet_images"]
        quartet_stats = {}
        for size in output_sizes:
            states = quartet_images.reshape(-1, 3, 64, 64)
            flat = _mask_stats_for(states, size)
            quartet_stats[size] = flat
            erased_flags = _per_quartet_erasure(quartet_images, size, mode="nearest")
            quartet_stats[f"{size}_quartets_with_any_erased_state"] = int(erased_flags.sum())
            quartet_stats[f"{size}_quartet_count"] = int(len(erased_flags))
            quartet_stats[f"{size}_erased_state_indices"] = np.flatnonzero(erased_flags).tolist()
        payload["quartet_states"] = quartet_stats
    payload["palette_ambiguity"] = _palette_ambiguity(
        np.load(bank_dir / "data.npz", allow_pickle=False)["test_images"]
    )
    payload["interpretation"] = {
        "area_rule": "mass-preserving; the fixed contract used by the v3 visual pool",
        "nearest_rule": "legacy Route-2 rule; kept only to quantify erasure",
        "claim": "none; this is an input-contract audit, not a performance result",
    }
    return payload


def _per_quartet_erasure(quartet_images: np.ndarray, output_size: int, mode: str = "nearest") -> np.ndarray:
    """True when any of a quartet's four states has a natively visible erased channel."""
    n_quartets = quartets = quartet_images.shape[0]
    flags = np.zeros(quartets, dtype=bool)
    for index in range(n_quartets):
        states = quartet_images[index].reshape(-1, 3, 64, 64)
        block = as_float01(states)
        masks = visual_pool.color_masks(block)
        erased_any = False
        for color in COLORS:
            stats = visual_pool.mask_mass_stats(
                masks[color], (output_size, output_size), mode=mode
            )
            if bool(stats["native_visible_but_pooled_zero"].any()):
                erased_any = True
        flags[index] = erased_any
    return flags


def _palette_ambiguity(images_uint8: np.ndarray) -> dict:
    """Confirm the frozen thresholds are unambiguous on this renderer's palette."""
    block = as_float01(images_uint8)
    red, green, blue = block[:, 0], block[:, 1], block[:, 2]
    red_mask = (red > 0.15) & (green < 0.35) & (blue < 0.35)
    blue_mask = (blue > 0.15) & (red < 0.35) & (green < 0.35)
    green_gray_zone = int(((green >= 0.30) & (green <= 0.40)).sum())
    return {
        "n_pixels": int(red.size),
        "green_in_0.30_0.40": green_gray_zone,
        "red_and_blue_both": int((red_mask & blue_mask).sum()),
        "neither_mask_nor_near_background": int(
            (~red_mask & ~blue_mask & (np.abs(red - 0.5) > 0.05)).sum()
        ),
        "unique_values_red": int(len(np.unique(np.round(red, 3)))),
        "note": (
            "a nonzero red/blue overlap or a populated green gray zone would mean the frozen "
            "thresholds are ambiguous on this renderer; zero is expected"
        ),
    }


def quartet_pixel_filter_audit(pilot_bank: Path = PILOT_BANK, bank_dir: Path = DEFAULT_BANK) -> dict:
    """How many archived quartets would fail the frozen changed-pixel filter.

    Measured with the same rule the generator applies (max over RGB channels of
    |state - P| > 0.2, require >= ``min_changed_pixels``). The archived pilot
    bank has no such filter, so this measures the exposure the filter removes.
    """

    def measure(images: np.ndarray) -> dict:
        rows = np.asarray(
            [
                [
                    changed_pixels(images[index, 0], images[index, state])
                    for state in (1, 2, 3)
                ]
                for index in range(len(images))
            ]
        )
        below = rows < bank_contract.FROZEN_MIN_CHANGED_PIXELS
        return {
            "n_quartets": int(len(rows)),
            "quartets_with_any_state_below_threshold": int(below.any(axis=1).sum()),
            "below_indices": np.flatnonzero(below.any(axis=1)).tolist(),
            "minimum_changed_pixels_per_quartet": rows.min(axis=1).tolist(),
            "changed_pixels_matrix": rows.tolist(),
            "states_a_b_ab_minimum": rows.min(axis=0).tolist(),
        }

    payload = {
        "schema": "mechanism_transfer_v3.m2.quartet_pixel_filter_audit/1",
        "threshold": bank_contract.FROZEN_MIN_CHANGED_PIXELS,
        "rule": "max over RGB channels of |state - P| > 0.2, counted",
        "note": (
            "the archived pilot renders each quartet state with a different background seed, but "
            "the background amplitude (0.45-0.55) stays below the 0.2 channel threshold, so the "
            "count still isolates geometry change"
        ),
        "pilot_bank": str(pilot_bank),
        "expanded_bank": str(bank_dir),
    }
    with np.load(pilot_bank, allow_pickle=False) as data:
        payload["pilot"] = measure(data["quartet_images"])
    with np.load(Path(bank_dir) / "data.npz", allow_pickle=False) as data:
        payload["expanded"] = measure(data["quartet_images"])
    return payload


def pilot_error_correlation(
    pilot_bank: Path = PILOT_BANK, run_root: Path = V2_RUNS, output_size: int = 7
) -> dict:
    """Descriptive audit: archived pilot-bank erasure vs archived v2 J3 correctness."""
    with np.load(pilot_bank, allow_pickle=False) as data:
        quartet_images = data["quartet_images"]
        labels = data["quartet_labels"].astype(np.float64)
        parents = data["quartet_parents"]
    erased = _per_quartet_erasure(quartet_images, output_size, mode="nearest")
    arms = {}
    for arm in ("direct", "additive", "representation", "interaction"):
        arm_rows = {}
        for seed in (803, 805, 806):
            path = run_root / "gpu_run_v2" / f"{arm}_s{seed}" / "predictions.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as prediction:
                logits = prediction["logits"].astype(np.float64)
                pred_labels = prediction["labels"].astype(np.float64)
            if len(pred_labels) != len(labels):
                raise ValueError(f"prediction length mismatch in {path}")
            result = metrics.joint_metrics(logits, labels)
            correct = ((logits > 0) == (labels > 0.5))
            j3_row = (correct[:, 1] & correct[:, 2] & correct[:, 3])
            record = {
                "J3_all": float(j3_row.mean()),
                "J3_erased_subset": float(j3_row[erased].mean()) if erased.any() else None,
                "J3_non_erased_subset": float(j3_row[~erased].mean()) if (~erased).any() else None,
                "n_erased": int(erased.sum()),
                "n_non_erased": int((~erased).sum()),
                "P": result["P"],
            }
            arm_rows[str(seed)] = record
        arms[arm] = arm_rows
    return {
        "schema": "mechanism_transfer_v3.m2.mask_audit.pilot_error_correlation/1",
        "bank": str(pilot_bank),
        "predictions": str(run_root),
        "output_size": output_size,
        "mode": "nearest",
        "n_quartets": int(len(erased)),
        "n_erased_quartets": int(erased.sum()),
        "erased_quartet_indices": np.flatnonzero(erased).tolist(),
        "arms": arms,
        "parents": int(len(set(parents.tolist()))),
        "claim": "none",
        "caveat": (
            "descriptive audit on the 28-quartet / 19-parent archived pilot bank only; the "
            "erased subset has 7 quartets, so no effect is estimable and none is claimed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "mechanism_transfer_v3" / "m2" / "mask_audit.json")
    parser.add_argument("--skip-pilot", action="store_true")
    args = parser.parse_args()
    payload = audit_bank(args.bank)
    payload["quartet_pixel_filter_audit"] = quartet_pixel_filter_audit(bank_dir=args.bank)
    if not args.skip_pilot:
        payload["pilot_error_correlation"] = pilot_error_correlation()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    summary = {
        "out": str(args.out),
        "test_7x7": {
            mode: {
                color: {
                    "erased_rate_of_native_visible": round(
                        payload["splits"]["test"][7][mode][color]["erased_rate_of_native_visible"], 4
                    ),
                    "erased_native_visible": payload["splits"]["test"][7][mode][color][
                        "erased_native_visible"
                    ],
                    "native_visible": payload["splits"]["test"][7][mode][color]["native_visible"],
                }
                for color in COLORS
            }
            for mode in MODES
        },
        "quartets_with_any_erased_state_7x7": payload["quartet_states"][
            "7_quartets_with_any_erased_state"
        ],
        "quartet_count": payload["quartet_states"]["7_quartet_count"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
