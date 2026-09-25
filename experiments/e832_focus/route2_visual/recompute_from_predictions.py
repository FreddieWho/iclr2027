#!/usr/bin/env python3
"""Recompute Route2 metrics from immutable archived predictions.

The first formal runner wrote J3 as atomic-joint accuracy. Predictions were
saved correctly, so this postprocessor derives corrected J3/J4 and valid
clean-only flow contrasts without retraining or changing test labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("direct", "additive", "representation", "interaction")
SEEDS = (803, 805, 806)


def correct_metrics(logits, labels):
    ok = (logits > 0) == (labels > 0.5)
    atomic = ok[:, 1] & ok[:, 2]
    joint_abc = atomic & ok[:, 3]
    joint_pabc = joint_abc & ok[:, 0]
    return {
        "n_quartets": int(len(labels)),
        "P": float(ok[:, 0].mean()),
        "A": float(ok[:, 1].mean()),
        "B": float(ok[:, 2].mean()),
        "AB": float(ok[:, 3].mean()),
        "atomic_joint": float(atomic.mean()),
        "atomic_denominator": int(atomic.sum()),
        "J3": float(joint_abc.mean()),
        "J4": float(joint_pabc.mean()),
    }


def load_prediction(path):
    with np.load(path, allow_pickle=False) as z:
        return z["logits"].copy(), z["labels"].copy(), z["parents"].copy()


def flow(base_logits, base_labels, candidate_logits, candidate_labels):
    if not np.array_equal(base_labels, candidate_labels):
        raise ValueError("labels differ")
    b = (base_logits > 0) == (base_labels > .5)
    c = (candidate_logits > 0) == (candidate_labels > .5)
    base_atomic = b[:, 1] & b[:, 2]
    cand_atomic = c[:, 1] & c[:, 2]
    h = base_atomic & ~b[:, 3]
    b111 = base_atomic & b[:, 3]
    full = h & cand_atomic & c[:, 3]
    migration = h & c[:, 3] & ~cand_atomic
    return {
        "baseline110_n": int(h.sum()),
        "full_repair_n": int(full.sum()),
        "migration_n": int(migration.sum()),
        "endpoint_gain_n": int((h & c[:, 3]).sum()),
        "baseline111_n": int(b111.sum()),
        "candidate111_n": int((cand_atomic & c[:, 3]).sum()),
        "regression111_n": int((b111 & ~(cand_atomic & c[:, 3])).sum()),
        "J3_delta": float(cand_atomic.mean() - base_atomic.mean()) if False else float(((cand_atomic & c[:, 3]).mean() - (base_atomic & b[:, 3]).mean())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    raw = args.raw_run
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    core = raw / "e832_focus/route2/gpu_run"
    static = raw / "static_baseline/static_baseline"
    corrected = {}
    mismatches = []
    for arm in ARMS:
        corrected[arm] = {}
        for seed in SEEDS:
            d = core / f"{arm}_s{seed}"
            logits, labels, parents = load_prediction(d / "predictions.npz")
            old = json.loads((d / "result.json").read_text())
            m = correct_metrics(logits, labels)
            corrected[arm][str(seed)] = {
                "arm": arm, "seed": seed, "metrics": m,
                "parents": int(len(np.unique(parents))),
                "best_epoch": old["best_epoch"], "best_dev_BCE": old["best_dev_BCE"],
                "train_seconds": old["train_seconds"],
                "parameter_counts": old["parameter_counts"],
                "checkpoint_sha256": old["checkpoint_sha256"],
                "initial_shared_backbone_sha256": old["initial_shared_backbone_sha256"],
            }
            if abs(float(old["metrics"]["J3"]) - m["J3"]) > 1e-12:
                mismatches.append({"arm": arm, "seed": seed, "raw_result_J3": old["metrics"]["J3"], "corrected_J3": m["J3"]})
    baseline = {}
    for seed in SEEDS:
        logits, labels, parents = load_prediction(static / f"direct_static_s{seed}/predictions.npz")
        baseline[str(seed)] = {"logits": logits, "labels": labels, "parents": parents, "metrics": correct_metrics(logits, labels)}
    flows = {}
    for arm in ARMS:
        flows[f"{arm}_vs_clean_static"] = []
        for seed in SEEDS:
            d = core / f"{arm}_s{seed}"
            logits, labels, parents = load_prediction(d / "predictions.npz")
            f = flow(baseline[str(seed)]["logits"], labels, logits, labels)
            f["seed"] = seed
            f["candidate_J3"] = corrected[arm][str(seed)]["metrics"]["J3"]
            f["baseline_J3"] = baseline[str(seed)]["metrics"]["J3"]
            flows[f"{arm}_vs_clean_static"].append(f)
    summary = {"status": "CORRECTED_COMPUTED_NOT_YET_INTERPRETED", "source": str(raw),
               "metric_correction": "J3 now means A&B&AB; raw runner incorrectly used atomic_joint",
               "mismatch_count": len(mismatches), "mismatches": mismatches,
               "arms": corrected, "clean_baseline": {k: v["metrics"] for k, v in baseline.items()},
               "valid_clean_flow": flows}
    (out / "corrected_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    receipt = {"status": "METRIC_CORRECTED_FROM_PREDICTIONS", "retraining": False,
               "source_prediction_root": str(raw), "mismatch_count": len(mismatches),
               "reason": "archived predictions were correct; result J3 field used atomic_joint",
               "claims": "none until independent review"}
    (out / "CORRECTION_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for arm in ARMS:
        vals = [corrected[arm][str(s)]["metrics"]["J3"] for s in SEEDS]
        print(arm, "J3", [round(v, 6) for v in vals], "mean", round(float(np.mean(vals)), 6))
    print("baseline", [round(baseline[str(s)]["metrics"]["J3"], 6) for s in SEEDS])
    print("mismatches", len(mismatches), "out", out)


if __name__ == "__main__":
    main()
