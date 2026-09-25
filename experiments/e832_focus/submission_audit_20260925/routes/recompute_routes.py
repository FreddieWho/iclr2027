#!/usr/bin/env python3
"""Recompute available e832_focus evidence from saved, permitted artifacts.

This script reads only Route 1-4 and structure artifacts under e832_focus.
Route 2 is recomputed from archived quartet logits/labels/parent IDs. Routes 1,
3, 4 and the coordinate-structure runs did not retain per-case prediction
arrays, so their formal prediction recomputations are explicitly NOT_RUN.
No training, model loading, data generation, or sealed-pool access occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "artifacts/e832_focus/route2/data/data.npz"
DATA_MANIFEST = ROOT / "artifacts/e832_focus/route2/data/data_manifest.json"
V2_ROOT = ROOT / "artifacts/e832_focus/route2/gpu_run_v2_remote_20260925/e832_focus/route2/gpu_run_v2"
V2_STATIC = ROOT / "artifacts/e832_focus/route2/gpu_run_v2_remote_20260925/e832_focus/route2/gpu_run/static_baseline_v2"
V3_ROOT = ROOT / "artifacts/e832_focus/route2/gpu_run_v3_remote_20260925/gpu_run_v3"
MATCHED_STATIC = ROOT / "artifacts/submission_audit_20260925/gpu_route2/out/static"
ARMS = ("direct", "additive", "representation", "interaction")
SEEDS = (803, 805, 806)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_prediction(path: Path, expected_labels, expected_parents):
    with np.load(path, allow_pickle=False) as z:
        logits = z["logits"].copy()
        labels = z["labels"].copy()
        parents = z["parents"].copy()
    if logits.shape != labels.shape or logits.shape != (len(expected_labels), 4):
        raise ValueError(f"bad prediction shape: {path}: {logits.shape} / {labels.shape}")
    if not np.isfinite(logits).all():
        raise ValueError(f"non-finite logits: {path}")
    if not np.array_equal(labels, expected_labels):
        raise ValueError(f"prediction labels differ from fresh Route-2 bank: {path}")
    if not np.array_equal(parents, expected_parents):
        raise ValueError(f"prediction parents differ from fresh Route-2 bank: {path}")
    return logits, labels, parents


def correctness(logits, labels):
    return (logits > 0) == (labels > 0.5)


def metrics(logits, labels, parents):
    ok = correctness(logits, labels)
    atomic = ok[:, 1] & ok[:, 2]
    joint_abc = atomic & ok[:, 3]
    joint_pabc = joint_abc & ok[:, 0]
    return {
        "n_quartets": int(len(labels)),
        "n_parents": int(len(np.unique(parents))),
        "P": float(ok[:, 0].mean()), "A": float(ok[:, 1].mean()),
        "B": float(ok[:, 2].mean()), "AB": float(ok[:, 3].mean()),
        "atomic_joint": float(atomic.mean()),
        "atomic_denominator": int(atomic.sum()),
        "J3": float(joint_abc.mean()),
        "J4": float(joint_pabc.mean()),
        "CCM_miss_given_atomic": float(1.0 - joint_abc.sum() / atomic.sum()) if atomic.any() else None,
    }


def bootstrap_paired_flow(base_logits, candidate_logits, labels, parents, seed, n_boot=10000):
    """Paired parent-cluster bootstrap; each draw resamples 19 parent groups."""
    b = correctness(base_logits, labels)
    c = correctness(candidate_logits, labels)
    b_atomic = b[:, 1] & b[:, 2]
    c_atomic = c[:, 1] & c[:, 2]
    b_joint = b_atomic & b[:, 3]
    c_joint = c_atomic & c[:, 3]
    baseline_110 = b_atomic & ~b[:, 3]
    full = baseline_110 & c_atomic & c[:, 3]
    migration = baseline_110 & ~c_atomic & c[:, 3]
    baseline_111 = b_joint
    regression_111 = baseline_111 & ~c_joint
    unique = np.unique(parents)
    groups = [np.flatnonzero(parents == parent) for parent in unique]

    point = {
        "baseline110_n": int(baseline_110.sum()),
        "full_repair_n": int(full.sum()),
        "migration_n": int(migration.sum()),
        "endpoint_gain_n": int((baseline_110 & c[:, 3]).sum()),
        "baseline111_n": int(baseline_111.sum()),
        "regression111_n": int(regression_111.sum()),
        "full_repair_fraction_of_110": float(full.sum() / baseline_110.sum()) if baseline_110.any() else None,
        "migration_fraction_of_110": float(migration.sum() / baseline_110.sum()) if baseline_110.any() else None,
        "J3_delta_quartet_weighted": float(c_joint.mean() - b_joint.mean()),
        "CCM_baseline": float(1.0 - b_joint.sum() / b_atomic.sum()) if b_atomic.any() else None,
        "CCM_candidate": float(1.0 - c_joint.sum() / c_atomic.sum()) if c_atomic.any() else None,
        "CCM_delta_candidate_minus_baseline": (
            float((1.0 - c_joint.sum() / c_atomic.sum()) - (1.0 - b_joint.sum() / b_atomic.sum()))
            if b_atomic.any() and c_atomic.any() else None
        ),
    }
    point["endpoint_gain_n_check"] = point["full_repair_n"] + point["migration_n"]
    if point["endpoint_gain_n"] != point["endpoint_gain_n_check"]:
        raise AssertionError("endpoint gain != full repair + migration")

    rng = np.random.default_rng(seed)
    samples = {name: [] for name in (
        "J3_delta_quartet_weighted", "CCM_baseline", "CCM_candidate",
        "CCM_delta_candidate_minus_baseline", "full_repair_fraction_of_110",
        "migration_fraction_of_110",
    )}
    for _ in range(n_boot):
        draw = rng.integers(0, len(groups), size=len(groups))
        ix = np.concatenate([groups[i] for i in draw])
        ba = b_atomic[ix]; ca = c_atomic[ix]
        bj = b_joint[ix]; cj = c_joint[ix]
        h = baseline_110[ix]
        f = full[ix]; m = migration[ix]
        values = {
            "J3_delta_quartet_weighted": float(cj.mean() - bj.mean()),
            "CCM_baseline": float(1.0 - bj.sum() / ba.sum()) if ba.any() else np.nan,
            "CCM_candidate": float(1.0 - cj.sum() / ca.sum()) if ca.any() else np.nan,
            "CCM_delta_candidate_minus_baseline": (
                float((1.0 - cj.sum() / ca.sum()) - (1.0 - bj.sum() / ba.sum()))
                if ba.any() and ca.any() else np.nan
            ),
            "full_repair_fraction_of_110": float(f.sum() / h.sum()) if h.any() else np.nan,
            "migration_fraction_of_110": float(m.sum() / h.sum()) if h.any() else np.nan,
        }
        for name, value in values.items():
            if np.isfinite(value):
                samples[name].append(value)
    intervals = {}
    for name, vals in samples.items():
        intervals[name] = {
            "ci95_parent_cluster_percentile": np.quantile(vals, [.025, .975]).tolist() if vals else None,
            "valid_draws": len(vals),
        }
    return {
        **point,
        "parent_bootstrap": intervals,
        "bootstrap_unit": "19 test parents, paired within seed; repeated parent draws include all that parent's quartets",
        "uncertainty_note": "within-seed interval only; 3 model seeds and 28 quartets do not establish broad replication",
    }


def read_run(version, arm, seed, labels, parents):
    root = V2_ROOT if version == "v2" else V3_ROOT
    prediction = root / f"{arm}_s{seed}/predictions.npz"
    result_path = root / f"{arm}_s{seed}/result.json"
    logits, plabels, pparents = load_prediction(prediction, labels, parents)
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    saved = result.get("metrics") if result else None
    recomputed = metrics(logits, plabels, pparents)
    comparison = None
    if saved is not None:
        comparison = {
            "saved_J3": saved.get("J3"), "recomputed_J3": recomputed["J3"],
            "saved_J4": saved.get("J4"), "recomputed_J4": recomputed["J4"],
            "J3_matches": bool(np.isclose(saved.get("J3", np.nan), recomputed["J3"])),
            "J4_matches": bool(np.isclose(saved.get("J4", np.nan), recomputed["J4"])),
        }
    return {
        "metrics": recomputed,
        "prediction_path": str(prediction.relative_to(ROOT)),
        "prediction_sha256": sha256(prediction),
        "saved_summary_comparison": comparison,
        "logits": logits,
    }


def read_static(root, seed, labels, parents):
    prediction = root / f"direct_static_s{seed}/predictions.npz"
    result_path = root / f"direct_static_s{seed}/result.json"
    if not prediction.exists():
        return None
    logits, plabels, pparents = load_prediction(prediction, labels, parents)
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    return {
        "metrics": metrics(logits, plabels, pparents),
        "prediction_path": str(prediction.relative_to(ROOT)),
        "prediction_sha256": sha256(prediction),
        "result_metadata": {k: result.get(k) for k in (
            "best_epoch", "selection_dev_mode", "selection_dev_n", "train_clean_unique_n",
            "train_exposure_total_per_epoch", "train_exposure_total", "input_contract",
        )},
        "logits": logits,
    }


def route2_audit():
    with np.load(DATA, allow_pickle=False) as data:
        labels = data["quartet_labels"].copy()
        parents = data["quartet_parents"].copy()
    manifest = json.loads(DATA_MANIFEST.read_text())
    versions = {}
    by_version = {}
    prediction_logits = {}
    for version in ("v2", "v3"):
        by_version[version] = {}
        prediction_logits[version] = {}
        summaries = {}
        for arm in ARMS:
            by_version[version][arm] = {}
            prediction_logits[version][arm] = {}
            summaries[arm] = []
            for seed in SEEDS:
                entry = read_run(version, arm, seed, labels, parents)
                prediction_logits[version][arm][str(seed)] = entry["logits"]
                summaries[arm].append(entry["metrics"]["J3"])
                by_version[version][arm][str(seed)] = {k: v for k, v in entry.items() if k != "logits"}
        deltas = {}
        for arm in ARMS[1:]:
            deltas[arm] = {
                "interaction_arm_minus_direct_J3_by_seed": [
                    by_version[version][arm][str(s)]["metrics"]["J3"]
                    - by_version[version]["direct"][str(s)]["metrics"]["J3"] for s in SEEDS
                ]
            }
        versions[version] = {
            "arms_by_seed": by_version[version],
            "mean_J3_by_arm": {arm: float(np.mean(values)) for arm, values in summaries.items()},
            "interaction_contrasts": deltas,
            "n_quartets": int(len(labels)), "n_parents": int(len(np.unique(parents))),
            "seed_note": "means summarize 3 fitted models on the same 28-quartet/19-parent bank; seeds are not data replicates",
        }

    v2_static = {}
    for seed in SEEDS:
        v2_static[str(seed)] = read_static(V2_STATIC, seed, labels, parents)
    matched_static = {}
    for seed in SEEDS:
        matched_static[str(seed)] = read_static(MATCHED_STATIC, seed, labels, parents)
    static_comparison = {
        "legacy_v2_clean_reference": {
            "rows": {s: {k: v for k, v in e.items() if k != "logits"} if e else None for s, e in v2_static.items()},
            "status": "LOWER_EXPOSURE_AND_CLEAN_DEV_SELECTION; descriptive only",
            "training_budget_from_archived_metadata": {
                "main_arm_images_per_epoch": int(manifest["counts"]["train_singleton_images"]),
                "main_arm_epochs": 20,
                "legacy_clean_unique_per_epoch": int(len(json.loads((V2_STATIC / "direct_static_s803/result.json").read_text())["train_clean_indices"])),
                "legacy_clean_dev_n": int(len(json.loads((V2_STATIC / "direct_static_s803/result.json").read_text())["dev_clean_indices"])),
                "main_dev_singleton_n": int(manifest["counts"]["dev_singleton_images"]),
            },
        },
        "matched_v3_clean_reference": {
            "rows": {s: {k: v for k, v in e.items() if k != "logits"} if e else None for s, e in matched_static.items()},
            "status": "AVAILABLE" if all(matched_static.values()) else "NOT_RUN_OR_INCOMPLETE",
            "matching_contract": "128 unique clean train images repeated to 323 presentations per epoch; all 81 singleton dev labels select; 20 epochs; same seed, initialization hash, fresh bank",
            "direct_v3_delta_and_flow_by_seed": {},
        },
    }
    if all(matched_static.values()):
        direct_v3 = by_version["v3"]["direct"]
        seed_flows = {}
        for seed in SEEDS:
            base = matched_static[str(seed)]["logits"]
            candidate = prediction_logits["v3"]["direct"][str(seed)]
            seed_flows[str(seed)] = {
                "baseline_J3": matched_static[str(seed)]["metrics"]["J3"],
                "baseline_CCM_miss_given_atomic": matched_static[str(seed)]["metrics"]["CCM_miss_given_atomic"],
                "direct_v3_J3": direct_v3[str(seed)]["metrics"]["J3"],
                "direct_v3_CCM_miss_given_atomic": direct_v3[str(seed)]["metrics"]["CCM_miss_given_atomic"],
                "paired_direct_v3_vs_clean": bootstrap_paired_flow(
                    base, candidate, labels, parents, seed=832925 + seed,
                ),
            }
        static_comparison["matched_v3_clean_reference"]["direct_v3_delta_and_flow_by_seed"] = seed_flows

    corr_path = ROOT / "artifacts/e832_focus/route2/gpu_run_v2_remote_20260925/corrected_results/corrected_results.json"
    corr = json.loads(corr_path.read_text())
    legacy_correction = {
        "archived_correction_status": corr.get("status"),
        "archived_mismatch_count": corr.get("mismatch_count"),
        "archived_v2_corrected_results_path": str(corr_path.relative_to(ROOT)),
        "mapping": "the preserved v2 postprocessor wrote J3=A&B&AB but copied that same value into J4; recomputed J4 here additionally requires P correct",
        "v2_rows": {},
    }
    for arm in ARMS:
        legacy_correction["v2_rows"][arm] = {}
        for seed in SEEDS:
            saved = corr["rows"][arm][str(seed)]
            recomputed = by_version["v2"][arm][str(seed)]["metrics"]
            legacy_correction["v2_rows"][arm][str(seed)] = {
                "saved_corrected_J3": saved.get("J3"), "recomputed_J3": recomputed["J3"],
                "saved_corrected_J4": saved.get("J4"), "recomputed_J4_PABC": recomputed["J4"],
                "legacy_J4_matches_correct_PABC": bool(np.isclose(saved.get("J4", np.nan), recomputed["J4"])),
            }
    return {
        "status": "RECOMPUTED_FROM_ARCHIVED_PREDICTIONS",
        "retraining": False,
        "data_archive": {
            "path": str(DATA.relative_to(ROOT)), "sha256": sha256(DATA),
            "manifest_sha256": sha256(DATA_MANIFEST),
            "manifest_counts": manifest["counts"],
        },
        "versions": versions,
        "clean_baseline_comparison": static_comparison,
        "legacy_metric_mapping": legacy_correction,
        "denominator_note": "28 quartets nested in 19 parents; CIs resample the 19 parent groups within each seed; do not pool seeds as independent data",
    }


def route1_summary_audit():
    out = {}
    for task in ("source", "T1", "T2"):
        path = ROOT / f"artifacts/e832_focus/route1/round2_{task}.json"
        data = json.loads(path.read_text())
        rows = data["arms"]
        contrasts = {}
        for supervision in ("clean", "flip"):
            by_arm = {}
            for arm in ("rich_unsorted", "rich_sorted"):
                by_arm[arm] = {
                    str(row["seed"]): float(row["metric"]["J3"])
                    for row in rows if row["arm"] == arm and row["supervision"] == supervision
                }
            common_seeds = sorted(set(by_arm["rich_sorted"]) & set(by_arm["rich_unsorted"]))
            contrasts[supervision] = {
                "rich_sorted_minus_rich_unsorted_J3_by_seed": {
                    seed: by_arm["rich_sorted"][seed] - by_arm["rich_unsorted"][seed]
                    for seed in common_seeds
                },
                "all_seed_deltas_positive": all(
                    by_arm["rich_sorted"][s] > by_arm["rich_unsorted"][s] for s in common_seeds
                ),
            }
        out[task] = {
            "status": "SUMMARY_ARITHMETIC_ONLY_RAW_PREDICTIONS_NOT_SAVED",
            "E": data["E"], "E_parents": data["E_parents"],
            "n_test_parents": data["n_test_parents"],
            "seed_metric_contrasts": contrasts,
            "formal_prediction_recomputation": "NOT_RUN: no per-case logits or correctness arrays are archived",
        }
    out["scope_note"] = "Arithmetic re-aggregates saved per-seed JSON metrics; it is not a raw-prediction recomputation or retraining."
    return out


def summary_only(route_name, path, row_count=None):
    p = ROOT / path
    if not p.exists():
        return {"status": "NOT_FOUND", "path": path}
    d = json.loads(p.read_text())
    return {
        "status": "SUMMARY_ARTIFACT_PRESENT_RAW_PREDICTIONS_NOT_ARCHIVED",
        "path": path,
        "prediction_array_files_found_in_expected_root": sorted(
            str(q.relative_to(ROOT)) for q in p.parent.rglob("predictions.npz")
        ),
        "recomputation": "NOT_RUN: only aggregate JSON metrics/contrasts are preserved; re-reading them would not independently reproduce predictions",
        "metadata": {
            "E": d.get("E"), "parents": d.get("parents"),
            "row_count": len(d.get("rows", [])) if isinstance(d.get("rows"), list) else row_count,
            "analytic_parser": d.get("analytic_parser"),
        },
    }


def route3_route4_structure_gaps():
    route3 = ROOT / "artifacts/e832_focus/route3/results.json"
    r3 = json.loads(route3.read_text())
    route3_report = (ROOT / "reports/e832_focus/route3/REPORT.md").read_text().splitlines()
    route3_line9 = route3_report[8] if len(route3_report) >= 9 else ""
    route4 = ROOT / "artifacts/e832_focus/route4/results.json"
    r4 = json.loads(route4.read_text())
    structure = ROOT / "artifacts/e832_focus/structure/metrics_3seed.json"
    continued = ROOT / "artifacts/e832_focus/structure/metrics_continued.json"
    expressivity = ROOT / "artifacts/e832_focus/structure/EXPRESSIVITY_REPO_AUDIT.json"
    return {
        "route3_parser": {
            "status": "SUMMARY_PRESENT_FORMAL_PREDICTION_RECOMPUTATION_NOT_RUN",
            "archived_metadata": {
                "E": r3.get("E"), "unique_source_test_parents": r3.get("parents"),
                "parser_J3": r3.get("analytic_parser", {}).get("J3"),
            },
            "formal_predictions": "NOT_RUN: route3 did not save bank states, parser predictions, or per-case labels alongside the summary",
            "denominator_conflict": {
                "report_line_9": route3_line9,
                "stored_metadata": f"E={r3.get('E')}, unique parent indices={r3.get('parents')}",
                "ledger_wording": "reports/e832_focus/CLAIM_LEDGER.csv line 25 says 230 E / 65 parents",
                "resolution": "mine_E stores the test-scene row index as parent (common_runner.py:79-112); unique stored parent IDs=65. The 142 in report line 9 is a typo; the line should say 65.",
            },
        },
        "route4_rigid_robustness": {
            "status": "SUMMARY_PRESENT_FORMAL_PREDICTION_RECOMPUTATION_NOT_RUN",
            "archived_metadata": {"E": r4.get("E"), "parents": r4.get("parents"), "conditions": r4.get("conditions"), "rows": len(r4.get("rows", []))},
            "formal_predictions": "NOT_RUN: route4 saved aggregate row scores but no logits or per-case correctness arrays",
            "noise": r4.get("noise_diagnostic"),
            "visual_renderer": r4.get("visual_renderer_contract"),
            "small_case_transform_contract": "see targeted test_robustness.py; this is a coordinate-only algebraic check, not a rerun of route4 performance",
        },
        "coordinate_structure": {
            "status": "SUMMARY_AND_ALGEBRAIC_WITNESS_PRESENT_RAW_PREDICTIONS_NOT_ARCHIVED",
            "metrics_3seed": str(structure.relative_to(ROOT)),
            "metrics_continued": str(continued.relative_to(ROOT)),
            "expressivity_audit": str(expressivity.relative_to(ROOT)),
            "formal_prediction_recomputation": "NOT_RUN: stored per-seed summaries/parent-bootstrap contrasts and a four-case algebraic witness are not per-quartet predictions",
            "typed_model_scope": "the segment-sum plus linear readout witness applies to the tested two-segment source architecture; it does not cover TypedTri, TypedDisk, or vision",
        },
        "route3_freshbank": {
            "status": "SUMMARY_PRESENT_FORMAL_PREDICTION_RECOMPUTATION_NOT_RUN",
            "path": "artifacts/e832_focus/route3/round3_freshbank.json",
            "metadata": json.loads((ROOT / "artifacts/e832_focus/route3/round3_freshbank.json").read_text()),
            "formal_predictions": "NOT_RUN: fresh bank states and per-case predictions are not archived",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/submission_audit_20260925/routes/recomputed.json")
    args = parser.parse_args()
    payload = {
        "status": "COMPLETE_WITH_EXPLICIT_NOT_RUN_GAPS",
        "audit_date": "2026-09-25",
        "recomputation_scope": "read-only route evidence audit; no training/download/GPU/model checkpoint use",
        "route1_summary_arithmetic": route1_summary_audit(),
        "route2": route2_audit(),
        "route3_route4_structure": route3_route4_structure_gaps(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.out),
        "route2_bank": payload["route2"]["data_archive"]["manifest_counts"],
        "v3_direct_mean_J3": payload["route2"]["versions"]["v3"]["mean_J3_by_arm"]["direct"],
        "matched_baseline_status": payload["route2"]["clean_baseline_comparison"]["matched_v3_clean_reference"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
