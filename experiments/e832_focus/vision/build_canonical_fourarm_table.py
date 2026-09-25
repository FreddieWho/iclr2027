#!/usr/bin/env python3
"""Read-only four-arm table for the finished canonical224 ResNet18 GPU runs.

No training, no GPU, no new renderer, no sealed-pool reads. Missing fields stay
the word ``missing``. They are not filled from the 178-quartet N02 bank, and
J4 is not derived from logits. The stored field named CCM is not copied into
this table: in these files it is conditional joint accuracy, not the paper's
conditional composition miss.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "artifacts/e1a933_review/vision_gpu_interpretation"
MATRIX = SRC / "cuda_matrix"
DATA = ROOT / "artifacts/e1a933_review/vision_canonical224_v2/data.npz"
N02_ARM = ROOT / "artifacts/e1a933_review/N02_interpretation/arm_parent_values.csv"
N02_MANIFEST = ROOT / "artifacts/e1a933_review/N02_interpretation/run_manifest.json"
OUT = ROOT / "artifacts/e832_focus/vision"
MISSING = "missing"
SEEDS = (803, 805, 806)
ARMS = (
    ("pretrained", "static"),
    ("pretrained", "flip"),
    ("random", "static"),
    ("random", "flip"),
)
CONTRASTS = (
    ("pretrained", "pretrained_flip", "pretrained_static"),
    ("random", "random_flip", "random_static"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def arm_name(init: str, regimen: str) -> str:
    return f"{init}_{regimen}"


def load_predictions(init: str, regimen: str, seed: int):
    path = MATRIX / f"{arm_name(init, regimen)}_s{seed}" / "predictions.npz"
    with np.load(path) as blob:
        return {
            "logits": blob["logits"].astype(np.float64),
            "labels": blob["labels"].astype(np.int64),
            "parent": blob["parent"].astype(np.int64),
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }


def decisions(pred: dict) -> np.ndarray:
    return (pred["logits"] > 0) == pred["labels"]


def e_mask(labels: np.ndarray) -> np.ndarray:
    # Paper E: both atomics preserve the base label and the joint flips it.
    return (labels[:, 1] == labels[:, 0]) & (labels[:, 2] == labels[:, 0]) & (labels[:, 3] != labels[:, 0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((SRC / "independent_summary.json").read_text())
    paired = json.loads((MATRIX / "paired_analysis.json").read_text())
    by_run = {(row["arm"], row["seed"]): row for row in summary["per_run"]}

    loaded = {
        (init, regimen, seed): load_predictions(init, regimen, seed)
        for init, regimen in ARMS
        for seed in SEEDS
    }
    reference = loaded[("pretrained", "static", 803)]
    labels = reference["labels"]
    parents = reference["parent"]
    e_rows = e_mask(labels)
    if not bool(e_rows.all()):
        raise SystemExit("quartet file is not entirely E; stored J must not be renamed J3")
    n_quartets = int(len(labels))
    parent_ids = np.unique(parents)
    n_parents = int(len(parent_ids))
    checks = []

    for init, regimen, seed in loaded:
        pred = loaded[(init, regimen, seed)]
        if not np.array_equal(pred["labels"], labels) or not np.array_equal(pred["parent"], parents):
            raise SystemExit(f"test labels or parents differ: {init} {regimen} {seed}")
        ok = decisions(pred)
        atomic = ok[:, 1] & ok[:, 2]
        joint = atomic & ok[:, 3]
        result_path = MATRIX / f"{arm_name(init, regimen)}_s{seed}" / "result.json"
        result = json.loads(result_path.read_text())
        recomputed = {
            "n": int(len(labels)),
            "P": float(ok[:, 0].mean()),
            "A": float(ok[:, 1].mean()),
            "B": float(ok[:, 2].mean()),
            "atomic_joint": float(atomic.mean()),
            "AB": float(ok[:, 3].mean()),
            "J": float(joint.mean()),
        }
        for key, value in recomputed.items():
            if key == "n":
                if int(result[key]) != value or int(by_run[(arm_name(init, regimen), seed)]["n"]) != value:
                    raise SystemExit(f"n mismatch {init} {regimen} {seed}")
            else:
                if abs(float(result[key]) - value) > 1e-12:
                    raise SystemExit(f"result.json {key} mismatch {init} {regimen} {seed}")
                if abs(float(by_run[(arm_name(init, regimen), seed)][key]) - value) > 1e-12:
                    raise SystemExit(f"per_run {key} mismatch {init} {regimen} {seed}")
        if "J3" in result or "J4" in result:
            raise SystemExit("unexpected J3/J4 field; table rule must be revisited")
        checks.append({
            "arm": arm_name(init, regimen),
            "seed": seed,
            "matches_result_json": True,
            "matches_per_run": True,
            "J_count": int(joint.sum()),
            "atomic_count": int(atomic.sum()),
            "AB_count": int(ok[:, 3].sum()),
            "source_result": str(result_path.relative_to(ROOT)),
        })

    arm_rows = []
    arm_parent_rows = []
    for init, regimen in ARMS:
        for seed in SEEDS:
            pred = loaded[(init, regimen, seed)]
            ok = decisions(pred)
            atomic = ok[:, 1] & ok[:, 2]
            joint = atomic & ok[:, 3]
            official = by_run[(arm_name(init, regimen), seed)]
            arm_rows.append({
                "bank": "canonical224_159",
                "init": init,
                "regimen": regimen,
                "seed": seed,
                "scope": "pooled_quartets",
                "n_quartets": n_quartets,
                "n_parents": n_parents,
                "J3_count": int(joint.sum()),
                "J3_denominator": n_quartets,
                "J3": official["J"],
                "J3_source_field": "J",
                "J4": MISSING,
                "atomic_correct_n": int(atomic.sum()),
                "atomic_denominator": n_quartets,
                "atomic_accuracy": official["atomic_joint"],
                "atomic_source_field": "atomic_joint",
                "AB_correct_n": int(ok[:, 3].sum()),
                "AB_denominator": n_quartets,
                "AB_accuracy": official["AB"],
                "AB_source_field": "AB",
                "full_repair": MISSING,
                "migration": MISSING,
                "baseline111_regression": MISSING,
                "paper_CCM": MISSING,
                "backend": official["backend"],
                "dtype": official["dtype"],
                "source_file": f"artifacts/e1a933_review/vision_gpu_interpretation/cuda_matrix/{arm_name(init, regimen)}_s{seed}/result.json",
            })
            for parent in parent_ids:
                mask = parents == parent
                denom = int(mask.sum())
                arm_parent_rows.append({
                    "bank": "canonical224_159",
                    "init": init,
                    "regimen": regimen,
                    "seed": seed,
                    "scope": "parent",
                    "parent": int(parent),
                    "n_quartets": denom,
                    "J3_count": int(joint[mask].sum()),
                    "J3_denominator": denom,
                    "J3": float(joint[mask].mean()) if denom else MISSING,
                    "J3_source_field": "predictions.npz logits/labels disaggregated; sums locked to result.json J",
                    "J4": MISSING,
                    "atomic_correct_n": int(atomic[mask].sum()),
                    "atomic_denominator": denom,
                    "atomic_accuracy": float(atomic[mask].mean()) if denom else MISSING,
                    "atomic_source_field": "atomic_joint",
                    "AB_correct_n": int(ok[mask, 3].sum()),
                    "AB_denominator": denom,
                    "AB_accuracy": float(ok[mask, 3].mean()) if denom else MISSING,
                    "AB_source_field": "AB",
                    "full_repair": MISSING,
                    "migration": MISSING,
                    "baseline111_regression": MISSING,
                    "paper_CCM": MISSING,
                    "source_file": f"artifacts/e1a933_review/vision_gpu_interpretation/cuda_matrix/{arm_name(init, regimen)}_s{seed}/predictions.npz",
                })
        mean_j = summary["classification"][arm_name(init, regimen)]["J_mean"]
        arm_rows.append({
            "bank": "canonical224_159",
            "init": init,
            "regimen": regimen,
            "seed": "seed_mean",
            "scope": "unweighted_mean_of_3_seed_rates",
            "n_quartets": n_quartets,
            "n_parents": n_parents,
            "J3_count": MISSING,
            "J3_denominator": "3 seed rates, each over 159; not a 477 pool",
            "J3": mean_j,
            "J3_source_field": "classification.J_mean",
            "J4": MISSING,
            "atomic_correct_n": MISSING,
            "atomic_denominator": MISSING,
            "atomic_accuracy": MISSING,
            "atomic_source_field": MISSING,
            "AB_correct_n": MISSING,
            "AB_denominator": MISSING,
            "AB_accuracy": MISSING,
            "AB_source_field": MISSING,
            "full_repair": MISSING,
            "migration": MISSING,
            "baseline111_regression": MISSING,
            "paper_CCM": MISSING,
            "backend": "cuda",
            "dtype": "FP32",
            "source_file": "artifacts/e1a933_review/vision_gpu_interpretation/independent_summary.json",
        })

    contrast_rows = []
    contrast_parent_rows = []
    answer = []
    for init, candidate, baseline in CONTRASTS:
        block = paired[f"{candidate} minus {baseline}"]
        for row in block["seeds"]:
            seed = int(row["seed"])
            base = decisions(loaded[(init, "static", seed)])
            flip = decisions(loaded[(init, "flip", seed)])
            h = base[:, 1] & base[:, 2] & ~base[:, 3]
            full = flip[:, 1:].all(1) & h
            migration = flip[:, 3] & ~(flip[:, 1] & flip[:, 2]) & h
            neither = h & ~flip[:, 3]
            baseline111 = base[:, 1:].all(1)
            regressed = baseline111 & ~flip[:, 1:].all(1)
            recomputed_counts = {
                "baseline110_n": int(h.sum()),
                "full_repair_n": int(full.sum()),
                "migration_n": int(migration.sum()),
                "baseline111_n": int(baseline111.sum()),
                "111_regression_n": int(regressed.sum()),
            }
            for key, value in recomputed_counts.items():
                if int(row[key]) != value:
                    raise SystemExit(f"paired count mismatch {init} {seed} {key}")
            if int(full.sum() + migration.sum() + neither.sum()) != int(h.sum()):
                raise SystemExit(f"H partition failed {init} {seed}")
            if int((full & migration).sum()) != 0:
                raise SystemExit(f"full repair and migration overlap {init} {seed}")
            endpoint = int(full.sum() + migration.sum())
            contrast_rows.append({
                "bank": "canonical224_159",
                "init": init,
                "contrast": "flip_minus_static",
                "seed": seed,
                "scope": "pooled_quartets",
                "n_quartets": n_quartets,
                "n_parents": n_parents,
                "baseline110_n": int(h.sum()),
                "baseline110_denominator_note": "static A and B correct, static AB wrong, among 159 E-quartets",
                "full_repair_n": int(full.sum()),
                "full_repair_denominator": int(h.sum()),
                "full_repair": f"{int(full.sum())}/{int(h.sum())}",
                "migration_n": int(migration.sum()),
                "migration_denominator": int(h.sum()),
                "migration": f"{int(migration.sum())}/{int(h.sum())}",
                "endpoint_gain_n": endpoint,
                "endpoint_gain_denominator": int(h.sum()),
                "full_repair_among_endpoint_gains": (
                    f"{int(full.sum())}/{endpoint}" if endpoint else MISSING
                ),
                "H_neither_endpoint_n": int(neither.sum()),
                "baseline111_n": int(baseline111.sum()),
                "baseline111_denominator_note": "static A, B, and AB all correct, among 159 E-quartets",
                "regression111_n": int(regressed.sum()),
                "baseline111_regression": f"{int(regressed.sum())}/{int(baseline111.sum())}",
                "J_difference": row["J_difference"],
                "parent_bootstrap95_low": row["parent_bootstrap95"][0],
                "parent_bootstrap95_high": row["parent_bootstrap95"][1],
                "J4": MISSING,
                "paper_CCM": MISSING,
                "source_file": "artifacts/e1a933_review/vision_gpu_interpretation/cuda_matrix/paired_analysis.json",
            })
            answer.append({
                "init": init,
                "seed": seed,
                "baseline110_n": int(h.sum()),
                "full_repair_n": int(full.sum()),
                "migration_n": int(migration.sum()),
                "endpoint_gain_n": endpoint,
                "full_among_endpoint": f"{int(full.sum())}/{endpoint}",
                "full_exceeds_migration": bool(full.sum() > migration.sum()),
                "regression111_n": int(regressed.sum()),
                "baseline111_n": int(baseline111.sum()),
            })
            for parent in parent_ids:
                mask = parents == parent
                h_n = int(h[mask].sum())
                full_n = int(full[mask].sum())
                mig_n = int(migration[mask].sum())
                end_n = full_n + mig_n
                b111 = int(baseline111[mask].sum())
                reg_n = int(regressed[mask].sum())
                contrast_parent_rows.append({
                    "bank": "canonical224_159",
                    "init": init,
                    "contrast": "flip_minus_static",
                    "seed": seed,
                    "scope": "parent",
                    "parent": int(parent),
                    "n_quartets": int(mask.sum()),
                    "baseline110_n": h_n,
                    "full_repair_n": full_n,
                    "full_repair_denominator": h_n,
                    "full_repair": f"{full_n}/{h_n}" if h_n else MISSING,
                    "migration_n": mig_n,
                    "migration_denominator": h_n,
                    "migration": f"{mig_n}/{h_n}" if h_n else MISSING,
                    "endpoint_gain_n": end_n,
                    "full_repair_among_endpoint_gains": (
                        f"{full_n}/{end_n}" if end_n else MISSING
                    ),
                    "H_neither_endpoint_n": int(neither[mask].sum()),
                    "baseline111_n": b111,
                    "regression111_n": reg_n,
                    "baseline111_regression": f"{reg_n}/{b111}" if b111 else MISSING,
                    "J4": MISSING,
                    "paper_CCM": MISSING,
                    "source_file": "artifacts/e1a933_review/vision_gpu_interpretation/cuda_matrix/paired_analysis.json",
                    "disaggregation_file": f"artifacts/e1a933_review/vision_gpu_interpretation/cuda_matrix/{init}_static_s{seed}/predictions.npz",
                })

    # Exposure caveat from the frozen training index rule, same data file.
    with np.load(DATA) as data:
        train_clean = data["train_clean"].astype(bool)
        train_parents = data["train_parents"].astype(np.int64)
        dev_parents = data["dev_parents"].astype(np.int64)
        quartet_parents = data["quartet_parents"].astype(np.int64)
        quartet_labels = data["quartet_labels"].astype(np.int64)
    if not np.array_equal(quartet_labels, labels) or not np.array_equal(quartet_parents, parents):
        raise SystemExit("prediction labels/parents differ from canonical224_v2 data.npz")
    clean_ix = np.flatnonzero(train_clean)
    static_ix = np.resize(clean_ix, len(train_parents))
    flip_ix = np.arange(len(train_parents))
    parent_max = int(max(train_parents.max(), 0))
    static_counts = np.bincount(train_parents[static_ix], minlength=parent_max + 1)
    flip_counts = np.bincount(train_parents[flip_ix], minlength=parent_max + 1)
    present = (static_counts + flip_counts) > 0
    mismatch = static_counts[present] != flip_counts[present]
    exposure = {
        "statement": "total exposure match is not per-parent exposure match",
        "source_data": str(DATA.relative_to(ROOT)),
        "data_sha256": sha256(DATA),
        "index_rule": "experiments/e1a933_review/vision_cuda_protocol.py train_arm: static uses np.resize(clean_indices, n_images); flip uses every training image once per epoch",
        "n_train_images": int(len(train_parents)),
        "n_clean_images": int(train_clean.sum()),
        "n_single_flip_images": int((~train_clean).sum()),
        "slots_per_epoch_static": int(len(static_ix)),
        "slots_per_epoch_flip": int(len(flip_ix)),
        "optimizer_steps_per_epoch": int(len(train_parents) // 32),
        "epochs": 20,
        "optimizer_steps_total": int(len(train_parents) // 32) * 20,
        "n_train_parents": int(len(np.unique(train_parents))),
        "n_train_parents_with_unequal_static_flip_slots": int(mismatch.sum()),
        "max_abs_slot_difference": int(np.abs(static_counts - flip_counts).max()),
        "n_dev_images": 707,
        "n_dev_parents": int(len(np.unique(dev_parents))),
        "note": "Same total slots and same step count. Static repeats clean images and never shows single-flip images; flip shows each training image once. Parent integer IDs restart inside each split, so they are not cross-split scene IDs.",
    }
    if exposure["slots_per_epoch_static"] != 2816 or exposure["optimizer_steps_per_epoch"] != 88:
        raise SystemExit("exposure totals do not match the already reported 2816 images / 88 steps")

    n02_parents = set()
    n02_rows = 0
    with N02_ARM.open() as handle:
        for row in csv.DictReader(handle):
            if row["arm"] == "A_standard_pretrained_flip" and row["seed"] == "803" and row["stratum"] == "all":
                n02_parents.add(row["parent"])
                n02_rows += int(row["n_rows"])
    n02_manifest = json.loads(N02_MANIFEST.read_text())
    bank = {
        "canonical224": {
            "n_quartets": n_quartets,
            "n_parents": n_parents,
            "quartets_per_parent_min": int(np.bincount(parents)[parent_ids].min()),
            "quartets_per_parent_max": int(np.bincount(parents)[parent_ids].max()),
            "data_sha256": exposure["data_sha256"],
            "renderer": "canonical endpoint order, visible red/blue segments, native 64 bilinear to 224",
            "backend": "cuda",
            "dtype": "FP32",
            "used_for_table": True,
        },
        "N02": {
            "n_quartets": n02_rows,
            "n_parents_one_arm_seed_stratum_all": len(n02_parents),
            "identity_only_source": str(N02_ARM.relative_to(ROOT)),
            "data_sha256": n02_manifest["inputs"]["data_npz_sha256"],
            "used_for_table": False,
            "note": "Different observation matrix and different data file. No N02 number was copied into the four-arm table. A difference from the 159 bank is not an improvement.",
        },
    }
    if n02_rows != 178:
        raise SystemExit(f"unexpected N02 row count {n02_rows}")
    if bank["N02"]["data_sha256"] == bank["canonical224"]["data_sha256"]:
        raise SystemExit("159 and 178 banks unexpectedly share a data hash")

    field_presence = {
        "J3": "stored field is J. All 159 prediction rows match paper E (base label preserved by both atomics, flipped by AB), so this J is paper J3. Denominator 159.",
        "J4": "missing. No J4 field in result.json, per_run, or paired_analysis.json. Base-state marginal P is not J4 and was not substituted. J4 was not derived from logits.",
        "atomic_accuracy": "stored field atomic_joint: both atomic states correct. Marginal A and B exist but are not this column.",
        "AB_accuracy": "stored field AB, quartet state 3.",
        "full_repair": "contrast field only, paired_analysis.json full_repair_n over baseline110_n. Absent on a single arm; those cells are missing.",
        "migration": "contrast field only, paired_analysis.json migration_n over baseline110_n. Absent on a single arm; those cells are missing.",
        "baseline111_regression": "contrast field only, paired_analysis.json 111_regression_n over baseline111_n. Absent on a single arm; those cells are missing.",
        "paper_CCM": "missing. The paper CCM is a conditional composition miss rate. The stored field named CCM is joint_count / atomic_count, which is conditional joint accuracy. It was not renamed CCM and was not inverted to fill the miss rate.",
        "per_parent": "not stored as rows. Disaggregated from the same predictions.npz files. Seed sums are locked to result.json and paired_analysis.json. This is the 159 bank, not the 178 bank.",
    }
    verification = {
        "status": "PASS",
        "e_rows": int(e_rows.sum()),
        "e_denominator": n_quartets,
        "n_parents": n_parents,
        "arm_checks": checks,
        "paired_counts_match": True,
        "h_partition_identity": "full_repair_n + migration_n + H_neither_endpoint_n = baseline110_n",
        "prediction_sha256": {
            f"{init}_{regimen}_s{seed}": loaded[(init, regimen, seed)]["sha256"]
            for init, regimen, seed in loaded
        },
    }
    counts = np.bincount(parents)
    numbers = {
        "bank": "canonical224_159",
        "n_quartets": n_quartets,
        "n_parents": n_parents,
        "parent_quartet_count_histogram": {
            str(k): int((counts == k).sum()) for k in range(1, int(counts.max()) + 1)
        },
        "endpoint_gains": answer,
        "all_six_seeds_full_repair_exceeds_migration": all(item["full_exceeds_migration"] for item in answer),
        "J_seed_means": {
            arm_name(init, regimen): summary["classification"][arm_name(init, regimen)]["J_mean"]
            for init, regimen in ARMS
        },
    }

    write_csv(OUT / "arm_pooled.csv", arm_rows)
    write_csv(OUT / "arm_per_parent.csv", arm_parent_rows)
    write_csv(OUT / "contrast_pooled.csv", contrast_rows)
    write_csv(OUT / "contrast_per_parent.csv", contrast_parent_rows)
    (OUT / "field_presence.json").write_text(json.dumps(field_presence, indent=2) + "\n")
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    (OUT / "exposure_caveat.json").write_text(json.dumps(exposure, indent=2) + "\n")
    (OUT / "bank_separation.json").write_text(json.dumps(bank, indent=2) + "\n")
    (OUT / "numbers_for_decision.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "out": str(OUT.relative_to(ROOT)), "endpoint": answer}, indent=2))


if __name__ == "__main__":
    main()
