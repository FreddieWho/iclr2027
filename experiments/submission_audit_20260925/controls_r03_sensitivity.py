#!/usr/bin/env python3
"""CPU-only exclusion sensitivity for R03's d09fresh665 single-state hits.

This scores the existing single-edit endpoint bank with already-trained
checkpoints. It does not train models and does not recompute D09 quartet J.
Only the exact bank rows identified by the frozen lineage audit are excluded
for checkpoints trained on the corresponding 4N or 16N mined-flip pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / "artifacts/p123_upgrade/bank/bank_d09fresh665.npz"
LINEAGE = ROOT / "artifacts/e1a933_review/data_lineage/AUGMENTATION_STATE_AUDIT.json"
OUT = ROOT / "artifacts/submission_audit_20260925/controls/r03_single_sensitivity.json"
MODEL_ROOT = ROOT / "artifacts"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_bank():
    b = np.load(BANK, allow_pickle=True)
    required = {"Sx", "Se", "Smeta", "Qx", "Qe", "Qmeta"}
    if not required.issubset(b.files):
        raise ValueError(f"bank fields incomplete: {sorted(set(b.files) ^ required)}")
    sx = np.asarray(b["Sx"], dtype=np.float64)
    se = np.asarray(b["Se"], dtype=np.float64)
    meta = json.loads(str(b["Smeta"]))
    if sx.shape != se.shape or sx.ndim != 3 or sx.shape[1:] != (4, 2):
        raise ValueError(f"unexpected single-bank shapes: {sx.shape}, {se.shape}")
    if len(meta) != len(sx):
        raise ValueError(f"single metadata mismatch: {len(meta)} vs {len(sx)}")
    # The evaluated single-edit state is the endpoint Sx + Se. Sx alone is the
    # parent/start state and is not the overlap target in the lineage audit.
    states = sx + se
    eids = np.asarray([int(m["eid"]) for m in meta], dtype=np.int64)
    labels = np.asarray([int(m["y1"]) for m in meta], dtype=np.int64)
    parents = np.asarray([int(m["parent"]) for m in meta], dtype=np.int64)
    flips = np.asarray([bool(m["flip"]) for m in meta], dtype=bool)
    if len(np.unique(eids)) != len(eids):
        raise ValueError("single-bank eid is not unique")
    return states, eids, labels, parents, flips, meta


def hit_ids_for_checkpoint(lineage: dict, checkpoint: str, record: dict,
                           id_to_row: dict[int, int], meta: list[dict]):
    setkey = record.get("setkey")
    pool = str(record.get("pool"))
    if pool not in {"4N", "16N"}:
        raise ValueError(f"unexpected hit checkpoint mapping: {checkpoint}: {pool=} {setkey=}")
    setkey = setkey if ":" in str(setkey) else f"{pool}:{setkey}"
    if setkey not in lineage.get("sets", {}):
        raise ValueError(f"unknown augmentation set for {checkpoint}: {setkey}")
    bank_record = lineage["sets"][setkey]["banks"]["d09fresh665"]
    detail = bank_record["exact_match_detail"]
    ids = sorted({int(d["bank_id"][1:]) for d in detail["details"]})
    n_model = int(record["banks"]["d09fresh665"]["single"]["n_exact"])
    if len(ids) != n_model:
        raise ValueError(f"hit id count differs for {checkpoint}: {len(ids)} vs {n_model}")
    missing = sorted(set(ids) - id_to_row.keys())
    if missing:
        raise ValueError(f"unknown bank eids for {checkpoint}: {missing[:5]}")
    hit_rows = np.asarray([id_to_row[i] for i in ids], dtype=np.int64)
    if not all(bool(meta[i]["flip"]) and int(meta[i]["y0"]) != int(meta[i]["y1"])
               for i in hit_rows):
        raise ValueError(f"not all hit rows are label-flip single states: {checkpoint}")
    q = bank_record["quartet"]
    if int(q["n_exact"]) != 0 or int(q["n_near"]) != 0:
        raise ValueError(f"quartet bank is not zero-overlap for {checkpoint}")
    return pool, setkey, ids, hit_rows


def feature_batch(x: np.ndarray, ck: dict, fe: str | None, featurize_raw,
                  rel_features):
    if fe == "sixdist":
        f = np.asarray(rel_features(x), dtype=np.float32)[:, :6]
        mu = np.asarray(ck["mu"], dtype=np.float32)
        sd = np.asarray(ck["sd"], dtype=np.float32)
        return torch.from_numpy(((f - mu) / sd).astype(np.float32))
    if fe == "rel12":
        f = np.asarray(rel_features(x), dtype=np.float32)
        mu = np.asarray(ck["mu"], dtype=np.float32)
        sd = np.asarray(ck["sd"], dtype=np.float32)
        return torch.from_numpy(((f - mu) / sd).astype(np.float32))
    return featurize_raw(x, ck)


def summarize(pred: np.ndarray, labels: np.ndarray, parents: np.ndarray,
              flips: np.ndarray, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    if not n:
        return {"n_rows": 0, "accuracy": None, "n_parents": 0}
    ok = pred[mask] == labels[mask]
    fmask = mask & flips
    return {
        "n_rows": n,
        "n_parents": int(len(np.unique(parents[mask]))),
        "n_correct": int(ok.sum()),
        "accuracy": float(ok.mean()),
        "flip_rows": int(fmask.sum()),
        "flip_accuracy": float((pred[fmask] == labels[fmask]).mean()) if fmask.any() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4096)
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 4:
        parser.error("--threads must be between 1 and 4")
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.out.exists():
        parser.error(f"output already exists; choose a new path: {args.out}")
    torch.set_num_threads(args.threads)

    sys.path.insert(0, str(ROOT / "experiments/f095_campaign"))
    sys.path.insert(0, str(ROOT / "experiments/discovery_campaign"))
    from common import rel_features  # noqa: E402
    from u01_eval import featurize_raw, load_arm  # noqa: E402

    states, eids, labels, parents, flips, meta = read_bank()
    lineage = json.loads(LINEAGE.read_text())
    id_to_row = {int(eid): i for i, eid in enumerate(eids)}
    affected = []
    for checkpoint, record in sorted(lineage["checkpoints"].items()):
        single = record.get("banks", {}).get("d09fresh665", {}).get("single", {})
        if int(single.get("n_exact", 0)) > 0:
            affected.append((checkpoint, record))
    if len(affected) != 21:
        raise ValueError(f"expected 21 affected checkpoints, got {len(affected)}")

    exclusion_by_set: dict[str, list[int]] = {}
    exclusions = {}
    prepared = []
    for checkpoint, record in affected:
        pool, setkey, ids, hit_rows = hit_ids_for_checkpoint(
            lineage, checkpoint, record, id_to_row, meta)
        previous = exclusion_by_set.setdefault(setkey, ids)
        if previous != ids:
            raise ValueError(f"same augmentation set has inconsistent bank ids: {setkey}")
        if pool in exclusions and exclusions[pool] != ids:
            raise ValueError(f"models in train pool {pool} do not share one hit list")
        exclusions[pool] = ids
        ckpt = MODEL_ROOT / Path(checkpoint).relative_to("artifacts")
        model_dir = ckpt.parent
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        prepared.append((checkpoint, record, pool, setkey, ids, hit_rows, model_dir, ckpt))

    # Check the two pools separately. They may share bank eids; each model is
    # filtered only against its own training pool, never the union.
    n_by_pool = {p: len(ids) for p, ids in exclusions.items()}
    started = time.time()
    result = {
        "schema": "submission_audit_20260925.r03_single_sensitivity/1",
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "computation": {
            "device": "cpu", "threads": args.threads, "batch_size": args.batch,
            "training": "NOT_RUN", "evaluated_state": "Sx + Se (single-edit endpoint)",
            "start_states_scored": False,
            "n_models": len(prepared), "n_single_rows_per_model": len(states),
            "state_forward_count_expected": len(prepared) * len(states),
            "quartet_j_reestimated": False,
        },
        "inputs": {
            str(BANK.relative_to(ROOT)): sha256(BANK),
            str(LINEAGE.relative_to(ROOT)): sha256(LINEAGE),
        },
        "single_bank": {
            "n_rows": len(states), "n_unique_eids": int(len(np.unique(eids))),
            "n_parents": int(len(np.unique(parents))),
            "n_flip_rows": int(flips.sum()),
        },
        "overlap_by_train_pool": {
            p: {"n_exact_eids": n_by_pool[p],
                "excluded_eids": ids,
                "n_overlap_parents": int(len({meta[id_to_row[i]]["parent"] for i in ids}))}
            for p, ids in sorted(exclusions.items())
        },
        "overlap_set_intersection": {
            "n_shared_4N_16N_eids": len(set(exclusions.get("4N", [])) & set(exclusions.get("16N", []))),
            "note": "intersection is descriptive; per-model exclusion uses its own train pool",
        },
        "d09_quartet_scope": {
            "n_quartets": 236,
            "quartet_overlap_exact": 0,
            "quartet_overlap_near": 0,
            "denominator_after_single_bank_exclusion": 236,
            "reestimated": False,
            "basis": "lineage audit quartet-state records; no single-row exclusion applied to quartets",
        },
        "models": [],
        "progress": {"completed_models": 0, "total_models": len(prepared)},
    }
    write_atomic(args.out, result)

    for ix, (checkpoint, record, pool, setkey, ids, hit_rows, model_dir, ckpt_path) in enumerate(prepared, 1):
        model, ck = load_arm(model_dir)
        fe = ck.get("featurize")
        yhat = np.empty(len(states), dtype=np.int8)
        for lo in range(0, len(states), args.batch):
            hi = min(len(states), lo + args.batch)
            f = feature_batch(states[lo:hi], ck, fe, featurize_raw, rel_features)
            with torch.inference_mode():
                scores = model(f).reshape(-1).cpu().numpy()
            yhat[lo:hi] = scores > 0

        hit = np.zeros(len(states), dtype=bool)
        hit[hit_rows] = True
        retained = ~hit
        all_metrics = summarize(yhat, labels, parents, flips, np.ones(len(states), bool))
        retained_metrics = summarize(yhat, labels, parents, flips, retained)
        hit_metrics = summarize(yhat, labels, parents, flips, hit)
        correct = (yhat == labels).astype(float)
        parent_deltas = []
        for parent in np.unique(parents):
            pm = parents == parent
            parent_deltas.append(float(correct[pm & retained].mean() - correct[pm].mean()))
        row = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": sha256(ckpt_path),
            "train_pool": pool,
            "augmentation_set": setkey,
            "featurize": fe or "raw_flattened",
            "n_exact_hits": int(hit.sum()),
            "excluded_bank_eids": ids,
            "full_bank": all_metrics,
            "excluded_hits": retained_metrics,
            "hit_subset": hit_metrics,
            "paired_differences": {
                "row_weighted_excluded_minus_full_accuracy":
                    retained_metrics["accuracy"] - all_metrics["accuracy"],
                "row_weighted_hit_minus_nonhit_accuracy":
                    hit_metrics["accuracy"] - retained_metrics["accuracy"],
                "parent_equal_excluded_minus_full_accuracy":
                    float(np.mean(parent_deltas)),
                "n_parent_clusters": int(len(parent_deltas)),
                "ci": "NOT_COMPUTED; rows are nested in parents and this diagnostic is not a D09 quartet estimand",
            },
        }
        result["models"].append(row)
        result["progress"] = {"completed_models": ix, "total_models": len(prepared)}
        result["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_atomic(args.out, result)
        print(json.dumps({"completed": ix, "of": len(prepared), "checkpoint": checkpoint,
                          "n_hit": int(hit.sum()),
                          "full_acc": round(all_metrics["accuracy"], 6),
                          "excluded_acc": round(retained_metrics["accuracy"], 6),
                          "hit_acc": round(hit_metrics["accuracy"], 6)},
                         separators=(",", ":")), flush=True)
        del model, ck, yhat

    result["status"] = "COMPLETE"
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["elapsed_seconds"] = round(time.time() - started, 3)
    result["computation"]["state_forward_count_actual"] = len(prepared) * len(states)
    result["limitations"] = [
        "This is a new single-edit endpoint accuracy diagnostic, not a re-estimate of D09 quartet J.",
        "The d09fresh665 single-state bank is evaluated row-weighted; rows share parent scenes.",
        "The 4N and 16N exclusion lists are applied separately; their union is never removed from every model.",
        "No statistical interval is claimed for the newly computed single-state diagnostic.",
    ]
    write_atomic(args.out, result)
    print(json.dumps({"status": result["status"], "output": str(args.out),
                      "models": len(result["models"]),
                      "state_forward_count": result["computation"]["state_forward_count_actual"],
                      "elapsed_seconds": result["elapsed_seconds"]}, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
