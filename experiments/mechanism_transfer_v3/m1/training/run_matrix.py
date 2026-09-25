#!/usr/bin/env python3
"""Driver for the M1.2 source representation matrix (CPU only).

Stages:
  probe    time a few representative cells and project the full grid
  train    run the frozen recipe grid (6 arms x 3 seeds x 2 supervisions x 3 lrs)
  select   pick the dev-BCE-minimal lr per cell; never reads test J
  eval     score the frozen bank and write contrasts, flows, and intervals
  summary  print the compact per-arm / per-contrast table
  all      train -> select -> eval -> summary

No stage trains the zero-train group-mean control, touches a GPU, or reads a
sealed pool.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
for path in (str(ROOT / "experiments" / "mechanism_transfer_v3"), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluate import CONTRASTS, evaluate  # noqa: E402
from features import ARMS, orbit_tie_fraction  # noqa: E402
from recipe import (  # noqa: E402
    ART,
    EPOCHS,
    LRS,
    NON_CLAIMS,
    SEEDS,
    SUPERVISIONS,
    THREADS,
    batch_contract,
    load_data,
    select,
    sha256_file,
    train_cell,
)
from models import COORD_MLP_SOURCE, build_model, count_parameters, macs_per_example  # noqa: E402


def probe(data: dict, cells: int = 2) -> dict:
    """Time representative expensive cells and project the full grid cost."""
    timings = []
    plan = [
        ("repaired_segment_rho", 11, "flip", 0.001),
        ("segment_moment", 11, "flip", 0.001),
        ("rich8_sorted", 11, "flip", 0.001),
        ("orbit_distance", 11, "flip", 0.001),
    ][:cells]
    for arm, seed, supervision, lr in plan:
        t0 = time.time()
        receipt = train_cell(arm, seed, supervision, lr, data, force=True)
        timings.append(
            {
                "arm": arm,
                "supervision": supervision,
                "seconds": time.time() - t0,
                "epochs": receipt["epochs"],
                "n_train": receipt["n_train_clean"] + receipt["n_train_flip"],
                "macs_per_example": receipt["macs_per_example"],
            }
        )
    per_cell = float(np.mean([row["seconds"] for row in timings]))
    n_cells = len(ARMS) * len(SEEDS) * len(SUPERVISIONS) * len(LRS)
    record = {
        "schema": "mechanism_transfer_v3.m1.training_probe/1",
        "threads": THREADS,
        "epochs": EPOCHS,
        "timings": timings,
        "mean_cell_seconds": per_cell,
        "n_grid_cells": n_cells,
        "projected_grid_seconds": per_cell * n_cells,
        "projected_grid_hours": per_cell * n_cells / 3600.0,
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "probe.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return record


def seam_diagnostic(data: dict) -> dict:
    """Tie/seam rate of the lexicographic orbit representative on real populations."""
    bank = data["bank"]
    populations = {
        "train_scenes": data["train_positions"],
        "bank_A_states": np.asarray(bank["states"][1]),
        "bank_P_states": np.asarray(bank["states"][0]),
    }
    record = {
        "schema": "mechanism_transfer_v3.m1.orbit_seam/1",
        "atol": 1e-9,
        "populations": {
            name: {
                "n_scenes": int(len(scenes)),
                "tie_fraction": orbit_tie_fraction(scenes),
            }
            for name, scenes in populations.items()
        },
        "note": (
            "the lexicographic whole-orbit representative is discontinuous where the "
            "minimum is not unique; the rate has to be reported next to the arm's "
            "numbers because the arm is only piecewise smooth"
        ),
    }
    (ART / "orbit_seam.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return record


def capacity(data: dict) -> dict:
    """Parameter and MAC accounting per arm, plus input hashes."""
    rows = {}
    for arm in ARMS:
        model, input_dim = build_model(arm)
        rows[arm] = {
            "in_dim": input_dim,
            "n_parameters": count_parameters(model),
            "macs_per_example": macs_per_example(model, input_dim),
        }
    record = {
        "schema": "mechanism_transfer_v3.m1.training_capacity/1",
        "arms": rows,
        "coord_mlp_source": str(COORD_MLP_SOURCE.relative_to(ROOT)),
        "coord_mlp_sha256": sha256_file(COORD_MLP_SOURCE),
        "note": (
            "raw, rich8_*, and orbit_distance share CoordMLP(64, 32); "
            "segment_moment counts its symmetric double evaluation; "
            "macs are counted from actual Linear calls, not estimated"
        ),
    }
    (ART / "capacity.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def write_manifest(data: dict, cap: dict) -> dict:
    manifest = {
        "schema": "mechanism_transfer_v3.m1.training_manifest/1",
        "queue": "EXPERIMENTS.md Queue 2 (M1 role-preserving invariance, source contract)",
        "recipe": {
            "epochs": EPOCHS,
            "lrs": list(LRS),
            "seeds": list(SEEDS),
            "supervisions": list(SUPERVISIONS),
            "threads": THREADS,
            "optimizer": "Adam, full batch, one step per epoch",
            "loss": "BCEWithLogits (clean only, or clean + all mined single flips)",
            "selection": "min final pooled static+single dev BCE; smaller lr wins ties",
            "decision_rule": "logit > 0",
            "frozen_origin": "reports/e832_focus/STRUCTURE_DECISION.md training contract",
        },
        "arms": list(ARMS),
        "role_preserving_references": ["orbit_distance", "segment_moment"],
        "zero_train_control_not_trained": {
            "name": "group_mean",
            "reference": "artifacts/mechanism_transfer_v3/m1/reference_eval.json",
            "why": "finite-group mean invariance is constructed; it is only a budgeted zero-train control",
        },
        "sources": data["sources"],
        "capacity": cap["arms"],
        "batch_contract": batch_contract(data),
        "code_sha256": {
            name: sha256_file(HERE / name)
            for name in ("features.py", "models.py", "recipe.py", "evaluate.py", "run_matrix.py")
        },
        "group_elements_are_not_replicates": True,
        "seeds_are_not_datasets": True,
        "non_claims": list(NON_CLAIMS),
    }
    (ART / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    return manifest


def summary(tag: str = "3seed") -> dict:
    payload = json.loads((ART / f"metrics_{tag}.json").read_text())
    table = {}
    for row in payload["arms"]:
        table.setdefault(row["arm"], {}).setdefault(row["supervision"], {})[
            str(row["seed"])
        ] = {
            "J3": row["J3"]["row_mean"],
            "J3_CI": row["J3"]["ci95"],
            "J4": row["J4"]["row_mean"],
            "A": row["A_accuracy"]["row_mean"],
            "B": row["B_accuracy"]["row_mean"],
            "AB": row["AB_accuracy"]["row_mean"],
            "dev_bce": row["dev_bce"],
        }
    contrast_table = {}
    for row in payload["contrasts"]:
        contrast_table.setdefault(row["contrast"], {}).setdefault(row["supervision"], {})[
            str(row["seed"])
        ] = {
            "delta_J3": row["row_mean"],
            "CI": row["ci95"],
            "direction": row["direction"],
        }
    compact = {
        "arms": table,
        "contrasts": contrast_table,
        "selection_rule": payload["selection_rule"],
    }
    (ART / f"SUMMARY_{tag}.json").write_text(json.dumps(compact, indent=2, default=float) + "\n")
    print(json.dumps(compact["arms"], indent=2, default=float))
    print(json.dumps(compact["contrasts"], indent=2, default=float))
    return compact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["probe", "train", "select", "eval", "summary", "seam", "all"],
        required=True,
    )
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--force", action="store_true", help="retrain existing cells")
    parser.add_argument("--probe-cells", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(THREADS)
    ART.mkdir(parents=True, exist_ok=True)
    data = load_data()
    cap = capacity(data)

    if args.stage == "probe":
        probe(data, cells=args.probe_cells)
        return 0
    if args.stage == "seam":
        seam_diagnostic(data)
        return 0
    if args.stage in ("train", "all"):
        manifest = write_manifest(data, cap)
        print(json.dumps({"manifest_arms": manifest["arms"]}, indent=2))
        for arm in args.arms:
            for seed in args.seeds:
                for supervision in SUPERVISIONS:
                    for lr in LRS:
                        receipt = train_cell(arm, seed, supervision, lr, data, force=args.force)
                        print(
                            json.dumps(
                                {
                                    "cell": f"{arm}_s{seed}_{supervision}_lr{lr}",
                                    "dev_bce": receipt["dev_bce"],
                                    "final_train_loss": receipt["final_train_loss"],
                                    "seconds": receipt["seconds"],
                                }
                            ),
                            flush=True,
                        )
    if args.stage in ("select", "all"):
        select(data, args.arms, args.seeds)
    if args.stage in ("eval", "all"):
        evaluate(args.seeds, args.arms)
    if args.stage in ("summary", "all"):
        summary()
    if args.stage == "all":
        seam_diagnostic(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
