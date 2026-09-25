#!/usr/bin/env python3
"""Recompute compact controls/lineage checks from stored summary artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def o01():
    path = ROOT / "artifacts/e1a933_review/data_o01/O01_FACTORIAL.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    lookup = {(r["size"], int(r["seed"]), r["arch"], int(r["frac"])): r for r in rows}
    cells = []
    for size in ("N", "4N"):
        for seed in (11, 23, 47):
            a = lookup[(size, seed, "sixdist", 25)]
            b = lookup[(size, seed, "raw", 100)]
            cells.append({"size": size, "seed": seed, "sixdist25_J": float(a["J"]),
                          "raw100_J": float(b["J"]), "difference": float(a["J"]) - float(b["J"]),
                          "n_quartets": int(a["n"]), "n_parents": int(a["n_parents"])})
    return {"cells": cells, "positive_cells": sum(x["difference"] > 0 for x in cells),
            "oracle_query_savings": "NOT_ESTIMATED; candidate mining precedes retained-label selection"}


def o02():
    main_path = ROOT / "artifacts/e1a933_review/data_u10_o02/O02_BUDGET_MATCH.json"
    rerun_path = ROOT / "artifacts/e1a933_review/data_u10_o02_numorder2/O02_BUDGET_MATCH.json"
    main, rerun = read_json(main_path), read_json(rerun_path)
    out = {}
    for task in ("T1", "T2"):
        out[task] = {"n_quartets": main[task]["n"], "n_parents": main[task]["n_parents"], "seeds": {}}
        for seed in ("s11", "s23", "s47"):
            key = "six_flip_minus_typed_m_flip"
            a = main[task]["contrasts"][seed][key]
            b = rerun[task]["contrasts"][seed][key]
            out[task]["seeds"][seed] = {
                "primary_estimate": a["estimate"][0], "primary_ci95": a["ci95"][0],
                "two_thread_estimate": b["estimate"][0], "two_thread_ci95": b["ci95"][0],
                "both_positive": a["estimate"][0] > 0 and b["estimate"][0] > 0,
            }
    return out


def o04():
    path = ROOT / "artifacts/e1a933_review/readout_forward_20260924/results.json"
    d = read_json(path)["predictions"]["primary"]["cells"]
    cells = []
    for name, row in sorted(d.items()):
        delta = {k: float(row[k]["seed_mean"]) for k in ("S", "J", "J_star")}
        p1 = abs(delta["J"]) > abs(delta["S"]) and abs(delta["J"]) > abs(delta["J_star"])
        p2 = abs(delta["J_star"]) < abs(delta["J"])
        cells.append({"cell": name, "delta": delta, "P1": p1, "P2": p2})
    return {"cells": cells, "P1_supported_cells": sum(x["P1"] for x in cells),
            "P2_supported_cells": sum(x["P2"] for x in cells),
            "primary_verdict": "SUPPORTED" if all(x["P1"] and x["P2"] for x in cells) else "MISS"}


def r08():
    path = ROOT / "artifacts/e1a933_review/football/NATURAL_E_RATE.json"
    d = read_json(path)
    cell = d["pooled"]["cells"]["W0.2|contract|forward"]
    search = d["controlled_search_reference"]["pooled"]
    n_sit = int(d["pooled"]["n_situations"])
    n_events = int(d["pooled"]["n_events"])
    return {"primary_cell": d["primary_cell"], "situations": {"E": cell["n_situations_with_E"], "n": n_sit,
            "rate": cell["natural_E_situation_rate"]},
            "events": {"E": cell["n_events_with_E"], "n": n_events, "rate": cell["natural_E_event_rate"]},
            "frame_pairs": {"E": cell["natural_E_pairs"], "n": cell["n_pairs_in_cap"],
            "rate": cell["natural_E_pair_rate"]}, "controlled_search": search}


def r09():
    path = ROOT / "artifacts/e1a933_review/frontier/FRONTIER_FULLSCENE.json"
    d = read_json(path)
    u = d["scene_universe"]
    return {"scene_universe": u,
            "infeasible_share_full": u["n_oracle_infeasible"] / u["n_scenes"],
            "infeasible_share_eval": u["eval_full_oracle_infeasible"] / u["eval_full_scenes"],
            "conditional_eval_feasible_n": u["eval_full_oracle_feasible"],
            "fullscene_coverage_range_at_target_0_5_from_claim_ledger": [0.1455, 0.3239]}


def r03():
    path = ROOT / "artifacts/submission_audit_20260925/controls/r03_single_sensitivity.json"
    d = read_json(path)
    deltas = [m["paired_differences"]["row_weighted_excluded_minus_full_accuracy"] * 100
              for m in d["models"]]
    return {"status": d["status"], "models": len(d["models"]), "state_forward_count": d["computation"]["state_forward_count_actual"],
            "row_weighted_delta_pp_range": [min(deltas), max(deltas)],
            "excluded_eids_by_pool": {k: v["n_exact_eids"] for k, v in d["overlap_by_train_pool"].items()},
            "quartets": d["d09_quartet_scope"]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/submission_audit_20260925/controls/summary_recomputation.json")
    a = p.parse_args()
    inputs = [
        "artifacts/e1a933_review/data_o01/O01_FACTORIAL.csv",
        "artifacts/e1a933_review/data_o02/O02_BUDGET_MATCH.json",
        "artifacts/e1a933_review/data_u10_o02/O02_BUDGET_MATCH.json",
        "artifacts/e1a933_review/data_u10_o02_numorder2/O02_BUDGET_MATCH.json",
        "artifacts/e1a933_review/readout_forward_20260924/results.json",
        "artifacts/e1a933_review/football/NATURAL_E_RATE.json",
        "artifacts/e1a933_review/frontier/FRONTIER_FULLSCENE.json",
        "artifacts/submission_audit_20260925/controls/r03_single_sensitivity.json",
    ]
    result = {"schema": "submission_audit_20260925.controls_summary/1",
              "inputs": {s: sha(ROOT / s) for s in inputs},
              "O01": o01(), "O02": o02(), "O04": o04(), "R03": r03(), "R08": r08(), "R09": r09(),
              "notes": ["summary-only recomputation; no training", "parent and model seeds are not independent data replications"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
