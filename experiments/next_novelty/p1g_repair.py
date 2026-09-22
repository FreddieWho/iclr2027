#!/usr/bin/env python3
"""P1-G: paired repair residual on clean-defined high-risk group.

High-risk = clean start-correct paths with clean conf >= dev top-quartile
threshold (frozen). NO repair-start-correct requirement for primary.
Report 4 cells (clean start/end x repair start/end) + paired delta with
parent-cluster bootstrap + common-correct secondary.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(26092243)


def pcb(parents, vals, n_boot=2000):
    ups = np.unique(parents)
    pmap = {u: np.nonzero(parents == u)[0] for u in ups}
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(ups, size=len(ups), replace=True)
        idx = np.concatenate([pmap[u] for u in draw])
        boots.append(vals[idx].mean())
    q = np.quantile(boots, [0.025, 0.975])
    return [round(float(q[0]), 4), round(float(q[1]), 4)]


def load_threshold(freeze, model_key, quantile="t25", mode="confirm",
                   require_fields=("thr",)):
    """Production threshold loader (R1 schema contract).

    freeze: parsed p1_freeze.json. Keys are '<seed>_<model>' with
    {'t25':, 't50':, 't75':}. Confirm mode MUST read the frozen table:
    a missing key, missing quantile, or a wrong value type is an error
    (never a silent fallback to current-data quantiles). Dev mode may
    compute, but its output is stamped mode='dev_computed' by the caller.
    """
    if not isinstance(freeze, dict):
        raise ValueError("freeze table must be a mapping of '<seed>_<model>' entries")
    ent = freeze.get(model_key)
    if ent is None:
        raise KeyError(f"freeze entry missing for '{model_key}' (confirm mode hard-fails)")
    if not isinstance(ent, dict) or quantile not in ent:
        raise KeyError(f"freeze entry '{model_key}' lacks quantile '{quantile}'")
    val = ent[quantile]
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        raise ValueError(f"freeze value '{model_key}.{quantile}' is not numeric")
    return float(val)


def check_manifest(indir, expected_bank):
    """Hard-fail bank-identity check for confirm-mode analyses."""
    mpath = Path(indir) / "manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(f"{mpath} required for confirm-mode analysis")
    man = json.load(open(mpath))
    if man.get("bank") != expected_bank:
        raise ValueError(f"bank mismatch: indir bank={man.get('bank')!r}, "
                         f"expected {expected_bank!r}")
    return man


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--freeze", default="")
    p.add_argument("--mode", choices=("dev", "confirm"), default="confirm",
                   help="confirm: frozen thresholds + manifest check (default). "
                        "dev: compute thresholds on the run, stamped in output.")
    a = p.parse_args()
    indir, out = Path(a.indir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.load(open(a.freeze)) if a.freeze else {}
    man = check_manifest(indir, "confirm1007") if a.mode == "confirm" else {}
    res = {}
    for seed in (11, 23, 47):
        dc = json.load(open(indir / ("unified_s%d_clean.json" % seed)))
        df = json.load(open(indir / ("unified_s%d_flipmine.json" % seed)))
        pc = [r for r in dc["paths"] if r["start_correct"] and r["semantic_change"] == 1]
        if a.mode == "confirm":
            thr = load_threshold(freeze, f"s{seed}_clean", "t25")
        else:
            cfc = np.array([r["confidence"] for r in pc])
            thr = float(np.quantile(cfc, 0.75))
        hi = [r for r in pc if r["confidence"] >= thr]
        fq = {(r["qid"], r["ptype"]): r for r in df["paths"]}
        rows = []
        for r in hi:
            b = fq.get((r["qid"], r["ptype"]))
            if b is None:
                continue
            rows.append((r["parent"], int(r["end_correct"]), int(b.get("start_correct", 0)),
                         int(b["end_correct"])))
        R = {"n_high": len(rows), "thr": round(float(thr), 4)}
        if rows:
            par = np.array([x[0] for x in rows])
            ce = np.array([x[1] for x in rows])
            # MAIN comparison (R1): full baseline-fixed cohort, no
            # repair-start-correct screening.
            be = np.array([x[3] for x in rows])
            se = np.array([x[2] for x in rows])
            R["main"] = {
                "clean_end_err": round(float(1 - ce.mean()), 4),
                "repair_end_err": round(float(1 - be.mean()), 4),
                "paired_delta": round(float((1 - ce.mean()) - (1 - be.mean())), 4),
                "paired_delta_ci": pcb(par, (1 - ce) - (1 - be)),
                "repair_start_err": round(float(1 - se.mean()), 4),
                "repair_start_err_ci": pcb(par, 1 - se),
                "n_repair_start_wrong": int((se == 0).sum()),
            }
            R["clean_end_err"] = R["main"]["clean_end_err"]
            R["cells"] = {}
            for name, sel in (("cs_re", lambda x: x[2] == 1), ("cs_rw", lambda x: x[2] == 0)):
                idx = [i for i, x in enumerate(rows) if sel(x)]
                be_s = np.array([rows[i][3] for i in idx])
                pa_s = np.array([rows[i][0] for i in idx])
                R["cells"][name] = {"n": len(idx),
                    "repair_end_err": round(float(1 - be_s.mean()), 4) if len(idx) else None,
                    "ci": pcb(pa_s, 1 - be_s) if len(idx) else None}
            # SECONDARY (renamed, R1): common-start-correct paired delta.
            # Not a substitute for the main cohort comparison.
            both = [i for i, x in enumerate(rows) if x[2] == 1]
            be2 = np.array([rows[i][3] for i in both])
            ce2 = np.array([rows[i][1] for i in both])
            pa2 = np.array([rows[i][0] for i in both])
            d = (1 - ce2) - (1 - be2)
            R["common_correct_n"] = len(both)
            R["common_start_correct_secondary"] = {
                "paired_delta": round(float(d.mean()), 4) if len(d) else None,
                "paired_delta_ci": pcb(pa2, d) if len(d) else None,
            }
            R["paired_delta"] = R["common_start_correct_secondary"]["paired_delta"]
            R["paired_delta_ci"] = R["common_start_correct_secondary"]["paired_delta_ci"]
        res[f"s{seed}"] = R
        print(f"s{seed}: {R}", flush=True)
    json.dump({"mode": a.mode, "manifest": man, "freeze_thr": freeze, "results": res},
              open(out / "P1G_CORRECTED.json", "w"), indent=1)
    print("DONE ->", out / "P1G_CORRECTED.json")


if __name__ == "__main__":
    main()
