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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--freeze", default="")
    a = p.parse_args()
    indir, out = Path(a.indir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.load(open(a.freeze)) if a.freeze else {}
    res = {}
    for seed in (11, 23, 47):
        dc = json.load(open(indir / ("unified_s%d_clean.json" % seed)))
        df = json.load(open(indir / ("unified_s%d_flipmine.json" % seed)))
        pc = [r for r in dc["paths"] if r["start_correct"] and r["semantic_change"] == 1]
        cfc = np.array([r["confidence"] for r in pc])
        thr = freeze.get(f"s{seed}", float(np.quantile(cfc, 0.75)))
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
            R["clean_end_err"] = round(float(1 - ce.mean()), 4)
            R["cells"] = {}
            for name, sel in (("cs_re", lambda x: x[2] == 1), ("cs_rw", lambda x: x[2] == 0)):
                idx = [i for i, x in enumerate(rows) if sel(x)]
                be = np.array([rows[i][3] for i in idx])
                R["cells"][name] = {"n": len(idx),
                    "repair_end_err": round(float(1 - be.mean()), 4) if len(idx) else None,
                    "ci": pcb(par[idx], 1 - be) if len(idx) else None}
            both = [i for i, x in enumerate(rows) if x[2] == 1]
            be = np.array([rows[i][3] for i in both])
            ce2 = np.array([rows[i][1] for i in both])
            d = (1 - ce2) - (1 - be)
            R["common_correct_n"] = len(both)
            R["paired_delta"] = round(float(d.mean()), 4) if len(d) else None
            R["paired_delta_ci"] = pcb(par[both], d) if len(d) else None
        res[f"s{seed}"] = R
        print(f"s{seed}: {R}", flush=True)
    json.dump({"freeze_thr": freeze, "results": res},
              open(out / "P1G.json", "w"), indent=1)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
