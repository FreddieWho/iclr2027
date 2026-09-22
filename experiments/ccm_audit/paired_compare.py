#!/usr/bin/env python3
"""P3-C: paired same-quartet repair comparison on the frozen bank.

Per quartet (qid): atomic A/B + AB predictions for clean vs flipmine
(s11/s23/s47). Four cells by atomic-pass status (both/clean-only/
repair-only/neither); AB paired changes within cells; own vs
baseline-fixed vs common-correct deltas; overall AB err, atomic pass,
joint-all-correct. Plus selection-artifact unit test (synthetic).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(4)
from common import load_model, preprocess  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
OUT = ROOT / "artifacts" / "p123_upgrade" / "p3"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="dev512")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    b = np.load(BANK / ("bank_" + a.bank + ".npz"), allow_pickle=True)
    Qx, Qe = b["Qx"], b["Qe"]
    Qmeta = json.loads(str(b["Qmeta"]))
    # group paths by qid (A/B pair)
    by_q = {}
    for qi, m in enumerate(Qmeta):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    res = {"n_quartets": len(quads)}
    for seed in (11, 23, 47):
        preds = {}
        for mname in ("clean", "flipmine"):
            model, stats = load_model(ART / ("r04b_s%d" % seed) / mname)
            model.eval()
            mu = np.asarray(stats["mu"], np.float32).reshape(-1)
            sd = np.asarray(stats["sd"], np.float32).reshape(-1)

            def pf(xs):
                xs = np.asarray(xs, np.float32)
                with torch.no_grad():
                    out = []
                    for s in range(0, len(xs), 256):
                        out.append(model(preprocess(xs[s:s + 256], stats)).numpy().reshape(-1))
                return (np.concatenate(out) > 0).astype(int)
            # per quartet: need x+A, x+B states: reconstruct from path A (x=x+A? no:
            # path A: start = x+ea, edit = eb. xa = start; xb = x+eb unknown directly.
            # Reconstruct base x via meta? Bank stores path starts only. Use:
            # pa = pred(start_A)==yA, pb: endpoint of path B minus its edit...
            # Simplest correct: pa from pathA start, pb from pathB start,
            # pc from either endpoint (same AB state).
            R = {}
            for qid, (i1, m1), (i2, m2) in quads:
                xa = Qx[i1]
                xb = Qx[i2]
                xab = (np.asarray(xa, float) + np.asarray(Qe[i1], float))
                pa = int(pf(xa[None])[0] == m1["yA"] if m1["ptype"] == "A" else pf(xa[None])[0] == m1["yB"])
                pb = int(pf(xb[None])[0] == m2["yA"] if m2["ptype"] == "A" else pf(xb[None])[0] == m2["yB"])
                pc = int(pf(xab[None])[0] == m1["yAB"])
                R[qid] = (pa, pb, pc)
            preds[mname] = R
        C, F = preds["clean"], preds["flipmine"]
        cells = {"both": [], "clean_only": [], "repair_only": [], "neither": []}
        for qid in C:
            ca = C[qid][0] and C[qid][1]
            fa = F[qid][0] and F[qid][1]
            k = "both" if (ca and fa) else ("clean_only" if ca else ("repair_only" if fa else "neither"))
            cells[k].append((C[qid][2], F[qid][2]))  # 1 = AB correct
        out = {}
        for k, v in cells.items():
            v = np.array(v)
            out[k] = {"n": len(v),
                      "clean_ABerr": round(float(1 - v[:, 0].mean()), 4) if len(v) else None,
                      "repair_ABerr": round(float(1 - v[:, 1].mean()), 4) if len(v) else None,
                      "fixed": int(((v[:, 0] == 0) & (v[:, 1] == 1)).sum()) if len(v) else 0,
                      "broken": int(((v[:, 0] == 1) & (v[:, 1] == 0)).sum()) if len(v) else 0}
        # triple denominators
        own_c = [q for q in C if C[q][0] and C[q][1]]
        own_f = [q for q in F if F[q][0] and F[q][1]]
        com = [q for q in C if (C[q][0] and C[q][1]) and (F[q][0] and F[q][1])]
        def err(P, Q):
            return round(float(np.mean([1 - P[q][2] for q in Q])), 4) if Q else None
        out["denoms"] = {
            "own_clean": [err(C, own_c), len(own_c)],
            "own_repair": [err(F, own_f), len(own_f)],
            "baseline_fixed_clean": [err(C, own_c), len(own_c)],
            "baseline_fixed_repair": [err(F, own_c), len(own_c)],
            "common_clean": [err(C, com), len(com)],
            "common_repair": [err(F, com), len(com)]}
        out["overall"] = {
            "clean_ABerr": round(float(np.mean([1 - C[q][2] for q in C])), 4),
            "repair_ABerr": round(float(np.mean([1 - F[q][2] for q in F])), 4),
            "clean_atomic_pass": round(float(np.mean([C[q][0] and C[q][1] for q in C])), 4),
            "repair_atomic_pass": round(float(np.mean([F[q][0] and F[q][1] for q in F])), 4)}
        res["s%d" % seed] = {"cells": out, "denoms": out.pop("denoms"), "overall": out.pop("overall")}
        print("SAW s%d cells=%s" % (seed, {k: v["n"] for k, v in out.items()}), flush=True)
    # unit test: AB fixed, atomics vary -> own-rate moves (metric property)
    # AB predictions IDENTICAL across models; only atomic-pass sets differ.
    ab_ok = np.array([0] * 50 + [1] * 50)  # 1 = AB correct; fixed for both
    a_c = np.array([True] * 30 + [False] * 70)   # clean passes first 30
    a_f = np.array([True] * 10 + [False] * 40 + [True] * 30 + [False] * 20)
    own_c = 1 - ab_ok[a_c].mean()
    own_f = 1 - ab_ok[a_f].mean()
    res["unit_test"] = {"note": "AB predictions identical; only atomic-pass changes",
                        "own_miss_clean": round(float(own_c), 4),
                        "own_miss_repair": round(float(own_f), 4),
                        "pass": bool(abs(own_c - own_f) > 0.2)}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "P3_PAIRED.json", "w"), indent=1)
    print("DONE paired_compare")


if __name__ == "__main__":
    main()
