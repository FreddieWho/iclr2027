#!/usr/bin/env python3
"""P3 final (§10-11): paired repair comparison with PARENT-CLUSTER primary CIs.

Per quartet (qid, one parent each): A/B/AB correctness for clean vs flipmine
(s11/s23/s47). Cells, triple denominators, 8-state transition table,
J/R_full/R_endpoint/M + parent-cluster bootstrap CIs (2000). Raw counts kept.
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
OUT = ROOT / "artifacts" / "next_novelty" / "p3"
rng = np.random.default_rng(26092247)


def pcb_groups(groups, stat_fn, n_boot=2000):
    # groups: list of arrays (one per parent); stat_fn(list-of-arrays)->scalar
    ups = np.arange(len(groups))
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(ups, size=len(ups), replace=True)
        boots.append(stat_fn([groups[i] for i in draw]))
    q = np.quantile(boots, [0.025, 0.975])
    return [round(float(q[0]), 4), round(float(q[1]), 4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="dev512")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    b = np.load(BANK / ("bank_" + a.bank + ".npz"), allow_pickle=True)
    Qx, Qe = b["Qx"], b["Qe"]
    Qmeta = json.loads(str(b["Qmeta"]))
    by_q = {}
    for qi, m in enumerate(Qmeta):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    qpar = {qid: ps[0][1]["parent"] for qid, ps in by_q.items() if len(ps) == 2}
    models = {}
    for seed in (11, 23, 47):
        for mname in ("clean", "flipmine"):
            model, stats = load_model(ART / ("r04b_s%d" % seed) / mname)
            model.eval()
            models[(seed, mname)] = (model, stats)

    def preds(seed, mname):
        model, stats = models[(seed, mname)]
        R = {}
        with torch.no_grad():
            xs_all, keys = [], []
            for qid, (i1, m1), (i2, m2) in quads:
                xa = np.asarray(Qx[i1], float)
                xb = np.asarray(Qx[i2], float)
                xab = xa + np.asarray(Qe[i1], float)
                xs_all += [xa, xb, xab]
                keys.append((qid, m1, m2))
            out = []
            for s in range(0, len(xs_all), 512):
                xx = np.stack(xs_all[s:s + 512]).astype(np.float32)
                out.append(model(preprocess(xx, stats)).numpy().reshape(-1))
            lg = np.concatenate(out)
            pl = (lg > 0).astype(int)
        for k, (qid, m1, m2) in enumerate(keys):
            la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
            lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
            R[qid] = (int(pl[3 * k] == la), int(pl[3 * k + 1] == lb),
                      int(pl[3 * k + 2] == m1["yAB"]))
        return R
    res = {"n_quartets": len(quads)}
    for seed in (11, 23, 47):
        C = preds(seed, "clean")
        F = preds(seed, "flipmine")
        Q = sorted(C)
        par = np.array([qpar[q] for q in Q])
        # cells
        cells = {}
        for name, sel in (("both", lambda q: (C[q][0] and C[q][1]) and (F[q][0] and F[q][1])),
                          ("clean_only", lambda q: (C[q][0] and C[q][1]) and not (F[q][0] and F[q][1])),
                          ("repair_only", lambda q: not (C[q][0] and C[q][1]) and (F[q][0] and F[q][1])),
                          ("neither", lambda q: not (C[q][0] and C[q][1]) and not (F[q][0] and F[q][1]))):
            qs = [q for q in Q if sel(q)]
            cv = np.array([1 - C[q][2] for q in qs])
            fv = np.array([1 - F[q][2] for q in qs])
            pp = np.array([qpar[q] for q in qs])
            cells[name] = {"n": len(qs),
                           "clean_ABerr": round(float(cv.mean()), 4) if len(qs) else None,
                           "repair_ABerr": round(float(fv.mean()), 4) if len(qs) else None,
                           "clean_ABerr_ci": pcb_groups([np.nonzero(pp == u)[0] for u in np.unique(pp)],
                                                 lambda idx: cv[np.concatenate(idx)].mean()) if len(qs) else None,
                           "repair_ABerr_ci": pcb_groups([np.nonzero(pp == u)[0] for u in np.unique(pp)],
                                                  lambda idx: fv[np.concatenate(idx)].mean()) if len(qs) else None,
                           "fixed": int(((cv == 1) & (fv == 0)).sum()) if len(qs) else 0,
                           "broken": int(((cv == 0) & (fv == 1)).sum()) if len(qs) else 0}
        # transition table 110 -> *
        trans = {}
        for q in Q:
            k = "".join(map(str, C[q])) + "->" + "".join(map(str, F[q]))
            trans[k] = trans.get(k, 0) + 1
        base110 = [q for q in Q if C[q] == (1, 1, 0)]
        n = len(base110)
        bp = np.array([qpar[q] for q in base110])
        Rf = np.array([1.0 if F[q] == (1, 1, 1) else 0.0 for q in base110])
        Re = np.array([float(F[q][2]) for q in base110])
        Mg = np.array([1.0 if (F[q][2] == 1 and (F[q][0] == 0 or F[q][1] == 0)) else 0.0 for q in base110])
        G = [np.nonzero(bp == u)[0] for u in np.unique(bp)]
        met = {}
        for nm, vv in (("R_full", Rf), ("R_endpoint", Re), ("M_migrate", Mg)):
            met[nm] = [round(float(vv.mean()), 4) if n else None,
                       pcb_groups(G, lambda idx: vv[np.concatenate(idx)].mean()) if n else None]
        Jc = np.array([1.0 if C[q] == (1, 1, 1) else 0.0 for q in Q])
        Jf = np.array([1.0 if F[q] == (1, 1, 1) else 0.0 for q in Q])
        Pall = [np.nonzero(par == u)[0] for u in np.unique(par)]
        met["J_clean"] = [round(float(Jc.mean()), 4), pcb_groups(Pall, lambda idx: Jc[np.concatenate(idx)].mean())]
        met["J_repair"] = [round(float(Jf.mean()), 4), pcb_groups(Pall, lambda idx: Jf[np.concatenate(idx)].mean())]
        # triple denominators
        own_c = [q for q in Q if C[q][0] and C[q][1]]
        own_f = [q for q in Q if F[q][0] and F[q][1]]
        com = [q for q in Q if (C[q][0] and C[q][1]) and (F[q][0] and F[q][1])]
        def err(P, QQ):
            return round(float(np.mean([1 - P[q][2] for q in QQ])), 4) if QQ else None
        met["denoms"] = {"own_clean": [err(C, own_c), len(own_c)],
                         "own_repair": [err(F, own_f), len(own_f)],
                         "baseline_fixed_repair": [err(F, own_c), len(own_c)],
                         "common_clean": [err(C, com), len(com)],
                         "common_repair": [err(F, com), len(com)]}
        met["overall"] = {
            "clean_ABerr": round(float(1 - np.mean([C[q][2] for q in Q])), 4),
            "repair_ABerr": round(float(1 - np.mean([F[q][2] for q in Q])), 4),
            "clean_atomic_pass": round(float(np.mean([C[q][0] and C[q][1] for q in Q])), 4),
            "repair_atomic_pass": round(float(np.mean([F[q][0] and F[q][1] for q in Q])), 4)}
        res["s%d" % seed] = {"cells": cells,
                             "trans_top": sorted(trans.items(), key=lambda kv: -kv[1])[:12],
                             "metrics": met}
        print("s%d: n110=%d R_full=%s M=%s J=%s->%s" % (
            seed, n, met["R_full"], met["M_migrate"],
            met["J_clean"][0], met["J_repair"][0]), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "P3_FINAL.json", "w"), indent=1)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
