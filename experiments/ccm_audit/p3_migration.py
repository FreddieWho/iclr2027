#!/usr/bin/env python3
"""P3-A/B: quartet state-transition table + migration metrics.

States = (A_ok, B_ok, AB_ok). Per quartet, clean vs repair state pair.
Metrics on baseline-110 subset: J, R_full, R_endpoint, M + quartet-paired
bootstrap CIs. Models: s11 x6 checkpoints + s23/s47 clean/flipmine.
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
rng = np.random.default_rng(26092244)
MODELS = [("11", m) for m in ("clean", "flipmine", "fliprand", "supmine", "auxmargin", "relfeat")] + \
         [("23", m) for m in ("clean", "flipmine")] + \
         [("47", m) for m in ("clean", "flipmine")]


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
    states = {}
    for seed, mname in MODELS:
        try:
            model, stats = load_model(ART / ("r04b_s%s" % seed) / mname)
        except Exception as e:
            print("SKIP s%s %s: %s" % (seed, mname, e), flush=True)
            continue
        model.eval()

        def pf(xs):
            xs = np.asarray(xs, np.float32)
            with torch.no_grad():
                out = []
                for s in range(0, len(xs), 512):
                    out.append(model(preprocess(xs[s:s + 512], stats)).numpy().reshape(-1))
            return (np.concatenate(out) > 0).astype(int)
        R = {}
        for qid, (i1, m1), (i2, m2) in quads:
            xa = np.asarray(Qx[i1], float)
            xb = np.asarray(Qx[i2], float)
            xab = xa + np.asarray(Qe[i1], float)
            la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
            lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
            pa = int(pf(xa[None])[0] == la)
            pb = int(pf(xb[None])[0] == lb)
            pc = int(pf(xab[None])[0] == m1["yAB"])
            R[qid] = (pa, pb, pc)
        states["s%s_%s" % (seed, mname)] = R
        print("SAW s%s %s: %d quartets" % (seed, mname, len(R)), flush=True)
    res = {"n_quartets": len(quads)}
    for seed in ("11", "23", "47"):
        C = states.get(f"s{seed}_clean")
        F = states.get(f"s{seed}_flipmine")
        if C is None or F is None:
            continue
        trans = {}
        for q in C:
            k = "".join(map(str, C[q])) + "->" + "".join(map(str, F[q]))
            trans[k] = trans.get(k, 0) + 1
        base110 = [q for q in C if C[q] == (1, 1, 0)]
        n = len(base110)
        rf = sum(1 for q in base110 if F[q] == (1, 1, 1))
        re_ = sum(1 for q in base110 if F[q][2] == 1)
        mg = sum(1 for q in base110 if F[q][2] == 1 and (F[q][0] == 0 or F[q][1] == 0))
        j_c = sum(1 for q in C if C[q] == (1, 1, 1)) / len(C)
        j_f = sum(1 for q in F if F[q] == (1, 1, 1)) / len(F)
        def ci(vals):
            vals = np.array(vals)
            if len(vals) == 0:
                return None
            boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(2000)]
            q = np.quantile(boots, [0.025, 0.975])
            return [round(float(q[0]), 4), round(float(q[1]), 4)]
        Rf = np.array([1.0 if F[q] == (1, 1, 1) else 0.0 for q in base110])
        Re = np.array([float(F[q][2]) for q in base110])
        Mg = np.array([1.0 if (F[q][2] == 1 and (F[q][0] == 0 or F[q][1] == 0)) else 0.0 for q in base110])
        res[f"s{seed}"] = {
            "n_base110": n,
            "trans_top": sorted(trans.items(), key=lambda kv: -kv[1])[:12],
            "R_full": [round(rf / n, 4) if n else None, ci(Rf)],
            "R_endpoint": [round(re_ / n, 4) if n else None, ci(Re)],
            "M_migrate": [round(mg / n, 4) if n else None, ci(Mg)],
            "J_clean": round(j_c, 4), "J_repair": round(j_f, 4)}
        print("s%s: n110=%d R_full=%s R_end=%s M=%s J=%.3f->%.3f" % (
            seed, n, res[f"s{seed}"]["R_full"], res[f"s{seed}"]["R_endpoint"],
            res[f"s{seed}"]["M_migrate"], j_c, j_f), flush=True)
    # cross-checkpoint J comparison (s11 breadth)
    if "s11_clean" in states:
        J = {}
        for k, R in states.items():
            if k.startswith("s11_"):
                J[k] = round(float(np.mean([v == (1, 1, 1) for v in R.values()])), 4)
        res["J_breadth_s11"] = J
        print("J breadth:", J, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "P3_MIGRATION.json", "w"), indent=1)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
