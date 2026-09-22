#!/usr/bin/env python3
"""P3-D eval: J/R_full/R_endpoint/M + static/single/preserveFA/comp for
fliprep/fliphalf/keepbal (s11/s23/s47) on dev bank. Same metric code as P3.
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
from coord_mlp import CoordMLP  # noqa: E402

BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
P3D = ROOT / "artifacts" / "next_novelty" / "p3d"
OUT = ROOT / "artifacts" / "next_novelty" / "p3"
rng = np.random.default_rng(26092246)


def pfwd(model, stats, xs):
    xs = np.asarray(xs, np.float32)
    with torch.no_grad():
        out = []
        for s in range(0, len(xs), 512):
            out.append(model(preprocess(xs[s:s + 512], stats)).numpy().reshape(-1))
    lg = np.concatenate(out)
    return lg, (lg > 0).astype(int)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    b = np.load(BANK / "bank_dev512.npz", allow_pickle=True)
    Qx, Qe = b["Qx"], b["Qe"]
    Qmeta = json.loads(str(b["Qmeta"]))
    Sx, Se = b["Sx"], b["Se"]
    Smeta = json.loads(str(b["Smeta"]))
    by_q = {}
    for qi, m in enumerate(Qmeta):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    res = {}
    for seed in (11, 23, 47):
        for arm in ("fliprep", "fliphalf", "keepbal"):
            ck = torch.load(P3D / ("s%d_%s" % (seed, arm)) / "model.pt",
                            map_location="cpu", weights_only=False)
            model = CoordMLP(ck["hidden"], ck["feat"], ck.get("in_dim", 8))
            model.load_state_dict(ck["state"])
            model.eval()
            stats = {"mu": ck["mu"], "sd": ck["sd"]}
            R = {}
            for qid, (i1, m1), (i2, m2) in quads:
                xa = np.asarray(Qx[i1], float)
                xb = np.asarray(Qx[i2], float)
                xab = xa + np.asarray(Qe[i1], float)
                la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
                lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
                _, p = pfwd(model, stats, np.stack([xa, xb, xab]))
                R[qid] = (int(p[0] == la), int(p[1] == lb), int(p[2] == m1["yAB"]))
            base110 = [q for q in R if R[q] == (1, 1, 0)]
            n = len(base110)
            # NOTE: R_full needs repair-vs-baseline; here arm IS the model:
            # report within-arm predictive rates + J; cross-arm compare later.
            # For migration metrics we need clean baseline states: reload r04b clean
            J = float(np.mean([v == (1, 1, 1) for v in R.values()]))
            # singles metrics
            Xs = np.asarray(Sx, np.float32)
            Xe = np.asarray(Se, np.float32)
            _, p0 = pfwd(model, stats, Xs)
            _, p1 = pfwd(model, stats, (Xs.reshape(len(Xs), -1) + Xe.reshape(len(Xe), -1)).reshape(-1, 4, 2))
            y0 = np.array([m["y0"] for m in Smeta])
            y1 = np.array([m["y1"] for m in Smeta])
            fl = np.array([m["flip"] == 1 for m in Smeta])
            static_err = float((p1 != y1).mean())
            single_err = float((p1[fl] != y1[fl]).mean())
            st = (~fl) & (p0 == y0)
            fa = float(((p1[st] != p0[st])).mean()) if st.sum() else None
            res[f"s{seed}_{arm}"] = {"n_q": len(R), "n_base110": n, "J": round(J, 4),
                                     "static": round(static_err, 4),
                                     "single": round(single_err, 4),
                                     "preserve_FA": round(fa, 4) if fa is not None else None}
            print(f"s{seed}_{arm}: J={J:.4f} static={static_err:.4f} single={single_err:.4f} FA={fa}", flush=True)
    # cross-arm migration vs r04b clean baseline states: recompute clean states
    for seed in (11, 23, 47):
        model, stats = load_model(Path("artifacts/discovery_campaign") / ("r04b_s%d" % seed) / "clean")
        model.eval()
        C = {}
        for qid, (i1, m1), (i2, m2) in quads:
            xa = np.asarray(Qx[i1], float)
            xb = np.asarray(Qx[i2], float)
            xab = xa + np.asarray(Qe[i1], float)
            la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
            lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
            _, p = pfwd(model, stats, np.stack([xa, xb, xab]))
            C[qid] = (int(p[0] == la), int(p[1] == lb), int(p[2] == m1["yAB"]))
        base110 = [q for q in C if C[q] == (1, 1, 0)]
        for arm in ("fliprep", "fliphalf", "keepbal"):
            ck = torch.load(P3D / ("s%d_%s" % (seed, arm)) / "model.pt",
                            map_location="cpu", weights_only=False)
            model2 = CoordMLP(ck["hidden"], ck["feat"], ck.get("in_dim", 8))
            model2.load_state_dict(ck["state"])
            model2.eval()
            st2 = {"mu": ck["mu"], "sd": ck["sd"]}
            F = {}
            for qid, (i1, m1), (i2, m2) in quads:
                xa = np.asarray(Qx[i1], float)
                xb = np.asarray(Qx[i2], float)
                xab = xa + np.asarray(Qe[i1], float)
                la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
                lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
                _, p = pfwd(model2, st2, np.stack([xa, xb, xab]))
                F[qid] = (int(p[0] == la), int(p[1] == lb), int(p[2] == m1["yAB"]))
            n = len(base110)
            rf = sum(1 for q in base110 if F[q] == (1, 1, 1))
            re_ = sum(1 for q in base110 if F[q][2] == 1)
            mg = sum(1 for q in base110 if F[q][2] == 1 and (F[q][0] == 0 or F[q][1] == 0))
            def ci(vals):
                vals = np.array(vals)
                if len(vals) == 0:
                    return None
                boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(2000)]
                q = np.quantile(boots, [0.025, 0.975])
                return [round(float(q[0]), 4), round(float(q[1]), 4)]
            R = res[f"s{seed}_{arm}"]
            R["R_full"] = [round(rf / n, 4) if n else None,
                           ci([1.0 if F[q] == (1, 1, 1) else 0.0 for q in base110])]
            R["R_endpoint"] = [round(re_ / n, 4) if n else None,
                               ci([float(F[q][2]) for q in base110])]
            R["M_migrate"] = [round(mg / n, 4) if n else None,
                              ci([1.0 if (F[q][2] == 1 and (F[q][0] == 0 or F[q][1] == 0)) else 0.0 for q in base110])]
            print(f"s{seed}_{arm}: R_full={R['R_full']} R_end={R['R_endpoint']} M={R['M_migrate']}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "P3D_PILOT.json", "w"), indent=1)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
