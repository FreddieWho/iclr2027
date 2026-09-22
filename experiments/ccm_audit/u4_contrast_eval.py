#!/usr/bin/env python3
"""U4 contrast evaluation on dev512 (frozen U4 checkpoints, no retraining).

Arms: centered_clean / centered_flip / sixdist_clean / sixdist_flip.
Featurization is applied here (checkpoints store precomputed-space stats):
  centered: X - per-scene centroid; sixdist: rel_features(X)[:, :6].
Reports J, H, atomic/AB rates with parent-cluster CIs (same rng family as
U1/U2: fixed seed 20260925), plus the U1 reference arms for direct contrast.

NEW outputs only: artifacts/next_novelty/u4_contrast/U4_SUMMARY.json
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
sys.path.insert(0, str(ROOT / "experiments" / "ccm_audit"))
torch.set_num_threads(4)
from common import load_model, rel_features  # noqa: E402
from ordering import G_of  # noqa: E402

BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
U4 = ROOT / "artifacts" / "next_novelty" / "u4_contrast"
rng = np.random.default_rng(20260925)

ARMS = ("centered_clean", "centered_flip", "sixdist_clean", "sixdist_flip")


def ckpt(rep, method, seed):
    return U4 / ("%s_s%d" % (rep, seed)) / (
        "clean" if method == "clean" else "flipmine")


def featurize(X, rep):
    X = np.asarray(X, dtype=np.float32)
    if rep == "centered":
        return (X - X.mean(axis=1, keepdims=True)).reshape(len(X), 8)
    return rel_features(X)[:, :6]


def pcb(groups, stat_fn, n_boot=2000):
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
    a = p.parse_args()
    b = np.load(BANK / ("bank_" + a.bank + ".npz"), allow_pickle=True)
    Qx, Qe = b["Qx"], b["Qe"]
    Qmeta = json.loads(str(b["Qmeta"]))
    by_q = {}
    for qi, m in enumerate(Qmeta):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    qpar = {qid: ps[0][1]["parent"] for qid, ps in by_q.items()
            if len(ps) == 2}
    Q = sorted(qpar)
    par = np.array([qpar[q] for q in Q])
    Pall = [np.nonzero(par == u)[0] for u in np.unique(par)]
    yAB = np.array([by_q[q][0][1]["yAB"] for q in Q])
    out = {"bank": a.bank, "seeds": {}}
    for seed in (11, 23, 47):
        R = {}
        for arm in ARMS:
            rep, method = arm.split("_", 1)
            method = "clean" if method == "clean" else "flipmine"
            mp = ckpt(rep, method, seed)
            import hashlib
            sha = hashlib.sha256((mp / "model.pt").read_bytes()).hexdigest()
            model, stats = load_model(mp)
            model.eval()
            mu, sd = stats["mu"], stats["sd"]
            with torch.no_grad():
                fwd = []
                for qid, (i1, m1), (i2, m2) in quads:
                    xa = np.asarray(Qx[i1], float)
                    xb = np.asarray(Qx[i2], float)
                    xab = xa + np.asarray(Qe[i1], float)
                    F = (featurize(np.stack([xa, xb, xab]), rep) - mu) / sd
                    fwd.append(model(torch.from_numpy(
                        F.astype(np.float32))).numpy().reshape(-1))
                LG = np.stack(fwd)
            pl = (LG > 0).astype(int)
            corr = []
            for k, (qid, (i1, m1), (i2, m2)) in enumerate(quads):
                la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
                lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
                corr.append([int(pl[k][0] == la), int(pl[k][1] == lb),
                             int(pl[k][2] == m1["yAB"])])
            R[arm] = {"logits": LG, "correct": np.array(corr), "sha": sha}
        res = {}
        for arm in ARMS:
            C = R[arm]["correct"]
            LG = R[arm]["logits"]
            J = (C == 1).all(axis=1).astype(float)
            g = G_of(LG[:, 0], LG[:, 1], LG[:, 2], yAB)
            H = (g > 0).astype(float)
            res[arm] = {
                "J": [round(float(J.mean()), 4),
                      pcb(Pall, lambda idx: J[np.concatenate(idx)].mean())],
                "H": [round(float(H.mean()), 4),
                      pcb(Pall, lambda idx: H[np.concatenate(idx)].mean())],
                "atomic_pass": round(float((C[:, :2] == 1).all(axis=1).mean()), 4),
                "AB_correct": round(float(C[:, 2].mean()), 4),
                "model_sha256": R[arm]["sha"][:16],
            }
        out["seeds"]["s%d" % seed] = res
        print("s%d" % seed,
              {k: (v["J"][0], v["H"][0]) for k, v in res.items()},
              flush=True)
    json.dump(out, open(U4 / "U4_SUMMARY.json", "w"), indent=1)
    print("DONE ->", U4)


if __name__ == "__main__":
    main()
