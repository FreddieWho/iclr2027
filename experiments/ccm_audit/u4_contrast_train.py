#!/usr/bin/env python3
"""U4 bounded representation contrasts (max two, pre-registered).

rep=centered : 8-dim translation-removed raw coords (per-scene centroid out).
rep=sixdist  : 6-dim pairwise distances only (rel_features drop the 4 radii).
Each rep trains {clean, flipmine} with the EXACT r04b recipe (300ep, Adam
1e-2, lam 1.0, torch.manual_seed(seed), same mine_flips(seed=5) supervision
on train_101) so the only difference from the U1 arms is the input.
Features are precomputed and the checkpoint carries NO featurize flag;
in_dim/mu/sd describe the stored feature space (u4 eval featurizes first).

Outputs: artifacts/next_novelty/u4_contrast/{rep}_s{seed}/{clean,flipmine}/
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(4)
from common import rel_features  # noqa: E402
from coord_mlp import CoordMLP  # noqa: E402
from r04b_methods import mine_flips  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
OUT = ROOT / "artifacts" / "next_novelty" / "u4_contrast"


def feat(X, rep):
    X = np.asarray(X, dtype=np.float32)
    if rep == "centered":
        return (X - X.mean(axis=1, keepdims=True)).reshape(len(X), 8)
    if rep == "sixdist":
        return rel_features(X)[:, :6]
    raise ValueError(rep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rep", choices=["centered", "sixdist"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lam", type=float, default=1.0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X, y = d["positions"].astype(np.float32), d["labels"].astype(np.float32)
    mined = mine_flips(X.astype(float), y.astype(int), seed=5)
    ii = np.array([m["scene"] for m in mined])
    ee = np.array([m["edit"] for m in mined])
    Fc = feat(X, a.rep)
    Ff = feat((X[ii] + ee).astype(np.float32), a.rep)
    mu, sd = Fc.mean(0), Fc.std(0) + 1e-8
    Xc = torch.from_numpy(((Fc - mu) / sd).astype(np.float32))
    yc = torch.from_numpy(y)
    Xf = torch.from_numpy(((Ff - mu) / sd).astype(np.float32))
    yf = torch.from_numpy(np.array([m["new"] for m in mined],
                                   dtype=np.float32))
    h = hashlib.sha256()
    for mm in mined:
        h.update(np.asarray(mm["scene"]).tobytes())
        h.update(np.asarray(mm["edit"], float).tobytes())
        h.update(np.asarray(mm["new"]).tobytes())
    od = OUT / ("%s_s%d" % (a.rep, a.seed))
    for name in ("clean", "flipmine"):
        torch.manual_seed(a.seed)
        model = CoordMLP(64, 32, in_dim=Fc.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        bce = nn.BCEWithLogitsLoss()
        for ep in range(a.epochs):
            model.train()
            opt.zero_grad()
            loss = bce(model(Xc), yc)
            if name == "flipmine":
                loss = loss + a.lam * bce(model(Xf), yf)
            loss.backward()
            opt.step()
        model.eval()
        mdir = od / name
        mdir.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "hidden": 64, "feat": 32,
                    "in_dim": int(Fc.shape[1]), "rep": a.rep,
                    "mu": mu, "sd": sd, "seed": a.seed}, mdir / "model.pt")
    (od / "manifest.json").write_text(json.dumps(
        {"rep": a.rep, "methods": ["clean", "flipmine"], "epochs": a.epochs,
         "lr": 1e-2, "lam": a.lam, "seed": a.seed, "mine_seed": 5,
         "n_train": len(X), "n_flips": len(mined),
         "flip_list_sha256": h.hexdigest()}, indent=1) + "\n")
    print("DONE %s s%d" % (a.rep, a.seed), flush=True)


if __name__ == "__main__":
    main()
