#!/usr/bin/env python3
"""RelFeat+flipmine (§15 conditional extension): rel12 features, base + flip
BCE (lam=1.0), same recipe (300ep, Adam 1e-2), seeds 11/23/47. train_101.
"""
import argparse
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
OUT = ROOT / "artifacts" / "next_novelty" / "relflip"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lam", type=float, default=1.0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X, y = d["positions"].astype(np.float32), d["labels"].astype(np.float32)
    Fr = rel_features(X)
    mur, sdr = Fr.mean(0), Fr.std(0) + 1e-8
    mined = mine_flips(X.astype(float), y.astype(int), seed=5)
    ii = np.array([m["scene"] for m in mined])
    ee = np.array([m["edit"] for m in mined])
    Ef = rel_features((X[ii] + ee).astype(np.float32))
    Xr = torch.from_numpy(((Fr - mur) / sdr).astype(np.float32))
    yc = torch.from_numpy(y)
    Xf = torch.from_numpy(((Ef - mur) / sdr).astype(np.float32))
    yf = torch.from_numpy(np.array([m["new"] for m in mined], dtype=np.float32))
    model = CoordMLP(64, 32, in_dim=10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(a.epochs):
        model.train()
        opt.zero_grad()
        loss = bce(model(Xr), yc) + a.lam * bce(model(Xf), yf)
        loss.backward()
        opt.step()
    model.eval()
    od = OUT / ("s%d" % a.seed)
    od.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "hidden": 64, "feat": 32,
                "in_dim": 10, "featurize": "rel12", "mu": mur, "sd": sdr,
                "seed": a.seed}, od / "model.pt")
    json.dump({"seed": a.seed, "epochs": a.epochs, "lam": a.lam,
               "n_flips": len(mined)}, open(od / "manifest.json", "w"), indent=1)
    print("SAW relflip s%d final=%.4f" % (a.seed, float(loss)), flush=True)


if __name__ == "__main__":
    main()
