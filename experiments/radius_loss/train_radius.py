#!/usr/bin/env python3
"""Radius pilot training: clean / flipcov / radius arms, r04b recipe.

CoordMLP(64,32) from scratch, Adam lr=1e-2, 300 epochs, full-batch.
clean:   BCE(base states).
flipcov: BCE(base) + 1.0*BCE(750 flip endpoints).
radius:  BCE(base) + 1.0*(mean softplus(1 - s*dF) flips + mean |dF| stays),
         dF = f(x+e)-f(x), s = +1 if y1==1 else -1, m = 1.0 frozen.
Pairs subsampled (rng 7) from events_train.npz (train_101 parents).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
torch.set_num_threads(4)
from coord_mlp import CoordMLP  # noqa: E402
from paths import oracle_at  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
EU = ROOT / "artifacts" / "event_updater"
OUT = ROOT / "artifacts" / "radius_loss"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["clean", "flipcov", "radius"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=1.0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(7)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(np.float32).reshape(-1, 8)
    y = d["labels"].astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xc = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    yc = torch.from_numpy(y)
    ev = np.load(EU / "events_train.npz")
    U = ev["u"]
    fl = np.nonzero(U == 1)[0]
    st = np.nonzero(U == 0)[0]
    fl750 = rng.choice(fl, 750, replace=False)
    st750 = rng.choice(st, 750, replace=False)

    def std(v):
        return torch.from_numpy(((np.asarray(v, dtype=np.float32).reshape(-1, 8) - mu) / sd).astype(np.float32))

    Xf = std(ev["x1"][fl750])
    yf = torch.from_numpy(np.array([oracle_at(ev["x1"][i])[0] for i in fl750], dtype=np.float32))
    Xs0, Xs1 = std(ev["x0"][st750]), std(ev["x1"][st750])
    Xf0, Xf1 = std(ev["x0"][fl750]), std(ev["x1"][fl750])
    s_flip = torch.from_numpy(np.array([1.0 if oracle_at(ev["x1"][i])[0] == 1 else -1.0 for i in fl750], dtype=np.float32))
    model = CoordMLP(64, 32)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(a.epochs):
        model.train()
        opt.zero_grad()
        loss = bce(model(Xc), yc)
        if a.arm == "flipcov":
            loss = loss + a.lam * bce(model(Xf), yf)
        elif a.arm == "radius":
            dF_flip = model(Xf1) - model(Xf0)
            dF_stay = model(Xs1) - model(Xs0)
            loss = loss + a.lam * (F.softplus(a.margin - s_flip * dF_flip).mean() + dF_stay.abs().mean())
        loss.backward()
        opt.step()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "hidden": 64, "feat": 32,
                "in_dim": 8, "mu": mu, "sd": sd, "seed": a.seed, "arm": a.arm},
               OUT / f"radius_{a.arm}_s{a.seed}.pt")
    print(f"DONE radius_{a.arm}_s{a.seed} final_loss={float(loss):.4f}", flush=True)


if __name__ == "__main__":
    main()
