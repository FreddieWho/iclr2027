#!/usr/bin/env python3
"""L007 seed-variance lane: retrain flipmine recipe on fresh seeds.

Exact r04b flipmine contract (300ep, Adam 1e-2, lam 1.0, mine_flips(seed=5)
on train_101, torch.manual_seed(seed)); only the init seed varies.
Seeds 2001-2008 avoid r04b (11/23/47) and E7 (11/23/47/101/202) seeds.

Outputs: artifacts/next_novelty/l007_seed/s{seed}/model.pt + manifest.json
No sealed pools, no overwrites of existing artifacts.
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
from coord_mlp import CoordMLP  # noqa: E402
from r04b_methods import mine_flips  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
OUT = ROOT / "artifacts" / "next_novelty" / "l007_seed"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lam", type=float, default=1.0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X, y = d["positions"].astype(np.float32), d["labels"].astype(np.float32)
    mu8, sd8 = X.reshape(-1, 8).mean(0), X.reshape(-1, 8).std(0) + 1e-8
    mined = mine_flips(X.astype(float), y.astype(int), seed=5)
    ii = np.array([m["scene"] for m in mined])
    ee = np.array([m["edit"] for m in mined])
    h = hashlib.sha256()
    for mm in mined:
        h.update(np.asarray(mm["scene"]).tobytes())
        h.update(np.asarray(mm["edit"], float).tobytes())
        h.update(np.asarray(mm["new"]).tobytes())
    Xc = torch.from_numpy(((X.reshape(-1, 8) - mu8) / sd8).astype(np.float32))
    yc = torch.from_numpy(y)
    Xf = torch.from_numpy((((X[ii] + ee).reshape(-1, 8) - mu8) / sd8).astype(np.float32))
    yf = torch.from_numpy(np.array([m["new"] for m in mined], dtype=np.float32))
    model = CoordMLP(64, 32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(a.epochs):
        model.train()
        opt.zero_grad()
        loss = bce(model(Xc), yc).mean() + a.lam * bce(model(Xf), yf).mean()
        loss.backward()
        opt.step()
    model.eval()
    od = OUT / ("s%d" % a.seed)
    od.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "hidden": 64, "feat": 32,
                "in_dim": 8, "mu": mu8, "sd": sd8, "seed": a.seed},
               od / "model.pt")
    (od / "manifest.json").write_text(json.dumps(
        {"method": "flipmine", "recipe": "r04b-exact", "epochs": a.epochs,
         "lr": 1e-2, "lam": a.lam, "seed": a.seed, "mine_seed": 5,
         "n_train": len(X), "n_flips": len(mined),
         "flip_list_sha256": h.hexdigest(),
         "final_loss": round(float(loss.detach()), 6)}, indent=1) + "\n")
    print("DONE l007_seed s%d loss=%.4f" % (a.seed, float(loss)), flush=True)


if __name__ == "__main__":
    main()
