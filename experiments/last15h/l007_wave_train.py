#!/usr/bin/env python3
"""L007 wave retrain: flipmine recipe on seeds 2001-2008 WITH epoch
checkpoints (every 25 epochs + init) for P1 lock-in dynamics and P2
TracIn screening. Recipe byte-identical to l007_seedtrain.py (r04b-exact);
only checkpoint saving is added. Determinism check: final weights must
match l007_seed finals (reported, not hard-failed).

Outputs: artifacts/next_novelty/l007_wave/s{seed}/ckpt_{eee}.pt + manifest
No sealed pools, no overwrites.
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
OUT = ROOT / "artifacts" / "next_novelty" / "l007_wave"
SEEDS = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]
EVERY = 25


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lam", type=float, default=1.0)
    a = p.parse_args()
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
    flip_sha = h.hexdigest()
    Xc = torch.from_numpy(((X.reshape(-1, 8) - mu8) / sd8).astype(np.float32))
    yc = torch.from_numpy(y)
    Xf = torch.from_numpy((((X[ii] + ee).reshape(-1, 8) - mu8) / sd8).astype(np.float32))
    yf = torch.from_numpy(np.array([m["new"] for m in mined], dtype=np.float32))
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = CoordMLP(64, 32)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        bce = nn.BCEWithLogitsLoss()
        od = OUT / ("s%d" % seed)
        od.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "hidden": 64, "feat": 32,
                    "in_dim": 8, "mu": mu8, "sd": sd8, "seed": seed},
                   od / "ckpt_000.pt")
        for ep in range(a.epochs):
            model.train()
            opt.zero_grad()
            loss = bce(model(Xc), yc).mean() + a.lam * bce(model(Xf), yf).mean()
            loss.backward()
            opt.step()
            if (ep + 1) % EVERY == 0:
                torch.save({"state": model.state_dict(), "hidden": 64,
                            "feat": 32, "in_dim": 8, "mu": mu8, "sd": sd8,
                            "seed": seed}, od / ("ckpt_%03d.pt" % (ep + 1)))
        model.eval()
        (od / "manifest.json").write_text(json.dumps(
            {"method": "flipmine", "recipe": "r04b-exact+ckpts", "epochs": a.epochs,
             "lr": 1e-2, "lam": a.lam, "seed": seed, "mine_seed": 5,
             "n_train": len(X), "n_flips": len(mined),
             "flip_list_sha256": flip_sha,
             "ckpt_every": EVERY,
             "final_loss": round(float(loss.detach()), 6)}, indent=1) + "\n")
        print("DONE l007_wave s%d loss=%.4f" % (seed, float(loss)), flush=True)


if __name__ == "__main__":
    main()
