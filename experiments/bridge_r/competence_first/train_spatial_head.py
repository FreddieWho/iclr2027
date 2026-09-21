#!/usr/bin/env python3
"""Competence-first C2: static spatial conv head on frozen patch grids.

Head: 1x1 conv 1024->128, 2x Conv3x3 128->128 pad1 + ReLU,
AdaptiveAvgPool 1x1, Linear 128->1. Static state supervision ONLY
(no pairs, no flip/preserve loss, no composition). AdamW 1e-3/wd 1e-4,
batch 128, up to 30 epochs, best single-dev-accuracy checkpoint
(patience 5). Seeds 11/23/47. CPU.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
GF = ROOT / "artifacts" / "bridge_r" / "competence_first" / "feats"
OUT = ROOT / "artifacts" / "bridge_r" / "competence_first" / "static_head"


class SpatialHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1024, 128, 1)
        self.c1 = nn.Conv2d(128, 128, 3, padding=1)
        self.c2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        h = torch.relu(self.proj(x))
        h = torch.relu(self.c1(h))
        h = torch.relu(self.c2(h))
        return self.fc(self.pool(h).flatten(1)).squeeze(-1)


def load_grids(prefix):
    P, Y = [], []
    for nm in sorted(glob.glob(str(GF / f"grid_{prefix}.*.npz"))):
        d = np.load(nm)
        P.append(d["patches"])
        Y.append(d["labels"])
    P = np.concatenate(P).astype(np.float32)
    Y = np.concatenate(Y).astype(np.int64)
    n = len(P)
    X = torch.from_numpy(P.reshape(n, 14, 14, 1024).transpose(0, 3, 1, 2))
    return X, torch.from_numpy(Y).float()


def acc_of(net, X, Y):
    net.eval()
    with torch.no_grad():
        out = []
        for s in range(0, len(X), 512):
            out.append(net(X[s:s + 512]).numpy())
    pred = (np.concatenate(out) > 0).astype(int)
    return float((pred == Y.numpy()).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(6)
    g = torch.Generator().manual_seed(a.seed)
    Xtr, Ytr = load_grids("repair_train")
    Xs, Ys = load_grids("flipsup")
    Xtr = torch.cat([Xtr, Xs])
    Ytr = torch.cat([Ytr, Ys])
    assert len(Xtr) == 12000, len(Xtr)
    Xdv, Ydv = load_grids("repair_dev")
    net = SpatialHead()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.BCEWithLogitsLoss()
    best, best_ep, wait, hist = -1.0, -1, 0, []
    for ep in range(30):
        net.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tot = 0.0
        for s in range(0, len(Xtr), 128):
            idx = perm[s:s + 128]
            opt.zero_grad()
            loss = ce(net(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        dv = acc_of(net, Xdv, Ydv)
        hist.append({"epoch": ep, "train_loss": tot / len(Xtr), "dev_acc": dv})
        if dv > best:
            best, best_ep, wait = dv, ep, 0
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), OUT / f"static_s{a.seed}.pt")
        else:
            wait += 1
            if wait >= 5:
                break
    json.dump({"seed": a.seed, "best_epoch": best_ep, "best_dev_acc": best,
               "history": hist}, open(OUT / f"static_s{a.seed}.json", "w"), indent=1)
    print(f"DONE static_s{a.seed} best_dev={best:.4f} @ep{best_ep}")


if __name__ == "__main__":
    main()
