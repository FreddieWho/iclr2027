#!/usr/bin/env python3
"""Repair-transfer R7-R9: train one arm x one seed (CPU).

Adapter: Linear(1024,128)->ReLU->Linear(128,1024) residual (alpha=1)
+ Linear head. AdamW lr=1e-3 wd=1e-4, fixed 20 epochs, no early stop,
no composition-based selection. Equal budget: 40 steps/epoch x 20 = 800.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
RT = ROOT / "artifacts" / "bridge_r" / "repair_transfer"
FEAT = RT / "feats"
OUT = RT / "adapters"


class Adapter(nn.Module):
    def __init__(self, d=1024, h=128):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        g = self.fc2(torch.relu(self.fc1(x)))
        hp = x + g
        return self.head(hp).squeeze(-1), hp


def load_pairs(prefixes):
    """Concatenate feat files into (base, end) pair arrays + labels."""
    import glob
    Z0, Z1, Y0, Y1 = [], [], [], []
    for pre in prefixes:
        for nm in sorted(glob.glob(str(FEAT / f"{pre}.*.npz"))):
            d = np.load(nm)
            z, y = d["z"].astype(np.float32), d["y"].astype(np.int64)
            Z0.append(z[0::2])
            Z1.append(z[1::2])
            Y0.append(y[0::2])
            Y1.append(y[1::2])
    return (np.concatenate(Z0), np.concatenate(Z1),
            np.concatenate(Y0), np.concatenate(Y1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["static", "flip_only", "balanced"], required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    g = torch.Generator().manual_seed(a.seed)
    # ---- data per arm (protocol frozen) ----
    if a.arm == "static":
        Z0, Z1, Y0, Y1 = load_pairs(["repair_train"])
        assert len(Y0) == 4000, f"static expects 4000 pairs, got {len(Y0)}"
        X = np.concatenate([Z0, Z1])
        Y = np.concatenate([Y0, Y1])
        pair_mode = False
    elif a.arm == "flip_only":
        Z0t, Z1t, Y0t, Y1t = load_pairs(["repair_train"])
        Z0s, Z1s, Y0s, Y1s = load_pairs(["flipsup"])
        f = np.concatenate([Y0t != Y1t, np.ones(len(Y0s), bool)])
        Z0 = np.concatenate([Z0t, Z0s])[np.concatenate(
            [Y0t != Y1t, np.ones(len(Y0s), bool)])]
        Z1 = np.concatenate([Z1t, Z1s])[np.concatenate(
            [Y0t != Y1t, np.ones(len(Y0s), bool)])]
        Y0 = np.concatenate([Y0t, Y0s])[f]
        Y1 = np.concatenate([Y1t, Y1s])[f]
        assert (Y0 != Y1).all()
        assert len(Y0) == 4000, f"flip_only expects 4000 pairs, got {len(Y0)}"
        pair_mode = "flip"
    else:
        Z0, Z1, Y0, Y1 = load_pairs(["repair_train"])
        assert len(Y0) == 4000, f"balanced expects 4000 pairs, got {len(Y0)}"
        pair_mode = "balanced"
    net = Adapter()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.BCEWithLogitsLoss()
    n_epochs = 20
    hist = []
    if not pair_mode:
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y).float()
        for ep in range(n_epochs):
            perm = torch.randperm(len(X), generator=g)
            tot = 0.0
            for s in range(0, len(X), 200):
                idx = perm[s:s + 200]
                opt.zero_grad()
                logit, _ = net(X[idx])
                loss = ce(logit, Y[idx])
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            hist.append(tot / len(X))
    else:
        Z0t = torch.from_numpy(Z0)
        Z1t = torch.from_numpy(Z1)
        Y0t = torch.from_numpy(Y0).float()
        Y1t = torch.from_numpy(Y1).float()
        S0 = torch.from_numpy(2 * Y0 - 1).float()
        S1 = torch.from_numpy(2 * Y1 - 1).float()
        is_flip = (Y0 != Y1)
        n = len(Z0t)
        for ep in range(n_epochs):
            perm = torch.randperm(n, generator=g)
            tot = 0.0
            for s in range(0, n, 100):
                idx = perm[s:s + 100]
                opt.zero_grad()
                l0, _ = net(Z0t[idx])
                l1, _ = net(Z1t[idx])
                loss = ce(l0, Y0t[idx]) + ce(l1, Y1t[idx])
                fl = is_flip[idx]
                if pair_mode == "balanced" and (~fl).any():
                    j = idx[~fl]
                    lp0, _ = net(Z0t[j])
                    lp1, _ = net(Z1t[j])
                    loss = loss + ((lp0 - lp1) ** 2).mean()
                if fl.any():
                    j = idx[fl]
                    lf0, _ = net(Z0t[j])
                    lf1, _ = net(Z1t[j])
                    s0 = S0[j]
                    s1 = S1[j]
                    loss = loss + (torch.clamp(1 - s0 * lf0, min=0).mean()
                                   + torch.clamp(1 - s1 * lf1, min=0).mean())
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            hist.append(tot / n)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.arm}_s{a.seed}"
    torch.save(net.state_dict(), OUT / f"{tag}.pt")
    json.dump({"arm": a.arm, "seed": a.seed, "epochs": n_epochs,
               "optimizer": "AdamW lr=1e-3 wd=1e-4",
               "train_loss_per_epoch": hist}, open(OUT / f"{tag}.json", "w"), indent=1)
    print(f"DONE {tag} final_loss={hist[-1]:.4f}")


if __name__ == "__main__":
    main()
