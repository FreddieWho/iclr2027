#!/usr/bin/env python3
"""G1/G2: train event head on frozen encoder features; dev-gate eval.

Head: Linear(128,64)->ReLU->Linear(64,32)->ReLU->Linear(32,1) on
[z_t,z_{t+1},dz,|dz|]. Encoder frozen (no grad). BCE, Adam 1e-3, 100 epochs,
best dev-Acc_event checkpoint. Seeds 11/23/47.
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
from common import load_model, preprocess  # noqa: E402

EU = ROOT / "artifacts" / "event_updater"


class EventHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                               nn.Linear(64, 32), nn.ReLU(),
                               nn.Linear(32, 1))

    def forward(self, t):
        return self.m(t).squeeze(-1)


@torch.no_grad()
def encode(model, stats, X):
    X = np.asarray(X, dtype=np.float32)
    out = []
    for s in range(0, len(X), 512):
        _, z = model(preprocess(X[s:s + 512], stats), return_feat=True)
        out.append(z.numpy())
    return np.concatenate(out)


def trans_feats(Z0, Z1):
    DZ = Z1 - Z0
    return np.concatenate([Z0, Z1, DZ, np.abs(DZ)], axis=1).astype(np.float32)


def acc_of(head, T, U):
    head.eval()
    with torch.no_grad():
        out = []
        for s in range(0, len(T), 1024):
            out.append(head(T[s:s + 1024]).numpy())
    pred = (np.concatenate(out) > 0).astype(int)
    return float((pred == U).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=100)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    model, stats = load_model(ROOT / "artifacts" / "discovery_campaign" /
                              "r04b_s11" / "clean")
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    tr = np.load(EU / "events_train.npz")
    dv = np.load(EU / "events_dev.npz")
    Z0tr, Z1tr = encode(model, stats, tr["x0"]), encode(model, stats, tr["x1"])
    Z0dv, Z1dv = encode(model, stats, dv["x0"]), encode(model, stats, dv["x1"])
    Ttr = torch.from_numpy(trans_feats(Z0tr, Z1tr))
    Tdv = torch.from_numpy(trans_feats(Z0dv, Z1dv))
    Utr = torch.from_numpy(tr["u"].astype(np.float32))
    Udv = dv["u"].astype(int)
    head = EventHead()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    ce = nn.BCEWithLogitsLoss()
    best, best_ep = -1.0, -1
    for ep in range(a.epochs):
        head.train()
        perm = torch.randperm(len(Ttr), generator=g)
        for s in range(0, len(Ttr), 256):
            idx = perm[s:s + 256]
            opt.zero_grad()
            loss = ce(head(Ttr[idx]), Utr[idx])
            loss.backward()
            opt.step()
        dv_acc = acc_of(head, Tdv, Udv)
        if dv_acc > best:
            best, best_ep = dv_acc, ep
            torch.save(head.state_dict(), EU / f"eventhead_s{a.seed}.pt")
    json.dump({"seed": a.seed, "best_dev_acc": best, "best_epoch": best_ep},
              open(EU / f"eventhead_s{a.seed}.json", "w"), indent=1)
    print(f"DONE eventhead_s{a.seed} best_dev={best:.4f} @ep{best_ep}")


if __name__ == "__main__":
    main()
