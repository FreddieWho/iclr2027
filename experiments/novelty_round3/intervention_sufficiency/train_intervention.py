#!/usr/bin/env python3
"""WP-A training: A0 static / A1 same-data augmentation / A2 intervention.

Shared encoder = CoordMLP.net (64-d h); state head = feat(64->32)+cls;
intervention head MLP([h64, e8] -> 64 -> 32 -> 1) predicts y(x+e) WITHOUT
seeing x+e, h(x+e), or oracle margin. A2 loss = BCE(state) + 1.0*BCE(interv).
Same base + same pairs as radius pilot (750+750, rng 7). r04b recipe.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
torch.set_num_threads(4)
from coord_mlp import CoordMLP  # noqa: E402
from paths import oracle_at  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
EU = ROOT / "artifacts" / "event_updater"
OUT = ROOT / "artifacts" / "novelty_round3" / "intervention_sufficiency"


class IntervHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(72, 64), nn.ReLU(),
                               nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, h, e):
        return self.m(torch.cat([h, e], 1)).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["A0", "A1", "A2"], required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(7)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(np.float32).reshape(-1, 8)
    y = d["labels"].astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-8

    def std(v):
        return torch.from_numpy(((np.asarray(v, dtype=np.float32).reshape(-1, 8) - mu) / sd).astype(np.float32))

    Xc, yc = std(X), torch.from_numpy(y)
    ev = np.load(EU / "events_train.npz")
    U = ev["u"]
    fl = rng.choice(np.nonzero(U == 1)[0], 750, replace=False)
    st = rng.choice(np.nonzero(U == 0)[0], 750, replace=False)
    Xe = torch.cat([std(ev["x1"][fl]), std(ev["x1"][st])])
    ye = torch.from_numpy(np.array(
        [oracle_at(ev["x1"][i])[0] for i in list(fl) + list(st)], dtype=np.float32))
    Ee = torch.from_numpy(np.stack(
        [(ev["x1"][i] - ev["x0"][i]).reshape(-1) for i in list(fl) + list(st)]).astype(np.float32))
    Xp0 = torch.cat([std(ev["x0"][fl]), std(ev["x0"][st])])
    yp1 = torch.from_numpy(np.array(
        [oracle_at(ev["x1"][i])[0] for i in list(fl) + list(st)], dtype=np.float32))
    enc = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
    feat = nn.Linear(64, 32)
    cls = nn.Linear(32, 1)
    ihead = IntervHead()
    mods = [enc, feat, cls] + ([ihead] if a.arm == "A2" else [])
    opt = torch.optim.Adam([p_ for m in mods for p_ in m.parameters()], lr=1e-2)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(300):
        for m in mods:
            m.train()
        opt.zero_grad()
        h = enc(Xc)
        loss = bce(cls(feat(h)).squeeze(-1), yc)
        if a.arm in ("A1", "A2"):
            he = enc(Xe)
            loss = loss + bce(cls(feat(he)).squeeze(-1), ye)
        if a.arm == "A2":
            h0 = enc(Xp0)
            loss = loss + 1.0 * bce(ihead(h0, Ee), yp1)
        loss.backward()
        opt.step()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"enc": enc.state_dict(), "feat": feat.state_dict(),
                "cls": cls.state_dict(),
                "ihead": ihead.state_dict() if a.arm == "A2" else None,
                "mu": mu, "sd": sd, "seed": a.seed, "arm": a.arm},
               OUT / f"interv_{a.arm}_s{a.seed}.pt")
    print(f"DONE interv_{a.arm}_s{a.seed} loss={float(loss):.4f}", flush=True)


if __name__ == "__main__":
    main()
