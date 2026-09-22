#!/usr/bin/env python3
"""P3-D pilot: preserve-balanced repair vs flip-only (same augmented budget).

Arms (train_101 base + mined augmentation, CoordMLP 64/32, Adam 1e-2, 300ep):
 fliprep : base + K flips (flipmine replica)
 fliphalf: base + K/2 flips (budget control)
 keepbal : base + K/2 flips + K/2 preserves (1:1 balanced)
Eval on dev bank: J, R_full, R_endpoint, M + static/single/preserveFA/comp.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "docs" / "iclr2027_discovery_campaign_20260917"))
torch.set_num_threads(4)
from core.relations import segment_relation  # noqa: E402
from r02_search import candidates_for_scene  # noqa: E402
from coord_mlp import CoordMLP  # noqa: E402
from common import load_model, preprocess  # noqa: E402
from r04b_methods import mine_flips  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
OUT = ROOT / "artifacts" / "next_novelty" / "p3d"


def mine_preserves(X, y, seed=6, cap=12):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(len(X)):
        found = []
        for e, fam in candidates_for_scene(X[i], rng):
            try:
                o = segment_relation(X[i] + e)
            except ValueError:
                continue
            if o["ambiguous"] or o["margin"] < 0.03:
                continue
            if o["label"] == int(y[i]):
                found.append({"scene": i, "edit": e.astype(np.float32),
                              "new": int(o["label"])})
                if len(found) >= cap:
                    break
        out.extend(found)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X, y = d["positions"].astype(np.float32), d["labels"].astype(np.float32)
    mu8, sd8 = X.reshape(-1, 8).mean(0), X.reshape(-1, 8).std(0) + 1e-8
    flips = mine_flips(X.astype(float), y.astype(int), seed=5)
    preserves = mine_preserves(X.astype(float), y.astype(int), seed=6)
    K = len(flips)
    P = len(preserves)
    r = np.random.default_rng(a.seed + 4242)
    fh = sorted(r.choice(K, K // 2, replace=False).tolist())
    ph = sorted(r.choice(P, K // 2, replace=False).tolist())

    def tensors(items, key):
        ii = np.array([m["scene"] for m in items])
        ee = np.array([m["edit"] for m in items])
        Xt = torch.from_numpy((((X[ii] + ee).reshape(-1, 8) - mu8) / sd8).astype(np.float32))
        yt = torch.from_numpy(np.array([m[key] for m in items], dtype=np.float32))
        return Xt, yt
    Xc = torch.from_numpy(((X.reshape(-1, 8) - mu8) / sd8).astype(np.float32))
    yc = torch.from_numpy(y)
    Xf, yf = tensors(flips, "new")
    Xfh, yfh = tensors([flips[i] for i in fh], "new")
    Xph, yph = tensors([preserves[i] for i in ph], "new")
    bce = nn.BCEWithLogitsLoss()
    arms = {
        "fliprep": [(Xc, yc, 1.0), (Xf, yf, 1.0)],
        "fliphalf": [(Xc, yc, 1.0), (Xfh, yfh, 1.0)],
        "keepbal": [(Xc, yc, 1.0), (Xfh, yfh, 1.0), (Xph, yph, 1.0)],
    }
    a.out.mkdir(parents=True, exist_ok=True)
    for name, terms in arms.items():
        torch.manual_seed(a.seed)
        model = CoordMLP(64, 32)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for ep in range(a.epochs):
            model.train()
            opt.zero_grad()
            loss = sum(w * bce(model(Xt), yt) for Xt, yt, w in terms) / len(terms)
            loss.backward()
            opt.step()
        model.eval()
        od = a.out / ("s%d_%s" % (a.seed, name))
        od.mkdir(exist_ok=True)
        torch.save({"state": model.state_dict(), "hidden": 64, "feat": 32,
                    "in_dim": 8, "mu": mu8, "sd": sd8, "seed": a.seed},
                   od / "model.pt")
        print("SAW s%d %s final=%.4f (K=%d,P=%d)" % (a.seed, name, float(loss), K, P), flush=True)
    json.dump({"seed": a.seed, "K": K, "P": P},
              open(a.out / ("meta_s%d.json" % a.seed), "w"), indent=1)
    print("DONE s%d" % a.seed)


if __name__ == "__main__":
    main()
