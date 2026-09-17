#!/usr/bin/env python3
"""Method round 2: targeted mining, support guidance, aux supervision, relfeat.

Shared: train_101, mlpA arch (h64f32, raw 8-dim) except relfeat (10-dim
label-free relation features), 300 epochs, oracle labels everywhere.
Mining (once, seed-fixed): up to 12 flips/scene via candidate protocol.
  flipmine : clean + mean BCE over ALL mined flips
  supmine  : clean + mean BCE over LOW-SUPPORT half (top d_new), tests Track0
  fliprand : clean + mean BCE over RANDOM half (same count control)
  auxmargin: clean BCE + margin-regression head (oracle margins free)
  relfeat  : clean-only on relation features (sharpest arch test)
  clean    : clean-only raw reference (same seed)
Eval afterwards with r02_search (relfeat works via featurize-aware loader).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn

PACK = Path(__file__).resolve().parents[2] / "docs" / "iclr2027_discovery_campaign_20260917"
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.relations import segment_relation
from coord_mlp import CoordMLP, MarginMLP
from common import rel_features
from r02_search import candidates_for_scene


def mine_flips(X, y, seed=5, cap=12):
    rng = np.random.default_rng(seed)
    Pf = X.reshape(len(X), -1)
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
            if o["label"] != int(y[i]):
                xp = (X[i] + e).reshape(-1)
                d = np.linalg.norm(Pf[y == o["label"]] - xp, axis=1)
                found.append({"scene": i, "edit": e.astype(np.float32),
                              "new": int(o["label"]), "margin": float(o["margin"]),
                              "fam": fam, "d_new": float(d.min()) if len(d) else float("nan")})
                if len(found) >= cap:
                    break
        out.extend(found)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--methods", nargs="+",
                   default=["flipmine", "supmine", "fliprand", "auxmargin", "relfeat", "clean"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--mine-seed", type=int, default=5)
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)
    d = np.load(a.train / "scenes.npz")
    X, y = d["positions"].astype(np.float32), d["labels"].astype(np.float32)
    margins = d["oracle_margin"].astype(np.float32)
    N = len(X)
    mu8, sd8 = X.reshape(-1, 8).mean(0), X.reshape(-1, 8).std(0) + 1e-8
    Fr = rel_features(X)
    mur, sdr = Fr.mean(0), Fr.std(0) + 1e-8

    mined = mine_flips(X.astype(float), y.astype(int), seed=a.mine_seed)
    np.savez_compressed(a.output / "mined.npz",
                        **{f"edit_{k}": m["edit"] for k, m in enumerate(mined)},
                        meta=np.array([(m["scene"], m["new"], m["margin"], m["d_new"]) for m in mined]))
    K = len(mined)
    order = np.argsort([m["d_new"] for m in mined])[::-1]
    sup_idx = sorted(order[:K // 2].tolist())
    rrand = np.random.default_rng(a.seed + 777)
    rand_idx = sorted(rrand.choice(K, K // 2, replace=False).tolist())
    sets = {"flipmine": list(range(K)), "supmine": sup_idx, "fliprand": rand_idx}
    print(f"mined {K} flips; sup/rand halves {len(sup_idx)}/{len(rand_idx)}", flush=True)

    def flip_tensors(idx):
        ii = np.array([mined[k]["scene"] for k in idx])
        ee = np.array([mined[k]["edit"] for k in idx])
        Xf = torch.from_numpy((((X[ii] + ee).reshape(-1, 8) - mu8) / sd8).astype(np.float32))
        yf = torch.from_numpy(np.array([mined[k]["new"] for k in idx], dtype=np.float32))
        return Xf, yf

    Xc = torch.from_numpy(((X.reshape(-1, 8) - mu8) / sd8))
    yc = torch.from_numpy(y)
    Xr = torch.from_numpy(((Fr - mur) / sdr).astype(np.float32))
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss()
    models, opt = {}, {}
    for name in a.methods:
        torch.manual_seed(a.seed)
        if name == "auxmargin":
            models[name] = MarginMLP()
        elif name == "relfeat":
            models[name] = CoordMLP(64, 32, in_dim=10)
        else:
            models[name] = CoordMLP(64, 32)
        opt[name] = torch.optim.Adam(models[name].parameters(), lr=a.lr)
    hist = {n: [] for n in a.methods}
    for ep in range(a.epochs):
        for name, model in models.items():
            model.train()
            opt[name].zero_grad()
            if name == "relfeat":
                loss = nn.BCEWithLogitsLoss()(model(Xr), yc)
            elif name == "auxmargin":
                logit, _, mpred = model(Xc, return_margin=True)
                loss = bce(logit, yc).mean() + a.lam * mse(mpred, torch.from_numpy(margins))
            elif name == "clean":
                loss = bce(model(Xc), yc).mean()
            else:
                Xf, yf = flip_tensors(sets[name])
                loss = bce(model(Xc), yc).mean() + a.lam * bce(model(Xf), yf).mean()
            loss.backward()
            opt[name].step()
            hist[name].append(float(loss.detach()))
        if (ep + 1) % 100 == 0:
            print(f"ep {ep+1}: " + " ".join(f"{n}={hist[n][-1]:.4f}" for n in a.methods), flush=True)
    for name, model in models.items():
        model.eval()
        od = a.output / name
        od.mkdir()
        ck = {"state": model.state_dict(), "hidden": 64, "feat": 32,
              "seed": a.seed}
        if name == "relfeat":
            ck.update({"in_dim": 10, "featurize": "rel12", "mu": mur, "sd": sdr})
        else:
            ck.update({"in_dim": 8, "mu": mu8, "sd": sd8})
        torch.save(ck, od / "model.pt")
    (a.output / "manifest.json").write_text(json.dumps(
        {"methods": a.methods, "n_mined_flips": K, "sup_half": len(sup_idx),
         "epochs": a.epochs, "lr": a.lr, "lam": a.lam, "seed": a.seed,
         "mine_seed": a.mine_seed, "n_train": N,
         "note": "all flip sets oracle-labeled; supmine=top d_new half, fliprand=random half same count"},
        indent=2) + "\n")
    print("saved", a.output, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
