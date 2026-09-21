#!/usr/bin/env python3
"""WP-B transfer training: base train_101 + fixed repair set, r04b recipe.

Repair sets: random-N / flipmine-N / src11 / src23 / diverse (pooled).
Targets: seed11 (source sanity), seed47 (unseen seed, fresh init 64/32),
archB (fresh init 32/16). Adam 1e-2, 300ep full-batch, lam 1.0.
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
from r02_search import candidates_for_scene  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
OUT = ROOT / "artifacts" / "novelty_round3" / "transferable_repair"


def load_set(name, n, seed):
    rng = np.random.default_rng(seed)
    X = np.load(ART / "scenes" / "train_101" / "scenes.npz")["positions"].astype(np.float32)
    if name == "random":
        rows = []
        for pi in rng.permutation(len(X)):
            if len(rows) >= n:
                break
            x = X[pi].astype(float)
            for item in candidates_for_scene(x, rng, n_random=8):
                e = np.asarray(item[0], dtype=float)
                try:
                    y1, m1, _ = oracle_at(x + e)
                except ValueError:
                    continue
                if m1 < 0.005:
                    continue
                rows.append((x + e, y1))
                if len(rows) >= n:
                    break
        return rows
    if name == "flipmine":
        m = np.load(ART / "r04b_s11" / "mined.npz")
        meta = m["meta"]
        idx = rng.choice(len(meta), n, replace=False)
        rows = []
        for i in idx:
            pi = int(meta[i, 0])
            e = np.stack([m[f"edit_{i}"]]).astype(np.float32)[0] if f"edit_{i}" in m else None
            if e is None:
                continue
            x = X[pi].astype(float)
            try:
                y1, _, _ = oracle_at(x + e)
            except ValueError:
                continue
            rows.append((x + e, y1))
        return rows
    if name in ("src11", "src23"):
        d = np.load(OUT / f"repairset_{name}.npz")
        idx = rng.choice(len(d["scene"]), min(n, len(d["scene"])), replace=False)
        rows = []
        for i in idx:
            x = X[int(d["scene"][i])].astype(float)
            rows.append((x + np.asarray(d["edit"][i], dtype=float), int(d["y1"][i])))
        return rows
    if name == "diverse":
        pool = []
        for t in ("src11", "src23"):
            d = np.load(OUT / f"repairset_{t}.npz")
            for i in range(len(d["scene"])):
                pool.append((int(d["scene"][i]), np.asarray(d["edit"][i], dtype=float), int(d["y1"][i])))
        rng.shuffle(pool)
        kept = []
        for pi, e, y1 in pool:
            if all(np.linalg.norm(e - k[1]) >= 0.05 for k in kept):
                kept.append((pi, e, y1))
            if len(kept) >= n:
                break
        Xx = np.load(ART / "scenes" / "train_101" / "scenes.npz")["positions"].astype(np.float32)
        return [(Xx[pi].astype(float) + e, y1) for pi, e, y1 in kept]
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True)
    p.add_argument("--target", choices=["seed11", "seed47", "archB"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n", type=int, default=150)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(np.float32).reshape(-1, 8)
    y = d["labels"].astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xc = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    yc = torch.from_numpy(y)
    rep = load_set(a.set, a.n, a.seed + 1000)
    Xr = torch.from_numpy(((np.stack([r[0] for r in rep]).reshape(-1, 8) - mu) / sd).astype(np.float32))
    yr = torch.from_numpy(np.array([r[1] for r in rep], dtype=np.float32))
    hidden, feat = (32, 16) if a.target == "archB" else (64, 32)
    model = CoordMLP(hidden, feat)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(300):
        model.train()
        opt.zero_grad()
        loss = bce(model(Xc), yc) + 1.0 * bce(model(Xr), yr)
        loss.backward()
        opt.step()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "hidden": hidden, "feat": feat,
                "in_dim": 8, "mu": mu, "sd": sd, "seed": a.seed},
               OUT / f"transfer_{a.set}_{a.target}_s{a.seed}.pt")
    json.dump({"set": a.set, "target": a.target, "seed": a.seed,
               "n_repair": len(rep), "final_loss": float(loss)},
              open(OUT / f"transfer_{a.set}_{a.target}_s{a.seed}.json", "w"), indent=1)
    print(f"DONE transfer_{a.set}_{a.target}_s{a.seed} n={len(rep)} loss={float(loss):.4f}", flush=True)


if __name__ == "__main__":
    main()
