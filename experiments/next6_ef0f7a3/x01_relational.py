#!/usr/bin/env python3
"""X01 round1: learned explicit relation interactions vs flat MLP.

Arms (same data = train_101 clean + flipmine flips, same 300ep full-batch
Adam(0.01) recipe, same torch seeds):
 (a) raw: CoordMLP with hidden sized to match relation-net param budget.
 (b) rel-bilinear: shared point encoder -> segment displacements d_AB, d_CD
     -> r = phi([d_AB, d_CD, (W1 d_AB)*(W2 d_CD)]) -> logit.
 (c) rel-concat: identical to (b) but multiply path replaced by zeros
     (exact same params; isolates the multiplicative operator).
Inputs are raw 4 points only: no orientation truth, no predicate, no
hardcoded intersection. Eval: U01 fixed unseen quartets + static eval_202.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
from paths import oracle_at
from common import preprocess
from coord_mlp import CoordMLP

ART = ROOT / "artifacts" / "discovery_campaign"
U01 = ROOT / "artifacts" / "next6_ef0f7a3" / "u01"


class RelNet(nn.Module):
    def __init__(self, pe=32, pd=32, use_multiply=True):
        super().__init__()
        self.use_multiply = use_multiply
        self.enc = nn.Sequential(nn.Linear(2, pe), nn.ReLU(),
                                 nn.Linear(pe, pe), nn.ReLU())
        self.W1 = nn.Linear(pe, pd)
        self.W2 = nn.Linear(pe, pd)
        self.phi = nn.Sequential(nn.Linear(3 * pd, pd), nn.ReLU(),
                                 nn.Linear(pd, 1))
        self.pd = pd

    def forward(self, x):
        h = self.enc(x.view(-1, 4, 2))          # (B,4,pe)
        d_ab = h[:, 1] - h[:, 0]
        d_cd = h[:, 3] - h[:, 2]
        inter = (self.W1(d_ab)) * (self.W2(d_cd)) if self.use_multiply \
            else torch.zeros_like(d_ab)
        return self.phi(torch.cat([d_ab, d_cd, inter], -1)).squeeze(-1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    p.add_argument("--epochs", type=int, default=300)
    a = p.parse_args()
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(float)
    y = d["labels"].astype(np.float32)
    mu = X.reshape(-1, 8).mean(0)
    sd = X.reshape(-1, 8).std(0) + 1e-8
    stats = {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}
    mine = np.load(ART / "r04b_s11" / "mined.npz")
    meta = mine["meta"]
    flips = []
    for i in range(len(meta)):
        if f"edit_{i}" not in mine:
            break
        pi = int(meta[i, 0])
        x = X[pi]
        e = np.asarray(mine[f"edit_{i}"], dtype=float)
        try:
            y0, _, _ = oracle_at(x)
            y1, m1, _ = oracle_at(x + e)
        except ValueError:
            continue
        if y1 != y0 and m1 >= 0.005:
            flips.append((x + e, y1))
    print(f"flips kept: {len(flips)}", flush=True)
    Xc = torch.from_numpy(((X.reshape(-1, 8) - mu) / sd).astype(np.float32))
    yc = torch.from_numpy(y)
    Xf = torch.from_numpy(
        ((np.stack([f[0] for f in flips]).reshape(-1, 8) - mu) / sd).astype(np.float32))
    yf = torch.from_numpy(np.array([f[1] for f in flips], dtype=np.float32))

    # eval: U01 fixed unseen quartets
    qe = np.load(U01 / "quartets_eval.npz")
    Q = [(qe[f"q{i}"], qe["meta"][i]) for i in range(len(qe["meta"]))]
    ev_end = np.stack([(q[0] + q[1] + q[2]).astype(np.float32) for q, _ in Q])
    ev_end_y = np.array([m[3] for _, m in Q])
    ev_a = np.stack([(q[0] + q[1]).astype(np.float32) for q, _ in Q])
    ev_a_y = np.array([m[1] for _, m in Q])
    ev_b = np.stack([(q[0] + q[2]).astype(np.float32) for q, _ in Q])
    ev_b_y = np.array([m[2] for _, m in Q])
    de = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    Xe, ye = de["positions"].astype(np.float32), de["labels"].astype(int)

    ref = RelNet()
    budget = n_params(ref)
    h = 8
    while n_params(CoordMLP(h, h // 2)) < budget:
        h += 4
    print(f"budget={budget} raw_hidden={h} raw_params={n_params(CoordMLP(h, h // 2))}",
          flush=True)

    def run(net_fn, seed_id):
        torch.manual_seed(seed_id)
        m = net_fn()
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        bce = nn.BCEWithLogitsLoss()
        m.train()
        for _ in range(a.epochs):
            opt.zero_grad()
            if isinstance(m, CoordMLP):
                loss = bce(m(Xc), yc) + bce(m(Xf), yf)
            else:
                loss = bce(m(Xc.view(-1, 4, 2)), yc) + bce(m(Xf.view(-1, 4, 2)), yf)
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            if isinstance(m, CoordMLP):
                pe = (m(preprocess(ev_end, stats)).numpy() > 0).astype(int)
                pa = (m(preprocess(ev_a, stats)).numpy() > 0).astype(int)
                pb = (m(preprocess(ev_b, stats)).numpy() > 0).astype(int)
                ps = (m(preprocess(Xe, stats)).numpy() > 0).astype(int)
            else:
                fe = lambda v: (v.reshape(-1, 8) - mu) / sd
                pe = (m(torch.from_numpy(fe(ev_end).astype(np.float32))).numpy() > 0).astype(int)
                pa = (m(torch.from_numpy(fe(ev_a).astype(np.float32))).numpy() > 0).astype(int)
                pb = (m(torch.from_numpy(fe(ev_b).astype(np.float32))).numpy() > 0).astype(int)
                ps = (m(torch.from_numpy(fe(Xe).astype(np.float32))).numpy() > 0).astype(int)
        both = (pa == ev_a_y) & (pb == ev_b_y)
        return {"emergent_miss": float((pe != ev_end_y).mean()),
                "atomic_acc": float(both.mean()),
                "cond_miss": float((pe[both] != ev_end_y[both]).mean()) if both.sum() else None,
                "static_err": float((ps != ye).mean())}

    res = {"budget": budget, "raw_hidden": h,
           "raw_params": n_params(CoordMLP(h, h // 2)),
           "rel_params": budget, "n_eval_quartets": len(Q),
           "n_flips": len(flips)}
    arms = {"raw": lambda: CoordMLP(h, h // 2),
            "rel_bilinear": lambda: RelNet(use_multiply=True),
            "rel_concat": lambda: RelNet(use_multiply=False)}
    for seed_id in a.seeds:
        for name, fn in arms.items():
            r = run(fn, seed_id)
            res[f"{name}_s{seed_id}"] = r
            print(f"SAW {name} s{seed_id}: emerg={r['emergent_miss']:.3f} "
                  f"atom={r['atomic_acc']:.3f} cond={r['cond_miss']} "
                  f"static={r['static_err']:.3f}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print("DONE X01")


if __name__ == "__main__":
    main()
