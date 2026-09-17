#!/usr/bin/env python3
"""N07 round1: turn-type cover selection (16/32/64/128) vs random vs full.

Measures TRAINING-sample efficiency (pool oracle labels are sunk cost;
reported honestly, not as label efficiency).
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
from r02_search import candidates_for_scene

ART = ROOT / "artifacts" / "discovery_campaign"


def load_pool():
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(float)
    mine = np.load(ART / "r04c_budget" / "mined.npz")
    meta = mine["meta"]
    pool = []
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
        if y1 == y0 or m1 < 0.005:
            continue
        pool.append((pi, e, y0, y1))
    return X, pool


def select_cover(pool, K, rng):
    # stratify by (transition, edited-node mask); k-center on unit dirs
    groups = {}
    for idx, (pi, e, y0, y1) in enumerate(pool):
        mask = tuple((np.linalg.norm(e, axis=1) > 1e-9).astype(int).tolist())
        groups.setdefault((y0, y1, mask), []).append(idx)
    quota = np.array([len(v) for v in groups.values()], float)
    quota = np.maximum(1, np.round(quota / quota.sum() * K)).astype(int)
    while quota.sum() > K:  # trim largest
        quota[int(np.argmax(quota))] -= 1
    sel = []
    keys = list(groups.keys())
    for key, q in zip(keys, quota):
        idxs = groups[key]
        U = np.stack([(pool[i][1] / np.linalg.norm(pool[i][1])).reshape(-1)
                      for i in idxs])
        order = [int(rng.integers(len(idxs)))]
        for _ in range(min(q, len(idxs)) - 1):
            dd = ((U[:, None, :] - U[order][None, :, :]) ** 2).sum(-1).min(1)
            order.append(int(np.argmax(dd)))
        sel += [idxs[o] for o in order]
    return sel[:K]


def train_eval(X, y, flips, mu, sd, seed, epochs=300):
    torch.manual_seed(seed)
    m = CoordMLP(64, 32)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    Xc = torch.from_numpy(((X.reshape(-1, 8) - mu) / sd).astype(np.float32))
    yc = torch.from_numpy(y.astype(np.float32))
    bce = nn.BCEWithLogitsLoss()
    if flips:
        ii = np.array([f[0] for f in flips])
        ee = np.stack([f[1] for f in flips])
        Xf = torch.from_numpy((((X[ii] + ee).reshape(-1, 8) - mu) / sd).astype(np.float32))
        yf = torch.from_numpy(np.array([f[2] for f in flips], dtype=np.float32))
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = bce(m(Xc), yc)
        if flips:
            loss = loss + bce(m(Xf), yf).mean() if False else loss + bce(m(Xf), yf)
        loss.backward()
        opt.step()
    m.eval()
    return m


def mine_eval_flips(Xe, rng, n_want=400):
    out = []
    order = rng.permutation(len(Xe))
    for pi in order:
        if len(out) >= n_want:
            break
        x = Xe[pi].astype(float)
        for e, _ in candidates_for_scene(x, rng, n_random=8):
            e = np.asarray(e, dtype=float)
            try:
                y0, _, _ = oracle_at(x)
                y1, m1, _ = oracle_at(x + e)
            except ValueError:
                continue
            if y1 != y0 and m1 >= 0.005:
                out.append((x, e, y1))
                break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(7)
    X, pool = load_pool()
    mu = X.reshape(-1, 8).mean(0)
    sd = X.reshape(-1, 8).std(0) + 1e-8
    y = np.load(ART / "scenes" / "train_101" / "scenes.npz")["labels"].astype(np.float32)
    Xe = np.load(ART / "scenes" / "eval_202" / "scenes.npz")["positions"].astype(float)
    evflips = mine_eval_flips(Xe, rng)
    res = {"pool_n": len(pool), "eval_flip_n": len(evflips), "arms": {}}
    for K in (16, 32, 64, 128):
        cov = select_cover(pool, K, rng)
        rnd = sorted(rng.choice(len(pool), size=K, replace=False).tolist())
        for name, idxs in (("cover", cov), ("random", rnd)):
            fl = [(pi, e, y0, y1) for (pi, e, y0, y1) in (pool[i] for i in idxs)]
            m = train_eval(X, y, [(pi, e, y1) for (pi, e, _, y1) in fl], mu, sd, 11)
            xs = np.stack([(x + e).astype(np.float32) for x, e, _ in evflips])
            ys = np.array([t for _, _, t in evflips])
            st = {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}
            with torch.no_grad():
                lg = m(preprocess(xs, st)).numpy()
            miss = float((((lg > 0).astype(int)) != ys).mean())
            res["arms"][f"{name}_{K}"] = {"miss_flip": miss, "train_pairs": K}
    # full-pool reference
    m = train_eval(X, y, [(pi, e, y1) for (pi, e, _, y1) in pool], mu, sd, 11)
    xs = np.stack([(x + e).astype(np.float32) for x, e, _ in evflips])
    ys = np.array([t for _, _, t in evflips])
    st = {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}
    with torch.no_grad():
        lg = m(preprocess(xs, st)).numpy()
    res["arms"]["full_1534"] = {"miss_flip": float((((lg > 0).astype(int)) != ys).mean()),
                                "train_pairs": len(pool)}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1,
              default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
    print("SAW: " + "; ".join(f"{k} miss={v['miss_flip']:.3f}" for k, v in res["arms"].items()))
    print("NEXT: if cover_K ~= full at small K -> deepen with 3 seeds + N05 transfer; else drop")
    print("CLAIM: counts semantic turn types, not raw counterexample counts")


if __name__ == "__main__":
    main()
