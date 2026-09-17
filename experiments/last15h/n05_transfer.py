#!/usr/bin/env python3
"""N05 round1: frozen source-flip textbook -> fresh smaller MLP target.

Source textbook = fixed 1534 oracle flips (no target-side failure search).
Target = fresh CoordMLP(32,16): clean-only vs clean+textbook vs clean+random.
Honest limitation round1: same MLP family (capacity change only).
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[23])
    a = p.parse_args()
    rng = np.random.default_rng(5)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(float)
    y = d["labels"].astype(np.float32)
    mu = X.reshape(-1, 8).mean(0)
    sd = X.reshape(-1, 8).std(0) + 1e-8
    mine = np.load(ART / "r04c_budget" / "mined.npz")
    meta = mine["meta"]
    tb = []
    for i in range(len(meta)):
        if f"edit_{i}" not in mine:
            break
        pi = int(meta[i, 0])
        e = np.asarray(mine[f"edit_{i}"], dtype=float)
        try:
            y0, _, _ = oracle_at(X[pi])
            y1, m1, _ = oracle_at(X[pi] + e)
        except ValueError:
            continue
        if y1 != y0 and m1 >= 0.005:
            tb.append((pi, e, y1))
    # random same-count control: oracle-labeled random edits
    rnd = []
    while len(rnd) < len(tb):
        pi = int(rng.integers(len(X)))
        e = rng.normal(size=(4, 2))
        e *= rng.choice([0.03, 0.06, 0.10, 0.16]) * 2 / np.linalg.norm(e)
        try:
            o = oracle_at(X[pi] + e)
        except ValueError:
            continue
        if o[2] or o[1] < 0.005:
            continue
        rnd.append((pi, e, o[0]))
    # eval flips on unseen-parent scenes (oracle-only mining, no model queries)
    Xr = np.load(ART / "scenes" / "recheck_404" / "scenes.npz")["positions"].astype(float)
    ev = []
    for pi in rng.permutation(len(Xr)):
        if len(ev) >= 400:
            break
        x = Xr[pi]
        for e, _ in candidates_for_scene(x, rng, n_random=8):
            e = np.asarray(e, dtype=float)
            try:
                y0, _, _ = oracle_at(x)
                y1, m1, _ = oracle_at(x + e)
            except ValueError:
                continue
            if y1 != y0 and m1 >= 0.005:
                ev.append((x, e, y1))
                break

    def train(flips, seed):
        torch.manual_seed(seed)
        m = CoordMLP(32, 16)
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        Xc = torch.from_numpy(((X.reshape(-1, 8) - mu) / sd).astype(np.float32))
        yc = torch.from_numpy(y)
        bce = nn.BCEWithLogitsLoss()
        Xf = yf = None
        if flips:
            xx = np.stack([np.asarray(f[0], dtype=float) for f in flips])
            ee = np.stack([np.asarray(f[1], dtype=float) for f in flips])
            Xf = torch.from_numpy((((xx + ee).reshape(-1, 8) - mu) / sd).astype(np.float32))
            yf = torch.from_numpy(np.array([f[2] for f in flips], dtype=np.float32))
        m.train()
        for _ in range(300):
            opt.zero_grad()
            loss = bce(m(Xc), yc)
            if Xf is not None:
                loss = loss + bce(m(Xf), yf)
            loss.backward()
            opt.step()
        m.eval()
        return m

    # encode textbook/random with explicit base scenes
    tb_full = [(X[pi], e, y1) for (pi, e, y1) in tb]
    rnd_full = [(X[pi], e, y1) for (pi, e, y1) in rnd]
    st = {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}
    xs = np.stack([(x + e).astype(np.float32) for x, e, _ in ev])
    ys = np.array([t for _, _, t in ev])
    res = {"textbook_n": len(tb_full), "eval_n": len(ev)}
    for sd in a.seeds:
        for name, fl in (("clean_only", None), ("textbook", tb_full), ("random", rnd_full)):
            m = train(fl, sd)
            with torch.no_grad():
                lg = m(preprocess(xs, st)).numpy()
            res[f"{name}_s{sd}"] = {"miss_flip": float((((lg > 0).astype(int)) != ys).mean())}
    for sd in a.seeds:
        print(f"SAW seed{sd}: " + " ".join(
            f"{k.replace(f'_s{sd}','')}={res[k]['miss_flip']:.3f}" for k in res
            if k.endswith(f"_s{sd}")))
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print("NEXT: if confirmed over seeds -> DeepSets target; else drop")
    print("CLAIM: transfers the repair textbook, not the attack directions")


if __name__ == "__main__":
    main()
