#!/usr/bin/env python3
"""N04 round2: combo-cover curriculum. Train on emergent-combo flips,
eval on UNSEEN-parent emergent combos. Controls: clean-only, same-count
random flips. Same recipe (full-batch Adam 0.01, 300ep)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
from paths import atomic_edits, oracle_at
from common import preprocess
from coord_mlp import CoordMLP
from r02_search import candidates_for_scene

ART = ROOT / "artifacts" / "discovery_campaign"


def mine_emergent(X, rng):
    out = []
    for pi in range(len(X)):
        x = X[pi].astype(float)
        try:
            y0, _, _ = oracle_at(x)
        except ValueError:
            continue
        res = []
        for e, _ in atomic_edits(x, rng):
            try:
                y, m, _ = oracle_at(x + e)
            except ValueError:
                continue
            if m < 0.005:
                continue
            res.append((e, y))
        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                (ea, ya), (eb, yb) = res[i], res[j]
                if ya != y0 or yb != y0:
                    continue
                try:
                    yc, mc, _ = oracle_at(x + ea + eb)
                except ValueError:
                    continue
                if yc != y0 and mc >= 0.005:
                    out.append((x, ea + eb, yc))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[11])
    a = p.parse_args()
    rng = np.random.default_rng(44)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(float)
    y = d["labels"].astype(np.float32)
    mu = X.reshape(-1, 8).mean(0)
    sd = X.reshape(-1, 8).std(0) + 1e-8
    Etr = mine_emergent(X, rng)
    # random-flip control, same count
    Rnd = []
    for pi in rng.permutation(len(X)):
        if len(Rnd) >= len(Etr):
            break
        x = X[pi]
        for e, _ in candidates_for_scene(x, rng, n_random=16):
            e = np.asarray(e, dtype=float)
            try:
                y0, _, _ = oracle_at(x)
                y1, m1, _ = oracle_at(x + e)
            except ValueError:
                continue
            if y1 != y0 and m1 >= 0.005:
                Rnd.append((x, e, y1))
                break
    # unseen-parent emergent eval
    Eev = []
    for pool in ("recheck_404", "recheck_505", "recheck_606"):
        de = np.load(ART / "scenes" / pool / "scenes.npz")
        Eev += mine_emergent(de["positions"].astype(float), rng)

    def train(flips, seed):
        torch.manual_seed(seed)
        m = CoordMLP(64, 32)
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

    st = {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}
    xs = np.stack([(x + e).astype(np.float32) for x, e, _ in Eev])
    ys = np.array([t for _, _, t in Eev])
    res = {"n_train_emergent": len(Etr), "n_eval_emergent": len(Eev)}
    for sd in a.seeds:
        for name, fl in (("clean_only", None), ("combo_cover", Etr), ("random", Rnd)):
            m = train(fl, sd)
            with torch.no_grad():
                lg = m(preprocess(xs, st)).numpy()
            res[f"{name}_s{sd}"] = {"emergent_miss": float((((lg > 0).astype(int)) != ys).mean())}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    for sd in a.seeds:
        row = "; ".join(f"{k.replace(f'_s{sd}', '')}={res[k]['emergent_miss']:.3f}" for k in res if k.endswith(f"_s{sd}"))
        print(f"SAW seed{sd}: train_emerg={len(Etr)} eval_emerg={len(Eev)} " + row)
    print("NEXT: if combo_cover beats random on unseen emergent -> method lives, "
          "enter paper; else phenomenon-only")
    print("CLAIM: turn-type cover over compositions, not raw flips")


if __name__ == "__main__":
    main()
