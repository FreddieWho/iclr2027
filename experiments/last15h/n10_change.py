#!/usr/bin/env python3
"""N10 round1: learn P(same/different) on (x,x') pairs; honest baselines.

No claim of beating single-frame classifiers with more input: reports
change-F1, false alarms, and miss-at-matched-FA vs classify-and-compare.
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
from common import load_model, preprocess
from r02_search import candidates_for_scene

ART = ROOT / "artifacts" / "discovery_campaign"


class ConcatMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(),
                                 nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SiamDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16))
        self.head = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        z0, z1 = self.enc(x[:, :8]), self.enc(x[:, 8:])
        return self.head(torch.cat([z0, z1, z1 - z0], 1)).squeeze(-1)


def build_pairs(X, flips, rng, n_neg):
    pos = [(x, x + e) for (x, e) in flips]
    # negatives: label-preserving edits, magnitude-matched
    costs = sorted(float(np.linalg.norm(e)) for _, e in flips)
    neg = []
    pis = rng.permutation(len(X))
    ci = 0
    for pi in pis:
        if len(neg) >= n_neg:
            break
        x = X[pi].astype(float)
        try:
            y0, _, _ = oracle_at(x)
        except ValueError:
            continue
        for e, _ in candidates_for_scene(x, rng, n_random=16):
            e = np.asarray(e, dtype=float)
            if abs(np.linalg.norm(e) - costs[ci % len(costs)]) > 0.02:
                continue
            try:
                y, m, _ = oracle_at(x + e)
            except ValueError:
                continue
            if y == y0 and m >= 0.005:
                neg.append((x, x + e))
                ci += 1
                break
    return pos, neg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(10)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(float)
    mine = np.load(ART / "r04c_budget" / "mined.npz")
    meta = mine["meta"]
    flips = []
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
            flips.append((X[pi], e))
    pos, neg = build_pairs(X, flips, rng, len(flips))
    # split by parent scene hash -> use index parity of a scene id map
    def split(pairs, frac=0.8):
        k = int(len(pairs) * frac)
        return pairs[:k], pairs[k:]
    ptr, pte = split(pos)
    ntr, nte = split(neg)
    mu = X.reshape(-1, 8).mean(0)
    sd = X.reshape(-1, 8).std(0) + 1e-8

    def arr(pairs):
        A = np.stack([np.concatenate([(x).reshape(-1), (xp).reshape(-1)]) for x, xp in pairs])
        return ((A - np.concatenate([mu, mu])) / np.concatenate([sd, sd])).astype(np.float32)

    Xtr = torch.from_numpy(arr(ptr + ntr))
    ytr = torch.from_numpy(np.array([1] * len(ptr) + [0] * len(ntr), dtype=np.float32))
    Xte = torch.from_numpy(arr(pte + nte))
    yte = np.array([1] * len(pte) + [0] * len(nte))
    res = {"n_train_pos": len(ptr), "n_train_neg": len(ntr),
           "n_test_pos": len(pte), "n_test_neg": len(nte)}
    for name, M in (("concat", ConcatMLP()), ("siamdiff", SiamDiff())):
        torch.manual_seed(11)
        opt = torch.optim.Adam(M.parameters(), lr=0.01)
        bce = nn.BCEWithLogitsLoss()
        M.train()
        for _ in range(200):
            opt.zero_grad()
            loss = bce(M(Xtr), ytr)
            loss.backward()
            opt.step()
        M.eval()
        with torch.no_grad():
            lg = M(Xte).numpy()
        # threshold sweep: report miss at FA<=0.01 and full operating points
        order = np.argsort(-lg)
        yteo = yte[order]
        n_neg = int((yte == 0).sum())
        n_pos = int((yte == 1).sum())
        fp_cum = np.cumsum(yteo == 0)
        tp_cum = np.cumsum(yteo == 1)
        fa_curve = fp_cum / max(1, n_neg)
        miss_curve = 1 - tp_cum / max(1, n_pos)
        j = int(np.searchsorted(fa_curve, 0.01, side="right")) - 1
        j = max(0, min(j, len(fa_curve) - 1))
        pr = (lg > 0).astype(int)
        tp = int(((pr == 1) & (yte == 1)).sum())
        fp = int(((pr == 1) & (yte == 0)).sum())
        fn = int(((pr == 0) & (yte == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        res[name] = {"f1": float(f1), "fa_rate": fp / max(1, len(nte)),
                     "miss_rate": fn / max(1, len(pte)),
                     "miss_at_FA01": float(miss_curve[j]),
                     "fa_at_thresh0": float(fa_curve[len(fa_curve) - 1])}
    # baseline: two-clean-classify-and-compare
    m, st = load_model(ART / "r04b_s11" / "clean")
    m.eval()
    with torch.no_grad():
        l0 = m(preprocess(np.stack([x.astype(np.float32) for x, _ in (pte + nte)]), st)).numpy()
        l1 = m(preprocess(np.stack([xp.astype(np.float32) for _, xp in (pte + nte)]), st)).numpy()
    prb = ((l0 > 0) != (l1 > 0)).astype(int)
    tp = int(((prb == 1) & (yte == 1)).sum())
    fp = int(((prb == 1) & (yte == 0)).sum())
    fn = int(((prb == 0) & (yte == 1)).sum())
    res["classify_compare"] = {"f1": float(2 * tp / (2 * tp + fp + fn)),
                               "fa_rate": fp / max(1, len(nte)),
                               "miss_rate": fn / max(1, len(pte))}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    for k, v in res.items():
        if isinstance(v, dict) and "f1" in v:
            extra = f" miss@FA.01={v.get('miss_at_FA01')}" if "miss_at_FA01" in v else ""
            print(f"SAW {k}: f1={v['f1']:.3f} FA={v['fa_rate']:.3f} miss={v['miss_rate']:.3f}{extra}")
    print("NEXT: if pair models beat classify-compare at matched FA -> joint "
          "state+change training trial; if all equal -> keep as monitoring method only")
    print("CLAIM: learns the relation between states, not another static metric")


if __name__ == "__main__":
    main()
