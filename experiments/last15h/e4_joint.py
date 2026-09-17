#!/usr/bin/env python3
"""E4: N10-joint — shared encoder, state head + change head, multi-task.

Arms (same recipe: full-batch Adam 0.01, 300ep, seeds 11/23/47 from start):
  joint, state-only, pair-only, state+shuffled-pair (compute/label control).
Key test: does change supervision improve SINGLE-frame turn behavior
(N01 single-turn turn_miss/integral)? Plus change-F1/FA and clean acc.
Pairs rebuilt exactly as n10_change round1b (same rng/pool/split).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h"))
torch.set_num_threads(2)
from paths import oracle_at, scan_linear, locate_oracle_turns, locate_model_turns
from common import preprocess
from coord_mlp import CoordMLP
from r02_search import candidates_for_scene
from n01_brackets import aimed_single_turns

ART = ROOT / "artifacts" / "discovery_campaign"


class JointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 16))
        self.state = nn.Linear(16, 1)
        self.change = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))

    def state_logit(self, x):
        return self.state(self.enc(x)).squeeze(-1)

    def change_logit(self, x0, x1):
        z0, z1 = self.enc(x0), self.enc(x1)
        return self.change(torch.cat([z0, z1, z1 - z0], 1)).squeeze(-1)


def build_pairs(X, flips, rng, n_neg):
    pos = [(x, x + e) for (x, e) in flips]
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
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    p.add_argument("--epochs", type=int, default=300)
    a = p.parse_args()
    rng = np.random.default_rng(10)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(float)
    y = d["labels"].astype(np.float32)
    mu = X.reshape(-1, 8).mean(0)
    sd = X.reshape(-1, 8).std(0) + 1e-8
    st = {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}
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

    def std(A):
        return ((np.asarray(A, dtype=np.float32).reshape(len(A), -1) - mu) / sd)

    Xc = torch.from_numpy(std(X).astype(np.float32))
    yc = torch.from_numpy(y)
    Ap = np.stack([np.concatenate([x.reshape(-1), xp.reshape(-1)]) for x, xp in pos])
    An = np.stack([np.concatenate([x.reshape(-1), xp.reshape(-1)]) for x, xp in neg])
    bce = nn.BCEWithLogitsLoss()

    # eval assets (built once): N01-style single-turn paths on eval scenes
    de = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    Xe = de["positions"].astype(float)
    ev_paths = aimed_single_turns(Xe, np.random.default_rng(64), 200, seed=202)
    # change-test split (same 80/20 rule as n10)
    def split(pairs, frac=0.8):
        k = int(len(pairs) * frac)
        return pairs[:k], pairs[k:]
    ptr, pte = split(pos)
    ntr, nte = split(neg)
    yte = np.array([1] * len(pte) + [0] * len(nte))

    res = {"n_pos": len(pos), "n_neg": len(neg), "n_eval_paths": len(ev_paths)}
    for sd_ in a.seeds:
        for arm in ("joint", "state_only", "pair_only", "state_shuffled"):
            torch.manual_seed(sd_)
            m = JointNet()
            opt = torch.optim.Adam(m.parameters(), lr=0.01)
            Xp0 = torch.from_numpy(std([x for x, _ in pos]).astype(np.float32))
            Xp1 = torch.from_numpy(std([xp for _, xp in pos]).astype(np.float32))
            Xn0 = torch.from_numpy(std([x for x, _ in neg]).astype(np.float32))
            Xn1 = torch.from_numpy(std([xp for _, xp in neg]).astype(np.float32))
            yp = torch.ones(len(pos))
            yn = torch.zeros(len(neg))
            yall = torch.cat([yp, yn])[torch.randperm(len(pos) + len(neg))]
            ypsh, ynsh = yall[:len(pos)], yall[len(pos):]
            m.train()
            for _ in range(a.epochs):
                opt.zero_grad()
                loss = 0.0
                if arm in ("joint", "state_only", "state_shuffled"):
                    loss = loss + bce(m.state_logit(Xc), yc)
                if arm in ("joint", "pair_only"):
                    loss = loss + (bce(m.change_logit(Xp0, Xp1), yp) +
                                   bce(m.change_logit(Xn0, Xn1), yn)) / 2
                if arm == "state_shuffled":
                    loss = loss + (bce(m.change_logit(Xp0, Xp1), ypsh) +
                                   bce(m.change_logit(Xn0, Xn1), ynsh)) / 2
                loss.backward()
                opt.step()
            m.eval()
            # (i) single-frame turn behavior
            def predict(xs):
                with torch.no_grad():
                    lg = m.state_logit(preprocess(xs, st)).numpy()
                return lg, (lg > 0).astype(int)
            errs, miss, integ = [], 0, []
            for ep in ev_paths:
                x = Xe[ep["parent_id"]]
                sc = scan_linear(x, ep["edit"], predict, n_scan=17)
                ot = locate_oracle_turns(x, ep["edit"], sc)
                if len(ot) != 1:
                    continue
                integ.append(float((sc["pred_label"] != sc["oracle_label"]).mean()))
                mt = locate_model_turns(x, ep["edit"], sc, predict)
                same = [u for u in mt if u["old"] == 0 and u["new"] == 1]
                if not same:
                    miss += 1
                    errs.append(None)
                else:
                    errs.append(abs(same[0]["t_theta"] - ot[0]["t_star"]))
            got = [e for e in errs if e is not None]
            # (ii) change detection @FA<=0.01
            with torch.no_grad():
                te0 = torch.from_numpy(std([x for x, _ in (pte + nte)]).astype(np.float32))
                te1 = torch.from_numpy(std([xp for _, xp in (pte + nte)]).astype(np.float32))
                lg = m.change_logit(te0, te1).numpy()
            order = np.argsort(-lg)
            yo = yte[order]
            nn_ = max(1, int((yte == 0).sum()))
            np_ = max(1, int((yte == 1).sum()))
            fa_c = np.cumsum(yo == 0) / nn_
            miss_c = 1 - np.cumsum(yo == 1) / np_
            j = max(0, min(int(np.searchsorted(fa_c, 0.01, side="right")) - 1, len(fa_c) - 1))
            # (iii) clean acc on EVAL scenes (not train)
            de_ = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
            Xev_ = ((de_["positions"].astype(np.float32).reshape(-1, 8) - mu) / sd)
            yev_ = de_["labels"].astype(int)
            with torch.no_grad():
                lc = m.state_logit(torch.from_numpy(Xev_.astype(np.float32))).numpy()
            res[f"{arm}_s{sd_}"] = {
                "turn_miss": miss / len(errs) if errs else None,
                "mean_integral": float(np.mean(integ)) if integ else None,
                "change_miss_at_FA01": float(miss_c[j]),
                "clean_acc_eval": float((((lc > 0).astype(int)) == yev_).mean())}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    for sd_ in a.seeds:
        for arm in ("joint", "state_only", "pair_only", "state_shuffled"):
            v = res.get(f"{arm}_s{sd_}", {})
            print(f"SAW {arm} s{sd_}: turn_miss={v.get('turn_miss')} "
                  f"integ={v.get('mean_integral')} "
                  f"chg_miss@FA01={v.get('change_miss_at_FA01')} "
                  f"clean_eval={v.get('clean_acc_eval')}", flush=True)
    print("NEXT: joint beats state_only on turn_miss/integral, multi-seed same "
          "direction -> structural method lives; else dead")
    print("CLAIM: shared state+change encoder (or not)")


if __name__ == "__main__":
    main()
