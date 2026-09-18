#!/usr/bin/env python3
"""U01 round1: fixed quartet curriculum (N04 reopen, sd-shadowing bug fixed).

Fix vs n04_curriculum.py: feature_std (array) and seed_id (int) are distinct
names; train() binds the SAME stats object used at eval; one assert checks
train/eval normalization identity.

Quartets (x, a, b, y0, ya, yb, yc) mined ONCE on train_101, saved, shared.
Arms: clean_only | combo_endpoint | quartet(4 states) | random(same count).
Same init state_dict copy per seed, same 300ep full-batch Adam(0.01) recipe.
Eval: FIXED unseen-parent emergent list (recheck pools) + static + atomics.
Reports per-arm cond AND common-intersection cond (paired comparison).
"""
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


def mine_quartets(X, rng):
    """Emergent quartets: both atomics preserve, combo flips (margin>=0.005)."""
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
                    out.append((x, ea, eb, y0, ya, yb, yc))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    a = p.parse_args()
    rng = np.random.default_rng(44)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(float)
    y = d["labels"].astype(np.float32)
    mu = X.reshape(-1, 8).mean(0)
    feature_std = X.reshape(-1, 8).std(0) + 1e-8
    stats = {"mu": mu.astype(np.float32), "sd": feature_std.astype(np.float32)}

    Qtr = mine_quartets(X, rng)
    Etr = [(x, ea + eb, yc) for x, ea, eb, y0, ya, yb, yc in Qtr]
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
    # fixed unseen-parent eval quartets
    Qev = []
    for pool in ("recheck_404", "recheck_505", "recheck_606"):
        de = np.load(ART / "scenes" / pool / "scenes.npz")
        Qev += mine_quartets(de["positions"].astype(float), rng)
    # persist shared quartets
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / "quartets_train.npz",
             **{f"q{i}": np.stack([np.asarray(t, dtype=float).ravel()
                                   for t in (x, ea, eb)]) for i, (x, ea, eb, *_)
                in enumerate(Qtr)},
             meta=np.array([[y0, ya, yb, yc] for *_, y0, ya, yb, yc in Qtr]))
    np.savez(a.out / "quartets_eval.npz",
             **{f"q{i}": np.stack([np.asarray(t, dtype=float).ravel()
                                   for t in (x, ea, eb)]) for i, (x, ea, eb, *_)
                in enumerate(Qev)},
             meta=np.array([[y0, ya, yb, yc] for *_, y0, ya, yb, yc in Qev]))

    def encode(rows):
        xx = np.stack([np.asarray(r[0], dtype=float) for r in rows])
        return torch.from_numpy(
            ((xx.reshape(-1, 8) - mu) / feature_std).astype(np.float32))

    Xc_t = encode([(x,) for x in X])
    yc_t = torch.from_numpy(y)

    def train(flips, seed_id, init_state):
        torch.manual_seed(seed_id)
        m = CoordMLP(64, 32)
        m.load_state_dict({k: v.clone() for k, v in init_state.items()})
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        bce = nn.BCEWithLogitsLoss()
        Xf = yf = None
        if flips:
            xx = np.stack([np.asarray(f[0], dtype=float) for f in flips])
            ee = np.stack([np.asarray(f[1], dtype=float) for f in flips])
            Xf = torch.from_numpy(
                (((xx + ee).reshape(-1, 8) - mu) / feature_std).astype(np.float32))
            yf = torch.from_numpy(np.array([f[2] for f in flips], dtype=np.float32))
        # THE assert: train-time normalization reproduces eval-time stats
        assert np.allclose(stats["mu"], mu.astype(np.float32)) and \
            np.allclose(stats["sd"], feature_std.astype(np.float32)), \
            "train/eval standardization mismatch"
        m.train()
        for _ in range(300):
            opt.zero_grad()
            loss = bce(m(Xc_t), yc_t)
            if Xf is not None:
                loss = loss + bce(m(Xf), yf)
            loss.backward()
            opt.step()
        m.eval()
        return m

    arms = {
        "clean_only": None,
        "combo_endpoint": Etr,
        "random": Rnd,
        "quartet": ([(x, np.zeros((4, 2)), y0) for x, _, _, y0, *_ in Qtr]
                    + [(x, ea, ya) for x, ea, _, _, ya, *_ in Qtr]
                    + [(x, eb, yb) for x, _, eb, _, _, yb, _ in Qtr]
                    + Etr),
    }
    arm_counts = {k: (0 if v is None else len(v)) for k, v in arms.items()}
    # eval frames: combo endpoints + atomics per quartet
    ev_end = np.stack([(x + ea + eb).astype(np.float32) for x, ea, eb, *_ in Qev])
    ev_end_y = np.array([yc for *_, yc in Qev])
    ev_a = np.stack([(x + ea).astype(np.float32) for x, ea, *_ in Qev])
    ev_a_y = np.array([ya for *_, ya, _, _ in Qev])
    ev_b = np.stack([(x + eb).astype(np.float32) for x, _, eb, _, _, yb, _ in Qev])
    ev_b_y = np.array([yb for *_, _, yb, _ in Qev])

    res = {"n_train_quartets": len(Qtr), "n_eval_quartets": len(Qev),
           "arm_sample_counts": arm_counts}
    for seed_id in a.seeds:
        torch.manual_seed(seed_id)
        init_state = CoordMLP(64, 32).state_dict()
        preds = {}
        for name, fl in arms.items():
            m = train(fl, seed_id, init_state)
            with torch.no_grad():
                pe = (m(preprocess(ev_end, stats)).numpy() > 0).astype(int)
                pa = (m(preprocess(ev_a, stats)).numpy() > 0).astype(int)
                pb = (m(preprocess(ev_b, stats)).numpy() > 0).astype(int)
                ps = (m(preprocess(X.astype(np.float32), stats)).numpy() > 0).astype(int)
            preds[name] = (pe, pa, pb)
            res[f"{name}_s{seed_id}"] = {
                "emergent_miss": float((pe != ev_end_y).mean()),
                "atomic_acc": float((((pa == ev_a_y) & (pb == ev_b_y))).mean()),
                "static_err": float((ps != y.astype(int)).mean())}
        # common-intersection paired comparison: quartets where ALL arms'
        # atomics are correct -> combo miss per arm on that shared set
        common = np.ones(len(Qev), dtype=bool)
        for pe, pa, pb in preds.values():
            common &= (pa == ev_a_y) & (pb == ev_b_y)
        res[f"common_n_s{seed_id}"] = int(common.sum())
        for name, (pe, pa, pb) in preds.items():
            res[f"{name}_s{seed_id}"]["common_cond_miss"] = (
                float((pe[common] != ev_end_y[common]).mean()) if common.sum() else None)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    for seed_id in a.seeds:
        row = "; ".join(
            f"{k.replace(f'_s{seed_id}', '')}={res[k]['emergent_miss']:.3f}"
            for k in res if k.endswith(f"_s{seed_id}") and isinstance(res[k], dict))
        print(f"SAW seed{seed_id}: Qtr={len(Qtr)} Qev={len(Qev)} "
              f"common_n={res[f'common_n_s{seed_id}']} " + row, flush=True)
    print("SAMPLES:", arm_counts, flush=True)
    print("DONE U01")


if __name__ == "__main__":
    main()
