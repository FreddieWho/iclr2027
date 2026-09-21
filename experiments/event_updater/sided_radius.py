#!/usr/bin/env python3
"""Side D: semantic radius diagnostic (frozen clean, eval only).

r_s(x) = min ||d|| s.t. oracle flips, over K random directions (bisect).
r_m(x) = min ||d|| s.t. model logit flips sign, same directions.
R = r_m / r_s. Question: do transition-miss cases have stably larger R?
3 mining seeds. No stable separation -> RADIUS_KILL.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(4)
from paths import oracle_at  # noqa: E402
from common import load_model, preprocess  # noqa: E402
from r02_search import candidates_for_scene  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
OUT = ROOT / "artifacts" / "event_updater"


def bisect_radius(x, y0, d, is_oracle, model=None, stats=None, rmax=1.0):
    try:
        if is_oracle:
            y1, _, _ = oracle_at(x + rmax * d)
        else:
            with torch.no_grad():
                lg = model(preprocess((x + rmax * d).astype(np.float32)[None],
                                      stats)).numpy()[0]
            y1 = int(lg > 0)
    except ValueError:
        return None
    if y1 == y0:
        return None
    lo, hi = 0.0, rmax
    for _ in range(20):
        m = 0.5 * (lo + hi)
        try:
            if is_oracle:
                ym, _, _ = oracle_at(x + m * d)
            else:
                with torch.no_grad():
                    lg = model(preprocess((x + m * d).astype(np.float32)[None],
                                          stats)).numpy()[0]
                ym = int(lg > 0)
        except ValueError:
            hi = m
            continue
        if ym == y0:
            lo = m
        else:
            hi = m
    return hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n_parents", type=int, default=200)
    p.add_argument("--k_dirs", type=int, default=24)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    d = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    X = d["positions"].astype(np.float32)
    model, stats = load_model(ART / "r04b_s11" / "clean")
    model.eval()

    def mlogit(x):
        with torch.no_grad():
            return float(model(preprocess(x.astype(np.float32)[None],
                                          stats)).numpy()[0])

    R_hit, R_miss, Rs_hit, Rs_miss, Rm_hit, Rm_miss = [], [], [], [], [], []
    for pi in rng.permutation(len(X))[:a.n_parents]:
        x = X[pi].astype(float)
        try:
            y0, _, _ = oracle_at(x)
        except ValueError:
            continue
        dirs = []
        for item in candidates_for_scene(x, rng, n_random=8):
            e = np.asarray(item[0], dtype=float)
            n = np.linalg.norm(e)
            if n > 1e-9:
                dirs.append(e / n)
            if len(dirs) >= a.k_dirs:
                break
        rs = [bisect_radius(x, y0, dd, True) for dd in dirs]
        rs = [r for r in rs if r is not None]
        if not rs:
            continue
        r_s = min(rs)
        rm = [bisect_radius(x, int(mlogit(x) > 0), dd, False, model, stats)
              for dd in dirs]
        rm = [r for r in rm if r is not None]
        if not rm:
            continue
        R = min(rm) / r_s
        # transition probe: first valid flip from x
        frng = np.random.default_rng(a.seed)
        e1, y1 = None, None
        for item in candidates_for_scene(x, frng, n_random=32):
            e = np.asarray(item[0], dtype=float)
            try:
                ya, _, _ = oracle_at(x)
                yb, mb, _ = oracle_at(x + e)
            except ValueError:
                continue
            if yb != ya and mb >= 0.03:
                e1, y1 = e, yb
                break
        if e1 is None:
            continue
        miss = int((mlogit(x + e1) > 0) != y1)
        (R_miss if miss else R_hit).append(R)
        (Rs_miss if miss else Rs_hit).append(r_s)
        (Rm_miss if miss else Rm_hit).append(min(rm))
    R_hit, R_miss = np.array(R_hit), np.array(R_miss)
    def _q(v):
        v = np.asarray(v, dtype=float)
        return [round(float(x), 3) for x in np.quantile(v, [0.25, 0.5, 0.75])] if len(v) else None
    res = {"seed": a.seed, "n_hit": len(R_hit), "n_miss": len(R_miss),
           "R_hit_q": _q(R_hit), "R_miss_q": _q(R_miss),
           "rs_hit_q": _q(Rs_hit), "rs_miss_q": _q(Rs_miss),
           "rm_hit_q": _q(Rm_hit), "rm_miss_q": _q(Rm_miss)}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"sided_s{a.seed}.json", "w"), indent=1)
    print(f"SAW sideD s{a.seed}: {json.dumps(res)}")


if __name__ == "__main__":
    main()
