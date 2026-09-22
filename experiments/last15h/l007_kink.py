#!/usr/bin/env python3
"""L007-K: kink-density diagnostic (zero training).

Hypothesis: a ReLU coordinate MLP is piecewise-linear along a 1-D path;
kink (activation-pattern switch) density near t_star sets the resolution at
which the flip location can be placed. Sparser kinks near the oracle turn ->
larger |terr|. Spectral-bias literature motivates counting the kink budget,
not the slope.

Same paths, models, turn definitions and miss rule as L007_DIAGNOSTIC.
Dense grid n_scan=129; kinks = adjacent grid pairs with differing ReLU
patterns (both hidden layers, 128 units), recorded via forward hooks
(inference-side only, no training).

PRE-REGISTERED criterion: |Spearman| >= 0.3 with the same direction in 3/3
seeds for kink-distance (+) or kink-count (-) vs |terr| = positive;
both < 0.2 = dead (spectral-bias explanation closed).

NEW outputs only:
  artifacts/next_novelty/l007_kink/L007K_DIAG.csv
  artifacts/next_novelty/l007_kink/L007K_SUMMARY.json
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h"))
torch.set_num_threads(4)
from paths import (locate_model_turns, locate_oracle_turns,  # noqa: E402
                   scan_linear)
from common import load_model, preprocess  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
OUT = ROOT / "artifacts" / "next_novelty" / "l007_kink"


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return None
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return round(float((rx * ry).sum() / d), 4) if d > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_eval_paths", type=int, default=200)
    p.add_argument("--seed_eval", type=int, default=202)
    p.add_argument("--n_scan", type=int, default=129)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location(
        "n01b", str(ROOT / "experiments" / "last15h" / "n01_brackets.py"))
    n01b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(n01b)
    de = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    Xe = de["positions"].astype(float)
    ev_paths = n01b.aimed_single_turns(
        Xe, np.random.default_rng(a.seed_eval), a.n_eval_paths,
        seed=a.seed_eval)

    rows = []
    for seed in (11, 23, 47):
        mp = ART / ("r04b_s%d" % seed) / "flipmine"
        model, stats = load_model(mp)
        model.eval()
        acts = {}

        def hook(name):
            def fn(_m, _i, o):
                acts[name] = (np.asarray(o.detach()) > 0)
            return fn
        h1 = model.net[1].register_forward_hook(hook("r1"))
        h2 = model.net[3].register_forward_hook(hook("r2"))

        def predict(xs):
            with torch.no_grad():
                lg = model(preprocess(np.asarray(xs, np.float32),
                                      stats)).numpy()
            lg = np.asarray(lg, float).reshape(-1)
            return lg, (lg > 0).astype(int)

        for ep in ev_paths:
            x = Xe[ep["parent_id"]]
            e = np.asarray(ep["edit"], float)
            sc = scan_linear(x, e, predict, n_scan=a.n_scan)
            ot = locate_oracle_turns(x, e, sc)
            if len(ot) != 1:
                continue
            t_star = ot[0]["t_star"]
            t = sc["t"]
            # dense representation pass on the same grid
            xs = (x[None, :, :] + t[:, None, None] * e[None, :, :]
                  ).astype(np.float32)
            with torch.no_grad():
                model(preprocess(xs, stats))
            P = np.concatenate([acts["r1"], acts["r2"]], axis=1)
            switch = P[1:] != P[:-1]
            kink_here = switch.any(axis=1)
            tmid = 0.5 * (t[:-1] + t[1:])
            inwin = np.abs(tmid - t_star) <= a.delta
            kinks = tmid[inwin & kink_here]
            r = {"seed": seed, "parent": int(ep["parent_id"]),
                 "t_star": round(t_star, 6),
                 "kink_count": int(kink_here[inwin].sum()),
                 "kink_dist": round(float(np.abs(kinks - t_star).min())
                                    if len(kinks) else np.nan, 6),
                 "n_kinks_path": int(kink_here.sum())}
            mt = locate_model_turns(x, e, sc, predict)
            same = [m for m in mt if m["old"] == 0 and m["new"] == 1]
            if not same:
                r.update({"miss": 1, "abserr": np.nan,
                          "theta_on_kink": np.nan})
            else:
                best = min(same, key=lambda m: abs(m["t_theta"] - t_star))
                tt = best["t_theta"]
                step = float(t[1] - t[0])
                r.update({"miss": 0,
                          "abserr": round(abs(tt - t_star), 6),
                          "theta_on_kink": int(
                              bool((np.abs(tmid[kink_here] - tt)
                                    <= step / 2).any()))})
            rows.append(r)
        h1.remove()
        h2.remove()
        n = sum(1 for r in rows if r["seed"] == seed)
        print("s%d: n=%d" % (seed, n), flush=True)

    summary = {"seeds": {}, "criterion": (
        "|Spearman| >= 0.3 same direction in 3/3 seeds for kink-distance "
        "(+) or kink-count (-) vs |terr|")}
    for seed in (11, 23, 47):
        R = [r for r in rows if r["seed"] == seed and r["miss"] == 0]
        y = np.array([r["abserr"] for r in R])
        sp_d = spearman([r["kink_dist"] for r in R], y)
        sp_c = spearman([r["kink_count"] for r in R], y)
        hit = np.array([r["theta_on_kink"] for r in R])
        m = np.isfinite([r["kink_dist"] for r in R])
        d = {"n": len(R), "sp_kink_dist": sp_d, "sp_kink_count": sp_c,
             "theta_on_kink_rate": round(float(hit[m].mean()), 4)
             if m.sum() else None}
        # miss vs non-miss sparsity (informational)
        M = [r for r in rows if r["seed"] == seed and r["miss"] == 1]
        if M:
            d["miss_kink_count_mean"] = round(float(np.mean(
                [r["kink_count"] for r in M])), 4)
            d["hit_kink_count_mean"] = round(float(np.mean(
                [r["kink_count"] for r in R])), 4)
        summary["seeds"]["s%d" % seed] = d
        print("s%d" % seed, d, flush=True)
    pos = []
    for s in (11, 23, 47):
        d = summary["seeds"]["s%d" % s]
        pos.append((d["sp_kink_dist"] is not None and d["sp_kink_dist"] >= 0.3,
                    d["sp_kink_count"] is not None and d["sp_kink_count"] <= -0.3))
    summary["criterion_met"] = (all(p[0] for p in pos) or
                                all(p[1] for p in pos))
    summary["verdict"] = ("kink-resolution signal"
                          if summary["criterion_met"]
                          else "no kink signal (spectral-bias closed)")
    a.out.mkdir(parents=True, exist_ok=True)
    cols = ["seed", "parent", "t_star", "miss", "abserr", "kink_count",
            "kink_dist", "n_kinks_path", "theta_on_kink"]
    with open(a.out / "L007K_DIAG.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    json.dump(summary, open(a.out / "L007K_SUMMARY.json", "w"), indent=1,
              default=float)
    print("criterion_met:", summary["criterion_met"],
          "verdict:", summary["verdict"])
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
