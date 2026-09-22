#!/usr/bin/env python3
"""L007-G: grazing-angle x width interaction diagnostic (zero training).

Hypothesis: |terr| ~ w / a, where a = |g.hat e| / ||g|| is the cosine of the
path tangent with the boundary normal g = grad_x logit at the crossing, and
w is the transition width (|logit| -k -> +k in t). Same paths, models, turn
definitions and miss rule as L007_DIAGNOSTIC (E7 contract).

PRE-REGISTERED criterion (from literature-lane proposal): Spearman(w/a,
|terr|) >= 0.3 with the same sign in 3/3 seeds = positive; < 0.2 = dead.
Width-alone and angle-alone reported to decompose the interaction.

NEW outputs only:
  artifacts/next_novelty/l007_grazing/L007G_DIAG.csv
  artifacts/next_novelty/l007_grazing/L007G_SUMMARY.json
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
from common import load_model  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
OUT = ROOT / "artifacts" / "next_novelty" / "l007_grazing"


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


def crossing_width(t, lg, t_theta, k=1.0):
    """Delta-t from last logit<=-k before crossing to first logit>=+k after.
    Returns None if the band is never reached on either side."""
    t = np.asarray(t, float)
    lg = np.asarray(lg, float)
    pre = np.where((t <= t_theta) & (lg <= -k))[0]
    post = np.where((t >= t_theta) & (lg >= k))[0]
    if len(pre) == 0 or len(post) == 0:
        return None
    return round(float(t[post[0]] - t[pre[-1]]), 6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_eval_paths", type=int, default=200)
    p.add_argument("--seed_eval", type=int, default=202)
    p.add_argument("--n_scan", type=int, default=129)
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
        mu = torch.from_numpy(np.asarray(stats["mu"], dtype=np.float32))
        sd = torch.from_numpy(np.asarray(stats["sd"], dtype=np.float32))

        def predict(xs):
            with torch.no_grad():
                lg = model(preprocess_t(torch.as_tensor(
                    np.asarray(xs, np.float32)), mu, sd)).numpy()
            lg = np.asarray(lg, float).reshape(-1)
            return lg, (lg > 0).astype(int)

        for ep in ev_paths:
            x = Xe[ep["parent_id"]]
            e = np.asarray(ep["edit"], float)
            en = float(np.linalg.norm(e))
            ehat = (e / en).reshape(-1) if en > 0 else None
            sc = scan_linear(x, e, predict, n_scan=a.n_scan)
            ot = locate_oracle_turns(x, e, sc)
            if len(ot) != 1:
                continue
            t_star = ot[0]["t_star"]
            t = sc["t"]
            lg = sc["logit"]
            mt = locate_model_turns(x, e, sc, predict)
            same = [m for m in mt if m["old"] == 0 and m["new"] == 1]
            r = {"seed": seed, "parent": int(ep["parent_id"]),
                 "t_star": round(t_star, 6)}
            if not same:
                r.update({"miss": 1, "abserr": np.nan, "t_theta": np.nan,
                          "angle_cos": np.nan, "width_k1": np.nan,
                          "width_k2": np.nan, "woa": np.nan})
                rows.append(r)
                continue
            best = min(same, key=lambda m: abs(m["t_theta"] - t_star))
            t_theta = best["t_theta"]
            # boundary normal at the crossing (autograd through manual scale)
            xr = torch.from_numpy(
                (np.asarray(x, dtype=np.float32) +
                 np.float32(t_theta) * np.asarray(e, dtype=np.float32)
                 ).reshape(-1)).requires_grad_(True)
            with torch.enable_grad():
                logit = model(((xr - mu) / sd).unsqueeze(0)).squeeze(0)
                g = torch.autograd.grad(logit, xr)[0].detach().numpy()
            gn = float(np.linalg.norm(g))
            acos = (round(float(abs(g.reshape(-1) @ ehat) / gn), 6)
                    if gn > 1e-12 and ehat is not None else np.nan)
            w1 = crossing_width(t, lg, t_theta, k=1.0)
            w2 = crossing_width(t, lg, t_theta, k=2.0)
            w = w1 if w1 is not None else w2
            r.update({"miss": 0,
                      "abserr": round(abs(t_theta - t_star), 6),
                      "t_theta": round(t_theta, 6),
                      "angle_cos": acos,
                      "width_k1": w1 if w1 is not None else np.nan,
                      "width_k2": w2 if w2 is not None else np.nan,
                      "woa": round(float(w / acos), 6)
                      if w is not None and acos and acos > 1e-9 else np.nan})
            rows.append(r)
        n = sum(1 for r in rows if r["seed"] == seed)
        print("s%d: n=%d" % (seed, n), flush=True)

    summary = {"seeds": {}, "criterion": (
        "Spearman(w/a, |terr|) >= 0.3 same sign in 3/3 seeds")}
    for seed in (11, 23, 47):
        R = [r for r in rows if r["seed"] == seed and r["miss"] == 0]
        y = np.array([r["abserr"] for r in R])
        d = {"n": len(R),
             "sp_woa": spearman([r["woa"] for r in R], y),
             "sp_width": spearman(
                 [r["width_k1"] if np.isfinite(r["width_k1"])
                  else r["width_k2"] for r in R], y),
             "sp_angle": spearman([r["angle_cos"] for r in R], y)}
        summary["seeds"]["s%d" % seed] = d
        print("s%d" % seed, d, flush=True)
    sp = [summary["seeds"]["s%d" % s]["sp_woa"] for s in (11, 23, 47)]
    summary["criterion_met"] = (
        all(v is not None and v >= 0.3 for v in sp))
    summary["verdict"] = ("grazing-amplifier signal"
                          if summary["criterion_met"]
                          else "no interaction (dead)")

    a.out.mkdir(parents=True, exist_ok=True)
    cols = ["seed", "parent", "t_star", "t_theta", "miss", "abserr",
            "angle_cos", "width_k1", "width_k2", "woa"]
    with open(a.out / "L007G_DIAG.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    json.dump(summary, open(a.out / "L007G_SUMMARY.json", "w"), indent=1,
              default=float)
    print("criterion_met:", summary["criterion_met"],
          "verdict:", summary["verdict"])
    print("DONE ->", a.out)


def preprocess_t(xx, mu, sd):
    return (xx.reshape(xx.shape[0], -1) - mu) / sd


if __name__ == "__main__":
    main()
