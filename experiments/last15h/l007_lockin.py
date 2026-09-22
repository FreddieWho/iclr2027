#!/usr/bin/env python3
"""L007-P1: checkpoint lock-in dynamics (uses l007_wave checkpoints).

Question: is per-path |terr| ranking established early (critical period)
or drifting through training?

Per seed (2001-2008) x ckpt (000..300 step 25): |terr|/miss on the same
200 eval paths (E7 contract, n_scan=17). Metrics:
  (a) within-seed Spearman(|terr|(e), |terr|(final)) vs epoch (path-rank
      stability); mean pairwise cross-seed rank agreement vs epoch.
  (b) lock-in epoch: first e with |terr| inside final +/- tol band and
      never leaving (tol = max(0.03, 20% of final)); miss excluded.
  (c) loss-convergence epoch control: train BCE within 5% of final.

PRE-REGISTERED criterion: median lock-in epoch < 50% of median
loss-convergence epoch in >=6/8 seeds = early-lock positive.
Else drift/late = critical-period line closed.

NEW outputs only: artifacts/next_novelty/l007_lockin/
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h"))
torch.set_num_threads(4)
from paths import (locate_model_turns, locate_oracle_turns,  # noqa: E402
                   scan_linear)
from coord_mlp import CoordMLP  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
WAVE = ROOT / "artifacts" / "next_novelty" / "l007_wave"
OUT = ROOT / "artifacts" / "next_novelty" / "l007_lockin"
SEEDS = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]
CKPTS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]


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
    return round(float((rx * ry).sum() / d), 4) if d > 0 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_eval_paths", type=int, default=200)
    p.add_argument("--seed_eval", type=int, default=202)
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
    tr = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    Xtr = tr["positions"].astype(np.float32)
    ytr = tr["labels"].astype(np.float32)
    mu8 = Xtr.reshape(-1, 8).mean(0)
    sd8 = Xtr.reshape(-1, 8).std(0) + 1e-8
    Xtr_t = torch.from_numpy(((Xtr.reshape(-1, 8) - mu8) / sd8).astype(np.float32))
    ytr_t = torch.from_numpy(ytr)
    bce = nn.BCEWithLogitsLoss()

    def load_ckpt(seed, e):
        ck = torch.load(WAVE / ("s%d" % seed) / ("ckpt_%03d.pt" % e),
                        map_location="cpu", weights_only=False)
        m = CoordMLP(64, 32, in_dim=8)
        m.load_state_dict(ck["state"])
        m.eval()
        return m

    def path_terrs(model):
        def predict(xs):
            with torch.no_grad():
                lg = model(((torch.as_tensor(np.asarray(xs, np.float32))
                             .reshape(len(xs), -1) - torch.from_numpy(mu8)) /
                            torch.from_numpy(sd8))).numpy()
            lg = np.asarray(lg, float).reshape(-1)
            return lg, (lg > 0).astype(int)

        out = []
        for ep in ev_paths:
            x = Xe[ep["parent_id"]]
            e = np.asarray(ep["edit"], float)
            sc = scan_linear(x, e, predict, n_scan=17)
            ot = locate_oracle_turns(x, e, sc)
            if len(ot) != 1:
                out.append(np.nan)
                continue
            mt = locate_model_turns(x, e, sc, predict)
            same = [m for m in mt if m["old"] == 0 and m["new"] == 1]
            if not same:
                out.append(np.nan)
                continue
            best = min(same, key=lambda m: abs(m["t_theta"] - ot[0]["t_star"]))
            out.append(abs(best["t_theta"] - ot[0]["t_star"]))
        return np.array(out)

    rows = []
    grid = {}
    for seed in SEEDS:
        terr = {}
        for e in CKPTS:
            terr[e] = path_terrs(load_ckpt(seed, e))
        grid[seed] = terr
        final = terr[300]
        losses = {}
        for e in CKPTS:
            with torch.no_grad():
                lgv = load_ckpt(seed, e)(Xtr_t).numpy().reshape(-1)
            lgv = np.clip(lgv, -30, 30)
            pr = 1 / (1 + np.exp(-lgv))
            losses[e] = float(-(ytr * np.log(pr + 1e-9) + (1 - ytr) *
                                np.log(1 - pr + 1e-9)).mean())
        fin_loss = losses[300]
        conv = next((e for e in CKPTS
                     if abs(losses[e] - fin_loss) <= 0.05 * abs(fin_loss)
                     and all(abs(losses[u] - fin_loss) <= 0.05 * abs(fin_loss)
                             for u in CKPTS if u >= e)), 300)
        tol = max(0.03, 0.2 * float(np.nanmean(final)))
        lock = next((e for e in CKPTS
                     if np.isfinite(final).any() and all(
                         (not np.isfinite(terr[u][i])) or
                         abs(terr[u][i] - final[i]) <= tol
                         for u in CKPTS if u >= e
                         for i in range(len(final)) if np.isfinite(final[i]))
                     ), None)
        st = [{"epoch": e,
               "spear_vs_final": spearman(terr[e], final),
               "miss_rate": round(float(np.isnan(terr[e]).mean()), 4),
               "loss": round(losses[e], 4)} for e in CKPTS]
        rows.append({"seed": seed, "lock_epoch": lock,
                     "conv_epoch": conv, "curve": st})
        print("s%d: lock=%s conv=%s" % (seed, lock, conv), flush=True)

    # cross-seed rank agreement vs epoch (pairwise Spearman of path vectors)
    agree = {}
    for e in CKPTS:
        sps = []
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                v = spearman(grid[SEEDS[i]][e], grid[SEEDS[j]][e])
                if v is not None:
                    sps.append(v)
        agree[e] = round(float(np.mean(sps)), 4) if sps else None
    summary = {"seeds": {r["seed"]: {"lock_epoch": r["lock_epoch"],
                                     "conv_epoch": r["conv_epoch"],
                                     "curve": r["curve"]} for r in rows},
               "cross_seed_agreement": agree,
               "criterion": ("median lock < 50% median conv in >=6/8 seeds")}
    locks = [r["lock_epoch"] for r in rows if r["lock_epoch"] is not None]
    convs = [r["conv_epoch"] for r in rows]
    summary["median_lock"] = float(np.median(locks)) if locks else None
    summary["median_conv"] = float(np.median(convs))
    summary["criterion_met"] = (
        len(locks) >= 6 and summary["median_lock"] is not None and
        summary["median_lock"] < 0.5 * summary["median_conv"])
    summary["verdict"] = ("early-lock positive" if summary["criterion_met"]
                          else "drift/late (critical-period closed)")
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(a.out / "L007_LOCKIN.json", "w"), indent=1,
              default=float)
    print("criterion_met:", summary["criterion_met"],
          "verdict:", summary["verdict"])
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
