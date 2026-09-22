#!/usr/bin/env python3
"""L007-P1 v2: per-path lock-in dynamics under the corrected contract.

Fixes vs l007_lockin.py (see GPT6ASTRA_CODE_REVIEW.md §6):
  - tolerance is PER PATH: tol_i = max(0.03, 0.2 * final_i) (was per seed);
  - per-path lock epoch computed first, then the seed-level median
    (was: one global "all paths simultaneously stable" time);
  - a miss at any later epoch counts as LEAVING the band (was: pass);
  - convergence reference uses the FULL training objective
    (mean clean BCE + lam * mean flip BCE), not clean only.

PRE-REGISTERED criterion (unchanged in spirit): seed-level median lock
epoch < 50% of that seed's full-objective convergence epoch, in >= 6/8
seeds = early-lock positive; else drift/late.

NEW outputs only: artifacts/next_novelty/l007_lockin_v2/
"""
import argparse
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
from r04b_methods import mine_flips  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
WAVE = ROOT / "artifacts" / "next_novelty" / "l007_wave"
OUT = ROOT / "artifacts" / "next_novelty" / "l007_lockin_v2"
SEEDS = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]
CKPTS = list(range(0, 301, 25))


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
    X = tr["positions"].astype(np.float32)
    y = tr["labels"].astype(np.float32)
    mu8 = X.reshape(-1, 8).mean(0)
    sd8 = X.reshape(-1, 8).std(0) + 1e-8
    mined = mine_flips(X.astype(float), y.astype(int), seed=5)
    ii = np.array([m["scene"] for m in mined])
    ee = np.array([m["edit"] for m in mined])
    yf = np.array([m["new"] for m in mined], dtype=np.float32)
    Xc_t = torch.from_numpy(((X.reshape(-1, 8) - mu8) / sd8).astype(np.float32))
    yc_t = torch.from_numpy(y)
    Xf_t = torch.from_numpy((((X[ii] + ee).reshape(-1, 8) - mu8) / sd8).astype(np.float32))
    yf_t = torch.from_numpy(yf)
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

    summary = {"seeds": {}, "criterion": (
        "seed median per-path lock < 50% of seed full-objective convergence "
        "epoch, in >=6/8 seeds")}
    n_early = 0
    for seed in SEEDS:
        terr, full_loss = {}, {}
        for e in CKPTS:
            m = load_ckpt(seed, e)
            terr[e] = path_terrs(m)
            with torch.no_grad():
                l_full = float((bce(m(Xc_t), yc_t).mean()
                                + bce(m(Xf_t), yf_t).mean()).item())
            full_loss[e] = l_full
        fin = terr[300]
        fin_loss = full_loss[300]
        conv = next((e for e in CKPTS
                     if all(full_loss[u] <= 1.05 * fin_loss
                            for u in CKPTS if u >= e)), 300)
        idx = [i for i in range(len(fin)) if np.isfinite(fin[i])]
        locks = []
        for i in idx:
            tol = max(0.03, 0.2 * fin[i])
            lk = None
            for e in CKPTS:
                ok = True
                for u in CKPTS:
                    if u < e:
                        continue
                    v = terr[u][i]
                    if not np.isfinite(v) or abs(v - fin[i]) > tol:
                        ok = False
                        break
                if ok:
                    lk = e
                    break
            if lk is not None:
                locks.append(lk)
        med = float(np.median(locks)) if locks else None
        early = (med is not None and med < 0.5 * conv)
        n_early += bool(early)
        summary["seeds"]["s%d" % seed] = {
            "n_hit_final": len(idx), "lock_median_per_path": med,
            "lock_frac_resolved": round(len(locks) / len(idx), 4) if idx else None,
            "conv_epoch_full_objective": conv,
            "final_full_loss": round(fin_loss, 6),
            "early": bool(early)}
        print("s%d lock_med=%s conv=%s early=%s" % (seed, med, conv, early),
              flush=True)
    summary["n_early_seeds"] = n_early
    summary["criterion_met"] = n_early >= 6
    summary["verdict"] = ("early-lock positive" if summary["criterion_met"]
                          else "drift/late")
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(a.out / "L007_LOCKIN_V2.json", "w"), indent=1,
              default=float)
    print("criterion_met:", summary["criterion_met"],
          "verdict:", summary["verdict"])
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
