#!/usr/bin/env python3
"""L007-D: zero-training turn-precision diagnostic (representation-side).

Question: does representation-space structure at the turn predict position
error |terr|, beyond what input geometry already predicts?

Frozen only: r04b_s{11,23,47}/flipmine via common.load_model (no training).
Paths: eval_202 aimed single turns, n=200, seed 202 (same as E7 evaluate).
Turn/scan contract: shared paths.py helpers (scan_linear n_scan=65,
bisect localization, E7's 0->1 match rule). Exactly-1-oracle-turn filter
kept from E7.

Per path, three feature families:
  ORACLE (input geometry): t_star, min oracle |margin| in +-0.1 window,
      oracle margin slope at turn, edit cost.
  SCORE (logit trajectory): max |dlogit/dt|, max |curvature| near turn,
      transition width (frac grid |logit|<1), slope asymmetry L/R,
      min |logit| near t_star.
  REPR (hidden trajectory z, return_feat): speed concentration near turn,
      max segment-angle near turn, normalized hyperplane distance at turn,
      total z path length.

Analysis: Spearman vs |terr| per feature per seed; family CV-R^2
(ridge, grouped-by-parent 5-fold) predicting |terr|; miss analyzed
separately (logistic AUC, informational).

PRE-REGISTERED criterion: REPR family CV-R^2 must exceed ORACLE family by
>=0.05 in >=2/3 seeds to claim a representation-side signal. Otherwise the
verdict is input-geometry-determined (clean boundary result, still
informative: it tells the next hypothesis where NOT to look).

NEW outputs only:
  artifacts/next_novelty/l007/L007_DIAG.csv
  artifacts/next_novelty/l007/L007_SUMMARY.json
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
OUT = ROOT / "artifacts" / "next_novelty" / "l007"

ORACLE_FEATS = ["t_star", "min_margin_win", "margin_slope", "cost"]
SCORE_FEATS = ["max_slope", "max_curv", "trans_width", "slope_asym",
               "min_abs_logit"]
REPR_FEATS = ["speed_conc", "max_angle", "hyper_dist", "z_length"]


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


def ridge_cv_r2(Xf, y, groups, lam=1.0, k=5):
    """Grouped k-fold CV R^2, features standardized in-fold. Returns None if
    degenerate."""
    Xf = np.asarray(Xf, float)
    y = np.asarray(y, float)
    ug = np.unique(groups)
    if len(ug) < k or len(y) < 10:
        return None
    folds = np.arange(len(ug)) % k
    pred = np.full_like(y, np.nan)
    for f in range(k):
        te_g = set(ug[folds == f])
        te = np.array([g in te_g for g in groups])
        tr = ~te
        if tr.sum() < Xf.shape[1] + 2 or te.sum() == 0:
            continue
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-8
        A = np.column_stack([(Xf[tr] - mu) / sd, np.ones(tr.sum())])
        B = np.column_stack([(Xf[te] - mu) / sd, np.ones(te.sum())])
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = B @ w
    m = np.isfinite(pred)
    if m.sum() < 10:
        return None
    ss = float(((y[m] - pred[m]) ** 2).sum())
    tt = float(((y[m] - y[m].mean()) ** 2).sum())
    return round(float(1 - ss / tt), 4) if tt > 0 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_eval_paths", type=int, default=200)
    p.add_argument("--seed_eval", type=int, default=202)
    p.add_argument("--n_scan", type=int, default=65)
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
    base = {}
    for seed in (11, 23, 47):
        mp = ART / ("r04b_s%d" % seed) / "flipmine"
        model, stats = load_model(mp)
        model.eval()
        w = model.cls.weight.detach().numpy().reshape(-1)
        b = float(model.cls.bias.detach().numpy().reshape(-1)[0])
        wn = float(np.linalg.norm(w)) + 1e-12

        def predict(xs, _m=model, _st=stats):
            with torch.no_grad():
                lg = _m(preprocess(np.asarray(xs, np.float32), _st)).numpy()
            lg = np.asarray(lg, float).reshape(-1)
            return lg, (lg > 0).astype(int)

        n_hit, errs, miss = 0, [], 0
        for ep in ev_paths:
            x = Xe[ep["parent_id"]]
            e = np.asarray(ep["edit"], float)
            sc = scan_linear(x, e, predict, n_scan=a.n_scan)
            ot = locate_oracle_turns(x, e, sc)
            if len(ot) != 1:
                continue
            n_hit += 1
            t_star = ot[0]["t_star"]
            t = sc["t"]
            lg = sc["logit"]
            omar = sc["oracle_margin"]
            dt = float(t[1] - t[0])
            # dense representation pass on the same grid
            xs = (x[None, :, :] + t[:, None, None] * e[None, :, :]).astype(np.float32)
            with torch.no_grad():
                _, Z = model(preprocess(xs, stats), return_feat=True)
            Z = np.asarray(Z, float)
            win = np.abs(t - t_star) <= 0.1
            # oracle family
            f = {"seed": seed, "parent": int(ep["parent_id"]),
                 "t_star": round(t_star, 6)}
            f["min_margin_win"] = round(float(np.abs(omar[win]).min())
                                        if win.any() else np.nan, 6)
            sl = np.gradient(omar, dt)
            i0 = int(np.argmin(np.abs(t - t_star)))
            f["margin_slope"] = round(float(sl[i0]), 6)
            f["cost"] = round(float(np.linalg.norm(e)), 6)
            # score family
            d1 = np.gradient(lg, dt)
            d2 = np.gradient(d1, dt)
            f["max_slope"] = round(float(np.abs(d1).max()), 6)
            f["max_curv"] = round(float(np.abs(d2[win]).max())
                                  if win.any() else np.nan, 6)
            f["trans_width"] = round(float(np.mean(np.abs(lg) < 1.0)), 6)
            lm = (t >= t_star - 0.15) & (t < t_star)
            rm = (t > t_star) & (t <= t_star + 0.15)
            f["slope_asym"] = round(float(abs(d1[lm].mean() - d1[rm].mean()))
                                    if lm.any() and rm.any() else np.nan, 6)
            f["min_abs_logit"] = round(float(np.abs(lg[win]).min())
                                       if win.any() else np.nan, 6)
            # representation family
            seg = np.diff(Z, axis=0)
            spd = np.linalg.norm(seg, axis=1) / dt
            tmid = 0.5 * (t[:-1] + t[1:])
            winmid = np.abs(tmid - t_star) <= 0.1
            f["speed_conc"] = round(float(spd[winmid].sum() / spd.sum())
                                    if spd.sum() > 0 else np.nan, 6)
            ang = []
            for k in range(len(seg) - 1):
                n1 = float(np.linalg.norm(seg[k]))
                n2 = float(np.linalg.norm(seg[k + 1]))
                if n1 > 1e-12 and n2 > 1e-12:
                    c = float(np.clip(seg[k] @ seg[k + 1] / (n1 * n2), -1, 1))
                    ang.append((k, float(np.arccos(c))))
            f["max_angle"] = round(max((v for k, v in ang
                                        if abs(float(t[k + 1]) - t_star) <= 0.1),
                                       default=np.nan), 6)
            f["hyper_dist"] = round(float(abs(Z[i0] @ w + b) / wn), 6)
            f["z_length"] = round(float(np.linalg.norm(seg, axis=1).sum()), 6)
            # target (E7 0->1 match rule)
            mt = locate_model_turns(x, e, sc, predict)
            same = [m for m in mt if m["old"] == 0 and m["new"] == 1]
            if not same:
                f["miss"] = 1
                f["abserr"] = np.nan
                miss += 1
            else:
                best = min(same, key=lambda m: abs(m["t_theta"] - t_star))
                f["miss"] = 0
                f["abserr"] = round(abs(best["t_theta"] - t_star), 6)
                errs.append(f["abserr"])
            rows.append(f)
        got = [e for e in errs]
        base["s%d" % seed] = {
            "n_single_turn": n_hit,
            "turn_miss_rate": round(miss / n_hit, 4) if n_hit else None,
            "mean_abs_terr": round(float(np.mean(got)), 4) if got else None,
        }
        print("s%d: n=%d miss=%s |terr|=%s" % (
            seed, n_hit, base["s%d" % seed]["turn_miss_rate"],
            base["s%d" % seed]["mean_abs_terr"]), flush=True)

    # analysis per seed on non-miss rows
    summary = {"baseline": base, "seeds": {}, "criterion": (
        "REPR family CV-R^2 must exceed ORACLE by >=0.05 in >=2/3 seeds")}
    for seed in (11, 23, 47):
        R = [r for r in rows if r["seed"] == seed and r["miss"] == 0]
        y = np.array([r["abserr"] for r in R])
        g = np.array([r["parent"] for r in R])
        fam = {}
        for name, feats in (("ORACLE", ORACLE_FEATS), ("SCORE", SCORE_FEATS),
                            ("REPR", REPR_FEATS)):
            Xf = np.array([[r[k] for k in feats] for r in R])
            fam[name] = {
                "cv_r2": ridge_cv_r2(Xf, y, g),
                "spearman": {k: spearman([r[k] for r in R], y)
                             for k in feats},
                "n": len(R),
            }
        ro = fam["ORACLE"]["cv_r2"]
        rr = fam["REPR"]["cv_r2"]
        fam["repr_minus_oracle"] = (round(rr - ro, 4)
                                    if ro is not None and rr is not None
                                    else None)
        summary["seeds"]["s%d" % seed] = fam
    wins = sum(1 for s in (11, 23, 47)
               if (summary["seeds"]["s%d" % s]["repr_minus_oracle"]
                   is not None and
                   summary["seeds"]["s%d" % s]["repr_minus_oracle"] >= 0.05))
    summary["criterion_met"] = wins >= 2
    summary["verdict"] = ("representation-side signal" if summary["criterion_met"]
                          else "input-geometry-determined (boundary)")

    a.out.mkdir(parents=True, exist_ok=True)
    cols = (["seed", "parent", "t_star"] + ORACLE_FEATS[1:] + SCORE_FEATS +
            REPR_FEATS + ["miss", "abserr"])
    with open(a.out / "L007_DIAG.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    json.dump(summary, open(a.out / "L007_SUMMARY.json", "w"), indent=1,
              default=float)
    print("criterion_met:", summary["criterion_met"],
          "verdict:", summary["verdict"])
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
