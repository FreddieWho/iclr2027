#!/usr/bin/env python3
"""L007 seed-variance lane: per-path |terr| across 8 fresh flipmine seeds.

Same 200 eval paths (seed_eval=202), shared paths.py contract, E7 0->1
match rule. Computes per-path |terr|/miss per seed plus an 8-seed
mean-logit ensemble. Analysis: cross-seed rank correlation, variance
decomposition, ensemble vs single-seed comparison.

PRE-REGISTERED decision: ensemble improves >=15% over mean single-seed
|terr| AND mean cross-seed Spearman >=0.4 -> systematic component exists
(report only, no follow-up training without authorization); otherwise ->
noise-dominated, mechanism search closed.

NEW outputs only:
  artifacts/next_novelty/l007_seed/L007_PATHS.csv
  artifacts/next_novelty/l007_seed/L007_SEEDVARIANCE.json
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
SEEDDIR = ROOT / "artifacts" / "next_novelty" / "l007_seed"
SEEDS = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]


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


def match_terr(mt, t_star):
    same = [m for m in mt if m["old"] == 0 and m["new"] == 1]
    if not same:
        return None
    best = min(same, key=lambda m: abs(m["t_theta"] - t_star))
    return abs(best["t_theta"] - t_star)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_eval_paths", type=int, default=200)
    p.add_argument("--seed_eval", type=int, default=202)
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

    models = {}
    for seed in SEEDS:
        model, stats = load_model(SEEDDIR / ("s%d" % seed))
        model.eval()
        models[seed] = (model, stats)

    def make_predict(seeds):
        def predict(xs):
            acc = None
            for sd in seeds:
                model, stats = models[sd]
                with torch.no_grad():
                    lg = model(preprocess(np.asarray(xs, np.float32),
                                          stats)).numpy()
                lg = np.asarray(lg, float).reshape(-1)
                acc = lg if acc is None else acc + lg
            acc = acc / len(seeds)
            return acc, (acc > 0).astype(int)
        return predict

    rows = []
    for ep in ev_paths:
        x = Xe[ep["parent_id"]]
        e = np.asarray(ep["edit"], float)
        # oracle turn from any single-seed scan (oracle part identical)
        sc0 = scan_linear(x, e, make_predict([SEEDS[0]]), n_scan=17)
        ot = locate_oracle_turns(x, e, sc0)
        if len(ot) != 1:
            continue
        t_star = ot[0]["t_star"]
        r = {"parent": int(ep["parent_id"]), "t_star": round(t_star, 6)}
        for seed in SEEDS:
            sc = scan_linear(x, e, make_predict([seed]), n_scan=17)
            mt = locate_model_turns(x, e, sc, make_predict([seed]))
            v = match_terr(mt, t_star)
            r["terr_s%d" % seed] = round(v, 6) if v is not None else ""
        sc_e = scan_linear(x, e, make_predict(SEEDS), n_scan=17)
        mt_e = locate_model_turns(x, e, sc_e, make_predict(SEEDS))
        ve = match_terr(mt_e, t_star)
        r["terr_ens"] = round(ve, 6) if ve is not None else ""
        rows.append(r)

    # analysis
    T = {}
    for seed in SEEDS:
        v = np.array([float(r["terr_s%d" % seed]) if r["terr_s%d" % seed] != ""
                      else np.nan for r in rows])
        T[seed] = v
    Te = np.array([float(r["terr_ens"]) if r["terr_ens"] != "" else np.nan
                   for r in rows])
    pairs, ns = [], []
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            sp = spearman(T[SEEDS[i]], T[SEEDS[j]])
            if sp is not None:
                pairs.append(sp)
                m = np.isfinite(T[SEEDS[i]]) & np.isfinite(T[SEEDS[j]])
                ns.append(int(m.sum()))
    M = np.stack([T[s] for s in SEEDS])
    with np.errstate(invalid="ignore"):
        across_var = float(np.nanmean(np.nanvar(M, axis=0)))
        path_means = np.nanmean(M, axis=0)
        between_var = float(np.nanvar(path_means))
    single_means = {s: round(float(np.nanmean(T[s])), 4) for s in SEEDS}
    ens_mean = round(float(np.nanmean(Te)), 4)
    miss_single = {s: round(float(np.isnan(T[s]).mean()), 4) for s in SEEDS}
    res = {
        "n_paths": len(rows),
        "seeds": SEEDS,
        "mean_pairwise_spearman": round(float(np.mean(pairs)), 4),
        "pairwise_n_mean": round(float(np.mean(ns)), 1),
        "across_seed_var_mean": round(across_var, 6),
        "between_path_var": round(between_var, 6),
        "single_seed_mean_terr": single_means,
        "mean_single_terr": round(float(np.mean(list(single_means.values()))), 4),
        "best_single_terr": min(single_means.values()),
        "ensemble_mean_terr": ens_mean,
        "ensemble_improvement_vs_mean": round(
            1 - ens_mean / np.mean(list(single_means.values())), 4),
        "miss_single": miss_single,
        "miss_ens": round(float(np.isnan(Te).mean()), 4),
    }
    crit = (res["ensemble_improvement_vs_mean"] >= 0.15
            and res["mean_pairwise_spearman"] >= 0.4)
    res["criterion_met"] = bool(crit)
    res["verdict"] = ("systematic-component (report only, no follow-up "
                      "without authorization)" if crit else
                      "noise-dominated, mechanism search closed")
    SEEDDIR.mkdir(parents=True, exist_ok=True)
    cols = (["parent", "t_star"] + ["terr_s%d" % s for s in SEEDS]
            + ["terr_ens"])
    with open(SEEDDIR / "L007_PATHS.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    json.dump(res, open(SEEDDIR / "L007_SEEDVARIANCE.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("DONE ->", SEEDDIR)


if __name__ == "__main__":
    main()
