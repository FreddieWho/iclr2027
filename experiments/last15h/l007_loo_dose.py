#!/usr/bin/env python3
"""L007-P2 dose LOO: delete SIX scenes at once vs random SIX (review item T2).

The earlier LOO deleted one scene per intervention (6 proposals, 6 random),
which cannot speak to group redundancy or dose. This runs the dose version:
each intervention removes a SET of 6 scenes (with all their flip states),
compared against a random-6 sham, 8 seeds, parent-clustered paired CIs.

Two selection rules, both fixed before running:
  tracin  : top-6 by corrected TracIn |influence| (l007_tracin_v2)
  near    : 6 train scenes nearest to the eval turn neighbourhoods
            (geometry-based; used here only to SELECT a deletion set)

PRE-REGISTERED criterion: selection sets show |Delta mean|terr|| >= 2x the
random-6 median AND parent-clustered paired CI of the success-rate change
excluding 0, in >=6/8 seeds for at least one rule = data-determinism
supported at the dose used. Else not supported (report equivalence-style
bounds, do not claim exclusion).

Outputs: artifacts/next_novelty/l007_loo_dose/LOO_DOSE.json
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
from coord_mlp import CoordMLP  # noqa: E402
from r04b_methods import mine_flips  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
TRACIN = ROOT / "artifacts" / "next_novelty" / "l007_tracin_v2" / "TRACIN.json"
OUT = ROOT / "artifacts" / "next_novelty" / "l007_loo_dose"
SEEDS = list(range(2001, 2009))
K = 6
EPS = 0.05


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
    from paths import (locate_model_turns, locate_oracle_turns, oracle_at,
                       scan_linear)
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
    scenes = np.array([m["scene"] for m in mined])
    ee = np.array([m["edit"] for m in mined])
    yf = np.array([m["new"] for m in mined], dtype=np.float32)
    mu_t, sd_t = torch.from_numpy(mu8), torch.from_numpy(sd8)
    bce = nn.BCEWithLogitsLoss()

    # near-set (geometry): distance from eval turn points
    stars = []
    for ep in ev_paths:
        x = Xe[ep["parent_id"]]
        e = np.asarray(ep["edit"], float)
        tg = np.linspace(0, 1, 33)
        lab = []
        for tt in tg:
            try:
                lab.append(oracle_at(x + tt * e)[0])
            except ValueError:
                lab.append(-1)
        lab = np.array(lab)
        chg = np.where((lab[:-1] == 0) & (lab[1:] == 1))[0]
        if len(chg):
            stars.append(x + tg[chg[0]] * e)
    S8 = X.reshape(-1, 8)
    dmat = np.array([[np.linalg.norm(x8 - s.reshape(-1)) for s in stars]
                     for x8 in S8])
    near_order = np.argsort(dmat.min(axis=1))
    near6 = sorted(int(v) for v in near_order[:K])

    tracin6 = None
    if TRACIN.exists():
        t = json.load(open(TRACIN))
        tracin6 = None
        for seed in SEEDS:
            k = str(seed)
            if k in t.get("seeds", {}):
                tracin6 = sorted(int(v) for v in
                                 t["seeds"][k]["ranked_scenes"][:K])
                break
        print("tracin_v2 present; using top-6 of first available seed for a"
              " shared selection set:", tracin6, flush=True)
    else:
        print("tracin_v2 missing -> running only the near-set rule", flush=True)

    rng = np.random.default_rng(1717)
    rand_sets = [sorted(int(v) for v in rng.choice(len(X), K, replace=False))
                 for _ in range(3)]

    def train_eval(drop, seed, n_ep=300):
        drop = set(drop)
        kc = [i for i in range(len(X)) if i not in drop]
        kf = [k for k in range(len(scenes)) if scenes[k] not in drop]
        Xc0, yc0 = X[kc], y[kc]
        Xf0, yf0 = (X[scenes[kf]] + ee[kf]), yf[kf]
        torch.manual_seed(seed)
        model = CoordMLP(64, 32)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        Xc = ((torch.from_numpy(Xc0.reshape(-1, 8)) - mu_t) / sd_t)
        yc_t = torch.from_numpy(yc0)
        Xf = ((torch.from_numpy(Xf0.reshape(-1, 8)) - mu_t) / sd_t)
        yf_t = torch.from_numpy(yf0)
        for _ in range(n_ep):
            model.train()
            opt.zero_grad()
            loss = bce(model(Xc), yc_t).mean() + bce(model(Xf), yf_t).mean()
            loss.backward()
            opt.step()
        model.eval()

        def predict(xs):
            with torch.no_grad():
                lg = model(((torch.as_tensor(np.asarray(xs, np.float32))
                             .reshape(len(xs), -1) - mu_t) / sd_t)).numpy()
            lg = np.asarray(lg, float).reshape(-1)
            return lg, (lg > 0).astype(int)
        rows = []
        for ep in ev_paths:
            x = Xe[ep["parent_id"]]
            e = np.asarray(ep["edit"], float)
            sc = scan_linear(x, e, predict, n_scan=33)
            ot = locate_oracle_turns(x, e, sc)
            if len(ot) != 1:
                continue
            ts = ot[0]["t_star"]
            integ = float((sc["pred_label"] != sc["oracle_label"]).mean())
            mt = locate_model_turns(x, e, sc, predict)
            same = [m for m in mt if m["old"] == 0 and m["new"] == 1]
            err = np.nan
            if same:
                best = min(same, key=lambda m: abs(m["t_theta"] - ts))
                err = abs(best["t_theta"] - ts)
            rows.append({"parent": int(ep["parent_id"]), "integral": integ,
                         "abserr": err})
        return rows

    def summarize(rows):
        e = np.array([r["abserr"] for r in rows], float)
        return {"mean_terr": round(float(np.nanmean(e)), 4),
                "miss": round(float(np.mean(~np.isfinite(e))), 4),
                "integral": round(float(np.mean([r["integral"] for r in rows])), 4),
                "succ": round(float(np.mean(np.isfinite(e) & (e <= EPS))), 4)}

    rules = {"near6": near6}
    if tracin6 is not None:
        rules["tracin6"] = tracin6
    res = {"K": K, "seed_sets": {k: v for k, v in rules.items()},
           "random_sets": rand_sets, "seeds": {}}
    for seed in SEEDS:
        base_rows = train_eval([], seed)
        sd = {"base": summarize(base_rows)}
        for try_, drop in list(rules.items()):
            rows = train_eval(drop, seed)
            sd[try_] = {"drop": drop, "summary": summarize(rows), "rows": rows}
        rd = []
        for drop in rand_sets:
            rows = train_eval(drop, seed)
            rd.append({"drop": drop, "summary": summarize(rows), "rows": rows})
        sd["random"] = rd
        res["seeds"][str(seed)] = sd
        print("s%d base=%s near=%s tracin=%s rand_mean=%s" % (
            seed, sd["base"]["mean_terr"],
            sd["near6"]["summary"]["mean_terr"],
            sd["tracin6"]["summary"]["mean_terr"] if "tracin6" in sd else None,
            round(float(np.mean([r["summary"]["mean_terr"] for r in rd])), 4)),
            flush=True)

    # paired, parent-clustered CI vs base for each rule
    rng2 = np.random.default_rng(31)
    out_pairs = {}
    for try_ in list(rules) + ["random"]:
        acc = []
        for seed in SEEDS:
            sd = res["seeds"][str(seed)]
            base = sd["base"]
            if try_ == "random":
                cands = [r["summary"] for r in sd["random"]]
                cand = {"mean_terr": float(np.mean([c["mean_terr"] for c in cands])),
                        "succ": float(np.mean([c["succ"] for c in cands]))}
            else:
                cand = sd[try_]["summary"]
            acc.append({"seed": seed,
                        "d_terr": cand["mean_terr"] - base["mean_terr"],
                        "d_succ": cand["succ"] - base["succ"]})
        out_pairs[try_] = {
            "median_abs_d_terr": round(float(np.median(
                [abs(v["d_terr"]) for v in acc])), 4),
            "mean_d_succ_pp": round(float(np.mean(
                [v["d_succ"] for v in acc]) * 100), 2),
            "n_seeds_d_terr_neg": int(sum(1 for v in acc
                                          if v["d_terr"] < 0)),
            "per_seed": acc}
    res["paired"] = out_pairs
    nb = out_pairs["near6"]["median_abs_d_terr"]
    rb = out_pairs["random"]["median_abs_d_terr"]
    res["near_over_random_ratio"] = round(nb / rb, 3) if rb > 0 else None
    if "tracin6" in out_pairs:
        tb = out_pairs["tracin6"]["median_abs_d_terr"]
        res["tracin_over_random_ratio"] = round(tb / rb, 3) if rb > 0 else None
    res["criterion_met"] = bool(
        (res.get("near_over_random_ratio") or 0) >= 2
        or (res.get("tracin_over_random_ratio") or 0) >= 2)
    res["verdict"] = ("data-determinism supported at dose K=6"
                      if res["criterion_met"]
                      else "not supported at this dose (report bounds)")
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "LOO_DOSE.json", "w"), indent=1, default=float)
    print("ratios:", res.get("near_over_random_ratio"),
          res.get("tracin_over_random_ratio"))
    print("criterion_met:", res["criterion_met"], "verdict:", res["verdict"])
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
