#!/usr/bin/env python3
"""L007-C2: parameter-scale restart probe (review item C2).

Same function, same data, same objective, different parameter coordinates:
relu is positively homogeneous, so scaling layer-1 by c>0 and compensating
in layer-2 leaves the network function EXACTLY unchanged (asserted). If the
optimizer's trajectory then diverges in root location, the location error
partly reflects the parameterization/optimizer, not the learned function
class or the data.

Arms (seeds 2001-2003, start = l007_wave ckpt_150, 100 epochs):
  c=0.5, c=1.0 (control), c=2.0
All arms restart Adam from scratch (moments are not stored in checkpoints),
so the comparison is a matched restart; c=1.0 absorbs the restart effect.

PRE-REGISTERED criterion: initial function difference must be < 1e-4 max
logit; then if >=2/3 seeds show the parent-clustered paired CI of
P(root within 0.05) between c!=1 and c=1 entirely outside +/-5pp, scale
sensitivity is supported; if all CIs lie inside +/-5pp, negated; else
undetermined. Training loss comparability is reported as a condition.

Outputs: artifacts/next_novelty/l007_scale/SCALE.json
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
from paths import locate_model_turns, locate_oracle_turns, scan_linear  # noqa: E402
from coord_mlp import CoordMLP  # noqa: E402
from r04b_methods import mine_flips  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
WAVE = ROOT / "artifacts" / "next_novelty" / "l007_wave"
OUT = ROOT / "artifacts" / "next_novelty" / "l007_scale"
SEEDS = [2001, 2002, 2003]
CS = [0.5, 1.0, 2.0]
START_EPOCH = 150
N_EPOCHS = 100
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
    mu_t, sd_t = torch.from_numpy(mu8), torch.from_numpy(sd8)
    Xc = ((torch.from_numpy(X.reshape(-1, 8)) - mu_t) / sd_t)
    yc = torch.from_numpy(y)
    Xf = ((torch.from_numpy((X[ii] + ee).reshape(-1, 8)) - mu_t) / sd_t)
    yft = torch.from_numpy(yf)
    bce = nn.BCEWithLogitsLoss()

    def scaled_state(seed, c):
        ck = torch.load(WAVE / ("s%d" % seed) / ("ckpt_%03d.pt" % START_EPOCH),
                        map_location="cpu", weights_only=False)
        st = {k: v.clone() for k, v in ck["state"].items()}
        st["net.0.weight"] = st["net.0.weight"] * c
        st["net.0.bias"] = st["net.0.bias"] * c
        st["net.2.weight"] = st["net.2.weight"] / c
        return st

    def build(st):
        m = CoordMLP(64, 32, in_dim=8)
        m.load_state_dict(st)
        return m

    def path_stats(model):
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

    def succ(rows, eps=EPS):
        e = np.array([r["abserr"] for r in rows], float)
        return (np.isfinite(e) & (e <= eps)).astype(float)

    res = {"start_epoch": START_EPOCH, "epochs": N_EPOCHS, "cs": CS,
           "eps": EPS, "seeds": {},
           "criterion": ("parent-clustered paired CI of P(root<=0.05) "
                         "difference (c!=1 vs c=1) outside +/-5pp in >=2/3 seeds")}
    for seed in SEEDS:
        sd = {}
        base_st = scaled_state(seed, 1.0)
        m0 = build(base_st)
        m0.eval()
        with torch.no_grad():
            lg_ref = m0(Xc)
        for c in CS:
            st = scaled_state(seed, c)
            mm = build(st)
            mm.eval()
            with torch.no_grad():
                d = float((mm(Xc) - lg_ref).abs().max())
            assert d < 1e-4, "scale c=%s is not function-preserving (%.3g)" % (c, d)
            sd["c%s_func_preserving_maxdiff" % c] = d
        for c in CS:
            torch.manual_seed(seed)   # same restart seed across arms
            model = build(scaled_state(seed, c))
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            for _ in range(N_EPOCHS):
                model.train()
                opt.zero_grad()
                loss = bce(model(Xc), yc).mean() + bce(model(Xf), yft).mean()
                loss.backward()
                opt.step()
            rows = path_stats(model)
            with torch.no_grad():
                fl = float((bce(model(Xc), yc).mean()
                            + bce(model(Xf), yft).mean()).item())
            sd["c%s" % c] = {"rows": rows,
                             "final_loss": round(fl, 6),
                             "succ": round(float(succ(rows).mean()), 4),
                             "miss": round(float(np.mean(
                                 [0.0 if np.isfinite(r["abserr"]) else 1.0
                                  for r in rows])), 4)}
            print("s%d c=%s succ=%s miss=%s loss=%.4f" %
                  (seed, c, sd["c%s" % c]["succ"], sd["c%s" % c]["miss"], fl),
                  flush=True)
        # parent-clustered paired CI vs c=1
        s1 = succ(sd["c1.0"]["rows"])
        par = np.array([r["parent"] for r in sd["c1.0"]["rows"]])
        groups = [np.nonzero(par == u)[0] for u in np.unique(par)]
        rng = np.random.default_rng(999)
        for c in (0.5, 2.0):
            sc_ = succ(sd["c%s" % c]["rows"])
            d = sc_ - s1
            boots = []
            for _ in range(2000):
                draw = rng.choice(len(groups), size=len(groups), replace=True)
                idx = np.concatenate([groups[i] for i in draw])
                boots.append(d[idx].mean())
            lo, hi = np.quantile(boots, [0.025, 0.975])
            sd["c%s_vs_c1" % c] = {"delta_pp": round(float(d.mean()) * 100, 2),
                                   "ci_pp": [round(float(lo) * 100, 2),
                                             round(float(hi) * 100, 2)],
                                   "outside_5pp": bool(lo * 100 > 5 or hi * 100 < -5)}
            print("  c=%s vs c=1: d=%s pp CI=%s outside=%s" %
                  (c, sd["c%s_vs_c1" % c]["delta_pp"],
                   sd["c%s_vs_c1" % c]["ci_pp"],
                   sd["c%s_vs_c1" % c]["outside_5pp"]), flush=True)
        res["seeds"][str(seed)] = sd
    n_out = sum(1 for s in SEEDS for c in (0.5, 2.0)
                if res["seeds"][str(s)]["c%s_vs_c1" % c]["outside_5pp"]
                and abs(res["seeds"][str(s)]["c%s_vs_c1" % c]["delta_pp"]) >= 5)
    all_in = all(not res["seeds"][str(s)]["c%s_vs_c1" % c]["outside_5pp"]
                 for s in SEEDS for c in (0.5, 2.0))
    res["criterion_met"] = n_out >= 2
    res["verdict"] = ("scale-sensitivity supported" if res["criterion_met"]
                      else ("negated (all CIs within +/-5pp)" if all_in
                            else "undetermined"))
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "SCALE.json", "w"), indent=1, default=float)
    print("criterion_met:", res["criterion_met"], "verdict:", res["verdict"])
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
