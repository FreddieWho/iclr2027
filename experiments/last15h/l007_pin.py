#!/usr/bin/env python3
"""L007-P3: training-point geometry pinning test (max-margin direction).

Question: is t_theta pinned by the training-point envelope rather than the
oracle truth? Three groups, 4 seeds (2001-2004), same recipe otherwise:
  A (insert): 40 new oracle-labeled states at turn neighborhoods
      (x + t_star*e + small jitter) appended to clean supervision.
  B (remove-near): delete the 20 train scenes nearest to turn
      neighborhoods (+ their flips).
  C (sham): delete the 20 FARTHEST train scenes (+ their flips).
Baselines = l007_wave finals (same inits).

PRE-REGISTERED criterion: group A mean|terr| drops >=10% in >=3/4 seeds
AND group C |change| <5% in >=3/4 seeds = pinning positive (regardless of
B, reported informationally). Else pinning line closed.

Outputs: artifacts/next_novelty/l007_pin/PIN.json
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
OUT = ROOT / "artifacts" / "next_novelty" / "l007_pin"
SEEDS = [2001, 2002, 2003, 2004]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location(
        "n01b", str(ROOT / "experiments" / "last15h" / "n01_brackets.py"))
    n01b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(n01b)
    sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
    from paths import (locate_model_turns, locate_oracle_turns, oracle_at,
                       scan_linear)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X, y = d["positions"].astype(np.float32), d["labels"].astype(np.float32)
    mu8, sd8 = X.reshape(-1, 8).mean(0), X.reshape(-1, 8).std(0) + 1e-8
    mined = mine_flips(X.astype(float), y.astype(int), seed=5)
    scenes = np.array([m["scene"] for m in mined])
    ee = np.array([m["edit"] for m in mined])
    yf = np.array([m["new"] for m in mined], dtype=np.float32)
    de = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    Xe = de["positions"].astype(float)
    ev_paths = n01b.aimed_single_turns(
        Xe, np.random.default_rng(202), 200, seed=202)
    # turn neighborhoods + nearest train scenes
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
    stars = np.array(stars)
    S8 = X.reshape(-1, 8)
    dmat = np.array([[np.linalg.norm(s8 - st.reshape(-1)) for st in stars]
                     for s8 in S8])
    near_order = np.argsort(dmat.min(axis=1))
    rng = np.random.default_rng(4242)
    # group A inserts
    ins_X, ins_y = [], []
    for ep in rng.choice(ev_paths, 40, replace=False):
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
        if not len(chg):
            continue
        ts = tg[chg[0]]
        pt = (x + ts * e + rng.normal(scale=0.02, size=(4, 2))).astype(np.float32)
        try:
            l, _, _ = oracle_at(pt)
        except ValueError:
            continue
        ins_X.append(pt.reshape(-1))
        ins_y.append(l)
    ins_X = np.array(ins_X)
    ins_y = np.array(ins_y, dtype=np.float32)
    groups = {"A_insert": None, "B_remove_near": set(near_order[:20].tolist()),
              "C_remove_far": set(near_order[-20:].tolist())}
    assert len(groups["B_remove_near"]) == 20 and len(groups["C_remove_far"]) == 20
    assert groups["B_remove_near"].isdisjoint(groups["C_remove_far"]), \
        "near/far deletion sets must be disjoint"
    mu8_t = torch.from_numpy(mu8)
    sd8_t = torch.from_numpy(sd8)

    def train_eval(group, seed):
        torch.manual_seed(seed)
        model = CoordMLP(64, 32)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        bce = nn.BCEWithLogitsLoss()
        if group == "BASE":
            Xc0, yc0 = X, y
            Xf0, yf0 = (X[scenes] + ee), yf
        elif group == "B_remove_near":
            drop = groups["B_remove_near"]
            Xc0 = X[[i for i in range(len(X)) if i not in drop]]
            yc0 = y[[i for i in range(len(y)) if i not in drop]]
            kf = [k for k in range(len(scenes)) if scenes[k] not in drop]
            Xf0, yf0 = (X[scenes[kf]] + ee[kf]), yf[kf]
        elif group == "C_remove_far":
            drop = groups["C_remove_far"]
            Xc0 = X[[i for i in range(len(X)) if i not in drop]]
            yc0 = y[[i for i in range(len(y)) if i not in drop]]
            kf = [k for k in range(len(scenes)) if scenes[k] not in drop]
            Xf0, yf0 = (X[scenes[kf]] + ee[kf]), yf[kf]
        else:
            Xc0 = np.concatenate([X.reshape(-1, 8), ins_X]).reshape(-1, 4, 2)
            yc0 = np.concatenate([y, ins_y])
            Xf0, yf0 = (X[scenes] + ee), yf
        Xc = torch.from_numpy(((Xc0.reshape(-1, 8) - mu8) / sd8).astype(np.float32))
        yc_t = torch.from_numpy(yc0)
        Xf = torch.from_numpy(((Xf0.reshape(-1, 8) - mu8) / sd8).astype(np.float32))
        yf_t = torch.from_numpy(yf0)
        for ep_ in range(300):
            model.train()
            opt.zero_grad()
            loss = bce(model(Xc), yc_t).mean() + bce(model(Xf), yf_t).mean()
            loss.backward()
            opt.step()
        model.eval()

        def predict(xs):
            with torch.no_grad():
                lg = model(((torch.as_tensor(np.asarray(xs, np.float32))
                             .reshape(len(xs), -1) - mu8_t) / sd8_t)).numpy()
            lg = np.asarray(lg, float).reshape(-1)
            return lg, (lg > 0).astype(int)

        errs, miss, n = [], 0, 0
        for epp in ev_paths:
            x = Xe[epp["parent_id"]]
            e = np.asarray(epp["edit"], float)
            sc = scan_linear(x, e, predict, n_scan=17)
            ot = locate_oracle_turns(x, e, sc)
            if len(ot) != 1:
                continue
            n += 1
            mt = locate_model_turns(x, e, sc, predict)
            same = [m for m in mt if m["old"] == 0 and m["new"] == 1]
            if not same:
                miss += 1
            else:
                best = min(same, key=lambda m: abs(m["t_theta"] - ot[0]["t_star"]))
                errs.append(abs(best["t_theta"] - ot[0]["t_star"]))
        return {"mean_terr": round(float(np.mean(errs)), 4) if errs else None,
                "miss": round(miss / n, 4) if n else None, "n": n}

    res = {"groups": list(groups), "n_insert": len(ins_X), "seeds": {}}
    for seed in SEEDS:
        base = train_eval("BASE", seed)
        sd = {"baseline": base}
        for g in groups:
            v = train_eval(g, seed)
            chg = (round((v["mean_terr"] - base["mean_terr"]) / base["mean_terr"], 4)
                   if v["mean_terr"] and base["mean_terr"] else None)
            sd[g] = {"mean_terr": v["mean_terr"], "miss": v["miss"],
                     "rel_change": chg}
            print("s%d %s: terr=%s rel=%s" % (seed, g, v["mean_terr"], chg),
                  flush=True)
        res["seeds"][str(seed)] = sd
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "PIN.json", "w"), indent=1, default=float)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
