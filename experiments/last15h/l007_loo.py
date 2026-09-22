#!/usr/bin/env python3
"""L007-P2 LOO: exact leave-one-scene-out retrains for TracIn top-k vs
random-k scenes. Deleting a scene removes its clean state AND all its
mined flips. Same recipe otherwise (300ep, Adam 1e-2, lam 1.0,
torch.manual_seed(seed) => identical init; only data differs).
Evaluated on the same 200 eval paths (E7 contract).

Deletion sets from ../l007_tracin/TRACIN.json (top-6 scenes per seed) +
random-6 (rng 777, excluding top-40). Seeds 2001-2003 (cost bound).
Baseline = l007_wave finals (same inits).

PRE-REGISTERED criterion: median |Delta mean|terr|| over top-k deletions
>= 2x the random-k median, with the same sign of (deleted - baseline)
in 3/3 seeds = data-determinism positive. Else LOO line closed
(TracIn scores descriptive only, no more retrains).

Outputs: artifacts/next_novelty/l007_loo/LOO.json
"""
import argparse
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
OUT = ROOT / "artifacts" / "next_novelty" / "l007_loo"
SEEDS = [2001, 2002, 2003]
K = 6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "n01b", str(ROOT / "experiments" / "last15h" / "n01_brackets.py"))
    n01b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(n01b)
    from paths import locate_model_turns, locate_oracle_turns, scan_linear
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
    tracin = json.load(open(
        ROOT / "artifacts" / "next_novelty" / "l007_tracin" / "TRACIN.json"))
    mu8_t = torch.from_numpy(mu8)
    sd8_t = torch.from_numpy(sd8)

    def train_eval(drop, seed):
        keep_c = np.array([i for i in range(len(X)) if i not in drop])
        keep_f = np.array([k for k in range(len(scenes)) if scenes[k] not in drop])
        torch.manual_seed(seed)
        model = CoordMLP(64, 32)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        bce = nn.BCEWithLogitsLoss()
        Xc = torch.from_numpy(((X[keep_c].reshape(-1, 8) - mu8) / sd8).astype(np.float32))
        yc = torch.from_numpy(y[keep_c])
        Xf = torch.from_numpy((((X[scenes[keep_f]] + ee[keep_f]).reshape(-1, 8) - mu8) / sd8).astype(np.float32))
        yff = torch.from_numpy(yf[keep_f])
        for ep in range(300):
            model.train()
            opt.zero_grad()
            loss = bce(model(Xc), yc).mean() + bce(model(Xf), yff).mean()
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

    rng = np.random.default_rng(777)
    res = {"K": K, "seeds": {}}
    for seed in SEEDS:
        top = tracin["seeds"][str(seed)]["ranked_scenes"][:K]
        pool = [i for i in range(len(X))
                if i not in tracin["seeds"][str(seed)]["ranked_scenes"][:40]]
        rand = sorted(rng.choice(pool, K, replace=False).tolist())
        base = train_eval(set(), seed)
        sd = {"baseline": base, "top": [], "random": []}
        for s in top:
            v = train_eval({s}, seed)
            sd["top"].append({"scene": s, "d_terr": (
                round(v["mean_terr"] - base["mean_terr"], 4)
                if v["mean_terr"] is not None and base["mean_terr"] is not None
                else None), "miss": v["miss"]})
            print("s%d top drop %d: dterr=%s" % (seed, s, sd["top"][-1]["d_terr"]),
                  flush=True)
        for s in rand:
            v = train_eval({s}, seed)
            sd["random"].append({"scene": s, "d_terr": (
                round(v["mean_terr"] - base["mean_terr"], 4)
                if v["mean_terr"] is not None and base["mean_terr"] is not None
                else None), "miss": v["miss"]})
            print("s%d rand drop %d: dterr=%s" % (seed, s, sd["random"][-1]["d_terr"]),
                  flush=True)
        res["seeds"][str(seed)] = sd
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "LOO.json", "w"), indent=1, default=float)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
