#!/usr/bin/env python3
"""L007-P3 v2: train-only pinning test with a MATCHED sham.

Fixes vs l007_pin.py (GPT6ASTRA_CODE_REVIEW.md §8):
  - every inserted state is constructed from train_101 parents only
    (was: eval_202-derived states entering the training set -> contract
    violation);
  - the sham now matches the operation: both arms insert the SAME number of
    oracle-labeled states drawn from the SAME train paths, differing only in
    WHERE on the path they sit (at the oracle turn vs away from it), so
    label balance, dose and source distribution are matched;
  - primary metrics are full-path (miss, integral error, P(root within eps))
    with mean|terr| among hits as a secondary, clustered by parent.

Arms (4 seeds 2001-2004, identical init per seed):
  BASE        : stock flipmine recipe
  A_turn      : +40 oracle states in t_star +/- 0.05 windows
  A_offturn   : +40 oracle states at |t - t_star| >= 0.25 (matched sham)
  B_remove_near / C_remove_far : secondary deletion arms, matched on the
      number of removed flip states (not on raw distance alone).

PRE-REGISTERED criterion: A_turn reduces full-path miss OR raises
P(root within 0.05) relative to BOTH BASE and A_offturn, in >=6/8 seeds,
with parent-clustered paired CIs excluding 0 = pinning supported. Else the
pinning hypothesis is not supported by this design.

Outputs: artifacts/next_novelty/l007_pin_v2/PIN_V2.json
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
OUT = ROOT / "artifacts" / "next_novelty" / "l007_pin_v2"
SEEDS = list(range(2001, 2009))
N_INSERT = 40
EPS_LIST = (0.02, 0.05, 0.10)


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
    # ---- eval paths (evaluation only) ----
    de = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    Xe = de["positions"].astype(float)
    ev_paths = n01b.aimed_single_turns(
        Xe, np.random.default_rng(a.seed_eval), a.n_eval_paths,
        seed=a.seed_eval)
    # ---- train pool (all insertion material comes from here) ----
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

    # ---- build matched insert sets from TRAIN parents ----
    rng = np.random.default_rng(20260926)
    train_turns = []          # (x, e, t_star) on train parents
    for k in rng.permutation(len(X)):
        if len(train_turns) >= N_INSERT * 3:
            break
        x = X[k].astype(float)
        try:
            if oracle_at(x)[0] != 0:
                continue
        except ValueError:
            continue
        # aimed edit: move CD toward AB (same construction as n01_brackets)
        d = rng.normal(size=2)
        d /= (np.linalg.norm(d) + 1e-12)
        e = np.zeros((4, 2))
        e[2] = 0.4 * d
        e[3] = 0.4 * d
        tg = np.linspace(0, 1, 65)
        lab = []
        for tt in tg:
            try:
                lab.append(oracle_at(x + tt * e)[0])
            except ValueError:
                lab.append(-1)
        lab = np.array(lab)
        chg = np.where((lab[:-1] == 0) & (lab[1:] == 1))[0]
        if len(chg) == 0:
            continue
        ts = float(tg[chg[0]])
        if ts < 0.1 or ts > 0.9:
            continue
        train_turns.append((x, e, ts))
    train_turns = train_turns[:N_INSERT * 3]
    assert len(train_turns) >= N_INSERT, "not enough train turns"

    def build_insert(mode):
        """mode 'turn': |t-ts|<=0.05 ; 'offturn': |t-ts|>=0.25 (matched)."""
        pts, labs = [], []
        for x, e, ts in train_turns:
            if len(pts) >= N_INSERT:
                break
            for _ in range(20):
                if mode == "turn":
                    tt = float(np.clip(ts + rng.uniform(-0.05, 0.05), 0, 1))
                else:
                    lo = rng.uniform(0, max(1e-6, ts - 0.25))
                    hi = rng.uniform(min(1.0, ts + 0.25), 1.0)
                    tt = float(lo if rng.random() < 0.5 else hi)
                    if abs(tt - ts) < 0.25:
                        continue
                pt = (x + tt * e).astype(np.float32)
                try:
                    l, _, amb = oracle_at(pt)
                except ValueError:
                    continue
                if amb:
                    continue
                pts.append(pt.reshape(-1))
                labs.append(l)
                break
        return np.array(pts), np.array(labs, dtype=np.float32)

    ins_turn = build_insert("turn")
    ins_off = build_insert("offturn")
    # ---- deletion arms matched on removed flip count ----
    dmat = np.array([[np.linalg.norm(x8 - s.reshape(-1))
                      for s in [tt[0] for tt in train_turns]]
                     for x8 in X.reshape(-1, 8)])
    near_order = np.argsort(dmat.min(axis=1))
    far_order = np.argsort(-dmat.min(axis=1))
    flip_count = np.bincount(scenes, minlength=len(X))
    # greedily match cumulative removed flips between the two arms
    def pick(order, target):
        chosen, tot = [], 0
        for i in order:
            if tot >= target:
                break
            chosen.append(int(i))
            tot += int(flip_count[i])
        return set(chosen), tot
    near_set, near_flips = pick(near_order, 20)
    far_set, far_flips = pick(far_order, near_flips)
    groups = {"A_turn": ins_turn, "A_offturn": ins_off,
              "B_remove_near": near_set, "C_remove_far": far_set}
    print("insert sets: turn=%d offturn=%d ; removed flips near=%d far=%d"
          % (len(ins_turn[0]), len(ins_off[0]), near_flips, far_flips),
          flush=True)

    def train_eval(group, seed):
        torch.manual_seed(seed)
        model = CoordMLP(64, 32)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        bce = nn.BCEWithLogitsLoss()
        if group in ("B_remove_near", "C_remove_far"):
            drop = groups[group]
            kc = [i for i in range(len(X)) if i not in drop]
            Xc0, yc0 = X[kc], y[kc]
            kf = [k for k in range(len(scenes)) if scenes[k] not in drop]
            Xf0, yf0 = (X[scenes[kf]] + ee[kf]), yf[kf]
        else:
            add_X, add_y = groups[group]
            Xc0 = np.concatenate([X.reshape(-1, 8), add_X]).reshape(-1, 4, 2)
            yc0 = np.concatenate([y, add_y])
            Xf0, yf0 = (X[scenes] + ee), yf
        Xc = ((torch.from_numpy(Xc0.reshape(-1, 8)) - mu_t) / sd_t)
        yc_t = torch.from_numpy(yc0)
        Xf = ((torch.from_numpy(Xf0.reshape(-1, 8)) - mu_t) / sd_t)
        yf_t = torch.from_numpy(yf0)
        for _ in range(300):
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
            extra = max(0, len(mt) - 1)
            if not same:
                rows.append({"parent": int(ep["parent_id"]), "miss": 1,
                             "integral": integ, "extra": extra,
                             "abserr": np.nan})
                continue
            best = min(same, key=lambda m: abs(m["t_theta"] - ts))
            rows.append({"parent": int(ep["parent_id"]), "miss": 0,
                         "integral": integ, "extra": extra,
                         "abserr": abs(best["t_theta"] - ts)})
        return rows

    def summarize(rows):
        par = np.array([r["parent"] for r in rows])
        miss = np.array([r["miss"] for r in rows], float)
        integ = np.array([r["integral"] for r in rows], float)
        err = np.array([r["abserr"] for r in rows], float)
        out = {"n": len(rows), "n_parents": int(len(np.unique(par))),
               "miss": round(float(miss.mean()), 4),
               "integral": round(float(integ.mean()), 4),
               "mean_terr_hits": round(float(np.nanmean(err)), 4)
               if np.isfinite(err).any() else None,
               "extra_turns": round(float(np.mean([r["extra"] for r in rows])), 4)}
        for eps in EPS_LIST:
            ok = np.isfinite(err) & (err <= eps)
            out["P_root_within_%.2f" % eps] = round(float(ok.mean()), 4)
        return out, par, miss, err

    res = {"n_insert_requested": N_INSERT,
           "insert_built": {"turn": int(len(ins_turn[0])),
                            "offturn": int(len(ins_off[0]))},
           "removed_flips": {"near": int(near_flips), "far": int(far_flips)},
           "seeds": {}}
    for seed in SEEDS:
        sd = {}
        for g in ("BASE", "A_turn", "A_offturn", "B_remove_near", "C_remove_far"):
            rows = train_eval(g, seed)
            s, par, miss, err = summarize(rows)
            sd[g] = {"summary": s, "rows": rows}
            print("s%d %s: %s" % (seed, g, s), flush=True)
        res["seeds"][str(seed)] = {g: sd[g]["summary"] for g in sd}
        res["seeds"][str(seed)]["_rows"] = {g: sd[g]["rows"] for g in sd}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "PIN_V2.json", "w"), indent=1, default=float)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
