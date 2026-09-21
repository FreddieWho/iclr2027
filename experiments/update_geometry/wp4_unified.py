#!/usr/bin/env python3
"""WP4: failure propagation on a unified eval set (frozen checkpoints, eval only).

6 arms (r04b_s11 family) x same eval_202 scenes. Model-level: static_err,
transition miss (first valid flip, margin>=0.03), composition cond miss
(N04-style), action invalid lam=2 (N06-style). Scene-level: per feasible
action scene, static_correct + transition hit/miss (first flip from that
scene) + action valid/invalid -> risk diff/OR + logistic control with
parent-cluster bootstrap. No SEM (n=6 < 15): Spearman + scatter only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(4)
from paths import action_library, oracle_at, atomic_edits  # noqa: E402
from common import load_model, preprocess  # noqa: E402
from r02_search import candidates_for_scene  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
ARMS = ["clean", "flipmine", "fliprand", "supmine", "auxmargin", "relfeat"]
rng = np.random.default_rng(26092205)


def first_flip(x, frng):
    for item in candidates_for_scene(x, frng, n_random=32):
        e = np.asarray(item[0], dtype=float)
        try:
            y0, _, _ = oracle_at(x)
            y1, m1, _ = oracle_at(x + e)
        except ValueError:
            continue
        if y1 != y0 and m1 >= 0.03:
            return e, y1
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_act", type=int, default=256)
    p.add_argument("--n_comp_parents", type=int, default=256)
    a = p.parse_args()
    d = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    X = d["positions"].astype(np.float32)
    lib = action_library()
    loaded = {}
    for k in ARMS:
        m, s = load_model(ART / "r04b_s11" / k)
        m.eval()
        loaded[k] = (m, s)

    def pred(m, s, xs):
        xs = np.asarray(xs, dtype=np.float32)
        with torch.no_grad():
            out = []
            for i in range(0, len(xs), 256):
                out.append(m(preprocess(xs[i:i + 256], s)).numpy())
        lg = np.concatenate(out)
        return lg, (lg > 0).astype(int)

    # static err on full pool
    lg_all = {k: pred(m, s, X)[1] for k, (m, s) in loaded.items()}
    y_all = np.array([oracle_at(x)[0] for x in X])
    static_err = {k: round(float((lg_all[k] != y_all).mean()), 4) for k in ARMS}

    # transition miss: first valid flip per parent
    trans = {}
    for k, (m, s) in loaded.items():
        miss = n = 0
        frng = np.random.default_rng(4)
        for pi in range(len(X)):
            x = X[pi].astype(float)
            e, y1 = first_flip(x, frng)
            if e is None:
                continue
            p1 = pred(m, s, (x + e)[None])[1][0]
            n += 1
            miss += (p1 != y1)
        trans[k] = {"miss": round(miss / n, 4) if n else None, "n": n}

    # composition cond miss (N04-style, capped parents)
    comp = {}
    order = np.random.default_rng(4).permutation(len(X))[:a.n_comp_parents]
    for k, (m, s) in loaded.items():
        cn = ce = 0
        erng = np.random.default_rng(4)
        for pi in order:
            x = X[pi].astype(float)
            try:
                y0, _, _ = oracle_at(x)
            except ValueError:
                continue
            res = []
            for e, _ in atomic_edits(x, erng):
                try:
                    y, _, _ = oracle_at(x + e)
                except ValueError:
                    continue
                res.append((e, y))
            for i in range(len(res)):
                for j in range(i + 1, len(res)):
                    (ea, ya), (eb, yb) = res[i], res[j]
                    if ya != y0 or yb != y0:
                        continue
                    try:
                        yc, _, _ = oracle_at(x + ea + eb)
                    except ValueError:
                        continue
                    if yc == y0:
                        continue
                    pa = pred(m, s, (x + ea)[None])[1][0]
                    pb = pred(m, s, (x + eb)[None])[1][0]
                    if pa == ya and pb == yb:
                        pc = pred(m, s, (x + ea + eb)[None])[1][0]
                        cn += 1
                        ce += (pc != yc)
        comp[k] = {"cond_miss": round(ce / cn, 4) if cn else None, "n": cn}

    # action invalid lam=2 on unified scene list + per-scene records
    cross = [i for i in np.random.default_rng(6).permutation(len(X))[:a.n_act * 2]
             if oracle_at(X[i])[0] == 1][:a.n_act]
    act = {}
    scenes = {k: [] for k in ARMS}
    for k, (m, s) in loaded.items():
        inv = tot = feas = 0
        for pi in cross:
            x = X[pi].astype(float)
            cands = []
            for act_ in lib:
                try:
                    y, _, _ = oracle_at(x + act_["edit"])
                except ValueError:
                    continue
                cands.append((act_, y))
            feas_oracle = [c for c in cands if c[1] == 0]
            if not feas_oracle:
                continue
            feas += 1
            xs = np.stack([(x + c[0]["edit"]).astype(np.float32) for c in cands])
            lg, _ = pred(m, s, xs)
            prob0 = 1 / (1 + np.exp(lg))
            scores = prob0 - 2.0 * np.array([c[0]["cost"] for c in cands])
            ystar = cands[int(np.argmax(scores))][1]
            tot += 1
            inv += (ystar != 0)
            scenes[k].append({"parent_id": int(pi), "action_invalid": int(ystar != 0),
                              "static_correct": int(pred(m, s, x[None])[1][0] == 1)})
        act[k] = {"invalid": round(inv / tot, 4) if tot else None, "feas": feas}

    # scene-level: transition proxy per action scene (first flip from that scene)
    for k, (m, s) in loaded.items():
        frng = np.random.default_rng(7)
        for rec in scenes[k]:
            x = X[rec["parent_id"]].astype(float)
            e, y1 = first_flip(x, frng)
            if e is None:
                rec["trans_miss"] = None
            else:
                rec["trans_miss"] = int(pred(m, s, (x + e)[None])[1][0] != y1)

    a.out.mkdir(parents=True, exist_ok=True)
    model_table = {k: {"static_err": static_err[k],
                       "trans_miss": trans[k]["miss"], "trans_n": trans[k]["n"],
                       "comp_cond": comp[k]["cond_miss"], "comp_n": comp[k]["n"],
                       "action_invalid": act[k]["invalid"],
                       "action_feas": act[k]["feas"]} for k in ARMS}
    json.dump({"model_table": model_table, "scenes": scenes},
              open(a.out / "rows.json", "w"))

    def spear(x, y):
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return round(float(np.corrcoef(rx, ry)[0, 1]), 4)

    mt = model_table
    K = ARMS
    res = {"model_table": mt,
           "spearman": {
               "static_vs_action": spear([mt[k]["static_err"] for k in K],
                                         [mt[k]["action_invalid"] for k in K]),
               "trans_vs_action": spear([mt[k]["trans_miss"] for k in K],
                                        [mt[k]["action_invalid"] for k in K]),
               "comp_vs_action": spear([mt[k]["comp_cond"] for k in K],
                                       [mt[k]["action_invalid"] for k in K])}}

    # scene-level pooled 2x2 + risk diff/OR + cluster bootstrap; logistic control
    all_rec = [r for k in K for r in scenes[k] if r["trans_miss"] is not None]
    a1 = np.array([r["action_invalid"] for r in all_rec if r["trans_miss"] == 1])
    a0 = np.array([r["action_invalid"] for r in all_rec if r["trans_miss"] == 0])
    pars = {}
    for r in all_rec:
        pars.setdefault(r["parent_id"], []).append(r)
    P = list(pars.values())

    def rd(sample):
        A = [r for q in sample for r in q]
        m1 = np.mean([r["action_invalid"] for r in A if r["trans_miss"] == 1])
        m0 = np.mean([r["action_invalid"] for r in A if r["trans_miss"] == 0])
        return m1 - m0

    obs = rd(P)
    boots = [rd([P[i] for i in rng.integers(0, len(P), len(P))]) for _ in range(10000)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p1 = float(a1.mean())
    p0 = float(a0.mean())
    OR = (p1 / (1 - p1)) / (p0 / (1 - p0)) if 0 < p0 < 1 and 0 < p1 < 1 else None
    res["scene_level"] = {
        "P_invalid_given_transmiss": round(p1, 4), "n1": int(len(a1)),
        "P_invalid_given_transhit": round(p0, 4), "n0": int(len(a0)),
        "risk_diff": round(float(obs), 4),
        "risk_diff_ci": [round(float(lo), 4), round(float(hi), 4)],
        "odds_ratio": round(float(OR), 4) if OR else None}
    try:
        from sklearn.linear_model import LogisticRegression
        Xa = np.array([[r["static_correct"], r["trans_miss"]] for r in all_rec])
        ya = np.array([r["action_invalid"] for r in all_rec])
        lr = LogisticRegression().fit(Xa, ya)
        res["scene_level"]["logistic_coef"] = [round(float(c), 4) for c in lr.coef_[0]]
    except ImportError:
        res["scene_level"]["logistic_coef"] = None
    json.dump(res, open(a.out / "WP4_RESULTS.json", "w"), indent=1)
    print(f"SAW WP4: {json.dumps(res, indent=1)[:2500]}")


if __name__ == "__main__":
    main()
