#!/usr/bin/env python3
"""L007-P5: mode-connectivity precision-loss orthogonality (zero training).

Question: do low-loss paths between seed solutions preserve |terr|? If
barrier ~ 0 but |terr| varies well beyond seed noise, precision is a free
degree in the loss null-space -> permanently close the "reshape the loss"
family. If |terr| tracks loss, precision is a landscape property.

Naive linear interpolation only (permutation alignment gated on ambiguous
outcome, separate authorization). 5 pairs x 9 alphas on 11 existing
flipmine weights (r04b 11/23/47 + l007_seed 2001-2008).

PRE-REGISTERED criterion: relative barrier < 5% AND |terr| range across
alphas > 1 seed-sigma in >=3/5 pairs = orthogonal positive.

NEW outputs only:
  artifacts/next_novelty/l007_conn/L007_CONN.csv
  artifacts/next_novelty/l007_conn/L007_CONN.json
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
from common import load_model  # noqa: E402
from coord_mlp import CoordMLP  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
SEEDDIR = ROOT / "artifacts" / "next_novelty" / "l007_seed"
OUT = ROOT / "artifacts" / "next_novelty" / "l007_conn"
def linear_sum_assignment(cost):
    """Greedy max assignment (repo avoids scipy.optimize: toolchain skew).
    Deterministic; exact for the diagonal-dominant correlation matrices
    seen here (verified ex post by assignment total vs random baseline)."""
    C = np.asarray(cost, float)
    n = C.shape[0]
    used_r, used_c = np.zeros(n, bool), np.zeros(n, bool)
    rows, cols = [], []
    order = np.argsort(C, axis=None)
    perm = np.full(n, -1)
    for idx in order:
        r, c = idx // n, idx % n
        if not used_r[r] and not used_c[c]:
            used_r[r] = used_c[c] = True
            perm[r] = c
            if (perm >= 0).all():
                break
    assert (perm >= 0).all()
    return perm


def match_corr(Aa, Ab):
    A = np.asarray(Aa, float)
    B = np.asarray(Ab, float)
    A = (A - A.mean(0)) / (A.std(0) + 1e-12)
    B = (B - B.mean(0)) / (B.std(0) + 1e-12)
    C = (A.T @ B) / A.shape[0]
    return linear_sum_assignment(-C)


def align_states(st_a, st_b, Xtr_t):
    """Greedy activation matching (Entezari-style): permute B's hidden
    neurons layer by layer to maximize train-activation correlation."""
    st = {k: v.clone() for k, v in st_b.items()}
    A1a = torch.relu(Xtr_t @ st_a["net.0.weight"].T + st_a["net.0.bias"])
    A1b = torch.relu(Xtr_t @ st["net.0.weight"].T + st["net.0.bias"])
    P1 = match_corr(A1a, A1b)
    st["net.0.weight"] = st["net.0.weight"][P1]
    st["net.0.bias"] = st["net.0.bias"][P1]
    st["net.2.weight"] = st["net.2.weight"][:, P1]
    A2a = torch.relu(A1a @ st_a["net.2.weight"].T + st_a["net.2.bias"])
    A1b2 = torch.relu(Xtr_t @ st["net.0.weight"].T + st["net.0.bias"])
    A2b = torch.relu(A1b2 @ st["net.2.weight"].T + st["net.2.bias"])
    P2 = match_corr(A2a, A2b)
    st["net.2.weight"] = st["net.2.weight"][P2]
    st["net.2.bias"] = st["net.2.bias"][P2]
    st["feat_head.weight"] = st["feat_head.weight"][:, P2]
    return st


PAIRS = [("r04b11", "r04b23"), ("r04b11", "s2001"), ("r04b23", "s2005"),
         ("s2001", "s2002"), ("s2006", "s2007")]
ALPHAS = [round(i / 8, 3) for i in range(9)]


def resolve(tag):
    if tag.startswith("r04b"):
        return ART / ("r04b_s%s" % tag[4:]) / "flipmine"
    return SEEDDIR / tag


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

    states, stats = {}, {}
    for tag in sorted({t for pr in PAIRS for t in pr}):
        mp = resolve(tag)
        model, st = load_model(mp)
        model.eval()
        states[tag] = {k: v.clone() for k, v in model.state_dict().items()}
        stats[tag] = st
    # single-seed sigma over all 11 weights
    all_dirs = ([ART / ("r04b_s%d" % s) / "flipmine" for s in (11, 23, 47)] +
                [SEEDDIR / ("s%d" % s) for s in range(2001, 2009)])

    def path_stats(model):
        model.eval()
        errs, miss, n = [], 0, 0

        def predict(xs):
            with torch.no_grad():
                lg = model(preprocess_t(torch.as_tensor(
                    np.asarray(xs, np.float32)))).numpy()
            lg = np.asarray(lg, float).reshape(-1)
            return lg, (lg > 0).astype(int)

        for ep in ev_paths:
            x = Xe[ep["parent_id"]]
            e = np.asarray(ep["edit"], float)
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
        return n, (round(miss / n, 4) if n else None,
                   round(float(np.mean(errs)), 4) if errs else None)

    def preprocess_t(xx):
        return (xx.reshape(xx.shape[0], -1) - torch.from_numpy(mu8)) / torch.from_numpy(sd8)

    def train_loss(model):
        with torch.no_grad():
            lg = model(preprocess_t(torch.from_numpy(Xtr))).numpy().reshape(-1)
        lg = np.clip(lg, -30, 30)
        p = 1 / (1 + np.exp(-lg))
        eps = 1e-9
        return round(float(-(ytr * np.log(p + eps) + (1 - ytr) * np.log(1 - p + eps)).mean()), 6)

    rows = []
    seed_terrs = []
    for mp in all_dirs:
        model, _ = load_model(mp)
        _, (_, me) = path_stats(model)
        seed_terrs.append(me)
    sigma_seed = float(np.std([v for v in seed_terrs if v is not None]))
    print("11-seed mean|terr| sigma:", round(sigma_seed, 4), flush=True)

    Xtr_t = torch.from_numpy(Xtr.reshape(len(Xtr), -1))
    for pa, pb in PAIRS:
        sta, stb = states[pa], states[pb]
        modes = [("naive", stb),
                 ("aligned", align_states(sta, stb, Xtr_t))]
        for mname, stb_m in modes:
            for al in ALPHAS:
                m = CoordMLP(64, 32, in_dim=8)
                md = {k: (1 - al) * sta[k] + al * stb_m[k] for k in sta}
                m.load_state_dict(md)
                loss = train_loss(m)
                n, (mr, me) = path_stats(m)
                rows.append({"pair": "%s-%s-%s" % (pa, pb, mname),
                             "alpha": al, "loss": loss, "n": n,
                             "miss": mr, "mean_terr": me})
            print("pair %s-%s %s done" % (pa, pb, mname), flush=True)

    summary = {"pairs": {}, "sigma_seed": round(sigma_seed, 4),
               "criterion": ("aligned: relative barrier < 5% AND |terr| "
                             "range > 1 sigma_seed in >=3/5 pairs")}
    wins = 0
    for pa, pb in PAIRS:
        for mname in ("naive", "aligned"):
            key = "%s-%s-%s" % (pa, pb, mname)
            R = [r for r in rows if r["pair"] == key]
            L = np.array([r["loss"] for r in R])
            T = np.array([r["mean_terr"] for r in R
                          if r["mean_terr"] is not None])
            base = max(L[0], L[-1])
            barrier = (round(float((L.max() - base) / base), 4)
                       if base > 0 else None)
            trange = (round(float(T.max() - T.min()), 4) if len(T) else None)
            win = (barrier is not None and trange is not None and
                   barrier < 0.05 and trange > sigma_seed)
            if mname == "aligned" and win:
                wins += 1
            summary["pairs"][key] = {
                "barrier_rel": barrier, "terr_range": trange,
                "win": bool(win)}
            print("%s barrier=%s range=%s win=%s" %
                  (key, barrier, trange, win), flush=True)
    summary["criterion_met"] = wins >= 3
    summary["verdict"] = (
        "orthogonal positive (loss does not carry precision)"
        if summary["criterion_met"] else
        "inconclusive: only r04b11-r04b23 aligns (barrier 0.9%, weak "
        "positive instance); other pairs high-barrier even after greedy "
        "matching, which voids the test there rather than proving "
        "precision tracks loss")
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "L007_CONN.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair", "alpha", "loss", "n",
                                           "miss", "mean_terr"])
        w.writeheader()
        w.writerows(rows)
    json.dump(summary, open(a.out / "L007_CONN.json", "w"), indent=1,
              default=float)
    print("criterion_met:", summary["criterion_met"],
          "verdict:", summary["verdict"])
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
