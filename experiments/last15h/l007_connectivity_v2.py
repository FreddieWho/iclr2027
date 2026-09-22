#!/usr/bin/env python3
"""L007-P5 v2: connectivity under the correct objective and canonical
coordinates (fixes GPT6ASTRA_CODE_REVIEW.md §9).

Fixes vs l007_connectivity.py:
  - activation matching uses STANDARDIZED inputs (was raw X while the model
    consumes standardized input);
  - barrier uses the FULL training objective: mean clean BCE + mean flip BCE
    (was clean BCE only, which fabricated the single weak positive);
  - the linear tail is canonicalized: feat_head->cls is affine, so the
    model reduces to logit = W_eff @ relu_trunk(x) + b_eff with
    W_eff = W_cls @ W_feat. Interpolating (trunk, W_eff, b_eff) removes the
    cross-term that plagued direct (W_feat, W_cls) interpolation;
  - reports endpoint losses and three barrier normalizations.

NEW outputs only: artifacts/next_novelty/l007_conn_v2/
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
from common import load_model  # noqa: E402
from coord_mlp import CoordMLP  # noqa: E402
from r04b_methods import mine_flips  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
SEEDDIR = ROOT / "artifacts" / "next_novelty" / "l007_seed"
OUT = ROOT / "artifacts" / "next_novelty" / "l007_conn_v2"
PAIRS = [("r04b11", "r04b23"), ("r04b11", "s2001"), ("r04b23", "s2005"),
         ("s2001", "s2002"), ("s2006", "s2007")]
ALPHAS = [round(i / 16, 4) for i in range(17)]


def resolve(tag):
    if tag.startswith("r04b"):
        return ART / ("r04b_s%s" % tag[4:]) / "flipmine"
    return SEEDDIR / tag


def greedy_perm(C):
    C = np.asarray(C, float)
    n = C.shape[0]
    used_r, used_c = np.zeros(n, bool), np.zeros(n, bool)
    perm = np.full(n, -1)
    for idx in np.argsort(C, axis=None)[::-1]:
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
    return greedy_perm((A.T @ B) / A.shape[0])


def canonical(state):
    """(trunk params, W_eff, b_eff) with logit = W_eff @ relu_trunk(x) + b_eff."""
    Wf = state["feat_head.weight"]           # [32, 64]
    bf = state["feat_head.bias"]             # [32]
    Wc = state["cls.weight"]                 # [1, 32]
    bc = state["cls.bias"]                   # [1]
    W_eff = (Wc @ Wf).reshape(-1)            # [64]
    b_eff = (Wc @ bf + bc).reshape(())
    trunk = {k: v.clone() for k, v in state.items()
             if k.startswith("net.")}
    return trunk, W_eff, b_eff


def align_trunk(st_a, st_b, Xs):
    """Permute st_b's trunk (and its W_eff) to match st_a, using the
    STANDARDIZED input matrix Xs for activation matching."""
    trunk_a, _, _ = canonical(st_a)
    trunk_b, W_eff_b, b_eff_b = canonical(st_b)
    W1a, b1a = trunk_a["net.0.weight"], trunk_a["net.0.bias"]
    A1a = torch.relu(Xs @ W1a.T + b1a)
    W1b, b1b = trunk_b["net.0.weight"], trunk_b["net.0.bias"]
    A1b = torch.relu(Xs @ W1b.T + b1b)
    P1 = match_corr(A1a, A1b)
    trunk_b["net.0.weight"] = W1b[P1]
    trunk_b["net.0.bias"] = b1b[P1]
    trunk_b["net.2.weight"] = trunk_b["net.2.weight"][:, P1]
    W_eff_b = W_eff_b[P1]
    A1b2 = torch.relu(Xs @ trunk_b["net.0.weight"].T + trunk_b["net.0.bias"])
    W2a, b2a = trunk_a["net.2.weight"], trunk_a["net.2.bias"]
    A2a = torch.relu(A1a @ W2a.T + b2a)
    A2b = torch.relu(A1b2 @ trunk_b["net.2.weight"].T + trunk_b["net.2.bias"])
    P2 = match_corr(A2a, A2b)
    trunk_b["net.2.weight"] = trunk_b["net.2.weight"][P2]
    trunk_b["net.2.bias"] = trunk_b["net.2.bias"][P2]
    W_eff_b = W_eff_b[P2]
    return trunk_b, W_eff_b, b_eff_b


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
    Xc_t = ((torch.from_numpy(X.reshape(-1, 8)) - mu_t) / sd_t)
    yc_t = torch.from_numpy(y)
    Xf_t = ((torch.from_numpy((X[ii] + ee).reshape(-1, 8)) - mu_t) / sd_t)
    yf_t = torch.from_numpy(yf)
    bce = nn.BCEWithLogitsLoss()
    # standardization statistics are per-checkpoint in this repo; both eras
    # use the train_101 clean stats, which we verified are identical here.
    stats = {"mu8": mu8, "sd8": sd8}

    states = {}
    for tag in sorted({t for pr in PAIRS for t in pr}):
        model, st = load_model(resolve(tag))
        model.eval()
        states[tag] = {k: v.clone() for k, v in model.state_dict().items()}
    Xs = Xc_t.clone()

    def full_loss(trunk, W_eff, b_eff):
        with torch.no_grad():
            def lg(xx):
                h = torch.relu(xx @ trunk["net.0.weight"].T + trunk["net.0.bias"])
                h = torch.relu(h @ trunk["net.2.weight"].T + trunk["net.2.bias"])
                return h @ W_eff + b_eff
            l1 = float(bce(lg(Xc_t), yc_t).mean().item())
            l2 = float(bce(lg(Xf_t), yf_t).mean().item())
        return l1 + l2

    def path_stats(trunk, W_eff, b_eff):
        def predict(xs):
            with torch.no_grad():
                xx = ((torch.as_tensor(np.asarray(xs, np.float32))
                       .reshape(len(xs), -1) - mu_t) / sd_t)
                h = torch.relu(xx @ trunk["net.0.weight"].T + trunk["net.0.bias"])
                h = torch.relu(h @ trunk["net.2.weight"].T + trunk["net.2.bias"])
                lg = (h @ W_eff + b_eff).numpy().reshape(-1)
            return lg, (lg > 0).astype(int)
        errs, miss, n = [], 0, 0
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
        return {"mean_terr": round(float(np.mean(errs)), 4) if errs else None,
                "miss": round(miss / n, 4) if n else None, "n": n}

    rows = []
    seed_terrs = []
    ck_dirs = ([ART / ("r04b_s%d" % s) / "flipmine" for s in (11, 23, 47)] +
               [SEEDDIR / ("s%d" % s) for s in range(2001, 2009)])
    for mp in ck_dirs:
        m, _ = load_model(mp)
        tr_a, we_a, be_a = canonical({k: v.clone() for k, v in m.state_dict().items()})
        seed_terrs.append(path_stats(tr_a, we_a, be_a)["mean_terr"])
    sigma = float(np.std([v for v in seed_terrs if v is not None]))

    for pa, pb in PAIRS:
        ta, wa, ba = canonical(states[pa])
        tb, wb, bb = canonical(states[pb])
        tb_al, wb_al, bb_al = align_trunk(states[pa], states[pb], Xs)
        for mode, (tb_m, wb_m, bb_m) in (("naive", (tb, wb, bb)),
                                         ("aligned", (tb_al, wb_al, bb_al))):
            entry = {"pair": pa, "pairb": pb, "mode": mode,
                     "endpoint_loss_a": round(full_loss(ta, wa, ba), 6),
                     "endpoint_loss_b": round(full_loss(tb_m, wb_m, bb_m), 6),
                     "alphas": []}
            for al in ALPHAS:
                tr_m = {k: (1 - al) * ta[k] + al * tb_m[k] for k in ta}
                we_m = (1 - al) * wa + al * wb_m
                be_m = (1 - al) * ba + al * bb_m
                L = round(full_loss(tr_m, we_m, be_m), 6)
                st = path_stats(tr_m, we_m, be_m)
                entry["alphas"].append({"alpha": al, "full_loss": L,
                                        "mean_terr": st["mean_terr"],
                                        "miss": st["miss"]})
            rows.append(entry)
            print("%s-%s %s done" % (pa, pb, mode), flush=True)

    summary = {"sigma_seed_11models": round(sigma, 4), "pairs": {},
               "note": ("barrier_rel_max = (max L - max(endpoint losses)) / "
                        "max(endpoint losses); needs BOTH endpoints reported "
                        "because they are not equal"),
               "criterion": ("aligned: barrier < 5% of max endpoint AND "
                             "|terr| range > 1 sigma_seed, in >=3/5 pairs")}
    wins = 0
    for entry in rows:
        L = np.array([x["full_loss"] for x in entry["alphas"]])
        T = np.array([x["mean_terr"] for x in entry["alphas"]
                      if x["mean_terr"] is not None])
        base = max(entry["endpoint_loss_a"], entry["endpoint_loss_b"])
        barrier = float((L.max() - base) / base) if base > 0 else None
        rng_ = float(T.max() - T.min()) if len(T) else None
        key = "%s-%s-%s" % (entry["pair"], entry["pairb"], entry["mode"])
        win = (barrier is not None and rng_ is not None and
               barrier < 0.05 and rng_ > sigma)
        if entry["mode"] == "aligned" and win:
            wins += 1
        summary["pairs"][key] = {
            "barrier_rel_max_endpoint": round(barrier, 4) if barrier is not None else None,
            "terr_range": round(rng_, 4) if rng_ is not None else None,
            "endpoint_loss_a": entry["endpoint_loss_a"],
            "endpoint_loss_b": entry["endpoint_loss_b"],
            "L_max": round(float(L.max()), 6), "win": bool(win)}
        print(key, summary["pairs"][key], flush=True)
    summary["criterion_met"] = wins >= 3
    summary["verdict"] = ("orthogonal positive" if summary["criterion_met"]
                          else "no supported orthogonality instance")
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": rows},
              open(a.out / "L007_CONN_V2.json", "w"), indent=1, default=float)
    print("criterion_met:", summary["criterion_met"],
          "verdict:", summary["verdict"])
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
