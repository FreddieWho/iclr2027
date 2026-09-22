#!/usr/bin/env python3
"""L007-P2 screening: TracIn influence of train scenes on turn-neighborhood
proxy loss, using l007_wave checkpoints. Zero new training.

Proxy: mean BCE over states within +-0.15 of oracle t_star on the same 200
eval paths (oracle-labeled). Influence per supervised train state:
  I(z) = sum_ckpts eta * <grad l(z; th_c), grad L_proxy(th_c)>
eta = 1e-2 constant (training LR). Checkpoints every 50 epochs.
Scores aggregated per train scene (clean state + its flips).

Outputs: artifacts/next_novelty/l007_tracin/TRACIN.json
  {seed: {top_scenes: [...], scores...}} + proxy manifest.
LOO deletion sets are chosen by l007_loo.py from this file.
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
from paths import locate_oracle_turns  # noqa: E402
from coord_mlp import CoordMLP  # noqa: E402
from r04b_methods import mine_flips  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
WAVE = ROOT / "artifacts" / "next_novelty" / "l007_wave"
OUT = ROOT / "artifacts" / "next_novelty" / "l007_tracin_v2"
SEEDS = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]
CKPTS = [0, 50, 100, 150, 200, 250, 300]
ETA = 1e-2


def params_of(model):
    return [p for p in model.parameters() if p.requires_grad]


def grad_vec(model, loss_fn):
    model.zero_grad()
    loss_fn().backward()
    return torch.cat([p.grad.detach().reshape(-1) for p in params_of(model)])


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
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X, y = d["positions"].astype(np.float32), d["labels"].astype(np.float32)
    mu8, sd8 = X.reshape(-1, 8).mean(0), X.reshape(-1, 8).std(0) + 1e-8
    mined = mine_flips(X.astype(float), y.astype(int), seed=5)
    ii = np.array([m["scene"] for m in mined])
    ee = np.array([m["edit"] for m in mined])
    yf = np.array([m["new"] for m in mined], dtype=np.float32)
    Xc = ((X.reshape(-1, 8) - mu8) / sd8).astype(np.float32)
    Xf = (((X[ii] + ee).reshape(-1, 8) - mu8) / sd8).astype(np.float32)
    N = len(X)

    # proxy states: +-0.15 of oracle t_star on eval paths (oracle-labeled)
    de = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    Xe = de["positions"].astype(float)
    ev_paths = n01b.aimed_single_turns(
        Xe, np.random.default_rng(a.seed_eval), a.n_eval_paths,
        seed=a.seed_eval)
    sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
    from paths import oracle_at
    PX, PY = [], []
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
        if len(chg) == 0:
            continue
        ts = tg[chg[0]]
        for tt in tg[np.abs(tg - ts) <= 0.15]:
            try:
                l, _, _ = oracle_at(x + tt * e)
            except ValueError:
                continue
            PX.append((x + tt * e).astype(np.float32))
            PY.append(l)
    PX = np.stack(PX)
    PY = np.array(PY, dtype=np.float32)
    print("proxy states:", len(PX), flush=True)

    bce = nn.BCEWithLogitsLoss()
    out = {"proxy": {"n_states": len(PX), "window": "+-0.15 (dense approx)",
                     "ckpts": CKPTS, "eta": ETA},
           "seeds": {}}
    for seed in SEEDS:
        infl = np.zeros(N + len(ii))
        for ec in CKPTS:
            ck = torch.load(WAVE / ("s%d" % seed) / ("ckpt_%03d.pt" % ec),
                            map_location="cpu", weights_only=False)
            model = CoordMLP(64, 32, in_dim=8)
            model.load_state_dict(ck["state"])
            model.train()
            Xc_t = torch.from_numpy(Xc)
            Xf_t = torch.from_numpy(Xf)
            PX_t = torch.from_numpy(
                ((PX.reshape(len(PX), -1) - mu8) / sd8).astype(np.float32))
            PY_t = torch.from_numpy(PY)
            gp = grad_vec(model, lambda: bce(
                model(PX_t).reshape(-1), PY_t).mean())
            # clean states
            # loss weights: mean clean BCE (1/512) + lam * mean flip BCE
            # (1/1534); per-state gradients must carry those weights or the
            # scene ranking is wrong (review item C8).
            w_clean = 1.0 / float(N)
            w_flip = 1.0 / float(len(ii))
            for i in range(N):
                gi = grad_vec(model, lambda i=i: bce(
                    model(Xc_t[i:i + 1]).reshape(-1),
                    torch.from_numpy(y[i:i + 1])).mean())
                infl[i] += ETA * w_clean * float(gi @ gp)
            # flip states
            yf_t = torch.from_numpy(yf)
            for k in range(len(ii)):
                gk = grad_vec(model, lambda k=k: bce(
                    model(Xf_t[k:k + 1]).reshape(-1),
                    yf_t[k:k + 1]).mean())
                infl[N + k] += ETA * w_flip * float(gk @ gp)
        # aggregate per scene
        scene_score = np.zeros(N)
        scene_score += infl[:N]
        for k, s in enumerate(ii):
            scene_score[s] += infl[N + k]
        order_abs = np.argsort(-np.abs(scene_score))
        order_pos = np.argsort(-scene_score)
        order_neg = np.argsort(scene_score)
        out["seeds"][str(seed)] = {
            "ranked_scenes": [int(v) for v in order_abs[:40]],
            "top_positive": [int(v) for v in order_pos[:20]],
            "top_negative": [int(v) for v in order_neg[:20]],
            "top_abs": round(float(np.abs(scene_score[order_abs[:10]]).mean()), 6),
            "loss_weights": {"clean": round(w_clean, 8),
                             "flip": round(w_flip, 8)},
        }
        print("s%d: top-|I| scenes %s" % (seed, order_abs[:10].tolist()), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out / "TRACIN.json", "w"), indent=1)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
