#!/usr/bin/env python3
"""U01 round2 (revise-once): ONLINE active quartet curriculum.

Round1 showed fixed combo/quartet/random coverage tie (~0.47-0.51, all only
slightly better than clean_only). This tests the actual model-conditioned
hypothesis: train on quartets the CURRENT model gets wrong
(both atomics right + combo wrong), vs equal-count RANDOM emergent quartets.
Same init copy, same total epochs (300), same extra-sample counts per stage;
only the SELECTION differs. Eval: fixed U01 unseen quartets + common cond.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
from common import preprocess
from coord_mlp import CoordMLP

ART = ROOT / "artifacts" / "discovery_campaign"
U01 = ROOT / "artifacts" / "next6_ef0f7a3" / "u01"
STAGES = [50, 100, 150, 200, 250, 300]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    a = p.parse_args()
    rng = np.random.default_rng(45)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(float)
    y = d["labels"].astype(np.float32)
    mu = X.reshape(-1, 8).mean(0)
    feature_std = X.reshape(-1, 8).std(0) + 1e-8
    stats = {"mu": mu.astype(np.float32), "sd": feature_std.astype(np.float32)}

    qt = np.load(U01 / "quartets_train.npz")
    QTR = [(qt[f"q{i}"], qt["meta"][i]) for i in range(len(qt["meta"]))]
    qe = np.load(U01 / "quartets_eval.npz")
    QEV = [(qe[f"q{i}"], qe["meta"][i]) for i in range(len(qe["meta"]))]
    # train-quartet frames (endpoints + atomics), precomputed
    tr_end = np.stack([(q[0] + q[1] + q[2]).astype(np.float32) for q, _ in QTR])
    tr_a = np.stack([(q[0] + q[1]).astype(np.float32) for q, _ in QTR])
    tr_b = np.stack([(q[0] + q[2]).astype(np.float32) for q, _ in QTR])
    Xc_t = torch.from_numpy(((X.reshape(-1, 8) - mu) / feature_std).astype(np.float32))
    yc_t = torch.from_numpy(y)
    ev_end = np.stack([(q[0] + q[1] + q[2]).astype(np.float32) for q, _ in QEV])
    ev_end_y = np.array([m[3] for _, m in QEV])

    def frames_of(model):
        with torch.no_grad():
            pe = (model(preprocess(tr_end, stats)).numpy() > 0).astype(int)
            pa = (model(preprocess(tr_a, stats)).numpy() > 0).astype(int)
            pb = (model(preprocess(tr_b, stats)).numpy() > 0).astype(int)
        return pe, pa, pb

    def run(mode, seed_id, init_state):
        torch.manual_seed(seed_id)
        m = CoordMLP(64, 32)
        m.load_state_dict({k: v.clone() for k, v in init_state.items()})
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        bce = nn.BCEWithLogitsLoss()
        extra_idx = np.zeros(len(QTR), dtype=bool)
        active_sizes = []
        m.train()
        prev = 0
        for stop in STAGES:
            for _ in range(stop - prev):
                opt.zero_grad()
                loss = bce(m(Xc_t), yc_t)
                if extra_idx.sum():
                    Xe_ = torch.from_numpy(
                        ((tr_end[extra_idx].reshape(-1, 8) - mu) / feature_std).astype(np.float32))
                    ye_ = torch.from_numpy(
                        np.array([QTR[i][1][3] for i in np.nonzero(extra_idx)[0]],
                                 dtype=np.float32))
                    loss = loss + bce(m(Xe_), ye_)
                loss.backward()
                opt.step()
            prev = stop
            if stop == STAGES[-1]:
                break
            pe, pa, pb = frames_of(m)
            ya = np.array([t[1][1] for t in QTR])
            yb = np.array([t[1][2] for t in QTR])
            yc = np.array([t[1][3] for t in QTR])
            wrong_active = (pa == ya) & (pb == yb) & (pe != yc)
            k = int(wrong_active.sum())
            active_sizes.append(k)
            if k == 0:
                extra_idx = np.zeros(len(QTR), dtype=bool)
                continue
            if mode == "active":
                extra_idx = wrong_active
            else:  # control: equal-count random emergent endpoints
                extra_idx = np.zeros(len(QTR), dtype=bool)
                extra_idx[rng.choice(len(QTR), size=k, replace=False)] = True
        m.eval()
        with torch.no_grad():
            pe = (m(preprocess(ev_end, stats)).numpy() > 0).astype(int)
        return {"emergent_miss": float((pe != ev_end_y).mean()),
                "active_sizes": active_sizes}

    res = {"n_train_quartets": len(QTR), "n_eval_quartets": len(QEV)}
    for seed_id in a.seeds:
        torch.manual_seed(seed_id)
        init_state = CoordMLP(64, 32).state_dict()
        for mode in ("active", "control"):
            r = run(mode, seed_id, init_state)
            res[f"{mode}_s{seed_id}"] = r
            print(f"SAW {mode} s{seed_id}: emerg={r['emergent_miss']:.3f} "
                  f"active_sizes={r['active_sizes']}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print("DONE U01b")


if __name__ == "__main__":
    main()
