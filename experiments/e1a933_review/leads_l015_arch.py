"""Learned G8 segment-set and S4 sum-pool controls for L-015.

Same train bank, flip mine seed, epoch budget and dev512 3-state J as the
feature/augmentation run. Does not read sealed pools.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "T", ROOT / "experiments" / "e1a933_review" / "leads_l014_l015_l006.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
spec2 = importlib.util.spec_from_file_location(
    "R", ROOT / "experiments" / "e1a933_review" / "leads_l015_reeval.py")
R = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(R)

ARMS = ("g8set", "s4set")


def nparams(arm):
    return sum(p.numel() for p in T.build(arm).parameters())


def main():
    out = Path(sys.argv[1])
    train = np.load(ROOT / "artifacts/f095_campaign/D01/scenes_N/scenes.npz")
    xtr = train["positions"].astype(np.float32)
    ytr = train["labels"].astype(np.int64)
    flips = T.mine_flips(xtr, ytr, seed=5, cap=8)
    Qx, Qe, quads = R.load_quads()
    ident = T.G8[0]
    rows = []
    for arm in ARMS:
        stats = T.fit_stats(arm, xtr)
        for seed in T.SEEDS:
            model, ck = T.train_one(arm, seed, xtr, ytr, flips, stats, out)
            model.eval()
            ok, parents = R.j_vec(model, arm, stats, Qx, Qe, quads, ident)
            js, changed = [], []
            gap = 0.0
            xs = []
            for qid, (i1, _m1), (i2, _m2) in quads:
                xs += [Qx[i1], Qx[i2], Qx[i1] + Qe[i1]]
            base = T.predict(model, arm, np.stack(xs), stats)
            for perm in T.G8:
                v, _ = R.j_vec(model, arm, stats, Qx, Qe, quads, perm)
                js.append(float(v.mean()))
                if perm != ident:
                    changed.append(float(np.mean(v != ok)))
                idx = list(perm)
                Xp, Ep = Qx[:, idx], Qe[:, idx]
                ys = []
                for qid, (i1, _m1), (i2, _m2) in quads:
                    ys += [Xp[i1], Xp[i2], Xp[i1] + Ep[i1]]
                gap = max(gap, float(np.max(np.abs(T.predict(model, arm, np.stack(ys), stats) - base))))
            row = {
                "arm": arm, "seed": int(seed), "J": float(ok.mean()),
                "J_min": float(min(js)), "J_max": float(max(js)),
                "J_swing": float(max(js) - min(js)),
                "status_flip_mean": float(np.mean(changed)),
                "logit_gap": gap,
                "n": int(len(ok)), "n_parents": int(len(np.unique(parents))),
                "seconds": float(ck["seconds"]), "n_params": nparams(arm),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    payload = {
        "arms": list(ARMS),
        "n_params": {a: nparams(a) for a in ("raw", "setmlp", "relfeat", "g8set", "s4set")},
        "rows": rows,
        "note": "S4 sum-pool is a negative control: invariant under regroupings that change the label.",
    }
    (out / "ARCH_REEVAL.json").write_text(json.dumps(payload, indent=2))
    print("wrote", out / "ARCH_REEVAL.json")


if __name__ == "__main__":
    main()
