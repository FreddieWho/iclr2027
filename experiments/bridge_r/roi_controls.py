#!/usr/bin/env python3
"""Bridge-R Task 2 closure controls (Am03): R5 leakage diagnostics.

R5b position-only: 2-dim normalized ROI coords (row/13, col/13), same
fixed Am03 LR recipe, no DINO features. Tests whether ROI position alone
carries the label.

R5c fixed-base ROI: one ROI computed from the BASE state per quartet,
applied to all four states (no per-state re-localization). Tests whether
ROI rescue survives without the selector re-aiming per state.

Dev diagnostic only. Holdout never loaded.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "bridge_r"))
from spatial_readout import (load_shards, evaluate, seg_closest_mid,
                             scene_to_patch, roi_features, LR_RECIPE,
                             TRAIN_N, FEAT, PGRID, ART, OUT)

KINDS = ["base", "A", "B", "AB"]


def full_states(split):
    """Per-state full (4,2) coordinates in render order + kinds + qid."""
    q = np.load(ART / f"quartets_{split}_v2.npz")
    nq = len(q["meta"])
    S = np.empty((nq * 4, 4, 2))
    K = np.tile(KINDS, nq)
    Q = np.repeat(np.arange(nq), 4)
    for qi in range(nq):
        x, ea, eb = q[f"q{qi}"].reshape(3, 4, 2)
        S[qi * 4:qi * 4 + 4] = [x, x + ea, x + eb, x + ea + eb]
    return S, K, Q


def roi_coords(S):
    return np.array([scene_to_patch(seg_closest_mid(s)) for s in S])


def run_lr(Ztr, ytr, Zdv):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ztr)
    Xtr = sc.transform(Ztr).astype(np.float64)
    Xdv = sc.transform(Zdv).astype(np.float64)
    clf = LogisticRegression(C=LR_RECIPE["C"], penalty=LR_RECIPE["penalty"],
                             solver=LR_RECIPE["solver"], tol=LR_RECIPE["tol"],
                             max_iter=LR_RECIPE["max_iter"])
    clf.fit(Xtr, ytr)
    return clf.predict(Xdv), float(clf.score(Xtr, ytr))


def save(bb, name, pred, ydev, kdev, qdev, uqdev, train_acc, dim, note):
    import sklearn
    dev = evaluate(pred, ydev, kdev, qdev)
    b1 = (dev["atomic_mean"] >= 0.85 and dev["both_frac"] >= 0.60
          and dev["both_n"] >= 100)
    res = {"backbone": bb, "readout": name, "probe": "lr-fixed-Am03",
           "lr_recipe": LR_RECIPE, "sklearn_version": sklearn.__version__,
           "train_n_quartets": TRAIN_N, "train_acc": train_acc, "dev": dev,
           "gate_B1": "ATOMIC_ACCESSIBILITY_RESCUED" if b1 else "FAIL"}
    od = OUT / f"{bb}_{name}"
    od.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(od / "result.json", "w"), indent=1)
    np.savez(od / "predictions.npz", pred=pred.astype(np.int64),
             labels=ydev.astype(np.int64), kinds=kdev, qid=qdev.astype(np.int64))
    json.dump({"backbone": bb, "readout": name, "probe": "lr-fixed-Am03",
               "train_n_quartets": TRAIN_N, "n_dev_quartets": len(uqdev),
               "feature_dim": dim, "holdout_touched": False, "note": note},
              open(od / "manifest.json", "w"), indent=1)
    print("SAW", bb, name, "atomic=%.4f" % dev["atomic_mean"], "B1=", res["gate_B1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["dinov3s", "dinov3b", "dinov3l"],
                    required=True)
    a = ap.parse_args()
    bb = a.backbone
    _, ktr, ytr, qtr, uqtr = load_shards(str(FEAT / f"feats_{bb}_train.shard*.npz"))
    _, kdev, ydev, qdev, uqdev = load_shards(str(FEAT / f"feats_{bb}_dev.shard*.npz"))
    keep_ids = uqtr[:TRAIN_N]
    keep = np.isin(qtr, keep_ids)
    Str, Ktr, Qtr = full_states("train")
    Sdev, Kdev, Qdev = full_states("dev")
    assert (Ktr == ktr).all() and (Kdev == kdev).all()
    assert (Qtr == qtr).all() and (Qdev == qdev).all()

    # R5b: position-only (normalized ROI coords, no visual features)
    Ctr = roi_coords(Str).astype(np.float64) / 13.0
    Cdev = roi_coords(Sdev).astype(np.float64) / 13.0
    pred, tacc = run_lr(Ctr[keep], ytr[keep], Cdev)
    save(bb, "R5b", pred, ydev, kdev, qdev, uqdev, tacc, 2,
         "position-only control: normalized ROI coords, no DINO features")

    # R5c: fixed-base ROI (one window per quartet from the base state)
    gtr, _, _, qgtr, _ = load_shards(str(PGRID / f"patchgrid_{bb}_train.shard*.npz"))
    gdev, _, _, qgdev, _ = load_shards(str(PGRID / f"patchgrid_{bb}_dev.shard*.npz"))
    Ptr, Pdev = gtr["patches"], gdev["patches"]
    base_roi_tr = np.array([scene_to_patch(seg_closest_mid(Str[i]))
                            for i in range(0, len(Str), 4)])
    base_roi_dev = np.array([scene_to_patch(seg_closest_mid(Sdev[i]))
                             for i in range(0, len(Sdev), 4)])

    def fixed_roi(P, Q, base_roi):
        N, _, C = P.shape
        grid = P.reshape(N, 14, 14, C)
        out = np.empty((N, C), np.float32)
        uq = np.unique(Q)
        lut = {int(q): tuple(base_roi[k]) for k, q in enumerate(uq)}
        for i in range(N):
            pr, pc = lut[int(Q[i])]
            r0, r1 = max(0, pr - 1), min(14, pr + 2)
            c0, c1 = max(0, pc - 1), min(14, pc + 2)
            out[i] = grid[i, r0:r1, c0:c1, :].reshape(-1, C).mean(axis=0)
        return out

    Ztr = fixed_roi(Ptr[keep], qtr[keep], base_roi_tr[:TRAIN_N])
    Zdv = fixed_roi(Pdev, qdev, base_roi_dev)
    pred, tacc = run_lr(Ztr, ytr[keep], Zdv)
    save(bb, "R5c", pred, ydev, kdev, qdev, uqdev, tacc, Ptr.shape[2],
         "fixed-base ROI: one window per quartet from base state only")


if __name__ == "__main__":
    main()
