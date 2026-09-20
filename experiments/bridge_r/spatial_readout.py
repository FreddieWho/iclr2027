#!/usr/bin/env python3
"""Bridge-R Task 2 (Am03): fixed-recipe spatial readout matrix.

R0/R1/R2 (cached) recomputed under the frozen Am03 LR recipe + new
R3 4x4 / R4 7x7 / R5 oracle-ROI diagnostics. Same v2 train/dev, same first
2000 train quartet IDs for every backbone. Dev only. No composition metric.
Holdout never loaded.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "bridge_r"))
from render_v1 import SAFE_SCALE, CANON

ART = ROOT / "artifacts" / "bridge_r"
FEAT = ART / "features"
PGRID = ART / "patchgrid"
OUT = ART / "spatial"
KINDS = ["base", "A", "B", "AB"]
TRAIN_N = 2000
LR_RECIPE = {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "tol": 1e-4,
             "max_iter": 5000}


def load_shards(pattern):
    files = sorted(glob.glob(pattern))
    assert files, f"no shards: {pattern}"
    parts = [np.load(f) for f in files]
    out = {}
    for k in ("patches", "R0_pooled", "R1_mean_patch", "R2_spatial2x2"):
        if k in parts[0]:
            out[k] = np.concatenate([pt[k] for pt in parts]).astype(np.float32)
    kinds = np.concatenate([pt["kinds"] for pt in parts])
    labels = np.concatenate([pt["labels"] for pt in parts])
    qid = np.concatenate([pt["qid"] for pt in parts]).astype(int)
    order = np.argsort(qid, kind="stable")
    uq = np.unique(qid)
    assert (np.sort(qid).reshape(-1, 4)[:, 0] == uq).all()
    return out, kinds[order], labels[order].astype(int), qid[order], uq


def spatial_pool(patches, out_grid):
    """patches [N,196,C] -> adaptive pool to [N,out_grid**2*C]."""
    N, _, C = patches.shape
    with torch.no_grad():
        y = F.adaptive_avg_pool2d(torch.from_numpy(patches.reshape(N, 14, 14, C))
                                  .permute(0, 3, 1, 2), out_grid)
    return y.permute(0, 2, 3, 1).reshape(N, out_grid * out_grid * C).numpy()


def seg_closest_mid(p):
    """Closest midpoint between segments AB (rows 0,1) and CD (rows 2,3).

    Input geometry only. Returns (row, col) in scene coords ~[-1,1].
    Standard segment-segment closest point with degenerate guards.
    """
    p1, p2, p3, p4 = (np.asarray(p[i], float) for i in range(4))
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    eps = 1e-12
    if a < eps and e < eps:
        return (p1 + p3) / 2
    if a < eps:
        s = 0.0
        t = float(np.clip((d2 @ r) / e, 0, 1))
    elif e < eps:
        t = 0.0
        s = float(np.clip(-(d1 @ r) / a, 0, 1))
    else:
        f = float(d2 @ r)
        c = float(d1 @ r)
        b = float(d1 @ d2)
        denom = a * e - b * b
        s = 0.0 if denom < eps else float(np.clip((b * f - c * e) / denom, 0, 1))
        t = float(np.clip((b * s + f) / e, 0, 1))
        s = float(np.clip((b * t - c) / a, 0, 1))
        t = float(np.clip((b * s + f) / e, 0, 1))
    return (p1 + d1 * s + p3 + d2 * t) / 2


def scene_to_patch(rc):
    """Scene (x,y) -> (patch_row, patch_col) on the 14x14 grid.

    Mirrors render_canonical (SAFE_SCALE similarity map into 512px,
    processor resizes 512->224 with no crop, 16px patches).
    """
    x, y = float(rc[0]), float(rc[1])
    col512 = ((x * SAFE_SCALE + 1) / 2 * (CANON - 1))
    row512 = ((y * SAFE_SCALE + 1) / 2 * (CANON - 1))
    pc = int(np.clip((col512 * 224 / 512) // 16, 0, 13))
    pr = int(np.clip((row512 * 224 / 512) // 16, 0, 13))
    return pr, pc


def roi_features(patches, states):
    """3x3 clipped patch average around the relation locus -> [N,C]."""
    N, _, C = patches.shape
    grid = patches.reshape(N, 14, 14, C)
    out = np.empty((N, C), np.float32)
    for i in range(N):
        pr, pc = scene_to_patch(seg_closest_mid(states[i]))
        r0, r1 = max(0, pr - 1), min(14, pr + 2)
        c0, c1 = max(0, pc - 1), min(14, pc + 2)
        out[i] = grid[i, r0:r1, c0:c1, :].reshape(-1, C).mean(axis=0)
    return out


def evaluate(pred, labels, kinds, qid):
    P = {k: pred[kinds == k] for k in KINDS}
    Y = {k: labels[kinds == k] for k in KINDS}
    acc = {k: float((P[k] == Y[k]).mean()) for k in KINDS}
    both = (P["A"] == Y["A"]) & (P["B"] == Y["B"])
    uq = np.unique(qid)
    qcorr = np.array([((pred[(qid == q) & ((kinds == "A") | (kinds == "B"))]) ==
                       (labels[(qid == q) & ((kinds == "A") | (kinds == "B"))])).mean()
                      for q in uq])
    return {
        "acc": acc,
        "atomic_mean": float(np.mean([(P["A"] == Y["A"]).mean(),
                                      (P["B"] == Y["B"]).mean()])),
        "min_atom": float(min((P["A"] == Y["A"]).mean(),
                              (P["B"] == Y["B"]).mean())),
        "both_frac": float(both.mean()),
        "both_n": int(both.sum()),
        "quartet_ids": uq.tolist(),
        "quartet_atomic_correct": qcorr.tolist(),
    }


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["dinov3s", "dinov3b", "dinov3l"],
                    required=True)
    ap.add_argument("--readout",
                    choices=["R0", "R1", "R2", "R3", "R4", "R5"], required=True)
    a = ap.parse_args()
    bb = a.backbone

    ftr, ktr, ytr, qtr, uqtr = load_shards(str(FEAT / f"feats_{bb}_train.shard*.npz"))
    fdev, kdev, ydev, qdev, uqdev = load_shards(str(FEAT / f"feats_{bb}_dev.shard*.npz"))
    # merged-file glob above matches shard files only; ensure full coverage
    assert len(uqtr) == 2164 and len(uqdev) == 760, (len(uqtr), len(uqdev))
    keep_ids = uqtr[:TRAIN_N]
    keep = np.isin(qtr, keep_ids)
    assert keep.sum() == TRAIN_N * 4

    key = {"R0": "R0_pooled", "R1": "R1_mean_patch",
           "R2": "R2_spatial2x2"}[a.readout] if a.readout in "R0R1R2" else None
    if a.readout in ("R0", "R1", "R2"):
        Ztr, Zdv = ftr[key][keep], fdev[key]
    else:
        gtr, _, _, qgtr, _ = load_shards(str(PGRID / f"patchgrid_{bb}_train.shard*.npz"))
        gdev, _, _, qgdev, _ = load_shards(str(PGRID / f"patchgrid_{bb}_dev.shard*.npz"))
        assert (qgtr == qtr).all() and (qgdev == qdev).all(), "patchgrid/qid mismatch"
        Ptr, Pdev = gtr["patches"], gdev["patches"]
        if a.readout == "R3":
            Ztr, Zdv = spatial_pool(Ptr[keep], 4), spatial_pool(Pdev, 4)
        elif a.readout == "R4":
            Ztr, Zdv = spatial_pool(Ptr[keep], 7), spatial_pool(Pdev, 7)
        else:
            # R5 oracle-localized diagnostic: per-state full coordinates
            # (base + edit delta by kind), input geometry only.
            q = np.load(ART / "quartets_train_v2.npz")
            Str_full = np.empty((keep.sum(), 4, 2))
            for i, (qi, k) in enumerate(zip(qtr[keep], ktr[keep])):
                x, ea, eb = q[f"q{qi}"].reshape(3, 4, 2)
                Str_full[i] = {"base": x, "A": x + ea, "B": x + eb,
                               "AB": x + ea + eb}[str(k)]
            qd = np.load(ART / "quartets_dev_v2.npz")
            Sdev_full = np.empty((len(qdev), 4, 2))
            for i, (qi, k) in enumerate(zip(qdev, kdev)):
                x, ea, eb = qd[f"q{qi}"].reshape(3, 4, 2)
                Sdev_full[i] = {"base": x, "A": x + ea, "B": x + eb,
                                "AB": x + ea + eb}[str(k)]
            Ztr, Zdv = roi_features(Ptr[keep], Str_full), roi_features(Pdev, Sdev_full)

    sc = StandardScaler().fit(Ztr)
    Xtr, Xdv = sc.transform(Ztr).astype(np.float64), sc.transform(Zdv).astype(np.float64)
    clf = LogisticRegression(C=LR_RECIPE["C"], penalty=LR_RECIPE["penalty"],
                             solver=LR_RECIPE["solver"], tol=LR_RECIPE["tol"],
                             max_iter=LR_RECIPE["max_iter"])
    clf.fit(Xtr, ytr[keep])
    pred = clf.predict(Xdv)
    dev = evaluate(pred, ydev, kdev, qdev)
    b1 = (dev["atomic_mean"] >= 0.85 and dev["both_frac"] >= 0.60
          and dev["both_n"] >= 100)
    import sklearn
    res = {"backbone": bb, "readout": a.readout, "probe": "lr-fixed-Am03",
           "lr_recipe": LR_RECIPE, "sklearn_version": sklearn.__version__,
           "train_n_quartets": TRAIN_N,
           "train_quartet_ids": [int(v) for v in keep_ids.tolist()],
           "train_acc": float(clf.score(Xtr, ytr[keep])), "dev": dev,
           "gate_B1": "ATOMIC_ACCESSIBILITY_RESCUED" if b1 else "FAIL"}
    od = OUT / f"{bb}_{a.readout}"
    od.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(od / "result.json", "w"), indent=1)
    np.savez(od / "predictions.npz", pred=pred.astype(np.int64),
             labels=ydev.astype(np.int64), kinds=kdev, qid=qdev.astype(np.int64))
    man = {"backbone": bb, "readout": a.readout, "probe": "lr-fixed-Am03",
           "train_n_quartets": TRAIN_N, "n_dev_quartets": len(uqdev),
           "feature_dim": int(Ztr.shape[1]),
           "holdout_touched": False,
           "note": "dev diagnostic only; R5 uses privileged input geometry"}
    json.dump(man, open(od / "manifest.json", "w"), indent=1)
    print("SAW", bb, a.readout, "atomic=%.4f" % dev["atomic_mean"],
          "B1=", res["gate_B1"])


if __name__ == "__main__":
    main()
