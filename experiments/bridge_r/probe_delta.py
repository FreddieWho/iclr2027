#!/usr/bin/env python3
"""Semantic-delta M6: fixed-recipe probes on train/dev (confirm separate).

Inputs: C1 geometry-only, C2 pixel-summary, T0 z1->flip, D0 delta->flip
(PRIMARY), D1 |delta| secondary, T1 [z0,z1] secondary, A z1->y1 state.
Fixed LR recipe (Am03). Includes train-label permutation sanity (§29).
Dev only. Writes results to semantic_delta/probes/.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "bridge_r"))

SEMD = ROOT / "artifacts" / "bridge_r" / "semantic_delta"
FEAT = SEMD / "feats"
OUT = SEMD / "probes"
RECIPE = {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "tol": 1e-4, "max_iter": 5000}


def load_split(split):
    files = sorted(glob.glob(str(FEAT / f"delta_{split}.*.npz")))
    assert files, f"no feats for {split}"
    P = [np.load(f) for f in files]
    F = {k: np.concatenate([p[k] for p in P]).astype(np.float64) for k in
         ("z0", "z1", "pix")}
    meta = {k: np.concatenate([p[k] for p in P]) for k in
            ("flip", "y0", "y1", "parent")}
    d = np.load(SEMD / f"{split}.npz")
    return F, meta, d


def geom_features(d):
    e = d["e"].astype(np.float64)
    ce = e.reshape(len(e), 4, 2).mean(axis=1)
    return np.stack([d["edit_norm"], d["m1"], d["m0"],
                     np.linalg.norm(ce, axis=1), ce[:, 0], ce[:, 1]], axis=1)


def run_probe(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=RECIPE["C"], penalty=RECIPE["penalty"],
                             solver=RECIPE["solver"], tol=RECIPE["tol"],
                             max_iter=RECIPE["max_iter"])
    clf.fit(sc.transform(Xtr).astype(np.float64), ytr)
    p = clf.predict(sc.transform(Xte).astype(np.float64))
    prob = clf.predict_proba(sc.transform(Xte).astype(np.float64))[:, 1]
    return {"acc": float((p == yte).mean()),
            "auroc": float(roc_auc_score(yte, prob)),
            "train_acc": float(clf.score(sc.transform(Xtr).astype(np.float64), ytr)),
            "pred": p.astype(np.int64), "prob": prob}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev"], default="dev")
    a = ap.parse_args()
    Ftr, Mtr, Dtr = load_split("train")
    Fte, Mte, Dte = load_split(a.split)
    Gtr, Gte = geom_features(Dtr), geom_features(Dte)
    ytr, yte = Mtr["flip"].astype(int), Mte["flip"].astype(int)
    inputs = {
        "C1_geom": (Gtr, Gte),
        "C2_pixel": (Ftr["pix"], Fte["pix"]),
        "T0_endpoint": (Ftr["z1"], Fte["z1"]),
        "D0_delta": (Ftr["z1"] - Ftr["z0"], Fte["z1"] - Fte["z0"]),
        "D1_absdelta": (np.abs(Ftr["z1"] - Ftr["z0"]), np.abs(Fte["z1"] - Fte["z0"])),
        "T1_concat": (np.concatenate([Ftr["z0"], Ftr["z1"]], 1),
                      np.concatenate([Fte["z0"], Fte["z1"]], 1)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (Xtr, Xte) in inputs.items():
        r = run_probe(Xtr, ytr, Xte, yte)
        summary[name] = {"acc": r["acc"], "auroc": r["auroc"], "train_acc": r["train_acc"]}
        np.savez(OUT / f"{name}_dev_pred.npz", pred=r["pred"], prob=r["prob"],
                 flip=yte, parent=Mte["parent"])
        print(f"SAW {name}: acc={r['acc']:.4f} auroc={r['auroc']:.4f} train={r['train_acc']:.4f}", flush=True)
    # Task A: endpoint state
    r = run_probe(Ftr["z1"], Mtr["y1"].astype(int), Fte["z1"], Mte["y1"].astype(int))
    summary["A_state"] = {"acc": r["acc"], "auroc": r["auroc"], "train_acc": r["train_acc"]}
    print(f"SAW A_state: acc={r['acc']:.4f} auroc={r['auroc']:.4f} train={r['train_acc']:.4f}", flush=True)
    # permutation sanity on PRIMARY
    rng = np.random.default_rng(7)
    yp = rng.permutation(ytr)
    Xtr, Xte = inputs["D0_delta"]
    r = run_probe(Xtr, yp, Xte, yte)
    summary["D0_permuted"] = {"acc": r["acc"], "auroc": r["auroc"]}
    print(f"SAW D0_permuted: acc={r['acc']:.4f} (expect ~0.5)", flush=True)
    json.dump({"recipe": RECIPE, "results": summary}, open(OUT / "dev_summary.json", "w"), indent=2)
    d0, c1, c2 = summary["D0_delta"]["acc"], summary["C1_geom"]["acc"], summary["C2_pixel"]["acc"]
    print(f"DEV GATE: AccD={d0:.4f} (need>=0.82) margin={d0-max(c1,c2):+.4f} (need>=0.08)")


if __name__ == "__main__":
    main()
