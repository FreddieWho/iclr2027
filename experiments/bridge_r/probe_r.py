#!/usr/bin/env python3
"""Bridge-R R5: probe training + dev evaluation + learning curve.

Trains on train-split quartet groups only, evaluates on dev. Probe configs
are FIXED by protocol (LR C=1.0 / MLP-128); the ONLY dev-driven choice is
training size {230,500,1000,2000} via the frozen plateau rule. Never touches
holdout. Never selects representation/readout by composition miss.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "bridge_r"
KINDS = ["base", "A", "B", "AB"]


def load_split(bb, split):
    d = np.load(ART / "features" / f"feats_{bb}_{split}.npz")
    return d


def quartet_ids(d):
    return d["qid"].astype(int)


def evaluate(pred, labels, kinds, qid):
    P = {k: pred[kinds == k] for k in KINDS}
    Y = {k: labels[kinds == k] for k in KINDS}
    acc = {k: float((P[k] == Y[k]).mean()) for k in KINDS}
    both = (P["A"] == Y["A"]) & (P["B"] == Y["B"])
    # compositional subset per protocol: atomics right, yA==yB==y0, AB flipped
    comp = both & (Y["A"] == Y["B"]) & (Y["AB"] != Y["A"])
    return {
        "acc": acc,
        "atomic_mean": float(np.mean([(P["A"] == Y["A"]).mean(),
                                      (P["B"] == Y["B"]).mean()])),
        "min_atom": float(min((P["A"] == Y["A"]).mean(),
                              (P["B"] == Y["B"]).mean())),
        "both_frac": float(both.mean()),
        "both_n": int(both.sum()),
        "comp_n": int(comp.sum()),
        "comp_miss_dev_exploratory": (
            float((P["AB"][comp] != Y["AB"][comp]).mean()) if comp.sum() else None),
    }


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--readout", choices=["R0_pooled", "R1_mean_patch", "R2_spatial2x2"],
                   required=True)
    p.add_argument("--probe", choices=["lr", "mlp128"], required=True)
    p.add_argument("--train-n", type=int, default=2000)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    tr = load_split(a.backbone, "train")
    dv = load_split(a.backbone, "dev")
    Ztr_all = tr[a.readout]
    qtr = quartet_ids(tr)
    uq = np.unique(qtr)
    if a.train_n > len(uq):
        raise SystemExit(f"FATAL: train has {len(uq)} quartets < {a.train_n}")
    keep = np.isin(qtr, uq[:a.train_n])
    Ztr, ytr = Ztr_all[keep], tr["labels"][keep].astype(int)
    Zdv, ydv = dv[a.readout], dv["labels"].astype(int)
    sc = StandardScaler().fit(Ztr)
    Xtr, Xdv = sc.transform(Ztr), sc.transform(Zdv)
    if a.probe == "lr":
        clf = LogisticRegression(C=1.0, max_iter=5000)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=2000, random_state=0)
    clf.fit(Xtr, ytr)
    res = {"backbone": a.backbone, "readout": a.readout, "probe": a.probe,
           "train_n": a.train_n,
           "train_acc": float(clf.score(Xtr, ytr)),
           "dev": evaluate(clf.predict(Xdv), ydv, dv["kinds"], dv["qid"])}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print("SAW", json.dumps(res))


if __name__ == "__main__":
    main()
