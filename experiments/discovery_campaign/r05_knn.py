#!/usr/bin/env python3
"""R05 round 2: kNN readout on frozen features vs fixed linear head.

For each R02 flip (X -> X', oracle label changed o -> o'): does the frozen
model's own linear head follow (pred == o')? Does 1-NN / 5-NN over a frozen
feature pool with an explicit shared rule follow? Same oracle budget for all.
Pools: train scenes (transfer for eval flips; leave-self-out for train flips).
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

PACK = Path(__file__).resolve().parents[2] / "docs" / "iclr2027_discovery_campaign_20260917"
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.relations import segment_relation
from common import load_model, preprocess
from r02_search import candidates_for_scene


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--flip-scenes", type=Path, required=True)
    p.add_argument("--flip-slice", type=str, default="")
    p.add_argument("--pool-slice", type=str, default="")
    p.add_argument("--leave-self-out", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=5)
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)
    model, stats = load_model(a.model)
    dp = np.load(a.pool / "scenes.npz")
    Xp, yp = dp["positions"].astype(float), dp["labels"].astype(int)
    if a.pool_slice:
        lo, hi = [int(v) for v in a.pool_slice.split(":")]
        Xp, yp = Xp[lo:hi], yp[lo:hi]
    df = np.load(a.flip_scenes / "scenes.npz")
    Xf, yf = df["positions"].astype(float), df["labels"].astype(int)
    if a.flip_slice:
        lo, hi = [int(v) for v in a.flip_slice.split(":")]
        Xf, yf = Xf[lo:hi], yf[lo:hi]
    rng = np.random.default_rng(a.seed)
    model.eval()

    def feats(X):
        with torch.no_grad():
            return model(preprocess(X, stats), return_feat=True)[1].numpy()

    def preds(X):
        with torch.no_grad():
            return (torch.sigmoid(model(preprocess(X, stats))).numpy() > .5).astype(int)

    Zp = feats(Xp)
    rows = []
    for i in range(len(Xf)):
        # rebuild first valid flip (same protocol as R02/R05; independent rng stream)
        flip = None
        for e, fam in candidates_for_scene(Xf[i], rng):
            try:
                o = segment_relation(Xf[i] + e)
            except ValueError:
                continue
            if o["ambiguous"] or o["margin"] < 0.03:
                continue
            if o["label"] != int(yf[i]):
                flip = (e, o["label"], fam)
                break
        if flip is None:
            continue
        e, new_label, fam = flip
        xp = (Xf[i] + e)[None]
        z = feats(xp)[0]
        head_pred = int(preds(xp)[0])
        d = np.linalg.norm(Zp - z, axis=1)
        if a.leave_self_out:
            # pool contains the flip scenes themselves: exclude exact self index
            d[i] = np.inf
        nn1 = int(yp[np.argmin(d)])
        k5 = np.argsort(d)[:5]
        vals, counts = np.unique(yp[k5], return_counts=True)
        nn5 = int(vals[np.argmax(counts)])
        rows.append({"scene": i, "orig": int(yf[i]), "new": int(new_label), "fam": fam,
                     "head_follows": head_pred == int(new_label),
                     "nn1_follows": nn1 == int(new_label),
                     "nn5_follows": nn5 == int(new_label)})
    n = len(rows)
    head = sum(r["head_follows"] for r in rows)
    nn1 = sum(r["nn1_follows"] for r in rows)
    nn5 = sum(r["nn5_follows"] for r in rows)
    summary = {"n_flips": n, "head_follow_rate": head / n if n else None,
               "nn1_follow_rate": nn1 / n if n else None,
               "nn5_follow_rate": nn5 / n if n else None,
               "head_miss_nn1_hit": sum(1 for r in rows if not r["head_follows"] and r["nn1_follows"]) / n if n else None,
               "pool": str(a.pool), "flip_scenes": str(a.flip_scenes),
               "note": "Zero backbone updates; only the readout rule changes (fixed head vs kNN)."}
    (a.output / "R05_knn_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(a.output / "knn_follow.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["scene"])
        w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
