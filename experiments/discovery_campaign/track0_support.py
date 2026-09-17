#!/usr/bin/env python3
"""Track0: does training-support density at the flipped point predict miss?

For each eval scene: rebuild first valid flip (fixed protocol/seed), then
  d_new  = min ||X' - X_train|| over train scenes carrying the NEW label
  d_any  = min ||X' - X_train|| over all train scenes
  d_self = ||X' - X|| (flip step size)
Target: miss = (model pred on X' != oracle new label), recomputed.
Report: AUC of each feature for miss + quartile table. Pure numpy + torch
forwards only; no training.
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


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    s1 = np.sort(scores[labels == 1])
    s0 = np.sort(scores[labels == 0])
    if len(s1) == 0 or len(s0) == 0:
        return float("nan")
    # Mann-Whitney U / (n1*n0)
    rank = np.searchsorted(s0, s1).sum()
    return float(rank / (len(s1) * len(s0)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--flip-scenes", type=Path, required=True)
    p.add_argument("--flip-slice", type=str, default="256:512")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=5)
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)
    model, stats = load_model(a.model)
    model.eval()
    dp = np.load(a.pool / "scenes.npz")
    Xp, yp = dp["positions"].astype(float), dp["labels"].astype(int)
    df = np.load(a.flip_scenes / "scenes.npz")
    lo, hi = [int(v) for v in a.flip_slice.split(":")]
    Xf, yf = df["positions"][lo:hi].astype(float), df["labels"][lo:hi].astype(int)
    Pf = Xp.reshape(len(Xp), -1)
    rng = np.random.default_rng(a.seed)
    rows = []
    for i in range(len(Xf)):
        flip = None
        for e, fam in candidates_for_scene(Xf[i], rng):
            try:
                o = segment_relation(Xf[i] + e)
            except ValueError:
                continue
            if o["ambiguous"] or o["margin"] < 0.03:
                continue
            if o["label"] != int(yf[i]):
                flip = (e, o["label"], fam, o["margin"])
                break
        if flip is None:
            continue
        e, new_label, fam, margin = flip
        xp = (Xf[i] + e).reshape(1, -1)
        with torch.no_grad():
            pred = int((torch.sigmoid(model(preprocess(xp, stats))).numpy()[0] > .5))
        d_all = np.linalg.norm(Pf - xp, axis=1)
        d_new = d_all[yp == new_label].min()
        d_old = d_all[yp == int(yf[i])].min()
        rows.append({"scene": i, "orig": int(yf[i]), "new": int(new_label),
                     "fam": fam, "margin": round(float(margin), 4),
                     "d_self": round(float(np.linalg.norm(e)), 4),
                     "d_new": round(float(d_new), 4),
                     "d_old": round(float(d_old), 4),
                     "d_any": round(float(d_all.min()), 4),
                     "miss": bool(pred == int(yf[i]))})
    y = np.array([r["miss"] for r in rows], dtype=int)
    dn = np.array([r["d_new"] for r in rows])
    da = np.array([r["d_any"] for r in rows])
    mg = np.array([r["margin"] for r in rows])
    ds = np.array([r["d_self"] for r in rows])
    ratio = dn / np.maximum(da, 1e-9)
    summary = {"n_flips": len(rows), "miss_rate": float(y.mean()),
               "auc_d_new": auc(dn, y), "auc_d_any": auc(da, y),
               "auc_margin_neg": auc(-mg, y), "auc_d_self": auc(ds, y),
               "auc_new_over_any_ratio": auc(ratio, y)}
    qs = np.quantile(dn, [0, .25, .5, .75, 1.0])
    qt = []
    for q in range(4):
        m = (dn >= qs[q]) & (dn <= qs[q + 1] + 1e-12)
        qt.append({"d_new_range": [round(float(qs[q]), 3), round(float(qs[q + 1]), 3)],
                   "n": int(m.sum()), "miss_rate": round(float(y[m].mean()), 3) if m.sum() else None})
    summary["quartile_by_d_new"] = qt
    (a.output / "TRACK0_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(a.output / "support_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
