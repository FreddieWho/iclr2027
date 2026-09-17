#!/usr/bin/env python3
"""R05 round 1: frozen-feature metric edit from relational diagnostic pairs.

Fit:  mlpA penultimate features. Semantic deltas = (orig, stealth-flip)
       diagnostic pairs from R02 search (train scenes). Nuisance deltas =
       oracle-verified label-preserving edits of the same scenes.
Controls (same pair budget): same initializer fit on RANDOM scene pairs;
       Euclidean (M=I) baseline.
Eval (untouched holdout scenes, clean only): pairwise same-vs-different
       relation ranking AUC + 1-NN leave-one-out accuracy with an explicit
       shared readout rule. Transfer: R02 eval-scene flips follow-rate.
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
from core.metric_edit import fit_metric_initializer, squared_distance
from common import load_model, preprocess


def features(model, stats, X):
    with torch.no_grad():
        return model(preprocess(X, stats), return_feat=True)[1].numpy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--scenes-train", type=Path, required=True)
    p.add_argument("--r02-train", type=Path, required=True)   # metamers.csv on train scenes
    p.add_argument("--scenes-holdout", type=Path, required=True)
    p.add_argument("--r02-eval", type=Path, required=True)    # metamers.csv on eval scenes (transfer)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--seed", type=int, default=5)
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(a.seed)
    model, stats = load_model(a.model)
    dtr = np.load(a.scenes_train / "scenes.npz")
    Xtr, ytr = dtr["positions"].astype(float), dtr["labels"].astype(int)
    dho = np.load(a.scenes_holdout / "scenes.npz")
    Xho, yho = dho["positions"].astype(float), dho["labels"].astype(int)

    # --- diagnostic pairs from R02 train flips ---
    diag_rows = [r for r in csv.DictReader(open(a.r02_train / "metamers.csv"))]
    Ztr = features(model, stats, Xtr)
    # NOTE: metamers.csv lacks the edit vector, so rebuild flips + nuisance edits here
    # with an independent rng stream and oracle verification (same protocol as R02).
    from r02_search import candidates_for_scene
    sem_edits, nui_edits = [], []
    for r in diag_rows:
        i = int(r["scene"])
        cands = candidates_for_scene(Xtr[i], rng)
        flip_e, pres_e = None, []
        for e, fam in cands:
            try:
                o = segment_relation(Xtr[i] + e)
            except ValueError:
                continue
            if o["ambiguous"] or o["margin"] < 0.03:
                continue
            if o["label"] != int(ytr[i]) and flip_e is None:
                flip_e = e
            elif o["label"] == int(ytr[i]) and len(pres_e) < 3:
                pres_e.append(e)
            if flip_e is not None and len(pres_e) >= 3:
                break
        if flip_e is not None:
            sem_edits.append((i, flip_e))
            nui_edits.extend((i, e) for e in pres_e)
    sem_d = np.array([features(model, stats, (Xtr[i] + e)[None])[0] - Ztr[i] for i, e in sem_edits])
    nui_d = np.array([features(model, stats, (Xtr[i] + e)[None])[0] - Ztr[i] for i, e in nui_edits])
    n_sem, n_nui = len(sem_d), len(nui_d)

    out_diag = fit_metric_initializer(sem_d, nui_d, rank=a.rank)
    M_diag = out_diag["metric"]
    # --- random-pair control, same budget ---
    idx = rng.permutation(len(Xtr))
    rs, rn = [], []
    j = 0
    while len(rs) < n_sem and j < len(idx) - 1:
        p1, p2 = idx[j], idx[j + 1]; j += 2
        if ytr[p1] != ytr[p2]:
            rs.append(Ztr[p2] - Ztr[p1])
    j = 0
    while len(rn) < n_nui and j < len(idx) - 1:
        p1, p2 = idx[j], idx[j + 1]; j += 2
        if ytr[p1] == ytr[p2]:
            rn.append(Ztr[p2] - Ztr[p1])
    out_rand = fit_metric_initializer(np.array(rs), np.array(rn), rank=a.rank)
    M_rand = out_rand["metric"]
    M_eye = np.eye(Ztr.shape[1])

    # --- eval on untouched holdout (clean scenes only) ---
    Zho = features(model, stats, Xho)
    N = len(Xho)
    same = yho[:, None] == yho[None, :]
    iu = np.triu_indices(N, k=1)
    ypair = same[iu].astype(int)
    res = {}
    for name, M in (("euclidean", M_eye), ("random_pairs", M_rand), ("diagnostic", M_diag)):
        D = np.zeros((N, N))
        for i in range(N):
            D[i] = squared_distance(Zho[i][None].repeat(N, 0), Zho, M)
        scores = -D[iu]
        order = np.argsort(scores)
        ys = ypair[order]
        tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
        auc = float(np.trapz(tp / tp[-1], fp / fp[-1])) if tp[-1] > 0 and fp[-1] > 0 else float("nan")
        # 1-NN LOO accuracy (explicit shared readout)
        np.fill_diagonal(D, np.inf)
        nn = D.argmin(axis=1)
        acc = float((yho[nn] == yho).mean())
        res[name] = {"pairwise_AUC": auc, "nn_loo_acc": acc}
    # --- transfer set-aside: eval-scene flip scenes are NOT used in fit ---
    for name, M in (("euclidean", M_eye), ("random_pairs", M_rand), ("diagnostic", M_diag)):
        ds = np.sqrt(np.einsum("ij,jk,ik->i", sem_d, M, sem_d))
        dn = np.sqrt(np.einsum("ij,jk,ik->i", nui_d, M, nui_d))
        res[name].update({"in_sample_flip_dist_med": float(np.median(ds)),
                          "in_sample_nuisance_dist_med": float(np.median(dn))})
    summary = {"n_semantic": n_sem, "n_nuisance": n_nui, "rank": a.rank,
               "diag_eigenvalues": out_diag["diagnostic_eigenvalues"].tolist(),
               "diag_gains": out_diag["gains"].tolist(),
               "rand_eigenvalues": out_rand["diagnostic_eigenvalues"].tolist(),
               "metrics": res,
               "notes": ["M fit on train-scene diagnostic pairs only; holdout never touched in fit.",
                         "Transfer rows are in-sample separation (labeled); cross-scene transfer needs R02-on-holdout, reserved."]}
    (a.output / "R05_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
