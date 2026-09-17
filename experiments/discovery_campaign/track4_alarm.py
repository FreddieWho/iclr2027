#!/usr/bin/env python3
"""Track4: test-time self-alarm + cross-arch ensemble, CSV-only (no forwards).

Alarm: among found flips, does feature motion (feat_dist, motion/null ratio,
margin, input_norm) detect misses? Report AUROC per feature + combined
(logistic on standardized features via numpy Newton).
Ensemble: join mlpA/B/C metamers by scene; majority-vote miss vs single rates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    s1 = np.sort(scores[labels == 1])
    s0 = np.sort(scores[labels == 0])
    if len(s1) == 0 or len(s0) == 0:
        return float("nan")
    return float(np.searchsorted(s0, s1).sum() / (len(s1) * len(s0)))


def logit_auc(X: np.ndarray, y: np.ndarray, iters: int = 200) -> float:
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-Xs @ w))
        g = Xs.T @ (pr - y) / len(y) + 1e-4 * w
        W = pr * (1 - pr)
        H = (Xs.T * W) @ Xs / len(y) + 1e-4 * np.eye(Xs.shape[1])
        try:
            w -= np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
    return auc(1 / (1 + np.exp(-Xs @ w)), y)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--r02a", type=Path, required=True)
    p.add_argument("--r02b", type=Path, required=True)
    p.add_argument("--r02c", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)
    A = pd.read_csv(a.r02a / "metamers.csv")
    B = pd.read_csv(a.r02b / "metamers.csv")
    C = pd.read_csv(a.r02c / "metamers.csv")
    y = (A["model_miss"].astype(str) == "True").to_numpy(dtype=int)
    fd = A["feat_dist"].to_numpy(float)
    nf = A["null_feat_dist"].to_numpy(float)
    mg = A["margin"].to_numpy(float)
    ino = A["input_norm"].to_numpy(float)
    ok = np.isfinite(nf) & (nf > 0)
    ratio = np.full_like(fd, np.nan)
    ratio[ok] = fd[ok] / nf[ok]
    res = {"n": len(A), "miss_rate_A": round(float(y.mean()), 4),
           "auc_feat_dist": round(auc(fd, y), 4),
           "auc_motion_null_ratio": round(auc(np.nan_to_num(ratio, nan=np.nanmedian(ratio)), y), 4),
           "auc_margin_neg": round(auc(-mg, y), 4),
           "auc_input_norm": round(auc(ino, y), 4)}
    Xc = np.column_stack([fd, np.nan_to_num(ratio, nan=np.nanmedian(ratio)), -mg, ino])
    res["auc_logistic_combined"] = round(logit_auc(Xc, y), 4)
    # ensemble join on scene (same candidate protocol => same flip sets)
    J = A[["scene", "oracle", "pred"]].merge(
        B[["scene", "pred"]].rename(columns={"pred": "predB"}), on="scene").merge(
        C[["scene", "pred"]].rename(columns={"pred": "predC"}), on="scene")
    for col, nm in (("pred", "A"), ("predB", "B"), ("predC", "C")):
        J[nm + "miss"] = (J[col] != J["oracle"]).astype(int)
    J["maj"] = (J[["pred", "predB", "predC"]].mode(axis=1)[0])
    J["majmiss"] = (J["maj"] != J["oracle"]).astype(int)
    J["disagree"] = (J[["pred", "predB", "predC"]].nunique(axis=1) > 1).astype(int)
    res["ensemble"] = {
        "n": len(J),
        "miss_A": round(float(J["Amiss"].mean()), 4),
        "miss_B": round(float(J["Bmiss"].mean()), 4),
        "miss_C": round(float(J["Cmiss"].mean()), 4),
        "miss_majority": round(float(J["majmiss"].mean()), 4),
        "disagreement_rate": round(float(J["disagree"].mean()), 4),
        "miss_given_agree": round(float(J.loc[J["disagree"] == 0, "majmiss"].mean()), 4),
        "miss_given_disagree": round(float(J.loc[J["disagree"] == 1, "majmiss"].mean()), 4)}
    (a.output / "TRACK4_result.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
