#!/usr/bin/env python3
"""L007 re-analysis under corrected statistics (review items A2/A3/A4).

Reads the EXISTING diagnostic CSVs (no forwards, no training) and recomputes
every reported correlation with average-rank Spearman (ties averaged), plus
the effective finite-pair n for each feature. Also reports the kink
gray-zone branch explicitly (success line 0.3 / negative line 0.2) instead
of collapsing all non-success into "closed".

NEW outputs only: artifacts/next_novelty/l007_reanalysis/L007_REANALYSIS.json
"""
import argparse
import numpy as np
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
from stats import spearman  # noqa: E402

U1 = ROOT / "artifacts" / "next_novelty" / "u1_factorial"
DIAG = ROOT / "artifacts" / "next_novelty" / "l007" / "L007_DIAG.csv"
GRAZ = ROOT / "artifacts" / "next_novelty" / "l007_grazing" / "L007G_DIAG.csv"
KINK = ROOT / "artifacts" / "next_novelty" / "l007_kink" / "L007K_DIAG.csv"
OUT = ROOT / "artifacts" / "next_novelty" / "l007_reanalysis"

SCORE = ["max_slope", "max_curv", "trans_width", "slope_asym", "min_abs_logit"]
REPR = ["speed_conc", "max_angle", "hyper_dist", "z_length"]
ORACLE = ["t_star", "min_margin_win", "margin_slope", "cost"]


def rows_of(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def num(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def old_spearman(x, y):
    """The superseded ordinal-rank variant, kept only to show the delta."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return None
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return round(float((rx * ry).sum() / d), 4) if d > 0 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    out = {"note": "average-rank Spearman over finite pairs; old = ordinal-rank "
                   "(argsort twice) for delta only",
           "families": {}, "kink_gray_zone": {}, "grazing": {}}

    R = rows_of(DIAG)
    if R:
        for seed in (11, 23, 47):
            base = [r for r in R if int(r["seed"]) == seed]
            hit = [r for r in base if int(r["miss"]) == 0]
            y = [num(r, "abserr") for r in hit]
            fam = {}
            for name, feats in (("ORACLE", ORACLE), ("SCORE", SCORE),
                                ("REPR", REPR)):
                fam[name] = {k: {"rho": spearman([num(r, k) for r in hit], y)[0],
                                 "n": spearman([num(r, k) for r in hit], y)[1],
                                 "rho_old": old_spearman([num(r, k) for r in hit], y)}
                             for k in feats}
            out["families"]["s%d" % seed] = {
                "n_hit": len(hit), "n_all": len(base), "features": fam}

    K = rows_of(KINK)
    if K:
        for seed in (11, 23, 47):
            base = [r for r in K if int(r["seed"]) == seed]
            hit = [r for r in base if int(r["miss"]) == 0]
            y = [num(r, "abserr") for r in hit]
            dstat = spearman([num(r, "kink_dist") for r in hit], y)
            cstat = spearman([num(r, "kink_count") for r in hit], y)
            # gray-zone branch: success if |rho|>=0.3 same direction 3/3;
            # negative-equivalent only if BOTH <0.2; else UNDETERMINED
            out["kink_gray_zone"]["s%d" % seed] = {
                "kink_dist_rho": dstat[0], "kink_dist_n": dstat[1],
                "kink_count_rho": cstat[0], "kink_count_n": cstat[1],
                "kink_count_rho_old": old_spearman(
                    [num(r, "kink_count") for r in hit], y),
                "n_hit": len(hit), "n_all": len(base)}

    G = rows_of(GRAZ)
    if G:
        for seed in (11, 23, 47):
            base = [r for r in G if int(r["seed"]) == seed]
            hit = [r for r in base if int(r["miss"]) == 0]
            y = [num(r, "abserr") for r in hit]
            woa = spearman([num(r, "woa") for r in hit], y)
            w1 = spearman([num(r, "width_k1") for r in hit], y)
            out["grazing"]["s%d" % seed] = {
                "woa_rho": woa[0], "woa_n": woa[1],
                "width_rho_k1": w1[0], "width_n_k1": w1[1],
                "n_hit": len(hit), "n_all": len(base),
                "woa_rho_old": old_spearman([num(r, "woa") for r in hit], y)}

    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out / "L007_REANALYSIS.json", "w"), indent=1,
              default=float)
    print(json.dumps(out, indent=1, default=float)[:4000])
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
