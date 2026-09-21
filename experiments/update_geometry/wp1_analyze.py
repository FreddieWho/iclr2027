#!/usr/bin/env python3
"""WP1 analysis: P(miss|I), lag|I, matched tangent-vs-normal, composition attribution.

Primary filter: oracle exactly one crossing + model start correct + incidence
available + clean gradient audit. Miss = no same-direction model turn
(MISS_NO_CROSS); abnormal = multi/wrong-target turns or late/early crossing.
Matched pairs: standardized-Euclidean NN on [edit_norm, start_margin,
endpoint_margin] + exact base_label/endpoint_label strata; |SMD|<0.10 gate.
Bootstrap 10k, resampling unit = parent (pairs for matched).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

rng = np.random.default_rng(26092202)


def load_rows(path):
    d = json.load(open(path))
    return d["rows"] if isinstance(d, dict) and "rows" in d else d


def primary(rows):
    out = []
    for r in rows:
        if r["oracle_n_crossings"] != 1:
            continue
        if not r["start_correct"]:
            continue
        if r["incidence"] is None:
            continue
        if r["incidence_audit_rel"] is not None and r["incidence_audit_rel"] > 1e-3:
            continue
        out.append(r)
    return out


def outcome_of(r):
    """miss / hit / abnormal + lag."""
    m = r["match"][0]
    if m["status"] == "missing":
        if m["n_model_turns"] == 0:
            return "miss_no_cross", None
        return "miss_abnormal", None
    lag = m.get("error")
    if m["status"] == "ok":
        return "hit", lag
    return ("abnormal_" + m["status"]), lag


def binned_curve(rows, nb=5):
    Is = np.array([r["incidence"] for r in rows])
    ms = np.array([1 if outcome_of(r)[0].startswith("miss") else 0 for r in rows])
    edges = np.quantile(Is, np.linspace(0, 1, nb + 1))
    edges[0], edges[-1] = 0.0, 1.0
    parents = np.array([r["parent_id"] for r in rows])
    out = []
    for b in range(nb):
        sel = (Is >= edges[b]) & (Is <= edges[b + 1] if b == nb - 1
                                  else Is < edges[b + 1])
        if sel.sum() == 0:
            continue
        x = ms[sel]
        # parent-cluster bootstrap: resample parents, take all their paths
        ups = np.unique(parents[sel])
        pmap = {u: np.nonzero(parents[sel] == u)[0] for u in ups}
        boots = []
        for _ in range(10000):
            draw = rng.choice(ups, size=len(ups), replace=True)
            idx = np.concatenate([pmap[u] for u in draw])
            boots.append(x[idx].mean())
        lo, hi = np.quantile(boots, [0.025, 0.975])
        out.append({"bin": [round(float(edges[b]), 3), round(float(edges[b + 1]), 3)],
                    "n": int(sel.sum()), "miss_rate": round(float(x.mean()), 4),
                    "ci": [round(float(lo), 4), round(float(hi), 4)]})
    return out


def logistic_slope(rows):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    X = np.array([[r["incidence"]] for r in rows])
    y = np.array([1 if outcome_of(r)[0].startswith("miss") else 0 for r in rows])
    if y.std() == 0:
        return None
    lr = LogisticRegression().fit(X, y)
    return {"coef": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0])}


def matched_tangent_normal(rows):
    tan = [r for r in rows if r["incidence"] is not None and r["incidence"] <= 0.25]
    nor = [r for r in rows if r["incidence"] is not None and r["incidence"] >= 0.75]
    tan = [r for r in tan if r["start_correct"]]
    nor = [r for r in nor if r["start_correct"]]
    if not tan or not nor:
        return {"n_tan": len(tan), "n_nor": len(nor), "pairs": 0}
    F = ["edit_norm", "start_margin", "endpoint_margin"]
    N = np.array([[r[f] for f in F] for r in nor])
    mu, sd = N.mean(0), N.std(0) + 1e-12
    pairs = []
    used = set()
    for t in tan:
        z = (np.array([t[f] for f in F]) - mu) / sd
        Zn = (N - mu) / sd
        cands = [(float(((Zn[i] - z) ** 2).sum()),
                  i) for i in range(len(nor))
                 if i not in used and nor[i]["start_label"] == t["start_label"]
                 and nor[i]["endpoint_label"] == t["endpoint_label"]]
        if not cands:
            continue
        cands.sort()
        used.add(cands[0][1])
        pairs.append((t, nor[cands[0][1]]))
    rec = []
    for t, n_ in pairs:
        mt = 1 if outcome_of(t)[0].startswith("miss") else 0
        mn = 1 if outcome_of(n_)[0].startswith("miss") else 0
        rec.append({"tan_miss": mt, "nor_miss": mn,
                    "parent_tan": t["parent_id"], "parent_nor": n_["parent_id"]})
    smds = {}
    for f in F:
        a = np.array([t[f] for t, _ in pairs])
        b = np.array([n_[f] for _, n_ in pairs])
        smds[f] = float(abs(a.mean() - b.mean()) /
                        max(1e-12, np.sqrt((a.var() + b.var()) / 2)))
    if pairs:
        d = np.array([r["tan_miss"] - r["nor_miss"] for r in rec])
        lo, hi = np.quantile(rng.choice(d, size=(10000, len(d))).mean(1),
                             [0.025, 0.975])
        res = {"n_tan": len(tan), "n_nor": len(nor), "pairs": len(pairs),
               "M_tan": round(float(np.mean([r["tan_miss"] for r in rec])), 4),
               "M_nor": round(float(np.mean([r["nor_miss"] for r in rec])), 4),
               "delta": round(float(d.mean()), 4),
               "delta_ci": [round(float(lo), 4), round(float(hi), 4)],
               "smd": {k: round(v, 4) for k, v in smds.items()},
               "smd_pass": bool(all(v < 0.10 for v in smds.values()))}
    else:
        res = {"n_tan": len(tan), "n_nor": len(nor), "pairs": 0}
    return res


def composition_attribution(rows):
    """Among AB-miss cases (endpoint wrong, atomics... start correct both paths
    proxy), fraction explained by path-level missed/abnormal update."""
    by_parent = {}
    for r in rows:
        by_parent.setdefault((r["parent_id"], r["fam_A"], r["fam_B"]), []).append(r)
    ab_miss = ab_hit = 0
    ab_miss_path_miss = ab_miss_path_abnormal = 0
    for k, ps in by_parent.items():
        if len(ps) != 2:
            continue
        if not all(p["start_correct"] for p in ps):
            continue
        ep_wrong = not ps[0]["endpoint_correct"]  # same AB endpoint
        if ep_wrong:
            ab_miss += 1
            outs = [outcome_of(p)[0] for p in ps]
            if any(o == "miss_no_cross" for o in outs):
                ab_miss_path_miss += 1
            elif any(o.startswith("abnormal") or o.startswith("miss") for o in outs):
                ab_miss_path_abnormal += 1
        else:
            ab_hit += 1
    return {"ab_miss": ab_miss, "ab_hit": ab_hit,
            "path_miss_frac": round(ab_miss_path_miss / ab_miss, 4) if ab_miss else None,
            "path_abnormal_frac": round(ab_miss_path_abnormal / ab_miss, 4) if ab_miss else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = load_rows(a.rows)
    prim = primary(rows)
    curve = binned_curve(prim)
    lags = [(r["incidence"], outcome_of(r)[1]) for r in prim
            if outcome_of(r)[1] is not None]
    matched = matched_tangent_normal(rows)
    attr = composition_attribution(rows)
    res = {"n_rows": len(rows), "n_primary": len(prim),
           "miss_curve": curve, "logistic": logistic_slope(prim),
           "lag_n": len(lags),
           "lag_median": round(float(np.median([l for _, l in lags])), 5) if lags else None,
           "matched": matched, "composition_attribution": attr}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "WP1_RESULTS.json", "w"), indent=1)
    with open(a.out / "WP1_RESULTS.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["n_primary", len(prim)])
        w.writerow(["matched_pairs", matched.get("pairs", 0)])
        w.writerow(["M_tan", matched.get("M_tan", "")])
        w.writerow(["M_nor", matched.get("M_nor", "")])
        w.writerow(["delta_M", matched.get("delta", "")])
        w.writerow(["delta_CI", matched.get("delta_ci", "")])
        w.writerow(["smd_pass", matched.get("smd_pass", "")])
        w.writerow(["ab_miss", attr["ab_miss"]])
        w.writerow(["path_miss_frac", attr["path_miss_frac"]])
        w.writerow(["path_abnormal_frac", attr["path_abnormal_frac"]])
    print(f"SAW WP1: n_primary={len(prim)} matched={matched.get('pairs', 0)} "
          f"dM={matched.get('delta')} attr={attr}")
    print(json.dumps({k: v for k, v in res.items()
                      if k in ("miss_curve", "matched")}, indent=1)[:2000])


if __name__ == "__main__":
    main()
