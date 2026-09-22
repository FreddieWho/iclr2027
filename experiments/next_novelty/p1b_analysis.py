#!/usr/bin/env python3
"""P1-B/C/D/E/F/G consolidated (dev-driven; thresholds frozen for confirm).

Rank base: path starts + ONE row per parent from singles (dedupe repeated
parent starts). Absolute thresholds frozen per model. Parent-cluster
bootstrap everywhere. No jittering of ties; tie groups merged.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(26092242)


def cluster_ci(parents, vals, n_boot=2000):
    ups = np.unique(parents)
    pmap = {u: np.nonzero(parents == u)[0] for u in ups}
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(ups, size=len(ups), replace=True)
        idx = np.concatenate([pmap[u] for u in draw])
        boots.append(vals[idx].mean())
    q = np.quantile(boots, [0.025, 0.975])
    return [round(float(q[0]), 4), round(float(q[1]), 4)]


def rankbase(paths, singles):
    # one row per parent from singles (x identical within parent)
    seen, rows = set(), []
    for r in singles:
        if r["parent"] not in seen:
            seen.add(r["parent"])
            rows.append((r["parent"], r["conf0"]))
    rows += [(r["parent"], r["confidence"]) for r in paths]
    return np.array([p for p, _ in rows]), np.array([c for _, c in rows])


def bin_by_thr(vals, q75, q50, q25):
    # mutually exclusive bins on ABSOLUTE cutoffs (top25/mid-high/mid-low/bottom25)
    # ties merged: a tied value belongs to exactly one bin (lower one)
    b = np.zeros(len(vals), int)
    b[vals >= q75] = 3
    b[(vals >= q50) & (vals < q75)] = 2
    b[(vals >= q25) & (vals < q50)] = 1
    return b  # 0..3


def analyze_model(paths, singles, out_thr=None):
    par_b, cf_b = rankbase(paths, singles)
    if out_thr is None:
        t25, t50, t75 = (float(np.quantile(cf_b, q)) for q in (0.75, 0.5, 0.25))
        thr = {"t25": t25, "t50": t50, "t75": t75}
    else:
        thr = out_thr
        t25, t50, t75 = thr["t25"], thr["t50"], thr["t75"]
    uv, cn = np.unique(cf_b, return_counts=True)
    R = {"n_paths": len(paths), "n_singles": len(singles),
         "n_rankbase": len(cf_b),
         "tie_share": round(float(cn[cn > 1].sum() / len(cf_b)), 4),
         "thresholds": {k: round(v, 4) for k, v in thr.items()}}
    # population items
    cur = ([(r["parent"], r["confidence"], r["start_correct"]) for r in paths] +
           [(r["parent"], r["conf0"], r["correct0"]) for r in singles])
    flp = ([(r["parent"], r["confidence"], r["end_correct"], r) for r in paths
            if r["start_correct"] and r["semantic_change"] == 1] +
           [(r["parent"], r["conf0"], r["correct1"], r) for r in singles
            if r["correct0"] == 1 and r["flip"] == 1])
    prv = [(r["parent"], r["conf0"], r["correct1"], r) for r in singles if r["y1"] == r["y0"]]
    sc = np.array([c for _, c, _ in cur]); sok = np.array([o for _, _, o in cur]).astype(int)
    bins = bin_by_thr(sc, t25, t50, t75)
    R["current_by_bin"] = []
    for i in range(4):
        sel = bins == i
        R["current_by_bin"].append({"bin": i, "n": int(sel.sum()),
            "err": round(float(1 - sok[sel].mean()), 4) if sel.sum() else None,
            "ci": cluster_ci(np.array([p for p, _, _ in cur])[sel], 1 - sok[sel]) if sel.sum() else None})
    for name, items, okidx in (("flip", flp, 2), ("preserve", prv, 2)):
        if not items:
            R[name + "_by_bin"] = []
            continue
        cf = np.array([c for _, c, _, *_ in items])
        ok = np.array([o for _, _, o, *_ in items]).astype(int)
        par = np.array([p for p, _, _, *_ in items])
        bb = bin_by_thr(cf, t25, t50, t75)
        rows = []
        for i in range(4):
            sel = bb == i
            rows.append({"bin": i, "n": int(sel.sum()),
                "err": round(float(1 - ok[sel].mean()), 4) if sel.sum() else None,
                "ci": cluster_ci(par[sel], 1 - ok[sel]) if sel.sum() else None})
        R[name + "_by_bin"] = rows
    # --- C/D: retention + rho scan (mixed AND uniformly-conditioned) ---
    R["retention_rho"] = []
    R["retention_update"] = []
    fall = [(r["parent"], r["confidence"], r["end_correct"]) for r in paths if r["start_correct"] and r["semantic_change"] == 1] + \
           [(r["parent"], r["conf0"], r["correct1"]) for r in singles if r["correct0"] == 1 and r["flip"] == 1]
    pall = [(r["parent"], r["conf0"], r["correct1"]) for r in singles if r["y1"] == r["y0"]]
    pall_u = [(r["parent"], r["conf0"], r["correct1"]) for r in singles
              if r["y1"] == r["y0"] and r["correct0"] == 1]
    jall = [(r["parent"], r["confidence"], r["start_correct"], r["end_correct"]) for r in paths] + \
           [(r["parent"], r["conf0"], r["correct0"], r["correct1"]) for r in singles]
    for frac, cutoff in (("top25", t25), ("top50", t50), ("top75", t75), ("all", -1e18)):
        fsel = [1 - o for _, c, o in fall if c >= cutoff]
        psel = [1 - o for _, c, o in pall if c >= cutoff]
        row = {"retain": frac, "n_flip": len(fsel), "n_prv": len(psel),
               "flip_err": round(float(np.mean(fsel)), 4) if fsel else None,
               "prv_err": round(float(np.mean(psel)), 4) if psel else None}
        for rho in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            if fsel and psel:
                row["R_rho_%.2f" % rho] = round(float((1 - rho) * np.mean(psel) + rho * np.mean(fsel)), 4)
        R["retention_rho"].append(row)
        # uniformly conditioned: preserve ALSO requires start-correct
        usel = [1 - o for _, c, o in pall_u if c >= cutoff]
        # joint: start wrong OR post-change wrong, same retained set
        # (retained = start states above cutoff; joint over starts+their ends)
        urow = {"retain": frac, "n_flip": len(fsel), "n_prv_u": len(usel),
                "flip_err": round(float(np.mean(fsel)), 4) if fsel else None,
                "prv_update_err": round(float(np.mean(usel)), 4) if usel else None}
        for rho in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            if fsel and usel:
                urow["Rupd_rho_%.2f" % rho] = round(float((1 - rho) * np.mean(usel) + rho * np.mean(fsel)), 4)
        jsel = [1 if (not sc) or (not ec) else 0 for _, c, sc, ec in jall if c >= cutoff]
        urow["n_joint"] = len(jsel)
        urow["R_joint"] = round(float(np.mean(jsel)), 4) if jsel else None
        R["retention_update"].append(urow)
    # --- E: regularized logistic + collinearity ---
    R["confound"] = confound(flp)
    # --- E2: matched support check on flip subset ---
    R["matched"] = matched_check(
        [r for r in paths if r["start_correct"] and r["semantic_change"] == 1] +
        [r for r in singles if r["correct0"] == 1 and r["flip"] == 1])
    # --- F: threshold control ---
    R["threshold_ctrl"] = thresh_ctrl([r for r in paths if r["start_correct"]])
    # --- temperature fit (dev singles NLL) ---
    R["temperature"] = temp_fit(singles)
    return R, thr


def temp_fit(singles):
    import math
    from scipy.optimize import minimize_scalar
    cf = np.array([r["conf0"] for r in singles])
    y = np.array([r["correct0"] for r in singles]).astype(float)
    # model P(correct) = sigmoid(a*conf + b) is saturated; temperature form:
    # P = sigmoid(conf / T) with T fit by NLL
    def nll(T):
        p = 1 / (1 + np.exp(-cf / max(T, 1e-9)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    r = minimize_scalar(nll, bounds=(0.05, 200.0), method="bounded")
    p = 1 / (1 + np.exp(-cf / r.x))
    brier = float(((p - y) ** 2).mean())
    base = float(-(y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean())).mean())
    return {"T_star": round(float(r.x), 3), "NLL": round(float(r.fun), 4),
            "NLL_null": round(base, 4), "Brier": round(brier, 4),
            "note": "ranks invariant under T>0 by construction"}


def confound(flp):
    # features: conf, m0, mAB?, dist, norm, class, family(one-hot top), parent size note
    import math
    rows = []
    for p, c, o, r in flp:
        rows.append(r)
    if len(rows) < 50:
        return {"n": len(rows), "note": "too few"}
    fams = sorted(set(r.get("edit_fam", r.get("fam", "?")) for r in rows))[:8]
    X, y, par = [], [], []
    for r in rows:
        cf = r.get("confidence", r.get("conf0", 0)) or 0
        # P1-C1: true PATH-start margin (m_start), never base m0 for paths
        m_start = r.get("m_start", r.get("m0", 0)) or 0
        m_end = r.get("m_end", r.get("m1", 0)) or 0
        v = [math.log1p(cf), m_start, m_end,
             r.get("cross_dist", 0) or 0, r.get("edit_norm", 0) or 0,
             float(r.get("start_lab", r.get("y0", 0)))]
        ef = r.get("edit_fam", r.get("fam", "?"))
        v += [1.0 if ef == f else 0.0 for f in fams]
        X.append(v)
        y.append(1 - int(r["end_correct"] if "end_correct" in r else r["correct1"]))
        par.append(r["parent"])
    X = np.array(X)
    y = np.array(y)
    C = np.corrcoef(X.T)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Zs = (X - mu) / sd
    m_full = LogisticRegression(C=1.0, max_iter=5000).fit(Zs, y)
    m_conf = LogisticRegression(C=1.0, max_iter=5000).fit(Zs[:, [0]], y)
    m_noconf = LogisticRegression(C=1.0, max_iter=5000).fit(Zs[:, 1:], y)
    def auc(m, Xx):
        try:
            return round(float(roc_auc_score(y, m.predict_proba(Xx)[:, 1])), 4)
        except Exception:
            return None
    # clustered CI on AUC gain via parent bootstrap (dev 500 for speed)
    gains = []
    ups = np.unique(par)
    for _ in range(500):
        draw = rng.choice(ups, size=len(ups), replace=True)
        idx = np.concatenate([np.nonzero(np.array(par) == u)[0] for u in draw])
        try:
            g = roc_auc_score(y[idx], m_full.predict_proba(Zs[idx])[:, 1]) - \
                roc_auc_score(y[idx], m_noconf.predict_proba(Zs[idx][:, 1:])[:, 1])
            gains.append(g)
        except Exception:
            pass
    ci = [round(float(x), 4) for x in np.quantile(gains, [0.025, 0.975])] if gains else None
    cv = grouped_cv(X, y, np.array(par))
    return {"n": len(y), "n_feat": X.shape[1],
            "grouped_cv": cv,
            "max_abs_corr": round(float(np.abs(C - np.eye(C.shape[0])).max()), 3),
            "auc_conf_only": auc(m_conf, Zs[:, [0]]),
            "auc_noconf": auc(m_noconf, Zs[:, 1:]),
            "auc_full": auc(m_full, Zs),
            "auc_gain_ci": ci,
            "coef_full": [round(float(c), 3) for c in m_full.coef_[0][:5]]}


def grouped_cv(X, y, par, k=5, n_boot=500):
    """5-fold grouped-by-parent CV; refit per bootstrap resample."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    ups = np.unique(par)
    folds = np.array_split(rng.permutation(ups), k)
    out = {}
    for name, cols in (("conf", [0]), ("noconf", list(range(1, X.shape[1]))),
                       ("full", list(range(X.shape[1])))):
        aucs = []
        oof = np.full(len(y), np.nan)
        for f in range(k):
            te = np.isin(par, folds[f])
            tr = ~te
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            mu, sd = X[tr][:, cols].mean(0), X[tr][:, cols].std(0) + 1e-12
            m = LogisticRegression(C=1.0, max_iter=5000).fit(
                (X[tr][:, cols] - mu) / sd, y[tr])
            p = m.predict_proba((X[te][:, cols] - mu) / sd)[:, 1]
            oof[te] = p
            try:
                aucs.append(roc_auc_score(y[te], p))
            except Exception:
                pass
        ok = ~np.isnan(oof)
        try:
            pooled = round(float(roc_auc_score(y[ok], oof[ok])), 4)
        except Exception:
            pooled = None
        # parent bootstrap with FULL refit per resample
        gains = []
        for _ in range(n_boot):
            draw = rng.choice(ups, size=len(ups), replace=True)
            idx = np.concatenate([np.nonzero(par == u)[0] for u in draw])
            Xa, ya = X[idx], y[idx]
            if len(np.unique(ya)) < 2:
                continue
            mua, sda = Xa.mean(0), Xa.std(0) + 1e-12
            try:
                ma = LogisticRegression(C=1.0, max_iter=2000).fit(
                    (Xa - mua) / sda, ya)
                gains.append(roc_auc_score(ya, ma.predict_proba((Xa - mua) / sda)[:, 1]))
            except Exception:
                pass
        ci = ([round(float(x), 4) for x in np.quantile(gains, [0.025, 0.975])]
              if gains else None)
        out[name] = {"pooled_oof_auc": pooled,
                     "fold_aucs": [round(float(a), 4) for a in aucs],
                     "refit_boot_ci": ci}
    return out


def matched_check(rows):
    """Coarsened matching on (start-margin tertile, norm tertile, class);
    reports matched n + SMDs; flags unresolved support."""
    import math
    recs = []
    for r in rows:
        cf = r.get("confidence", r.get("conf0", 0)) or 0
        recs.append((r["parent"],
                     r.get("m_start", r.get("m0", 0)) or 0,
                     r.get("edit_norm", 0) or 0,
                     int(r.get("start_lab", r.get("y0", 0))),
                     math.log1p(cf),
                     1 - int(r["end_correct"] if "end_correct" in r else r["correct1"])))
    if len(recs) < 100:
        return {"n": len(recs), "note": "too few"}
    mg = np.array([x[1] for x in recs])
    nm = np.array([x[2] for x in recs])
    mt = np.quantile(mg, [1 / 3, 2 / 3])
    nt = np.quantile(nm, [1 / 3, 2 / 3])
    cells = {}
    for i, (p, m, n, c, f, y) in enumerate(recs):
        key = (int(np.digitize(m, mt)), int(np.digitize(n, nt)), c)
        cells.setdefault(key, []).append((f, y, p))
    deltas, ws, smds = [], [], []
    for key, v in cells.items():
        if len(v) < 10:
            continue
        f = np.array([x[0] for x in v])
        med = np.median(f)
        hi = [x for x in v if x[0] >= med]
        lo = [x for x in v if x[0] < med]
        if len(hi) < 3 or len(lo) < 3:
            continue
        deltas.append((np.mean([x[1] for x in hi]) - np.mean([x[1] for x in lo]),
                       len(v)))
        ws.append(len(v))
    if not deltas or sum(ws) < 100:
        return {"n_cells": len(deltas), "n_matched": int(sum(ws)),
                "note": "support too thin: UNRESOLVED confound"}
    est = float(np.average([d for d, _ in deltas], weights=ws))
    return {"n_cells": len(deltas), "n_matched": int(sum(ws)),
            "matched_delta": round(est, 4), "note": "ok"}


def thresh_ctrl(sub):
    cf, df, y = [], [], []
    for r in sub:
        pred = r.get("start_pred", None)
        el = r.get("end_logit", None)
        if pred is None or el is None:
            continue
        s = r["confidence"] if pred == 1 else -r["confidence"]
        dd = el - s
        cf.append(abs(s))
        df.append(abs(s) / max(1e-9, abs(dd)))
        y.append(1 - int(r["end_correct"]))
    cf, df, y = map(np.array, (cf, df, y))
    if len(y) < 20 or len(np.unique(y)) < 2:
        return {"n": len(y), "note": "too few/single-class"}
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    a1 = LogisticRegression(C=1.0, max_iter=2000).fit(np.log1p(cf).reshape(-1, 1), y)
    a2 = LogisticRegression(C=1.0, max_iter=2000).fit(np.log1p(df).reshape(-1, 1), y)
    X = np.stack([np.log1p(cf), np.log1p(df)], 1)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    a3 = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, y)
    def auc(m, Xx):
        try:
            return round(float(roc_auc_score(y, m.predict_proba(Xx)[:, 1])), 4)
        except Exception:
            return None
    return {"n": len(y), "auc_conf": auc(a1, np.log1p(cf).reshape(-1, 1)),
            "auc_norm": auc(a2, np.log1p(df).reshape(-1, 1)),
            "auc_joint": auc(a3, (X - mu) / sd),
            "coef_joint": [round(float(c), 4) for c in a3.coef_[0]]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--freeze", default="")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.load(open(a.freeze)) if a.freeze else {}
    res = {"thresholds": {}}
    for seed in (11, 23, 47):
        for mname in ("clean", "flipmine"):
            d = json.load(open(Path(a.indir) / ("unified_s%d_%s.json" % (seed, mname))))
            ft = freeze.get(f"s{seed}_{mname}")
            R, thr = analyze_model(d["paths"], d["singles"], out_thr=ft)
            res[f"s{seed}_{mname}"] = R
            res["thresholds"][f"s{seed}_{mname}"] = {k: round(v, 4) for k, v in thr.items()}
            print(f"s{seed}_{mname}: flip={[b['err'] for b in R['flip_by_bin']]} thr_ok", flush=True)
    json.dump(res, open(out / "P1B.json", "w"), indent=1)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
