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
    R["_retention_rho_note"] = (
        "R(tau,rho) = (1-rho)*mean(preserve errors) + rho*mean(flip errors) on the "
        "retained subset; it is a mixed risk under an explicit reweighting of the "
        "retained population, NOT an unselected deployment change rate. "
        "flip cohort = start-correct semantic-change rows; preserve cohort = "
        "semantics-preserving singles (unscreened start correctness).")
    for frac, cutoff in (("top25", t25), ("top50", t50), ("top75", t75), ("all", -1e18)):
        fsel = [1 - o for _, c, o in fall if c >= cutoff]
        psel = [1 - o for _, c, o in pall if c >= cutoff]
        row = {"retain": frac, "n_flip": len(fsel), "n_prv": len(psel),
               "retained_frac_flip_cohort": round(len(fsel) / len(fall), 4) if fall else None,
               "retained_frac_preserve_cohort": round(len(psel) / len(pall), 4) if pall else None,
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
    # --- E2: stratified high-vs-low confidence comparison on flip subset ---
    R["stratified"] = stratified_check(
        [r for r in paths if r["start_correct"] and r["semantic_change"] == 1] +
        [r for r in singles if r["correct0"] == 1 and r["flip"] == 1])
    # kept under the old name for backward compatibility of artifacts
    R["matched"] = R["stratified"]
    # --- E3: coverage + population accounting + sensitivity (R2.2/R2.3) ---
    R["population"] = population_stats(paths, singles)
    R["coverage"] = coverage_report(paths, singles, (t25, t50, t75))
    R["sensitivity_stratified_cap"] = sensitivity_stratified_cap(
        [r for r in paths if r["start_correct"] and r["semantic_change"] == 1] +
        [r for r in singles if r["correct0"] == 1 and r["flip"] == 1],
        caps=(10, 20, 40))
    R["sensitivity_equal_parent"] = sensitivity_equal_parent_weight(
        [r for r in paths if r["start_correct"] and r["semantic_change"] == 1] +
        [r for r in singles if r["correct0"] == 1 and r["flip"] == 1])
    R["edit_family_distribution"] = edit_family_distribution(paths, singles)
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
        # P1-C1: true PATH-start margin (m_start), never base m0 for paths;
        # 0 is a legitimate margin value (tied logit), not a missing marker,
        # so we do not coerce it. Rows lacking the field get m0 only if
        # their record is a single (which carries the base margin).
        m_start = r.get("m_start") if r.get("m_start") is not None else r.get("m0", 0.0)
        m_end = r.get("m_end") if r.get("m_end") is not None else r.get("m1", 0.0)
        m_start = float(m_start) if m_start is not None else 0.0
        m_end = float(m_end) if m_end is not None else 0.0
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
    """5-fold grouped-by-parent CV (R2.1).

    Pooled OOF predictions are computed once per column-set (train-fold
    standardization only). Bootstrap CIs are PAIRED intervals of OOF AUC
    differences on identical parent resamples (conf-vs-noconf,
    full-vs-noconf), so the intervals are comparable per draw. A separate
    in-sample diagnostic (in_sample_diagnostic) is reported and must never
    be mixed with held-out numbers.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    ups = np.unique(par)
    folds = np.array_split(rng.permutation(ups), k)
    out = {}
    oof = {}
    for name, cols in (("conf", [0]), ("noconf", list(range(1, X.shape[1]))),
                       ("full", list(range(X.shape[1])))):
        aucs = []
        oo = np.full(len(y), np.nan)
        for f in range(k):
            te = np.isin(par, folds[f])
            tr = ~te
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            mu, sd = X[tr][:, cols].mean(0), X[tr][:, cols].std(0) + 1e-12
            m = LogisticRegression(C=1.0, max_iter=5000).fit(
                (X[tr][:, cols] - mu) / sd, y[tr])
            p = m.predict_proba((X[te][:, cols] - mu) / sd)[:, 1]
            oo[te] = p
            try:
                aucs.append(roc_auc_score(y[te], p))
            except Exception:
                pass
        ok = ~np.isnan(oo)
        try:
            pooled = round(float(roc_auc_score(y[ok], oo[ok])), 4)
        except Exception:
            pooled = None
        # in-sample diagnostic (NEVER reported as held-out)
        mu_a, sd_a = X[:, cols].mean(0), X[:, cols].std(0) + 1e-12
        m_a = LogisticRegression(C=1.0, max_iter=5000).fit(
            (X[:, cols] - mu_a) / sd_a, y)
        in_sample = round(float(roc_auc_score(
            y, m_a.predict_proba((X[:, cols] - mu_a) / sd_a)[:, 1])), 4)
        oof[name] = (oo, cols)
        out[name] = {"pooled_oof_auc": pooled,
                     "fold_aucs": [round(float(a), 4) for a in aucs],
                     "in_sample_diagnostic": in_sample}
    # paired OOF difference intervals on identical parent draws
    def _auc_on(mask, pred):
        yy = y[mask]
        try:
            return roc_auc_score(yy, pred[mask])
        except Exception:
            return None
    for pair, (a_nm, b_nm) in (("oof_auc_diff_conf_vs_noconf", ("conf", "noconf")),
                               ("oof_auc_diff_full_vs_noconf", ("full", "noconf"))):
        pa, pb = oof[a_nm][0], oof[b_nm][0]
        ok = ~np.isnan(pa) & ~np.isnan(pb)
        # parent-cluster resampling on the commonly-scored rows
        par_ok = par[ok]
        ups_ok = np.unique(par_ok)
        pmap = {u: np.nonzero(par_ok == u)[0] for u in ups_ok}
        diffs = []
        for _ in range(n_boot):
            draw = rng.choice(ups_ok, size=len(ups_ok), replace=True)
            idx = np.concatenate([pmap[u] for u in draw])
            ia = _auc_on(ok, pa)
            ib = _auc_on(ok, pb)
            if ia is None or ib is None:
                continue
            # recompute AUC on the bootstrapped row subset
            yy = y[ok][idx]
            try:
                aa = roc_auc_score(yy, pa[ok][idx])
                bb = roc_auc_score(yy, pb[ok][idx])
                diffs.append(aa - bb)
            except Exception:
                pass
        ci = ([round(float(x), 4) for x in np.quantile(diffs, [0.025, 0.975])]
              if diffs else None)
        pt = _auc_on(ok, pa); p2 = _auc_on(ok, pb)
        out[pair] = {"point_diff": round(float(pt - p2), 4) if pt is not None and p2 is not None else None,
                     "paired_oof_diff_ci": ci,
                     "n_common": int(ok.sum())}
    out["_note"] = ("Pooled OOF AUCs with train-fold standardization; paired CI "
                    "conditions on this CV fit (parent-cluster resampling of "
                    "commonly-scored OOF rows); in_sample_diagnostic is not held-out.")
    return out


def stratified_check(rows):
    """Stratified high-vs-low confidence comparison (formerly 'matched_check').

    Coarsens (start-margin tertile, norm tertile, class) and compares
    above/below-median confidence WITHIN each cell. This is a stratified
    analysis, not matched matching: we report per-cell balance (n per arm,
    standardized mean difference on log1p-confidence) and a support note.
    Large n does not imply common support or a matched cohort.
    """
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
    deltas, ws, per_cell = [], [], []
    for key, v in cells.items():
        if len(v) < 10:
            continue
        f = np.array([x[0] for x in v])
        med = np.median(f)
        hi = [x for x in v if x[0] >= med]
        lo = [x for x in v if x[0] < med]
        if len(hi) < 3 or len(lo) < 3:
            continue
        f_hi = np.array([x[0] for x in hi])
        f_lo = np.array([x[0] for x in lo])
        smd = float((f_hi.mean() - f_lo.mean()) /
                    np.sqrt((f_hi.var() + f_lo.var()) / 2 + 1e-12))
        deltas.append((np.mean([x[1] for x in hi]) - np.mean([x[1] for x in lo]),
                       len(v)))
        ws.append(len(v))
        per_cell.append({"cell": [int(k) for k in key], "n": len(v),
                         "n_hi": len(hi), "n_lo": len(lo),
                         "smd_log_conf": round(smd, 3)})
    if not deltas or sum(ws) < 100:
        return {"n_cells": len(deltas), "n_matched": int(sum(ws)),
                "cells": per_cell,
                "note": "support too thin: UNRESOLVED stratified comparison"}
    est = float(np.average([d for d, _ in deltas], weights=ws))
    return {"n_cells": len(deltas), "n_matched": int(sum(ws)),
            "stratified_delta": round(est, 4),
            "cells": per_cell,
            "note": "stratified (within-cell high-vs-low), NOT matched; see per-cell SMD"}


# alias so older artifacts/tests referencing matched_check still import
matched_check = stratified_check


def population_stats(paths, singles):
    """R2.2: row / distinct-start / quartet / parent counts (parent = CI unit)."""
    path_parents = set(r["parent"] for r in paths)
    sing_parents = set(r["parent"] for r in singles)
    starts = set()
    for r in paths:
        starts.add((r["parent"], r.get("ptype", "")))
    for r in singles:
        starts.add((r["parent"], r.get("eid", "")))
    return {"n_path_rows": len(paths),
            "n_single_rows": len(singles),
            "n_quartet_parents": len(path_parents),
            "n_single_parents": len(sing_parents),
            "n_distinct_starts": len(starts),
            "n_parents_total": len(path_parents | sing_parents),
            "ci_unit": "parent"}


def coverage_report(paths, singles, thr3):
    """R2.3: top25 is a DEV reference threshold; report ACTUAL retained coverage."""
    t25, t50, t75 = thr3
    pops = {
        "paths_start_states": [(r["parent"], r["confidence"]) for r in paths],
        "singles_start_states": [(r["parent"], r["conf0"]) for r in singles],
    }
    out = {"thresholds": {"t25": round(float(t25), 4),
                          "t50": round(float(t50), 4),
                          "t75": round(float(t75), 4)},
           "note": "top25 = dev reference quantile; actual retained coverage below",
           "actual_coverage": {}}
    for name, items in pops.items():
        cf = np.array([c for _, c in items])
        out["actual_coverage"][name] = {
            "n": len(cf),
            "retained_at_t25": int((cf >= t25).sum()),
            "share_at_t25": round(float((cf >= t25).mean()), 4),
            "retained_at_t50": int((cf >= t50).sum()),
            "retained_at_t75": int((cf >= t75).sum()),
        }
    return out


def sensitivity_stratified_cap(rows, caps=(10, 20, 40)):
    """R2.2: truncation sensitivity — keep at most cap rows per parent."""
    import collections
    res = {}
    for cap in caps:
        seen = collections.defaultdict(int)
        kept = []
        for r in rows:
            p = r["parent"]
            if seen[p] < cap:
                seen[p] += 1
                kept.append(r)
        errs = [(r["parent"],
                 1 - int(r["end_correct"] if "end_correct" in r else r["correct1"]))
                for r in kept]
        if errs:
            y = np.array([e for _, e in errs])
            res[f"cap{cap}"] = {"n": len(y),
                                "flip_err_overall": round(float(y.mean()), 4),
                                "n_parents": len(seen)}
        else:
            res[f"cap{cap}"] = {"n": 0}
    return res


def sensitivity_equal_parent_weight(rows):
    """R2.2: each parent contributes one vote (mean of its rows)."""
    import collections
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r["parent"]].append(
            1 - int(r["end_correct"] if "end_correct" in r else r["correct1"]))
    if not byp:
        return {"n": 0}
    per_parent = np.array([float(np.mean(v)) for v in byp.values()])
    return {"n_parents": len(byp),
            "flip_err_parent_weighted": round(float(per_parent.mean()), 4)}


def edit_family_distribution(paths, singles):
    """R2.2: edit-family/node/displacement distributions for the manuscript."""
    import collections
    def _dist(rows, fam_key):
        c = collections.Counter()
        nrm = []
        for r in rows:
            f = r.get(fam_key, r.get("edit_fam", r.get("fam", "?")))
            c[f] += 1
            nrm.append(float(r.get("edit_norm", 0) or 0))
        nrm = np.array(nrm)
        return {"families": dict(c.most_common()),
                "edit_norm": {"mean": round(float(nrm.mean()), 4),
                              "p25": round(float(np.quantile(nrm, 0.25)), 4),
                              "median": round(float(np.quantile(nrm, 0.5)), 4),
                              "p75": round(float(np.quantile(nrm, 0.75)), 4)}}
    return {"paths": _dist(paths, "edit_fam"), "singles": _dist(singles, "fam")}


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
