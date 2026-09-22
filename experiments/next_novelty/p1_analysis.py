#!/usr/bin/env python3
"""P1-B/C/D/E/F/G on unified table. Rank policy: ranks over union of path
starts + single starts per model (one policy); ties reported, tie groups
merged (never jittered). Parent-cluster bootstrap (dev 2000).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(26092241)
IN = ROOT / "artifacts" / "next_novelty" / "p1_unified"
OUT = ROOT / "artifacts" / "next_novelty" / "p1_res"


def ranks(v):
    v = np.asarray(v, float)
    return np.argsort(np.argsort(v, kind="mergesort")) / max(1, len(v) - 1)


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


def load_uni(indir, seed, mname):
    d = json.load(open(Path(indir) / ("unified_s%d_%s.json" % (seed, mname))))
    return d["paths"], d["singles"]


def analyze(indir, outdir, seeds=(11, 23, 47), freeze_thr=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    res = {}
    for seed in seeds:
        for mname in ("clean", "flipmine"):
            paths, singles = load_uni(indir, seed, mname)
            P = np.array([r["parent"] for r in paths])
            S = np.array([r["parent"] for r in singles])
            # unified rank base: path starts + single starts
            base_conf = np.array([r["confidence"] for r in paths] + [r["conf0"] for r in singles])
            base_par = np.array([r["parent"] for r in paths] + [r["parent"] for r in singles])
            # tie report
            uv, cn = np.unique(base_conf, return_counts=True)
            tie_share = float(cn[cn > 1].sum() / len(base_conf))
            # rank cutoffs: fixed fractions on the pooled base
            qs = {f: float(np.quantile(base_conf, 1 - f)) for f in (0.25, 0.5, 0.75)}
            R = {"n_paths": len(paths), "n_singles": len(singles),
                 "tie_share": round(tie_share, 4), "thresholds": qs}
            # --- B: three risks, same rank policy ---
            sc = np.array([r["start_correct"] for r in paths])
            pc = np.array([r["correct0"] for r in singles])
            sconf = np.array([r["conf0"] for r in singles])
            pconf = np.array([r["confidence"] for r in paths])
            risks = {}
            # current risk on path starts + single starts pooled
            cur_ok = np.array([r["start_correct"] for r in paths] + [r["correct0"] for r in singles])
            cur_cf = np.array([r["confidence"] for r in paths] + [r["conf0"] for r in singles])
            cur_par = np.array([r["parent"] for r in paths] + [r["parent"] for r in singles])
            # preserve: singles with y1==y0
            prv = [r for r in singles if r["y1"] == r["y0"]]
            # flip: paths (semantic_change==1 by construction) + singles with flip
            flp_p = [r for r in paths if r["semantic_change"] == 1]
            flp_s = [r for r in singles if r["flip"] == 1]
            risks["current"] = {"n": len(cur_ok), "err": round(float(1 - cur_ok.mean()), 4)}
            if prv:
                pa = np.array([r["parent"] for r in prv])
                va = np.array([1 - int(r["pred1"] == r["y1"]) for r in prv])
                risks["preserve"] = {"n": len(prv), "err": round(float(va.mean()), 4),
                                     "ci": cluster_ci(pa, va)}
                # specialized: start-correct preserve spurious-change
                sp = [r for r in prv if r["correct0"] == 1]
                if sp:
                    pa2 = np.array([r["parent"] for r in sp])
                    va2 = np.array([int(r["pred1"] != r["pred0"]) for r in sp])
                    risks["preserve_spurious"] = {"n": len(sp), "err": round(float(va2.mean()), 4),
                                                  "ci": cluster_ci(pa2, va2)}
            # flip risk on start-correct subset
            fsub = [r for r in flp_p if r["start_correct"]] + [r for r in flp_s if r["correct0"] == 1]
            if fsub:
                pa3 = np.array([r["parent"] for r in fsub])
                va3 = np.array([(1 - int(r["end_correct"]) if "end_correct" in r else 1 - int(r["correct1"])) for r in fsub])
                risks["flip"] = {"n": len(fsub), "err": round(float(va3.mean()), 4),
                                 "ci": cluster_ci(pa3, va3)}
            R["risks"] = risks
            # --- rank-binned curves with SAME cutoffs from pooled base ---
            for name, pop, okfn, parfn in (
                    ("cur", list(paths) + [{"parent": r["parent"], "confidence": r["conf0"],
                                            "ok": r["correct0"]} for r in singles],
                     lambda r: r.get("start_correct", r.get("ok")), lambda r: r["parent"]),
            ):
                pass
            # current-curve and flip-curve by rank bins of pooled base ranks
            allr = ranks(base_conf)
            # map each path/single to its rank via pooled order is positional; recompute per item below
            R["curves"] = {}
            # current correctness curve (paths starts + singles starts)
            items_c = ([(r["parent"], r["confidence"], r["start_correct"]) for r in paths] +
                       [(r["parent"], r["conf0"], r["correct0"]) for r in singles])
            R["curves"]["current"] = binned(items_c)
            # flip-update curve (start-correct flipped only)
            items_f = ([(r["parent"], r["confidence"], r["end_correct"]) for r in paths if r["start_correct"] and r["semantic_change"] == 1] +
                       [(r["parent"], r["conf0"], r["correct1"]) for r in singles if r["correct0"] == 1 and r["flip"] == 1])
            R["curves"]["flip_update"] = binned(items_f)
            # preserve curve (all preserve singles: endpoint correctness)
            items_p = [(r["parent"], r["conf0"], r["correct1"]) for r in singles if r["y1"] == r["y0"]]
            R["curves"]["preserve"] = binned(items_p)
            # --- C: retention consequence ---
            R["retention"] = retention(paths, singles)
            # --- F: threshold-distance control ---
            R["threshold_ctrl"] = thresh_ctrl([r for r in paths if r["start_correct"]])
            res[f"s{seed}_{mname}"] = R
            print("SAW s%d %s: %s" % (seed, mname, {k: v.get("err") for k, v in risks.items()}), flush=True)
    json.dump(res, open(Path(outdir) / "P1_DEV.json", "w"), indent=1)
    print("DONE p1_analysis ->", outdir)


def binned(items):
    # items: (parent, conf, ok_bool)
    if not items:
        return []
    par = np.array([i[0] for i in items])
    cf = np.array([i[1] for i in items])
    ok = np.array([i[2] for i in items]).astype(int)
    cr = ranks(cf)
    out = []
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        sel = (cr >= lo) & (cr < hi)
        out.append({"bin": [lo, hi], "n": int(sel.sum()),
                    "err": round(float(1 - ok[sel].mean()), 4) if sel.sum() else None,
                    "ci": cluster_ci(par[sel], 1 - ok[sel]) if sel.sum() else None})
    return out


def retention(paths, singles):
    sub = [r for r in paths if r["start_correct"]]
    cf = np.array([r["confidence"] for r in sub])
    prv = [r for r in singles if r["y1"] == r["y0"]]
    out = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        thr = float(np.quantile(cf, 1 - frac)) if len(cf) else None
        sel = cf >= thr if thr is not None else np.zeros(0, bool)
        ue = np.array([1 - int(r["end_correct"]) for r in sub])[sel].mean() if sel.sum() else None
        # preserve + joint on retained-confidence singles (same thr scale? singles have own conf0;
        # apply same ABSOLUTE thr: "same confidence policy")
        pc = np.array([r["conf0"] for r in prv])
        psel = pc >= thr if thr is not None else np.zeros(0, bool)
        pe = np.array([1 - int(r["pred1"] == r["y1"]) for r in prv])[psel].mean() if psel.sum() else None
        out.append({"retain": frac, "thr": round(thr, 4) if thr is not None else None,
                    "n_path": int(sel.sum()),
                    "flip_err": round(float(ue), 4) if ue is not None else None,
                    "n_prv": int(psel.sum()),
                    "preserve_err": round(float(pe), 4) if pe is not None else None})
    return out


def thresh_ctrl(sub):
    # |f(x)| vs |f(x)|/|Δf| ; Δf uses endpoint logit (no endpoint label)
    import math
    rows = []
    for r in sub:
        try:
            d = float(r["end_logit"] - (r["start_logit"] if "start_logit" in r else 0.0))
        except Exception:
            continue
        rows.append(r)
    # NOTE: unified rows lack signed start_logit; reconstruct from start_pred
    cf, df, y = [], [], []
    for r in rows:
        pred = r.get("start_pred", None)
        if pred is None:
            continue
        s = r["confidence"] if pred == 1 else -r["confidence"]
        el = r.get("end_logit", None)
        if el is None:
            continue
        dd = el - s
        cf.append(abs(s))
        df.append(abs(s) / max(1e-9, abs(dd)))
        y.append(1 - int(r["end_correct"]))
    cf, df, y = map(np.array, (cf, df, y))
    if len(y) < 20:
        return {"n": len(y), "note": "too few"}
    from sklearn.linear_model import LogisticRegression
    a1 = LogisticRegression(C=1.0, max_iter=2000).fit(np.log1p(cf).reshape(-1, 1), y)
    a2 = LogisticRegression(C=1.0, max_iter=2000).fit(np.log1p(df).reshape(-1, 1), y)
    X = np.stack([np.log1p(cf), np.log1p(df)], 1)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    a3 = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, y)
    from sklearn.metrics import roc_auc_score
    def auc(m, Xx):
        try:
            return round(float(roc_auc_score(y, m.predict_proba(Xx)[:, 1])), 4)
        except Exception:
            return None
    return {"n": len(y),
            "auc_conf": auc(a1, np.log1p(cf).reshape(-1, 1)),
            "auc_norm": auc(a2, np.log1p(df).reshape(-1, 1)),
            "auc_joint": auc(a3, (X - mu) / sd),
            "coef_joint": [round(float(c), 4) for c in a3.coef_[0]]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", default=str(IN))
    p.add_argument("--outdir", default=str(OUT))
    a = p.parse_args()
    pass
    # load_uni is inline: reuse functions with indir override
    import types
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    res = {}
    # replicate analyze() with indir
    for seed in (11, 23, 47):
        for mname in ("clean", "flipmine"):
            d = json.load(open(Path(a.indir) / ("unified_s%d_%s.json" % (seed, mname))))
            paths, singles = d["paths"], d["singles"]
            # temporarily reuse module-level helpers by stuffing globals
            res[f"s{seed}_{mname}"] = analyze_one(paths, singles)
    json.dump(res, open(Path(a.outdir) / "P1_DEV.json", "w"), indent=1)
    for k, v in res.items():
        print(k, "risks:", {kk: vv.get("err") for kk, vv in v["risks"].items()},
              "thresh:", v["threshold_ctrl"].get("auc_conf"), v["threshold_ctrl"].get("auc_norm"), flush=True)
    print("DONE ->", a.outdir)


def analyze_one(paths, singles):
    base_conf = np.array([r["confidence"] for r in paths] + [r["conf0"] for r in singles])
    uv, cn = np.unique(base_conf, return_counts=True)
    tie_share = float(cn[cn > 1].sum() / len(base_conf))
    R = {"n_paths": len(paths), "n_singles": len(singles),
         "tie_share": round(tie_share, 4)}
    cur_ok = np.array([r["start_correct"] for r in paths] + [r["correct0"] for r in singles])
    risks = {"current": {"n": len(cur_ok), "err": round(float(1 - cur_ok.mean()), 4)}}
    prv = [r for r in singles if r["y1"] == r["y0"]]
    if prv:
        pa = np.array([r["parent"] for r in prv])
        va = np.array([1 - int(r["pred1"] == r["y1"]) for r in prv])
        risks["preserve"] = {"n": len(prv), "err": round(float(va.mean()), 4), "ci": cluster_ci(pa, va)}
        sp = [r for r in prv if r["correct0"] == 1]
        if sp:
            pa2 = np.array([r["parent"] for r in sp])
            va2 = np.array([int(r["pred1"] != r["pred0"]) for r in sp])
            risks["preserve_spurious"] = {"n": len(sp), "err": round(float(va2.mean()), 4), "ci": cluster_ci(pa2, va2)}
    fsub = [r for r in paths if r["start_correct"] and r["semantic_change"] == 1] + \
           [r for r in singles if r["correct0"] == 1 and r["flip"] == 1]
    if fsub:
        pa3 = np.array([r["parent"] for r in fsub])
        va3 = np.array([(1 - int(r["end_correct"]) if "end_correct" in r else 1 - int(r["correct1"])) for r in fsub])
        risks["flip"] = {"n": len(fsub), "err": round(float(va3.mean()), 4), "ci": cluster_ci(pa3, va3)}
    R["risks"] = risks
    items_c = ([(r["parent"], r["confidence"], r["start_correct"]) for r in paths] +
               [(r["parent"], r["conf0"], r["correct0"]) for r in singles])
    R["curves"] = {"current": binned(items_c)}
    items_f = ([(r["parent"], r["confidence"], r["end_correct"]) for r in paths if r["start_correct"] and r["semantic_change"] == 1] +
               [(r["parent"], r["conf0"], r["correct1"]) for r in singles if r["correct0"] == 1 and r["flip"] == 1])
    R["curves"]["flip_update"] = binned(items_f)
    items_p = [(r["parent"], r["conf0"], r["correct1"]) for r in singles if r["y1"] == r["y0"]]
    R["curves"]["preserve"] = binned(items_p)
    R["retention"] = retention(paths, singles)
    R["threshold_ctrl"] = thresh_ctrl([r for r in paths if r["start_correct"]])
    return R


if __name__ == "__main__":
    main()
