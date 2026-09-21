#!/usr/bin/env python3
"""P1 unified analysis: dual test, stratified curves, selection (P1-C),
repair residual (P1-E), temperature check, analytic-oracle control.
Parent-cluster bootstrap (dev 2000). Per-model confidence ranks.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "artifacts" / "p123_upgrade" / "p1"
OUT = ROOT / "artifacts" / "p123_upgrade" / "p1"
rng = np.random.default_rng(26092231)


def ranks(v):
    return np.argsort(np.argsort(np.asarray(v, dtype=float))) / max(1, len(v) - 1)


def cluster_ci(parents, vals, n_boot=2000):
    ups = np.unique(parents)
    pmap = {u: np.nonzero(parents == u)[0] for u in ups}
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(ups, size=len(ups), replace=True)
        idx = np.concatenate([pmap[u] for u in draw])
        boots.append(vals[idx].mean())
    return [round(float(x), 4) for x in np.quantile(boots, [0.025, 0.975])]


def load_coord(seed, mname):
    d = json.load(open(IN / ("p1coord_s%d_%s.json" % (seed, mname))))
    return d["rows"], d["ref"]


def coord_dual(seed, mname):
    rows, ref = load_coord(seed, mname)
    rc, ro = np.array(ref["conf"]), np.array(ref["ok"])
    # (1) static curve on reference
    rr = ranks(rc)
    static_curve = []
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        sel = (rr >= lo) & (rr < hi)
        static_curve.append({"rank_bin": [lo, hi], "n": int(sel.sum()),
                             "static_err": round(float(1 - ro[sel].mean()), 4) if sel.sum() else None})
    # (2) update curve on start-correct + flipped
    sub = [r for r in rows if r["start_correct"]]
    par = np.array([r["parent"] for r in sub])
    cf = np.array([r["start_conf"] for r in sub])
    mg = np.array([r["start_margin"] for r in sub])
    upd_fail = np.array([1 - int(r["endpoint_correct"]) for r in sub])
    cr = ranks(cf)
    update_curve = []
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        sel = (cr >= lo) & (cr < hi)
        update_curve.append({"rank_bin": [lo, hi], "n": int(sel.sum()),
                             "update_err": round(float(upd_fail[sel].mean()), 4) if sel.sum() else None,
                             "ci": cluster_ci(par[sel], upd_fail[sel]) if sel.sum() else None})
    # margin-tertile stratified hi/lo
    mt = np.quantile(mg, [1 / 3, 2 / 3])
    strat = []
    for t in range(3):
        sel = (mg >= (mt[t - 1] if t else -1)) & (mg <= (mt[t] if t < 2 else 1e9))
        med = np.median(cr[sel])
        hi = sel & (cr >= med)
        lo = sel & (cr < med)
        strat.append({"tertile": t, "n_hi": int(hi.sum()), "n_lo": int(lo.sum()),
                      "err_hi": round(float(upd_fail[hi].mean()), 4) if hi.sum() else None,
                      "err_lo": round(float(upd_fail[lo].mean()), 4) if lo.sum() else None,
                      "ci_hi": cluster_ci(par[hi], upd_fail[hi]) if hi.sum() else None,
                      "ci_lo": cluster_ci(par[lo], upd_fail[lo]) if lo.sum() else None})
    return {"static_curve": static_curve, "update_curve": update_curve,
            "stratified": strat, "n_sub": len(sub)}


def coord_selection(seed, mname):
    """P1-C: retain top-X% by confidence; report static/update/preserve/joint."""
    rows, _ = load_coord(seed, mname)
    sub = [r for r in rows if r["start_correct"]]
    cf = np.array([r["start_conf"] for r in sub])
    out = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        thr = np.quantile(cf, 1 - frac)
        sel = cf >= thr
        eff = sel.mean()
        ue = np.array([1 - int(r["endpoint_correct"]) for r in sub])[sel].mean() if sel.sum() else None
        out.append({"retain": frac, "thr": round(float(thr), 4),
                    "eff_cov": round(float(eff), 4), "n": int(sel.sum()),
                    "update_err": round(float(ue), 4) if ue is not None else None})
    return out


def coord_repair_residual(seed):
    """P1-E: high-risk group fixed by clean confidence; compare repair."""
    rc, _ = load_coord(seed, "clean")
    rf, _ = load_coord(seed, "flipmine")
    sub_c = [r for r in rc if r["start_correct"]]
    sub_f = [r for r in rf if r["start_correct"]]
    cfc = np.array([r["start_conf"] for r in sub_c])
    thr = np.quantile(cfc, 0.75)
    # same quartets? match by (qid,ptype)
    fc = {(r["qid"], r["ptype"]): r for r in sub_f}
    pairs = [(r, fc[(r["qid"], r["ptype"])]) for r in sub_c
             if (r["qid"], r["ptype"]) in fc]
    hi = [(a, b) for a, b in pairs if a["start_conf"] >= thr]
    d = np.array([(1 - int(a["endpoint_correct"])) - (1 - int(b["endpoint_correct"])) for a, b in hi])
    lo, hi_ci = np.quantile(rng.choice(d, size=(2000, len(d))).mean(1), [0.025, 0.975]) if len(d) else (None, None)
    return {"n_pairs": len(pairs), "n_high": len(hi),
            "clean_high_err": round(float(np.mean([1 - int(a["endpoint_correct"]) for a, b in hi])), 4) if hi else None,
            "repair_high_err": round(float(np.mean([1 - int(b["endpoint_correct"]) for a, b in hi])), 4) if hi else None,
            "delta_ci": [round(float(lo), 4), round(float(hi_ci), 4)] if hi else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    res = {}
    for seed in (11, 23, 47):
        for mname in ("clean", "flipmine"):
            try:
                res[f"s{seed}_{mname}"] = coord_dual(seed, mname)
                res[f"s{seed}_{mname}"]["selection"] = coord_selection(seed, mname)
            except FileNotFoundError:
                res[f"s{seed}_{mname}"] = {"error": "missing"}
        try:
            res[f"s{seed}_repair_residual"] = coord_repair_residual(seed)
        except FileNotFoundError:
            pass
    # temperature note: binary |logit|/T is strictly monotonic in |logit| ->
    # ranks identical by construction; argmax unchanged. Analytic control:
    # oracle predictor has conf=margin, updates perfectly.
    res["temperature_note"] = ("binary |logit|/T strictly monotonic: ranks and "
                               "argmax identical by construction; P1 curves invariant.")
    res["analytic_oracle_control"] = ("predictor y=oracle(x), conf=margin: high "
                                      "confidence with perfect updating exists; high confidence need not imply "
                                      "update failure as a logical fact (sanity control only).")
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "P1_COORD.json", "w"), indent=1)
    for k, v in res.items():
        if isinstance(v, dict) and "update_curve" in v:
            print(k, [(b["rank_bin"], b["n"], b["update_err"]) for b in v["update_curve"]], flush=True)
        if k.endswith("repair_residual"):
            print(k, v, flush=True)


if __name__ == "__main__":
    main()
