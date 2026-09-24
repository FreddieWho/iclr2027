#!/usr/bin/env python3
"""D10 decision frontier: same-coverage / same-cost comparisons + full
coverage-cost-risk frontier. Reuses p2_table.npz (read-only) and the v2
dev/eval split exactly. Threshold selection structurally dev-only (unit
tested). No training, no sealed-pool reads.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "repair_decomposition"))
from p2_decompose import load  # noqa: E402

ART = ROOT / "artifacts" / "f095_campaign" / "D10"
SEEDS = (11, 23, 47)
N_BOOT = 2000
BOOT_SEED = 26092245


def split_scenes(d, feas):
    scenes_sorted = np.array(sorted(set(d["scene_of"])))
    F = np.array([s for s in np.unique(d["scene_of"])
                  if d["feas"][np.nonzero(d["scene_of"] == s)[0]].any()])
    dev = np.array([s for s in scenes_sorted[:86] if s in set(F)])
    ev = np.array([s for s in F if s not in set(dev)])
    return dev, ev


def scene_maxp0(P0, scene_of, scenes):
    return np.array([P0[np.nonzero(scene_of == s)[0]].max() for s in scenes])


def reachable_ladder(dev_maxp0):
    """Complete reachable ladder: all distinct dev maxP0 values, no grid."""
    return np.array(sorted(set(float(t) for t in dev_maxp0)))


def policy_outcomes(P0, scene_of, act_cost_row, feas, scenes, tau):
    """Per-scene (acted, success, cost). Act = cheapest row with P0>=tau."""
    acted, succ, cost = [], [], []
    for s in scenes:
        idx = np.nonzero(scene_of == s)[0]
        cand = idx[P0[idx] >= tau]
        if len(cand) == 0:
            acted.append(False); succ.append(False); cost.append(0.0)
        else:
            j = cand[int(np.argmin(act_cost_row[cand]))]
            acted.append(True); succ.append(bool(feas[j])); cost.append(float(act_cost_row[j]))
    return (np.array(acted), np.array(succ), np.array(cost))


def summarize(acted, succ, cost):
    n = len(acted)
    cov = float(acted.mean()) if n else 0.0
    q = float(succ[acted].mean()) if acted.any() else float("nan")
    return {"coverage": round(cov, 4),
            "own_quality": round(q, 4) if acted.any() else None,
            "total_cost": round(float(cost[acted].sum()), 4),
            "mean_cost": round(float(cost[acted].mean()), 4) if acted.any() else None,
            "n_acted": int(acted.sum()), "n": n}


def match_ladder(dev_cov_f, taus_f, dev_cov_c, taus_c):
    """Every flip rung -> nearest clean rung by dev coverage (tie: larger tau)."""
    out = []
    for tf, cf in zip(taus_f, dev_cov_f):
        gaps = np.abs(dev_cov_c - cf)
        best = np.nonzero(gaps == gaps.min())[0]
        tc = taus_c[best[-1]]
        out.append((float(tf), float(tc), round(float(gaps[best[-1]]), 4)))
    return out


def paired_own_quality_ci(f_succ, f_acted, c_succ, c_acted, scenes, rng):
    """Same-scene bootstrap: recompute each policy's OWN ratio per resample."""
    n = len(scenes)
    diffs = []
    for _ in range(N_BOOT):
        draw = rng.choice(n, size=n, replace=True)
        fa, ca = f_acted[draw], c_acted[draw]
        fq = f_succ[draw][fa].mean() if fa.any() else float("nan")
        cq = c_succ[draw][ca].mean() if ca.any() else float("nan")
        diffs.append(fq - cq)
    diffs = np.array(diffs)
    return [round(float(np.nanquantile(diffs, 0.025)), 4),
            round(float(np.nanquantile(diffs, 0.975)), 4)]


def fit_isotonic(x, y):
    """PAVA isotonic regression (increasing). Returns sorted xs + fitted values."""
    xs = np.asarray(x, float)
    ys = np.asarray(y, float)
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], ys[order]
    blocks = [[xs[i], ys[i], 1] for i in range(len(xs))]
    out = []
    for x, y, w in blocks:
        out.append([x, y, w])
        while len(out) >= 2 and out[-2][1] > out[-1][1]:
            x2, y2, w2 = out.pop()
            x1, y1, w1 = out.pop()
            w = w1 + w2
            out.append([(x1 * w1 + x2 * w2) / w, (y1 * w1 + y2 * w2) / w, w])
    bx = np.array([b[0] for b in out])
    by = np.array([b[1] for b in out])
    return bx, by


def apply_isotonic(bx, by, x):
    return np.interp(np.asarray(x, float), bx, by, left=by[0], right=by[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ART)
    a = p.parse_args()
    d = load()
    table_sha = hashlib.sha256(
        (ROOT / "artifacts" / "p123_upgrade" / "p2" / "p2_table.npz").read_bytes()
    ).hexdigest()
    feas = d["feas"].astype(bool)
    scene_of = d["scene_of"]
    act_cost_row = d["costs"][d["act_idx"]]
    dev, ev = split_scenes(d, feas)
    rng = np.random.default_rng(BOOT_SEED)
    res = {"dev_scenes": len(dev), "eval_scenes": len(ev),
           "table_sha256": table_sha,
           "split_rule": "sorted-first-86 ∩ feasible = dev (v2 identical)",
           "ladder_rule": "all distinct dev maxP0 values, no grid",
           "match_rule": "nearest dev coverage, tie -> larger tau",
           "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "scene, paired"},
           "seeds": {}}
    for seed in SEEDS:
        R = {"seed": seed, "models": {}}
        Lg = {}
        for m in ("clean", "flipmine"):
            Lg[m] = d["lg_s%d_%s" % (seed, "clean" if m == "clean" else "flipmine")]
        P0 = {m: 1 / (1 + np.exp(Lg[m])) for m in Lg}
        dev_max = {m: scene_maxp0(P0[m], scene_of, dev) for m in Lg}
        ladders = {m: reachable_ladder(dev_max[m]) for m in Lg}
        R["ladder_sizes"] = {m: int(len(ladders[m])) for m in Lg}
        # full frontier data per model per split
        for split, S in (("dev", dev), ("eval", ev)):
            for m in Lg:
                curve = []
                for t in ladders[m]:
                    ac, su, co = policy_outcomes(P0[m], scene_of, act_cost_row, feas, S, float(t))
                    s = summarize(ac, su, co)
                    s["tau"] = float(t)
                    curve.append(s)
                R["models"].setdefault(m, {})["frontier_" + split] = curve
        # dev stats per ladder rung (for matching)
        dev_cov = {}
        dev_cost = {}
        for m in Lg:
            dc, dco = [], []
            for t in ladders[m]:
                ac, _, co = policy_outcomes(P0[m], scene_of, act_cost_row, feas, dev, float(t))
                dc.append(float(ac.mean()))
                dco.append(float(co[ac].sum()))
            dev_cov[m] = np.array(dc)
            dev_cost[m] = np.array(dco)
        # same-coverage matching (flip rungs -> clean)
        pairs_cov = match_ladder(dev_cov["flipmine"], ladders["flipmine"],
                                 dev_cov["clean"], ladders["clean"])
        # same-cost matching (total dev cost)
        pairs_cost = match_ladder(dev_cost["flipmine"], ladders["flipmine"],
                                  dev_cost["clean"], ladders["clean"])
        R["matched_same_coverage"] = []
        for tf, tc, gap in pairs_cov:
            ac, suf, cof = policy_outcomes(P0["flipmine"], scene_of, act_cost_row, feas, ev, tf)
            acc, suc, coc = policy_outcomes(P0["clean"], scene_of, act_cost_row, feas, ev, tc)
            both = ac & acc
            qf = float(suf[ac].mean()) if ac.any() else None
            qc = float(suc[acc].mean()) if acc.any() else None
            row = {"tau_f": tf, "tau_c": tc, "dev_cov_gap": gap,
                   "eval_cov_f": round(float(ac.mean()), 4),
                   "eval_cov_c": round(float(acc.mean()), 4),
                   "eval_q_f": round(qf, 4) if qf is not None else None,
                   "eval_q_c": round(qc, 4) if qc is not None else None,
                   "eval_cost_f": round(float(cof[ac].sum()), 4),
                   "eval_cost_c": round(float(coc[acc].sum()), 4)}
            if ac.any() and acc.any():
                row["own_quality_diff_CI"] = paired_own_quality_ci(
                    suf, ac, suc, acc, ev, rng)
                qb = float(suf[both].mean()) if both.any() else None
                qc2 = float(suc[both].mean()) if both.any() else None
                row["common_acted_n"] = int(both.sum())
                row["common_quality_diff"] = round(qb - qc2, 4) if qb is not None else None
            R["matched_same_coverage"].append(row)
        R["matched_same_cost"] = []
        for tf, tc, gap in pairs_cost:
            ac, suf, cof = policy_outcomes(P0["flipmine"], scene_of, act_cost_row, feas, ev, tf)
            acc, suc, coc = policy_outcomes(P0["clean"], scene_of, act_cost_row, feas, ev, tc)
            qf = float(suf[ac].mean()) if ac.any() else None
            qc = float(suc[acc].mean()) if acc.any() else None
            row = {"tau_f": tf, "tau_c": tc, "dev_cost_gap": gap,
                   "eval_cost_f": round(float(cof[ac].sum()), 4),
                   "eval_cost_c": round(float(coc[acc].sum()), 4),
                   "eval_q_f": round(qf, 4) if qf is not None else None,
                   "eval_q_c": round(qc, 4) if qc is not None else None}
            if ac.any() and acc.any():
                row["own_quality_diff_CI"] = paired_own_quality_ci(
                    suf, ac, suc, acc, ev, rng)
            R["matched_same_cost"].append(row)
        # clean best-threshold control (dev own-quality, tie -> higher coverage)
        cq = [r for r in R["models"]["clean"]["frontier_dev"] if r["own_quality"] is not None]
        best = max(cq, key=lambda r: (r["own_quality"], r["coverage"]))
        ac, suc, _ = policy_outcomes(P0["clean"], scene_of, act_cost_row, feas, ev, best["tau"])
        R["clean_best"] = {"dev_tau": best["tau"], "dev_q": best["own_quality"],
                           "eval_q": round(float(suc[ac].mean()), 4) if ac.any() else None,
                           "eval_cov": round(float(ac.mean()), 4)}
        # isotonic calibrator on clean (dev rows), compare calibrated-clean vs flip
        devrows = np.concatenate([np.nonzero(scene_of == s)[0] for s in dev])
        bx, by = fit_isotonic(P0["clean"][devrows], feas[devrows].astype(float))
        R["iso_n_blocks"] = int(len(bx))
        P0c_cal = apply_isotonic(bx, by, P0["clean"])
        cal_max = scene_maxp0(P0c_cal, scene_of, dev)
        cal_ladder = reachable_ladder(cal_max)
        cal_rows = []
        for t in cal_ladder:
            ac, su, co = policy_outcomes(P0c_cal, scene_of, act_cost_row, feas, ev, float(t))
            cal_rows.append((float(t), float(ac.mean()),
                             float(su[ac].mean()) if ac.any() else None))
        R["iso_calibrated_clean_eval"] = [
            {"tau": t, "cov": round(c, 4), "q": round(q, 4) if q is not None else None}
            for t, c, q in cal_rows]
        # test-label-optimized frontier (diagnostic bound only)
        R["oracle_bound_note"] = ("test-label-optimized frontier over eval ladder rungs; "
                                  "diagnostic upper bound, NOT deployable")
        orb = {}
        for m in Lg:
            P0e = P0[m]
            uniq = sorted(set(float(v) for v in
                              [P0e[np.nonzero(scene_of == s)[0]].max() for s in ev]))
            pts = []
            for t in uniq:
                ac, su, co = policy_outcomes(P0e, scene_of, act_cost_row, feas, ev, float(t))
                pts.append({"tau": float(t), "cov": round(float(ac.mean()), 4),
                            "q": round(float(su[ac].mean()), 4) if ac.any() else None,
                            "cost": round(float(co[ac].sum()), 4)})
            orb[m] = pts
        R["oracle_bound_frontier_eval"] = orb
        res["seeds"]["s%d" % seed] = R
        print("s%d ladders" % seed, R["ladder_sizes"], "cov_pairs",
              len(pairs_cov), "cost_pairs", len(pairs_cost), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "D10_SUMMARY.json", "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out / "D10_SUMMARY.json")


if __name__ == "__main__":
    main()
