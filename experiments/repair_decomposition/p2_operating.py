#!/usr/bin/env python3
"""P2-A/B/C/D: equal-coverage comparison + affine-clean control + paired CIs.

1. Equal coverage: rank-based thresholds on dev scenes -> common achievable
   coverage levels; quality|coverage on eval with paired scenes (success,
   cost, candidate precision/recall).
2. Affine-clean: grid fit alpha>0,b on dev to match repair coverage +
   prevalence; eval clean/affine/flipmine on static/single/preserve/
   composition/joint/action (+paired CIs, scene-cluster bootstrap).
3. Rule dependence kept: lexico vs score reported separately, never merged.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "repair_decomposition"))
from p2_decompose import load, acts, feasible_scenes  # noqa: E402

OUT = ROOT / "artifacts" / "next_novelty" / "p2"
P2OLD = ROOT / "artifacts" / "p123_upgrade" / "p2"
rng = np.random.default_rng(26092245)
LAM = 2.0


def scene_bootstrap(scenes, fn, n_boot=2000):
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(scenes, size=len(scenes), replace=True)
        vals.append(fn(draw))
    q = np.quantile(vals, [0.025, 0.975])
    return [round(float(q[0]), 4), round(float(q[1]), 4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    d = load()
    feas = d["feas"].astype(bool)
    F = feasible_scenes(d)
    scenes_sorted = np.array(sorted(set(d["scene_of"])))
    dev = np.array([s for s in scenes_sorted[:86] if s in set(F)])
    ev = np.array([s for s in F if s not in set(dev)])
    res = {"n_dev": len(dev), "n_eval": len(ev)}
    for seed in (11, 23, 47):
        lgc = d["lg_s%d_clean" % seed]
        lgf = d["lg_s%d_flipmine" % seed]
        R = {"seed": seed}
        # ---- A: rank-based equal coverage ----
        P0c = 1 / (1 + np.exp(lgc))
        P0f = 1 / (1 + np.exp(lgf))

        def maxP0(lg, s):
            idx = np.nonzero(d["scene_of"] == s)[0]
            P0 = 1 / (1 + np.exp(lg[idx]))
            return float(P0.max())
        # coverage staircase: scene acts if max P0 >= tau
        taus = np.quantile([maxP0(lgc, s) for s in dev], np.linspace(0, 1, 21))
        taus = sorted(set([round(float(t), 4) for t in taus]))
        covc = {t: float(np.mean([maxP0(lgc, s) >= t for s in dev])) for t in taus}
        covf = {t: float(np.mean([maxP0(lgf, s) >= t for s in dev])) for t in taus}
        # common achievable levels: flipmine taus whose dev coverage is within
        # 3pp of some clean-tau dev coverage
        levels = []
        for tf in taus:
            diffs = [(abs(covf[tf] - c), tc) for tc, c in covc.items()]
            diffs.sort()
            if diffs[0][0] <= 0.03:
                levels.append((tf, diffs[0][1], covf[tf]))
        R["equal_cov_levels"] = [{"tau_f": t[0], "tau_c": t[1], "dev_cov": round(t[2], 4)} for t in levels[:6]]

        def succ_at(lg, tau, S):
            P0 = 1 / (1 + np.exp(lg))
            ok = suc = 0
            cost = []
            for s in S:
                idx = np.nonzero(d["scene_of"] == s)[0]
                cand = idx[P0[idx] >= tau]
                if len(cand) == 0:
                    continue
                rc = d["costs"][d["act_idx"]]
                j = cand[int(np.argmin(rc[cand]))]
                ok += 1
                suc += feas[j]
                cost.append(float(rc[j]))
            return ok, suc, cost
        R["equal_cov_eval"] = []
        for tf, tc, _ in levels[:6]:
            okc, sc, cc = succ_at(lgc, tc, ev)
            okf, sf, cf = succ_at(lgf, tf, ev)
            R["equal_cov_eval"].append({
                "tau_c": tc, "tau_f": tf,
                "clean": {"n": okc, "q": round(sc / okc, 4) if okc else None,
                          "cost": round(float(np.mean(cc)), 4) if cc else None},
                "repair": {"n": okf, "q": round(sf / okf, 4) if okf else None,
                           "cost": round(float(np.mean(cf)), 4) if cf else None}})
        # candidate P/R at matched coverage (micro, eval)
        # ---- B: affine-clean fit on dev ----
        # match repair coverage (lexico tau .5 act rate) + mean P0 prevalence
        def dev_stats(lg):
            P0 = 1 / (1 + np.exp(lg))
            act = np.mean([len(np.nonzero(d["scene_of"] == s)[0][P0[np.nonzero(d["scene_of"] == s)[0]] >= 0.5]) > 0 for s in dev])
            return act, float(P0.mean())
        tcov, tprev = dev_stats(lgf)
        best, bk = 1e9, None
        for alpha in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
            for b in (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0):
                lga = alpha * lgc + b
                acov, aprev = dev_stats(lga)
                loss = abs(acov - tcov) + abs(aprev - tprev)
                if loss < best:
                    best, bk = loss, (alpha, b)
        alpha, b0 = bk
        lga = alpha * lgc + b0
        R["affine_fit"] = {"alpha": alpha, "b": b0, "dev_loss": round(float(best), 4),
                           "dev_cov_repair": round(float(tcov), 4)}
        # eval: clean / affine / flipmine on action metrics
        for tag, lg in (("clean", lgc), ("affine", lga), ("flip", lgf)):
            sel, act = acts(d, lg, "lexico", LAM)
            ok = np.array([s for s in ev if sel[s]])
            suc = np.array([s for s in ok if feas[act[s]]])
            P0 = 1 / (1 + np.exp(lg))
            tp = fp = fn = 0
            for s in ev:
                idx = np.nonzero(d["scene_of"] == s)[0]
                pr = set(idx[P0[idx] >= 0.5])
                tr = set(idx[feas[idx]])
                tp += len(pr & tr)
                fp += len(pr - tr)
                fn += len(tr - pr)
            R[f"lex_{tag}"] = {"a": round(len(ok) / len(ev), 4),
                               "q": round(len(suc) / len(ok), 4) if len(ok) else None,
                               "s": round(len(suc) / len(ev), 4),
                               "prec": round(tp / max(1, tp + fp), 4),
                               "rec": round(tp / max(1, tp + fn), 4)}
            sel2, act2 = acts(d, lg, "score", LAM)
            ok2 = np.array([s for s in ev if sel2[s]])
            suc2 = np.array([s for s in ok2 if feas[act2[s]]])
            R[f"score_{tag}"] = {"s": round(len(suc2) / len(ev), 4), "n": len(ev)}
        # paired CIs for key deltas (scene bootstrap on eval)
        evl = list(ev)
        R["paired_ci"] = {
            "lex_s_flip-clean": scene_bootstrap(
                evl, lambda S: np.mean([1.0 if _lex_ok(d, lgf, s) else 0.0 for s in S]) -
                               np.mean([1.0 if _lex_ok(d, lgc, s) else 0.0 for s in S])),
            "lex_s_affine-clean": scene_bootstrap(
                evl, lambda S: np.mean([1.0 if _lex_ok(d, lga, s) else 0.0 for s in S]) -
                               np.mean([1.0 if _lex_ok(d, lgc, s) else 0.0 for s in S])),
        }
        res["s%d" % seed] = R
        print("s%d affine=%s lex=%s/%s/%s" % (
            seed, (alpha, b0), R["lex_clean"]["s"], R["lex_affine"]["s"], R["lex_flip"]["s"]), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "P2_OPERATING.json", "w"), indent=1)
    print("DONE ->", a.out)


def _lex_ok(d, lg, s):
    P0 = 1 / (1 + np.exp(lg))
    idx = np.nonzero(d["scene_of"] == s)[0]
    cand = idx[P0[idx] >= 0.5]
    if len(cand) == 0:
        return False
    rc = d["costs"][d["act_idx"]]
    return bool(d["feas"][cand[int(np.argmin(rc[cand]))]])


if __name__ == "__main__":
    main()
