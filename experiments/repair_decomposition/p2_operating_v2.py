#!/usr/bin/env python3
"""P2 R5 redo (v2): corrected controls + split estimands + complete-threshold
enumeration + dev-mask fix. NEW outputs under artifacts/next_novelty/p2_v2/
(does NOT overwrite artifacts/next_novelty/p2/*.json). Dev-only fitting;
eval reports the FIXED policy only. No model training.

Corrections vs p2_operating.py:
- dev_stats() prevalence now uses the same dev-row mask as coverage
  (previously P0.mean() averaged over ALL rows incl. eval candidates).
- probabilities are NOT round(4)-ed before ranking (no synthetic saturation ties).
- equal-coverage quality split into two estimands:
    policy_quality_difference (each policy on its OWN acted set, paired by scene
        where both act, plus per-policy quality on its own set)
    common_act_difference (both-act scenes only)
- dev-vs-eval coverage gap reported per level (dev-matched coverage does not
  guarantee eval-matched coverage).
- complete-threshold control: for binary feasibility, a positive affine
  transform of clean logits equals a shared threshold shift; enumerate every
  reachable threshold between ranked dev scores (no b-grid widening), pick on
  dev by task utility (lexico success at lam=2), evaluate the FIXED policy on eval.
- keeps the frozen wide-grid affine as a bounded control (s11 edge stated).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "repair_decomposition"))
from p2_decompose import load, acts, feasible_scenes  # noqa: E402

OUT = ROOT / "artifacts" / "next_novelty" / "p2_v2"
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
    devrows = np.concatenate([np.nonzero(d["scene_of"] == s)[0] for s in dev])
    res = {"n_dev_scenes": len(dev), "n_eval_scenes": len(ev),
           "n_dev_rows": int(len(devrows)),
           "n_eval_rows": int(sum((d["scene_of"] == s).sum() for s in ev)),
           "n_feasible_rows": int(feas.sum())}
    for seed in (11, 23, 47):
        lgc = d["lg_s%d_clean" % seed]
        lgf = d["lg_s%d_flipmine" % seed]
        R = {"seed": seed}
        P0c_all = 1 / (1 + np.exp(lgc))
        P0f_all = 1 / (1 + np.exp(lgf))
        # NOTE: raw logits used for ranking; no rounding (avoids fake ties)

        def maxP0(lg, s):
            idx = np.nonzero(d["scene_of"] == s)[0]
            P0 = 1 / (1 + np.exp(lg[idx]))
            return float(P0.max())
        taus_c = sorted(set([float(t) for t in
                             np.quantile([maxP0(lgc, s) for s in dev], np.linspace(0, 1, 21))]))
        taus_f = sorted(set([float(t) for t in
                             np.quantile([maxP0(lgf, s) for s in dev], np.linspace(0, 1, 21))]))
        covc = {t: float(np.mean([maxP0(lgc, s) >= t for s in dev])) for t in taus_c}
        covf = {t: float(np.mean([maxP0(lgf, s) >= t for s in dev])) for t in taus_f}
        levels = []
        for tf in taus_f:
            diffs = sorted([(abs(covf[tf] - c), tc) for tc, c in covc.items()])
            if diffs[0][0] <= 0.03:
                levels.append((tf, diffs[0][1], covf[tf], round(diffs[0][0], 4)))
        R["equal_cov_levels"] = [{"tau_f": t[0], "tau_c": t[1], "dev_cov_f": round(t[2], 4),
                                  "dev_cov_c": round(covc[t[1]], 4),
                                  "dev_cov_gap": t[3]} for t in levels[:6]]

        def q_at(lg, tau, S):
            P0 = 1 / (1 + np.exp(lg))
            out = {}
            for x in S:
                idx = np.nonzero(d["scene_of"] == x)[0]
                cand = idx[P0[idx] >= tau]
                if len(cand) == 0:
                    out[x] = None
                else:
                    rc = d["costs"][d["act_idx"]]
                    j = cand[int(np.argmin(rc[cand]))]
                    out[x] = float(feas[j])
            return out
        R["equal_cov_eval"] = []
        for tf, tc, _, gap in levels[:6]:
            qc = q_at(lgc, tc, ev)
            qf = q_at(lgf, tf, ev)
            acted_c = [x for x in ev if qc[x] is not None]
            acted_f = [x for x in ev if qf[x] is not None]
            both = [x for x in ev if qc[x] is not None and qf[x] is not None]
            # dev-matched coverage vs actual eval coverage (reported, not assumed)
            eval_cov_c = len(acted_c) / len(ev)
            eval_cov_f = len(acted_f) / len(ev)
            dv_own = np.array([qf[x] - qc[x] for x in both])  # paired, both-act
            if len(dv_own):
                boots = [rng.choice(dv_own, size=len(dv_own), replace=True).mean() for _ in range(2000)]
                qq = np.quantile(boots, [0.025, 0.975])
                common_ci = [round(float(qq[0]), 4), round(float(qq[1]), 4)]
            else:
                common_ci = None
            R["equal_cov_eval"].append({
                "tau_c": tc, "tau_f": tf, "dev_cov_gap": gap,
                "eval_cov_clean": round(eval_cov_c, 4),
                "eval_cov_repair": round(eval_cov_f, 4),
                "eval_cov_gap": round(eval_cov_f - eval_cov_c, 4),
                "policy_quality_clean": round(float(np.mean([qc[x] for x in acted_c])), 4) if acted_c else None,
                "policy_quality_repair": round(float(np.mean([qf[x] for x in acted_f])), 4) if acted_f else None,
                "policy_quality_difference": (round(float(np.mean([qf[x] for x in acted_f])) -
                                                         float(np.mean([qc[x] for x in acted_c])), 4)
                                              if acted_c and acted_f else None),
                "common_act_n": len(both),
                "common_act_difference_ci": common_ci})

        # ---- dev_stats with corrected mask (coverage & prevalence both dev-only) ----
        def dev_stats(lg):
            P0 = 1 / (1 + np.exp(lg))
            act = np.mean([len(np.nonzero(d["scene_of"] == s)[0][P0[np.nonzero(d["scene_of"] == s)[0]] >= 0.5]) > 0 for s in dev])
            prev = float(P0[devrows].mean())
            return act, prev
        tcov, tprev = dev_stats(lgf)
        acov_c, aprev_c = dev_stats(lgc)
        R["dev_stats"] = {"repair_coverage_lexico_tau0.5": round(float(tcov), 4),
                          "repair_prevalence_devrows": round(float(tprev), 4),
                          "clean_coverage_lexico_tau0.5": round(float(acov_c), 4),
                          "clean_prevalence_devrows": round(float(aprev_c), 4)}

        # ---- B1: frozen wide-grid affine (bounded control; s11 edge stated) ----
        best, bk = 1e9, None
        for alpha in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
            for b in (-8.0, -6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0,
                      0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
                lga = alpha * lgc + b
                acov, aprev = dev_stats(lga)
                loss = abs(acov - tcov) + abs(aprev - tprev)
                if loss < best:
                    best, bk = loss, (alpha, b)
        alpha, b0 = bk
        edge_alpha = (alpha == 4.0) or (b0 in (-8.0, 8.0))
        R["affine_fit_grid"] = {"alpha": alpha, "b": b0, "dev_loss": round(float(best), 4),
                                "at_grid_edge": bool(edge_alpha),
                                "note": "bounded control (frozen grid); fit is NOT utility-optimal"}
        # ---- B2: complete-threshold enumeration (positive affine == shared threshold shift) ----
        # all reachable thresholds between ranked dev scores; pick on dev by lexico success
        devP0 = 1 / (1 + np.exp(lgc[devrows]))
        sorted_scores = np.sort(devP0)
        cands = [(sorted_scores[i] + sorted_scores[i + 1]) / 2 for i in range(len(sorted_scores) - 1)]
        cands = [c for c in cands if 0 < c < 1]
        def lex_success_on(lg, tau, S):
            P0 = 1 / (1 + np.exp(lg))
            ok = suc = 0
            for s in S:
                idx = np.nonzero(d["scene_of"] == s)[0]
                cand = idx[P0[idx] >= tau]
                if len(cand) == 0:
                    continue
                rc = d["costs"][d["act_idx"]]
                j = cand[int(np.argmin(rc[cand]))]
                ok += 1
                suc += feas[j]
            return suc / max(1, ok)
        best_t, best_s = None, -1
        for c in cands:
            s = lex_success_on(lgc, c, dev)
            if s > best_s:
                best_s, best_t = s, c
        R["threshold_enum"] = {"best_tau_dev": round(float(best_t), 4),
                               "dev_lexico_success_at_tau": round(float(best_s), 4),
                               "n_thresholds_tried": len(cands)}

        # ---- eval: fixed policies only ----
        lga = alpha * lgc + b0
        for tag, lg, note in (("clean", lgc, "base"),
                              ("affine_grid", lga, "shape-approx control"),
                              ("thresh_enum", lgc, "best dev threshold (fixed tau on clean scores)"),
                              ("flip", lgf, "repair")):
            if tag == "thresh_enum":
                sel, act = acts(d, lg, "lexico", LAM, tau=float(best_t))
            else:
                sel, act = acts(d, lg, "lexico", LAM)
            ok = np.array([s for s in ev if sel[s]])
            suc = np.array([s for s in ok if feas[act[s]]])
            R[f"lex_{tag}"] = {"a": round(len(ok) / len(ev), 4),
                               "q": round(len(suc) / len(ok), 4) if len(ok) else None,
                               "s": round(len(suc) / len(ev), 4),
                               "control_kind": note}
        evl = list(ev)
        R["paired_ci"] = {
            "lex_s_flip-clean": scene_bootstrap(
                evl, lambda S: np.mean([1.0 if _lex_ok(d, lgf, s) else 0.0 for s in S]) -
                               np.mean([1.0 if _lex_ok(d, lgc, s) else 0.0 for s in S])),
            "lex_s_affine_grid-clean": scene_bootstrap(
                evl, lambda S: np.mean([1.0 if _lex_ok(d, lga, s) else 0.0 for s in S]) -
                               np.mean([1.0 if _lex_ok(d, lgc, s) else 0.0 for s in S])),
            "lex_s_thresh_enum-clean": scene_bootstrap(
                evl, lambda S: np.mean([1.0 if _lex_ok(d, lgc, s, tau=float(best_t)) else 0.0 for s in S]) -
                               np.mean([1.0 if _lex_ok(d, lgc, s) else 0.0 for s in S])),
        }
        res["s%d" % seed] = R
        print("s%d grid=edge:%s thresh_tau=%.4f lex s c/a/te/f=%s/%s/%s/%s" % (
            seed, edge_alpha, best_t, R["lex_clean"]["s"], R["lex_affine_grid"]["s"],
            R["lex_thresh_enum"]["s"], R["lex_flip"]["s"]), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "P2_OPERATING_V2.json", "w"), indent=1)
    print("DONE ->", a.out)


def _lex_ok(d, lg, s, tau=0.5):
    P0 = 1 / (1 + np.exp(lg))
    idx = np.nonzero(d["scene_of"] == s)[0]
    cand = idx[P0[idx] >= tau]
    if len(cand) == 0:
        return False
    rc = d["costs"][d["act_idx"]]
    return bool(d["feas"][cand[int(np.argmin(rc[cand]))]])


if __name__ == "__main__":
    main()
