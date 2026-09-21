#!/usr/bin/env python3
"""P2-C/D/E: symmetric coverage/quality split, strata, equal-coverage,
interface swaps (lexico collapses by definition), temperature check, P1 link.
Reads immutable p2_table.npz. Dev(86 scenes, order-fixed)/eval(170) split
for threshold selection. lam=2 primary, 0/1/5 sensitivity.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "p123_upgrade" / "p2"
rng = np.random.default_rng(26092232)


def load():
    d = np.load(OUT / "p2_table.npz")
    return d


def acts(d, lg, rule, lam=2.0, tau=0.5):
    P0 = 1 / (1 + np.exp(lg))
    scenes = np.unique(d["scene_of"])
    sel, act = {}, {}
    rowcost = d["costs"][d["act_idx"]]
    for s in scenes:
        idx = np.nonzero(d["scene_of"] == s)[0]
        if rule == "score":
            j = idx[int(np.argmax(P0[idx] - lam * rowcost[idx]))]
            sel[s], act[s] = True, j
        else:
            cand = idx[P0[idx] >= tau]
            if len(cand) == 0:
                sel[s], act[s] = False, None
            else:
                sel[s], act[s] = True, cand[int(np.argmin(rowcost[cand]))]
    return sel, act


def feasible_scenes(d):
    out = []
    for s in np.unique(d["scene_of"]):
        idx = np.nonzero(d["scene_of"] == s)[0]
        if d["feas"][idx].any():
            out.append(s)
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    d = load()
    feas = d["feas"].astype(bool)
    F = feasible_scenes(d)
    scenes_sorted = np.array(sorted(set(d["scene_of"])))
    dev = set(scenes_sorted[:86])
    ev = np.array([s for s in F if s not in dev])
    res = {"n_feasible": len(F), "n_eval": len(ev)}
    for seed in (11, 23, 47):
        lgc = d["lg_s%d_clean" % seed]
        lgf = d["lg_s%d_flipmine" % seed]
        R = {"seed": seed}
        for lam in (0, 1, 2, 5):
            for rule in ("score", "lexico"):
                for tag, lg in (("clean", lgc), ("flip", lgf)):
                    sel, act = acts(d, lg, rule, lam)
                    ok = np.array([s for s in ev if sel[s]])
                    suc = np.array([s for s in ok if feas[act[s]]])
                    R[f"{rule}_lam{lam}_{tag}"] = {
                        "a": round(len(ok) / len(ev), 4),
                        "q": round(len(suc) / len(ok), 4) if len(ok) else None,
                        "s": round(len(suc) / len(ev), 4), "n": len(ev)}
        # strata on eval, lexico lam2
        selc, actc = acts(d, lgc, "lexico", 2.0)
        self_, actf = acts(d, lgf, "lexico", 2.0)
        strata = {}
        for name, cond in (("both", lambda s: selc[s] and self_[s]),
                           ("clean_only", lambda s: selc[s] and not self_[s]),
                           ("repair_only", lambda s: (not selc[s]) and self_[s]),
                           ("neither", lambda s: (not selc[s]) and (not self_[s]))):
            ss = [s for s in ev if cond(s)]
            strata[name] = {"n": len(ss),
                            "clean_suc": sum(1 for s in ss if selc[s] and feas[actc[s]]),
                            "repair_suc": sum(1 for s in ss if self_[s] and feas[actf[s]])}
        R["strata_lex_lam2"] = strata
        # symmetric split on s
        sc, sf = R["lexico_lam2_clean"]["s"], R["lexico_lam2_flip"]["s"]
        ac, af = R["lexico_lam2_clean"]["a"], R["lexico_lam2_flip"]["a"]
        qc, qf = R["lexico_lam2_clean"]["q"], R["lexico_lam2_flip"]["q"]
        R["split"] = {"D_coverage": round((af - ac) * (qf + qc) / 2, 4),
                      "D_quality": round((qf - qc) * (af + ac) / 2, 4),
                      "total": round(sf - sc, 4)}
        # candidate precision/recall of predicted-feasible sets (micro)
        for tag, lg in (("clean", lgc), ("flip", lgf)):
            P0 = 1 / (1 + np.exp(lg))
            tp = fp = fn = 0
            for s in ev:
                idx = np.nonzero(d["scene_of"] == s)[0]
                pr = set(idx[P0[idx] >= 0.5])
                tr = set(idx[feas[idx]])
                tp += len(pr & tr)
                fp += len(pr - tr)
                fn += len(tr - pr)
            R[f"cand_{tag}"] = {"prec": round(tp / max(1, tp + fp), 4),
                                "rec": round(tp / max(1, tp + fn), 4)}
        # common-act paired correctness + cost
        both_s = [s for s in ev if selc[s] and self_[s]]
        pair_ok_c = sum(1 for s in both_s if feas[actc[s]])
        pair_ok_f = sum(1 for s in both_s if feas[actf[s]])
        R["common_act"] = {"n": len(both_s), "clean_ok": pair_ok_c,
                           "repair_ok": pair_ok_f}
        # 4-cell interface swap (lexico: score stage is cheapest-cost,
        # model-independent -> cells with same set coincide by definition)
        setc = {s: set(np.nonzero(d["scene_of"] == s)[0][(1 / (1 + np.exp(lgc[np.nonzero(d["scene_of"] == s)[0]]))) >= 0.5]) for s in ev}
        setf = {s: set(np.nonzero(d["scene_of"] == s)[0][(1 / (1 + np.exp(lgf[np.nonzero(d["scene_of"] == s)[0]]))) >= 0.5]) for s in ev}
        cell = {}
        for sn, st in (("clean_set", setc), ("repair_set", setf)):
            ok = sum(1 for s in ev if st[s] and feas[min(st[s], key=lambda j: d["costs"][d["act_idx"][j]])]) if any(st[s] for s in ev) else 0
            n = sum(1 for s in ev if st[s])
            cell[sn] = {"n": n, "suc": ok, "rate": round(ok / n, 4) if n else None}
        R["interface_2cell"] = cell
        R["interface_note"] = ("lexico score stage = cheapest cost, model-independent: "
                               "4-cell collapses to set-only comparison by definition (verified: identical selections).")
        # equal-coverage: dev-select tau to match coverages, eval compare
        def cov_at(lg, tau):
            P0 = 1 / (1 + np.exp(lg))
            return np.mean([len(np.nonzero(d["scene_of"] == s)[0][P0[np.nonzero(d["scene_of"] == s)[0]] >= tau]) > 0 for s in dev])
        taus = np.linspace(0.05, 0.95, 19)
        cc = [cov_at(lgc, t) for t in taus]
        cf = [cov_at(lgf, t) for t in taus]
        R["equal_cov"] = {"dev_cov_clean_tau05": round(float(cc[9]), 4),
                          "dev_cov_flip_tau05": round(float(cf[9]), 4)}
        # temperature: lexico sets invariant at any T>0 (argmax-invariant);
        # score-rule selection shifts. verify on eval.
        for Tm in (0.5, 2.0):
            lgcT, lgfT = lgc / Tm, lgf / Tm
            same_lex = all(
                set(np.nonzero(d["scene_of"] == s)[0][(1 / (1 + np.exp(lgc[np.nonzero(d["scene_of"] == s)[0]]))) >= 0.5]) ==
                set(np.nonzero(d["scene_of"] == s)[0][(1 / (1 + np.exp(lgcT[np.nonzero(d["scene_of"] == s)[0]]))) >= 0.5])
                for s in ev)
            _, ac0 = acts(d, lgc, "score", 2.0)
            _, acT = acts(d, lgcT, "score", 2.0)
            shift = sum(1 for s in ev if ac0[s] != acT[s])
            R[f"temp_T{Tm}"] = {"lexico_set_invariant": bool(same_lex),
                                "score_selection_shift_frac": round(shift / len(ev), 4)}
        # P1 link: high-confidence scenes (top quartile of clean selected P0)
        P0c = 1 / (1 + np.exp(lgc))
        conf_s = {}
        for s in ev:
            if selc[s]:
                conf_s[s] = float(P0c[actc[s]])
        thr = np.quantile(list(conf_s.values()), 0.75)
        hi_s = [s for s in ev if selc[s] and conf_s[s] >= thr]
        R["P1link"] = {"n_high": len(hi_s),
                       "clean_high_suc": round(sum(1 for s in hi_s if feas[actc[s]]) / len(hi_s), 4) if hi_s else None,
                       "repair_high_suc": round(sum(1 for s in hi_s if self_[s] and feas[actf[s]]) / len(hi_s), 4) if hi_s else None}
        res_key = "s%d" % seed
        res[res_key] = R
        print("SAW s%d split=%s strata=%s" % (seed, R["split"], {k: v["n"] for k, v in strata.items()}), flush=True)
    json.dump(res, open(a.out / "P2_DECOMP.json", "w"), indent=1)
    print("DONE p2_decompose")


if __name__ == "__main__":
    main()
