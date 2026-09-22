#!/usr/bin/env python3
"""P3 R4 redo: full 8x8 transition matrix + R_full/M/R_endpoint identity +
M/R_endpoint ratio + paired delta-J CIs (same parent resample) + zero-training
rank/sort diagnostic. Dev bank 512, s11/s23/s47, frozen checkpoints.
NEW outputs: artifacts/next_novelty/p3_v2/P3_FINAL.json (does NOT overwrite
artifacts/next_novelty/p3/P3_FINAL.json). No training, no sealed-holdout reads.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(4)
from common import load_model, preprocess  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
OUT = ROOT / "artifacts" / "next_novelty" / "p3_v2"
rng = np.random.default_rng(26092247)


def pcb_groups(groups, stat_fn, n_boot=2000):
    ups = np.arange(len(groups))
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(ups, size=len(ups), replace=True)
        boots.append(stat_fn([groups[i] for i in draw]))
    q = np.quantile(boots, [0.025, 0.975])
    return [round(float(q[0]), 4), round(float(q[1]), 4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="dev512")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    b = np.load(BANK / ("bank_" + a.bank + ".npz"), allow_pickle=True)
    Qx, Qe = b["Qx"], b["Qe"]
    Qmeta = json.loads(str(b["Qmeta"]))
    by_q = {}
    for qi, m in enumerate(Qmeta):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    qpar = {qid: ps[0][1]["parent"] for qid, ps in by_q.items() if len(ps) == 2}
    models = {}
    for seed in (11, 23, 47):
        for mname in ("clean", "flipmine"):
            model, stats = load_model(ART / ("r04b_s%d" % seed) / mname)
            model.eval()
            models[(seed, mname)] = (model, stats)

    def preds(seed, mname):
        model, stats = models[(seed, mname)]
        R = {}
        with torch.no_grad():
            xs_all, keys = [], []
            for qid, (i1, m1), (i2, m2) in quads:
                xa = np.asarray(Qx[i1], float)
                xb = np.asarray(Qx[i2], float)
                xab = xa + np.asarray(Qe[i1], float)
                xs_all += [xa, xb, xab]
                keys.append((qid, m1, m2))
            out = []
            for s in range(0, len(xs_all), 512):
                xx = np.stack(xs_all[s:s + 512]).astype(np.float32)
                out.append(model(preprocess(xx, stats)).numpy().reshape(-1))
            lg = np.concatenate(out)
            pl = (lg > 0).astype(int)
        for k, (qid, m1, m2) in enumerate(keys):
            la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
            lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
            R[qid] = (int(pl[3 * k] == la), int(pl[3 * k + 1] == lb),
                      int(pl[3 * k + 2] == m1["yAB"]))
        return R

    res = {"n_quartets": len(quads), "bank": a.bank}
    for seed in (11, 23, 47):
        C = preds(seed, "clean")
        F = preds(seed, "flipmine")
        Q = sorted(C)
        par = np.array([qpar[q] for q in Q])
        Pall = [np.nonzero(par == u)[0] for u in np.unique(par)]

        # ---- R4.1 FULL 8x8 transition matrix (state bits = A/B/AB correctness) ----
        trans = {}
        for q in Q:
            k = "".join(map(str, C[q])) + "->" + "".join(map(str, F[q]))
            trans[k] = trans.get(k, 0) + 1
        # order rows by state (000..111) for a complete matrix
        order = ["".join(map(str, s)) for s in
                 [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]]
        matrix = {c: {d: trans.get(c + "->" + d, 0) for d in order} for c in order}

        base110 = [q for q in Q if C[q] == (1, 1, 0)]
        n = len(base110)
        bp = np.array([qpar[q] for q in base110])
        Rf = np.array([1.0 if F[q] == (1, 1, 1) else 0.0 for q in base110])
        Re = np.array([float(F[q][2]) for q in base110])
        Mg = np.array([1.0 if (F[q][2] == 1 and (F[q][0] == 0 or F[q][1] == 0))
                       else 0.0 for q in base110])
        G = [np.nonzero(bp == u)[0] for u in np.unique(bp)]

        met = {}
        for nm, vv in (("R_full", Rf), ("R_endpoint", Re), ("M_migrate", Mg)):
            met[nm] = [round(float(vv.mean()), 4) if n else None,
                       pcb_groups(G, lambda idx: vv[np.concatenate(idx)].mean()) if n else None]

        # R4.2 identity + ratio (same estimands, same H set).
        # NOTE: R_endpoint = P(AB correct | H) counts AB-correct cases;
        # each can be decomposed into full-repair (all three correct) or
        # migration (AB correct but an atomic broken). Under the plan's
        # definition these coincide; here we verify and flag if not.
        # identity holds exactly on raw indicator means; rounding to 4dp for
        # the report can shift the sum by 1e-4, so verify on unrounded means.
        ident_ok = abs(float(Rf.mean() + Mg.mean() - Re.mean())) < 1e-9
        met["identity_R_endpoint_eq_R_full_plus_M"] = ident_ok
        met["identity_residual"] = round(float(Re.mean() - Rf.mean() - Mg.mean()), 12)
        met["M_over_R_endpoint"] = (round(float(Mg.mean() / Re.mean()), 4)
                                    if n and Re.mean() > 0 else None)

        Jc = np.array([1.0 if C[q] == (1, 1, 1) else 0.0 for q in Q])
        Jf = np.array([1.0 if F[q] == (1, 1, 1) else 0.0 for q in Q])
        met["J_clean"] = [round(float(Jc.mean()), 4), pcb_groups(Pall, lambda idx: Jc[np.concatenate(idx)].mean())]
        met["J_repair"] = [round(float(Jf.mean()), 4), pcb_groups(Pall, lambda idx: Jf[np.concatenate(idx)].mean())]

        # R4.3 paired delta-J CI on the SAME parent resample
        dJ = Jf - Jc
        met["delta_J_repair_minus_clean"] = [
            round(float(dJ.mean()), 4),
            pcb_groups(Pall, lambda idx: dJ[np.concatenate(idx)].mean())]

        own_c = [q for q in Q if C[q][0] and C[q][1]]
        own_f = [q for q in Q if F[q][0] and F[q][1]]
        com = [q for q in Q if (C[q][0] and C[q][1]) and (F[q][0] and F[q][1])]

        def err(P, QQ):
            return round(float(np.mean([1 - P[q][2] for q in QQ])), 4) if QQ else None
        met["denoms"] = {"own_clean": [err(C, own_c), len(own_c)],
                         "own_repair": [err(F, own_f), len(own_f)],
                         "baseline_fixed_repair": [err(F, own_c), len(own_c)],
                         "common_clean": [err(C, com), len(com)],
                         "common_repair": [err(F, com), len(com)]}
        met["overall"] = {
            "clean_ABerr": round(float(1 - np.mean([C[q][2] for q in Q])), 4),
            "repair_ABerr": round(float(1 - np.mean([F[q][2] for q in Q])), 4),
            "clean_atomic_pass": round(float(np.mean([C[q][0] and C[q][1] for q in Q])), 4),
            "repair_atomic_pass": round(float(np.mean([F[q][0] and F[q][1] for q in Q])), 4)}

        # ---- R4.6 zero-training rank/sort diagnostic for 110->001 dominance ----
        # check whether the AB-error pattern is explained by a shared score
        # threshold/rank ordering (monotone recalibration) on the SAME bank:
        # if AB errors of clean and flipmine on base110 are consistent with a
        # single monotone score order, a threshold story cannot be excluded.
        model_c, stats_c = models[(seed, "clean")]
        model_f, stats_f = models[(seed, "flipmine")]
        diag = {}
        with torch.no_grad():
            xs_all = []
            qid_to_i1 = {}
            for qi, m in enumerate(Qmeta):
                if m["ptype"] == "A":
                    qid_to_i1[m["qid"]] = qi
            for qid in base110:
                i1 = qid_to_i1[qid]
                xa = np.asarray(Qx[i1], float)
                xab = xa + np.asarray(Qe[i1], float)
                xs_all += [xa, xab]
            out = []
            for s in range(0, len(xs_all), 512):
                out.append(model_c(preprocess(np.stack(xs_all[s:s + 512]).astype(np.float32), stats_c)).numpy().reshape(-1))
            lg_c = np.concatenate(out)
            out = []
            for s in range(0, len(xs_all), 512):
                out.append(model_f(preprocess(np.stack(xs_all[s:s + 512]).astype(np.float32), stats_f)).numpy().reshape(-1))
            lg_f = np.concatenate(out)
        # AB state correctness for clean on base110 under its own and under
        # flipmine's score order
        cab_c = (lg_c[1::2] > 0).astype(int)  # clean AB pred correct flag proxy
        # Spearman between clean-AB-score and flip-AB-score on base110
        # rank-correlate via argsort to avoid scipy import-chain issues
        def _rank(x):
            return np.argsort(np.argsort(x)).astype(float)
        rc = _rank(lg_c[1::2]); rf = _rank(lg_f[1::2])
        sp_val = float(np.corrcoef(rc, rf)[0, 1])
        diag["spearman_ABscore_clean_vs_flipmine"] = (None if np.isnan(sp_val)
                                                       else round(sp_val, 4))
        diag["n_base110"] = n
        # fraction of base110 where both models give the SAME AB sign
        diag["same_AB_sign_frac"] = round(float(np.mean(
            (lg_c[1::2] > 0) == (lg_f[1::2] > 0))), 4)
        met["rank_diag"] = diag

        res["s%d" % seed] = {"transition_matrix_full8x8": matrix,
                             "trans_top": sorted(trans.items(), key=lambda kv: -kv[1])[:12],
                             "metrics": met}
        print("s%d: n110=%d R_full=%s R_endpoint=%s M=%s M/Re=%s dJ=%s ident=%s" % (
            seed, n, met["R_full"][0], met["R_endpoint"][0], met["M_migrate"][0],
            met["M_over_R_endpoint"], met["delta_J_repair_minus_clean"],
            ident_ok), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "P3_FINAL.json", "w"), indent=1)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
