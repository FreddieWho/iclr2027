#!/usr/bin/env python3
"""D03 evaluation: Bank-O endpoint/update-risk per order, Bank-E J/S + R_full/M,
Bank-EA subsets, orbit averages on E + singles. Reuses frozen four-arm
checkpoints (no training). Dev-fitted confidence = train_101 top-tertile
|logit| per model (no eval-bank leakage).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "ccm_audit"))
sys.path.insert(0, str(ROOT / "docs" / "f095dfa_review_pack" / "checks"))
from common import rel_features  # noqa: E402
from symmetry import GROUP, apply  # noqa: E402
from threshold_certificate import threshold_certificate  # noqa: E402
from u01_eval import load_arm  # noqa: E402
from u1_factorial_eval import arm_checkpoint  # noqa: E402

torch.set_num_threads(4)
BANKDIR = ROOT / "artifacts" / "p123_upgrade" / "bank"
OUTD = ROOT / "artifacts" / "f095_campaign" / "D03"
ARMS = ("raw_clean", "raw_flipmine", "relfeat", "relflip")
SEEDS = (11, 23, 47)
ORDERS = ("first", "random", "stratified")


def sha_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def feat_batch(X44, ck):
    mu = np.asarray(ck["mu"], dtype=np.float32)
    sd = np.asarray(ck["sd"], dtype=np.float32)
    if ck.get("featurize") == "rel12":
        F = rel_features(np.asarray(X44, np.float32))
    else:
        F = np.asarray(X44, np.float32).reshape(len(X44), -1)
    return torch.from_numpy(((F - mu) / sd).astype(np.float32))


@torch.no_grad()
def fwd(model, X44, ck, bs=4096):
    model.eval()
    out = []
    for s in range(0, len(X44), bs):
        out.append(model(feat_batch(X44[s:s + bs], ck)).numpy().reshape(-1))
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bankO", default="d03O")
    p.add_argument("--bankE", default="d03E")
    a = p.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    bo = np.load(BANKDIR / f"bank_{a.bankO}.npz", allow_pickle=True)
    Sx, Se = np.asarray(bo["Sx"], float), np.asarray(bo["Se"], float)
    Sm = json.loads(str(bo["Smeta"]))
    be = np.load(BANKDIR / f"bank_{a.bankE}.npz", allow_pickle=True)
    Qx, Qe = np.asarray(be["Qx"], float), np.asarray(be["Qe"], float)
    Qm = json.loads(str(be["Qmeta"]))
    sc = np.load(ROOT / "artifacts" / "f095_campaign" / "D01" / "scenes_16N" / "scenes.npz",
                 allow_pickle=True)
    X16, y16 = np.asarray(sc["positions"], float), np.asarray(sc["labels"], int)
    dtr = np.load(ROOT / "artifacts" / "discovery_campaign" / "scenes" / "train_101" / "scenes.npz")
    Xtr = np.asarray(dtr["positions"], float)
    res = {"bankO_sha256": sha_of(BANKDIR / f"bank_{a.bankO}.npz"),
           "bankE_sha256": sha_of(BANKDIR / f"bank_{a.bankE}.npz"),
           "seeds": {}}
    by_q = {}
    for qi, m in enumerate(Qm):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    for seed in SEEDS:
        R = {}
        # per-model base data
        for arm in ARMS:
            mp = arm_checkpoint(arm, seed)
            model, ck = load_arm(mp)
            # start states: base scenes of D03 parents present in singles
            parents = sorted(set(m["parent"] for m in Sm))
            X0 = np.stack([X16[p] for p in parents])
            y0 = np.array([y16[p] for p in parents])
            lg0 = fwd(model, X0, ck)
            pred0 = (lg0 > 0).astype(int)
            # singles predictions (+ orbit average)
            lgS = fwd(model, Sx, ck)
            predS = (lgS > 0).astype(int)
            # orbit-averaged logits on singles (selected states only; 
            # unselected singles enter no estimand)
            sel_any = np.array([bool(m["sel_flip"] or m["sel_keep"]) for m in Sm])
            Sx_sel = Sx[sel_any]
            lgO_sel = np.zeros(len(Sx_sel))
            for g in GROUP:
                Xg = np.stack([apply(s, g) for s in Sx_sel.reshape(-1, 4, 2)])
                lgO_sel += fwd(model, Xg, ck)
            lgO_sel /= len(GROUP)
            lgO = np.full(len(Sx), np.nan)
            lgO[sel_any] = lgO_sel
            predO = (lgO_sel > 0).astype(int)
            y1 = np.array([m["y1"] for m in Sm])
            y1sel = y1[sel_any]
            is_flip = np.array([m["flip"] for m in Sm], dtype=bool)
            sm_par = np.array([m["parent"] for m in Sm])
            p0map = {p: i for i, p in enumerate(parents)}
            start_ok = np.array([pred0[p0map[m["parent"]]] == m["y0"] for m in Sm])
            # confidence threshold: train_101 top-tertile |logit|
            lgtr = fwd(model, Xtr, ck)
            thr = float(np.quantile(np.abs(lgtr), 2 / 3))
            armR = {"model_sha": sha_of(mp / "model.pt"), "conf_thr": round(thr, 4),
                    "orders": {}}
            for order in ORDERS:
                fsel = np.array([order in m["sel_flip"] for m in Sm])
                ksel = np.array([order in m["sel_keep"] for m in Sm])
                o = {}
                # parent-equal primary + edit-equal secondary
                for wname, w in (("parent", None), ("edit", None)):
                    o[wname] = {}
                    for sname, sel, lab in (("start", None, None),
                                            ("preserve", ksel, [m["y1"] for m in Sm]),
                                            ("flip", fsel, [m["y1"] for m in Sm])):
                        if sname == "start":
                            if wname == "parent":
                                vals = [float(pred0[p0map[p]] != y16[p]) for p in parents]
                            else:
                                vals = [float(pred0[p0map[m["parent"]]] != m["y0"]) for m in Sm]
                            o[wname][sname] = round(float(np.mean(vals)), 4)
                            o[wname][sname + "_n"] = len(vals)
                        else:
                            lab = np.array(lab)
                            pr = (predS[sel] != lab[sel])
                            if wname == "parent":
                                ps = sorted(set(sm_par[sel].tolist()))
                                vals = [float(pr[sm_par[sel] == p].mean())
                                        for p in ps] if len(ps) else [float("nan")]
                            else:
                                vals = [pr]
                                vals = [float(np.concatenate(vals).mean())] if pr.size else [float("nan")]
                            o[wname][sname] = round(float(np.nanmean(vals)), 4)
                            o[wname][sname + "_n"] = len(ps) if wname == "parent" else int(sel.sum())
                # update risk | start-correct (pooled + paired)
                for lname, sel in (("preserve", ksel), ("flip", fsel)):
                    sub = sel & start_ok
                    o["risk_startcorrect_" + lname] = \
                        round(float((predS[sub] != y1[sub]).mean()), 4) if sub.any() else None
                    o["risk_startcorrect_" + lname + "_n"] = int(sub.sum())
                    hi = (np.abs(lgS[sel]) >= thr) & start_ok[sel]
                    ii = np.nonzero(sel)[0][hi]
                    o["risk_hiconf_" + lname] = \
                        round(float((predS[ii] != y1[ii]).mean()), 4) if len(ii) else None
                # orbit-averaged endpoint errors (selected states)
                for lname, sel in (("preserve", ksel), ("flip", fsel)):
                    ii = np.nonzero(sel)[0]
                    jj = np.searchsorted(np.nonzero(sel_any)[0], ii)
                    o["orbit_" + lname] = round(float((predO[jj] != y1sel[jj]).mean()), 4) \
                        if len(ii) else None
                # direction strata (y0->y1) on flips
                y0a = np.array([m["y0"] for m in Sm])
                for d0, d1 in ((0, 1), (1, 0)):
                    dd = fsel & (y0a == d0) & (y1 == d1)
                    o[f"flip_dir{d0}{d1}"] = round(float((predS[dd] != y1[dd]).mean()), 4) \
                        if dd.any() else None
                    o[f"flip_dir{d0}{d1}_n"] = int(dd.sum())
                armR["orders"][order] = o
            R[arm] = armR
        # Bank-E eval (all arms) + EA subsets + orbit
        # E states per quartet
        eq = {}
        for qid, (i1, m1), (i2, m2) in quads:
            xa = np.asarray(Qx[i1], float).reshape(4, 2)
            xb = np.asarray(Qx[i2], float).reshape(4, 2)
            xab = xa + np.asarray(Qe[i1], float).reshape(4, 2)
            la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
            lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
            eq[qid] = (xa, xb, xab, (la, lb, m1["yAB"]), m1["parent"])
        Q = sorted(eq)
        Eres = {}
        for arm in ARMS:
            mp = arm_checkpoint(arm, seed)
            model, ck = load_arm(mp)
            lgA = fwd(model, np.stack([eq[q][0] for q in Q]), ck)
            lgB = fwd(model, np.stack([eq[q][1] for q in Q]), ck)
            lgAB = fwd(model, np.stack([eq[q][2] for q in Q]), ck)
            lab = np.array([eq[q][3] for q in Q])
            scm = np.stack([lgA, lgB, lgAB], axis=1)
            pred = (scm > 0).astype(int)
            ok = (pred == lab)
            cert = threshold_certificate(scm, lab, 0.0)
            # orbit-averaged J
            gsum = np.zeros((len(Q), 3))
            for g in GROUP:
                ga = fwd(model, np.stack([apply(eq[q][0], g) for q in Q]), ck)
                gb = fwd(model, np.stack([apply(eq[q][1], g) for q in Q]), ck)
                gc = fwd(model, np.stack([apply(eq[q][2], g) for q in Q]), ck)
                gsum += np.stack([ga, gb, gc], axis=1)
            gavg = gsum / len(GROUP)
            Eres[arm] = {
                "J": round(float(ok.all(axis=1).mean()), 4),
                "atomic_pass": round(float(ok[:, :2].all(axis=1).mean()), 4),
                "AB_correct": round(float(ok[:, 2].mean()), 4),
                "S": round(cert.S_local_separable, 4),
                "J_star": round(cert.J_star_global_oracle, 4),
                "J_orbitavg": round(float(((gavg > 0).astype(int) == lab).all(axis=1).mean()), 4),
                "n": len(Q),
                "_okAE": [int(v) for v in ok[:, :2].all(axis=1)],
                "_okAB": [int(v) for v in ok[:, 2]],
                "_okAEB": [int(v) for v in ok.all(axis=1)],
            }
            print(f"s{seed} {arm} E-J={Eres[arm]['J']} S={Eres[arm]['S']} "
                  f"Jorb={Eres[arm]['J_orbitavg']}", flush=True)
        # R_full/M vs raw_clean B110 on Bank-E
        rcAE = np.array(Eres["raw_clean"]["_okAE"])
        rcAB = np.array(Eres["raw_clean"]["_okAB"])
        H = (rcAE == 1) & (rcAB == 0)
        for arm in ARMS:
            eAB = np.array(Eres[arm]["_okAB"])
            eAE = np.array(Eres[arm]["_okAE"])
            eAEB = np.array(Eres[arm]["_okAEB"])
            Eres[arm]["repair_vs_rawclean"] = {
                "n_H": int(H.sum()),
                "R_endpoint": round(float(eAB[H].mean()), 4) if H.sum() else None,
                "R_full": round(float(eAEB[H].mean()), 4) if H.sum() else None,
                "M": round(float(((eAB == 1) & (eAE == 0))[H].mean()), 4) if H.sum() else None,
            }
            for fk in ("_okAE", "_okAB", "_okAEB"):
                del Eres[arm][fk]
        # EA subsets: raw_clean atomic-correct (flags recomputed; deleted above)
        mp = arm_checkpoint("raw_clean", seed)
        model, ck = load_arm(mp)
        lgA = fwd(model, np.stack([eq[q][0] for q in Q]), ck)
        lgB = fwd(model, np.stack([eq[q][1] for q in Q]), ck)
        lgAB = fwd(model, np.stack([eq[q][2] for q in Q]), ck)
        lab = np.array([eq[q][3] for q in Q])
        oka = ((np.stack([lgA, lgB, lgAB], axis=1) > 0).astype(int) == lab)
        EAkeep = oka[:, :2].all(axis=1)
        EA = {}
        for arm in ARMS:
            mp2 = arm_checkpoint(arm, seed)
            model2, ck2 = load_arm(mp2)
            lgA2 = fwd(model2, np.stack([eq[q][0] for q in Q]), ck2)
            lgB2 = fwd(model2, np.stack([eq[q][1] for q in Q]), ck2)
            lgAB2 = fwd(model2, np.stack([eq[q][2] for q in Q]), ck2)
            sc2 = np.stack([lgA2, lgB2, lgAB2], axis=1)
            ok2 = ((sc2 > 0).astype(int) == lab)
            sub = ok2[EAkeep]
            cert = threshold_certificate(sc2[EAkeep], lab[EAkeep], 0.0) if EAkeep.sum() else None
            EA[arm] = {"n": int(EAkeep.sum()),
                       "J": round(float(sub.all(axis=1).mean()), 4) if EAkeep.sum() else None,
                       "S": round(cert.S_local_separable, 4) if cert else None}
        R["E"] = Eres
        R["EA"] = EA
        res["seeds"]["s%d" % seed] = R
    with open(OUTD / "D03_EVAL.json", "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUTD / "D03_EVAL.json")


if __name__ == "__main__":
    main()
