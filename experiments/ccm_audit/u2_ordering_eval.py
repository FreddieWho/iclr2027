#!/usr/bin/env python3
"""U2 ordering diagnostic: can a threshold fix the same quartet?

Reuses U1 per-quartet logits (no new forwards). Per arm/seed computes
J (fixed threshold 0), H = P(G>0), boundary share, per-label-class H,
G<=0 failure share and raw_clean -> repair state changes, plus a
parent-split threshold-transfer check (select a global threshold on half
the parents, evaluate J on the other half; no sealed-pool reads).

NEW outputs only:
  artifacts/next_novelty/u2_ordering/U2_SUMMARY.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "experiments" / "ccm_audit"))
from ordering import G_of, H_of, fixed_threshold_J  # noqa: E402

ARMS = ("raw_clean", "raw_flipmine", "relfeat", "relflip")
rng = np.random.default_rng(20260923)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--perquartet", type=Path, default=ROOT / "artifacts" /
                   "next_novelty" / "u1_factorial" / "U1_PERQUARTET.csv")
    p.add_argument("--bank", default="dev512")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" /
                   "next_novelty" / "u2_ordering")
    a = p.parse_args()
    b = np.load(ROOT / "artifacts" / "p123_upgrade" / "bank" /
                ("bank_" + a.bank + ".npz"), allow_pickle=True)
    Qmeta = json.loads(str(b["Qmeta"]))
    lab = {}
    for m in Qmeta:
        lab.setdefault(m["qid"], {"yAB": m["yAB"], "start": m["start_lab"],
                                  "parent": m["parent"]})
    rows = {}
    with open(a.perquartet) as f:
        for r in csv.DictReader(f):
            rows[(int(r["seed"]), r["arm"], int(r["qid"]))] = (
                float(r["logit_A"]), float(r["logit_B"]),
                float(r["logit_AB"]))
    out = {"bank": a.bank, "seeds": {}}
    for seed in (11, 23, 47):
        qids = sorted({q for (s, _, q) in rows if s == seed})
        arms = {}
        G = {}
        for arm in ARMS:
            fA = np.array([rows[(seed, arm, q)][0] for q in qids])
            fB = np.array([rows[(seed, arm, q)][1] for q in qids])
            fAB = np.array([rows[(seed, arm, q)][2] for q in qids])
            yAB = np.array([lab[q]["yAB"] for q in qids])
            yS = np.array([lab[q]["start"] for q in qids])
            g = G_of(fA, fB, fAB, yAB)
            G[arm] = g
            H, _ = H_of(fA, fB, fAB, yAB)
            J = fixed_threshold_J(fA, fB, fAB, yS, yS, yAB, 0.0)
            per_class = {}
            for c in (0, 1):
                m = yAB == c
                Hc, _ = H_of(fA[m], fB[m], fAB[m], yAB[m])
                Jc = fixed_threshold_J(fA[m], fB[m], fAB[m], yS[m], yS[m],
                                       yAB[m], 0.0)
                per_class["yAB_%d" % c] = {"n": int(m.sum()),
                                           "H": round(Hc, 4),
                                           "J": round(Jc, 4)}
            arms[arm] = {"n": len(qids), "J_t0": round(J, 4),
                         "H": round(H, 4),
                         "G_le0_frac": round(float(np.mean(g <= 0)), 4),
                         "boundary_frac": round(float(np.mean(g == 0)), 4),
                         "per_class": per_class}
        C = (G["raw_clean"] > 0)
        changes = {}
        for arm in ("raw_flipmine", "relfeat", "relflip"):
            F = (G[arm] > 0)
            changes[arm] = {
                "stay_sep": int(np.sum(C & F)),
                "become_sep": int(np.sum(~C & F)),
                "become_unsep": int(np.sum(C & ~F)),
                "stay_unsep": int(np.sum(~C & ~F)),
            }
        parents = np.array([lab[q]["parent"] for q in qids])
        ups = np.unique(parents)
        sel = set(rng.choice(ups, size=len(ups) // 2, replace=False).tolist())
        transfer = {}
        for arm in ARMS:
            fA = np.array([rows[(seed, arm, q)][0] for q in qids])
            fB = np.array([rows[(seed, arm, q)][1] for q in qids])
            fAB = np.array([rows[(seed, arm, q)][2] for q in qids])
            yAB = np.array([lab[q]["yAB"] for q in qids])
            yS = np.array([lab[q]["start"] for q in qids])
            sm = np.array([lab[q]["parent"] in sel for q in qids])
            cands = np.unique(np.concatenate([fA[sm], fB[sm], fAB[sm]]))
            best_t, best_j = 0.0, -1.0
            for t in cands[:: max(1, len(cands) // 400)]:
                j = fixed_threshold_J(fA[sm], fB[sm], fAB[sm], yS[sm],
                                      yS[sm], yAB[sm], float(t))
                if j > best_j:
                    best_j, best_t = j, float(t)
            em = ~sm
            He, _ = H_of(fA[em], fB[em], fAB[em], yAB[em])
            Je = fixed_threshold_J(fA[em], fB[em], fAB[em], yS[em], yS[em],
                                   yAB[em], best_t)
            transfer[arm] = {"dev_half_J": round(best_j, 4),
                             "eval_half_J": round(Je, 4),
                             "eval_half_H": round(He, 4),
                             "n_sel_parents": int(sm.sum()),
                             "n_eval_parents": int(em.sum())}
        out["seeds"]["s%d" % seed] = {"arms": arms,
                                      "separability_change_vs_rawclean": changes,
                                      "threshold_transfer_parent_split": transfer}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out / "U2_SUMMARY.json", "w"), indent=1)
    for seed in (11, 23, 47):
        d = out["seeds"]["s%d" % seed]["arms"]
        print("s%d" % seed,
              {k: (v["J_t0"], v["H"]) for k, v in d.items()}, flush=True)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
