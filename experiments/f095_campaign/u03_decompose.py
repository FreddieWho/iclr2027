#!/usr/bin/env python3
"""U03 three-part decomposition on archived four-arm quartet scores.

Builds per-quartet [n,3] score/label arrays (states A/B/AB; labels
yS/yS/yAB from the bank Qmeta) for each arm/seed in
artifacts/next_novelty/u1_factorial/U1_PERQUARTET.csv and runs the exact
threshold_certificate (S, J*, J(t0=0)):
  1 - J = (1 - S) + (S - J*) + (J* - J)
Plus a parent-split threshold-transfer check: select t* on half the
parents, report J on the other half (deployed thresholds must not be
test-oracles). No training, no new forwards, no sealed-pool reads.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "f095dfa_review_pack" / "checks"))
from threshold_certificate import threshold_certificate  # noqa: E402

ARMS = ("raw_clean", "raw_flipmine", "relfeat", "relflip")
SEEDS = (11, 23, 47)
OUT = ROOT / "artifacts" / "f095_campaign" / "U03"


def load():
    pq = ROOT / "artifacts" / "next_novelty" / "u1_factorial" / "U1_PERQUARTET.csv"
    rows = {}
    with open(pq) as f:
        for r in csv.DictReader(f):
            rows[(int(r["seed"]), r["arm"], int(r["qid"]))] = (
                float(r["logit_A"]), float(r["logit_B"]), float(r["logit_AB"]))
    b = np.load(ROOT / "artifacts" / "p123_upgrade" / "bank" / "bank_dev512.npz",
                allow_pickle=True)
    lab = {}
    for m in json.loads(str(b["Qmeta"])):
        lab.setdefault(m["qid"], (m["start_lab"], m["yAB"], m["parent"]))
    return rows, lab


def main():
    rows, lab = load()
    rng = np.random.default_rng(20260923)
    summary = {"bank": "dev512", "states": ["A", "B", "AB"], "seeds": {}}
    for seed in SEEDS:
        qids = sorted({q for (s, _, q) in rows if s == seed})
        parents = np.array([lab[q][2] for q in qids])
        ups = np.unique(parents)
        sel = set(rng.choice(ups, size=len(ups) // 2, replace=False).tolist())
        sm = np.array([lab[q][2] in sel for q in qids])
        arms = {}
        for arm in ARMS:
            sc = np.array([rows[(seed, arm, q)] for q in qids])
            lb = np.array([(lab[q][0], lab[q][0], lab[q][1]) for q in qids])
            cert = threshold_certificate(sc, lb, 0.0)
            # transfer: best t on selected parents -> J on held-out parents
            cands = np.unique(sc[sm].reshape(-1))
            step = max(1, len(cands) // 400)
            best_t, best_j = 0.0, -1.0
            for t in cands[::step]:
                j = float(np.mean([all((sc[i, k] > t) == lb[i, k] for k in range(3))
                                   for i in np.nonzero(sm)[0]]))
                if j > best_j:
                    best_j, best_t = j, float(t)
            em = ~sm
            je = float(np.mean([all((sc[i, k] > best_t) == lb[i, k] for k in range(3))
                                for i in np.nonzero(em)[0]]))
            arms[arm] = {
                "n": len(qids),
                "S": round(cert.S_local_separable, 4),
                "J_star": round(cert.J_star_global_oracle, 4),
                "J_t0": round(cert.J_at_threshold, 4),
                "ordering_failure": round(cert.ordering_failure, 4),
                "global_incompatibility": round(cert.global_incompatibility, 4),
                "operating_point_gap": round(cert.operating_point_gap, 4),
                "transfer_t_star": best_t,
                "transfer_J_eval_half": round(je, 4),
            }
        summary["seeds"]["s%d" % seed] = arms
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "U03_SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=1)
    for seed in SEEDS:
        print(f"--- s{seed} ---")
        for arm in ARMS:
            a = summary["seeds"]["s%d" % seed][arm]
            print(f"{arm:12s} S={a['S']:.4f} J*={a['J_star']:.4f} J={a['J_t0']:.4f} "
                  f"[ord={a['ordering_failure']:.3f} incomp={a['global_incompatibility']:.3f} "
                  f"op={a['operating_point_gap']:.3f}] transferJ={a['transfer_J_eval_half']:.4f}")


if __name__ == "__main__":
    main()
