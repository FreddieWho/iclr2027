#!/usr/bin/env python3
"""Competence-first C3: atomic-only evaluation on compdev base/A/B.

AB is NEVER loaded here. Writes per-seed atomic rows to RESULTS.csv
(composition_opened=False). A separate script opens AB only if C1 passes.
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "bridge_r" / "competence_first"))
from train_spatial_head import SpatialHead

GF = ROOT / "artifacts" / "bridge_r" / "competence_first" / "feats"
RT = ROOT / "artifacts" / "bridge_r" / "competence_first"


def main():
    torch.set_num_threads(4)
    P, Y, Q = [], [], []
    for nm in sorted(glob.glob(str(GF / "grid_compdev_abb.*.npz"))):
        d = np.load(nm)
        P.append(d["patches"])
        Y.append(d["labels"])
        Q.append(d["parent"])
    P = np.concatenate(P).astype(np.float32)
    Y = np.concatenate(Y)
    Q = np.concatenate(Q)
    nq = len(P) // 3
    assert len(P) == 3 * nq
    X = torch.from_numpy(P.reshape(len(P), 14, 14, 1024).transpose(0, 3, 1, 2))
    rows = []
    for seed in (11, 23, 47):
        net = SpatialHead()
        net.load_state_dict(torch.load(RT / "static_head" / f"static_s{seed}.pt",
                                       map_location="cpu"))
        net.eval()
        with torch.no_grad():
            out = []
            for s in range(0, len(X), 512):
                out.append(net(X[s:s + 512]).numpy())
        pred = (np.concatenate(out) > 0).astype(int).reshape(nq, 3)
        y = Y.reshape(nq, 3)
        b, a_, b_ = [(pred[:, k] == y[:, k]).mean() for k in range(3)]
        base_acc, a_acc, b_acc = b, a_, b_
        atomic_mean = float((a_acc + b_acc) / 2)
        both = (pred[:, 1] == y[:, 1]) & (pred[:, 2] == y[:, 2])
        rows.append({"seed": seed, "dev_state_acc": json.load(
            open(RT / "static_head" / f"static_s{seed}.json"))["best_dev_acc"],
            "base_acc": round(float(base_acc), 4), "A_acc": round(float(a_acc), 4),
            "B_acc": round(float(b_acc), 4), "atomic_mean": round(atomic_mean, 4),
            "min_atom": round(float(min(a_acc, b_acc)), 4),
            "both_frac": round(float(both.mean()), 4), "both_n": int(both.sum()),
            "composition_opened": False, "composition_miss": "",
            "composition_n": int(both.sum())})
        r = rows[-1]
        print(f"SAW s{seed}: dev={r['dev_state_acc']:.4f} base={r['base_acc']:.4f} "
              f"A={r['A_acc']:.4f} B={r['B_acc']:.4f} atomic={r['atomic_mean']:.4f} "
              f"min={r['min_atom']:.4f} both={r['both_frac']:.4f} n={r['both_n']}", flush=True)
    import csv
    with open(RT / "COMPETENCE_FIRST_RESULTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    c1 = sum(1 for r in rows if r["atomic_mean"] >= 0.85 and r["min_atom"] >= 0.82
             and r["both_frac"] >= 0.60)
    print(f"C1 GATE: {c1}/3 seeds pass")


if __name__ == "__main__":
    main()
