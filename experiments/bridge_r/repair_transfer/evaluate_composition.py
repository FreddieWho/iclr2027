#!/usr/bin/env python3
"""Repair-transfer R10-R11: evaluate all arm x seed on compdev + single-dev.

Metrics per protocol: base_acc, atomic_A/B/mean, both_frac, both_n,
M_comp (+n), M_single (+n, flip endpoint miss given base correct),
preserve_FA, Δ_comp. Writes RESULTS.csv + summary JSON.
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "bridge_r" / "repair_transfer"))
from train_adapter import Adapter

RT = ROOT / "artifacts" / "bridge_r" / "repair_transfer"
FEAT = RT / "feats"


def load_states(prefix):
    Z, Y, P, K = [], [], [], []
    for nm in sorted(glob.glob(str(FEAT / f"{prefix}.*.npz"))):
        d = np.load(nm)
        Z.append(d["z"])
        Y.append(d["y"])
        P.append(d["parent"])
        K.append(d["kind"])
    return (np.concatenate(Z).astype(np.float32), np.concatenate(Y),
            np.concatenate(P), np.concatenate(K))


def predict(net, Z):
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(Z), 512):
            logit, _ = net(torch.from_numpy(Z[s:s + 512]))
            out.append(logit.numpy())
    p = np.concatenate(out)
    return (p > 0).astype(int), p


def main():
    torch.set_num_threads(4)
    Zc, Yc, Pc, Kc = load_states("compdev")
    Zd, Yd, Pd, Kd = load_states("repair_dev")
    rows = []
    for arm in ("static", "flip_only", "balanced"):
        for seed in (11, 23, 47):
            net = Adapter()
            net.load_state_dict(torch.load(RT / "adapters" / f"{arm}_s{seed}.pt",
                                           map_location="cpu"))
            pred_c, _ = predict(net, Zc)
            pred_d, _ = predict(net, Zd)
            nq = len(Yc) // 4
            pc = pred_c.reshape(nq, 4)
            yc = Yc.reshape(nq, 4)
            base_ok = pc[:, 0] == yc[:, 0]
            a_ok = pc[:, 1] == yc[:, 1]
            b_ok = pc[:, 2] == yc[:, 2]
            ab_ok = pc[:, 3] == yc[:, 3]
            both = a_ok & b_ok
            both_n = int(both.sum())
            mcomp = float((~ab_ok[both]).mean()) if both_n else float("nan")
            # single-dev: pairs
            nd = len(Yd) // 2
            pd = pred_d.reshape(nd, 2)
            yd = Yd.reshape(nd, 2)
            is_flip = yd[:, 0] != yd[:, 1]
            is_pres = ~is_flip
            base_ok_d = pd[:, 0] == yd[:, 0]
            fl = is_flip & base_ok_d
            msingle = float((pd[fl][:, 1] != yd[fl][:, 1]).mean()) if fl.sum() else float("nan")
            fa = float((pd[is_pres][:, 1] != pd[is_pres][:, 0]).mean()) if is_pres.sum() else float("nan")
            rows.append({"arm": arm, "seed": seed,
                         "base_acc": float(base_ok.mean()),
                         "atomic_A": float(a_ok.mean()), "atomic_B": float(b_ok.mean()),
                         "atomic_mean": float((a_ok.mean() + b_ok.mean()) / 2),
                         "both_frac": float(both.mean()), "both_n": both_n,
                         "composition_miss": mcomp, "composition_n": both_n,
                         "single_flip_miss": msingle, "single_flip_n": int(fl.sum()),
                         "delta_comp": (mcomp - msingle
                                        if both_n and fl.sum() else float("nan")),
                         "preserve_FA": fa})
            r = rows[-1]
            print(f"SAW {arm} s{seed}: atomic={r['atomic_mean']:.3f} both={r['both_n']} "
                  f"Mcomp={r['composition_miss']:.3f} Msingle={r['single_flip_miss']:.3f} "
                  f"dComp={r['delta_comp']:.3f} FA={r['preserve_FA']:.3f}", flush=True)
    import csv
    with open(RT / "RESULTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(RT / "RESULTS.json", "w"), indent=2)
    print("DONE evaluate")


if __name__ == "__main__":
    main()
