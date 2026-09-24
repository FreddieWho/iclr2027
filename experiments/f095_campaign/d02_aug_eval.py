#!/usr/bin/env python3
"""D02-aug evaluation: identity J + orbit stats for augmented-training arms.

Loads checkpoints from artifacts/f095_campaign/D02/aug_{g8,rep}_s{seed}/{clean,flipmine},
evaluates identity-labeling J plus the D02 orbit panel (per-g J range, all-8,
disagreement, group-averaged J) on the dev bank. Same quartet/label contract.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
from u01_eval import load_arm, featurize_raw  # noqa: E402
from symmetry import GROUP, apply  # noqa: E402

torch.set_num_threads(4)
BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
D02D = ROOT / "artifacts" / "f095_campaign" / "D02"


def eval_dir(mp: Path):
    model, ck = load_arm(mp)
    model.eval()
    b = np.load(BANK / "bank_dev512.npz", allow_pickle=True)
    Qx, Qe = b["Qx"], b["Qe"]
    Qmeta = json.loads(str(b["Qmeta"]))
    by_q = {}
    for qi, m in enumerate(Qmeta):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    Q = sorted(qid for qid, _, _ in quads)
    base = {}
    for qid, (i1, m1), (i2, m2) in quads:
        xa = np.asarray(Qx[i1], float).reshape(8)
        xb = np.asarray(Qx[i2], float).reshape(8)
        xab = (np.asarray(Qx[i1], float) + np.asarray(Qe[i1], float)).reshape(8)
        la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
        lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
        base[qid] = (xa, xb, xab, (la, lb, m1["yAB"]))
    labels = np.array([base[q][3] for q in Q])
    per_g = np.zeros((len(Q), 3, len(GROUP)))
    with torch.no_grad():
        for gi, g in enumerate(GROUP):
            xs = []
            for q in Q:
                xa, xb, xab, _ = base[q]
                xs += [apply(xa, g), apply(xb, g), apply(xab, g)]
            lgs = []
            for s in range(0, len(xs), 512):
                xx = np.stack(xs[s:s + 512])
                lgs.append(model(featurize_raw(xx, ck)).numpy().reshape(-1))
            per_g[:, :, gi] = np.concatenate(lgs).reshape(-1, 3)
    ok_g = ((per_g > 0).astype(int) == labels[:, :, None])
    J_g = ok_g.all(axis=1).mean(axis=0)
    avg = per_g.mean(axis=2)
    ok_avg = ((avg > 0).astype(int) == labels)
    ident = [float(v) for v in J_g[[GROUP.index((0, 1, 2, 3))]]]
    return {"J_ident": round(ident[0], 4),
            "J_groupavg": round(float(ok_avg.all(axis=1).mean()), 4),
            "per_g_min": round(float(J_g.min()), 4),
            "per_g_max": round(float(J_g.max()), 4),
            "all8": round(float(ok_g.all(axis=(1, 2)).mean()), 4),
            "disagreement": round(float((ok_g[:, :2, :].all(axis=1).max(axis=1) !=
                                         ok_g[:, :2, :].all(axis=1).min(axis=1)).mean()), 4)}


def main():
    out = {}
    for aug in ("g8", "rep"):
        for seed in (11, 23, 47):
            for m in ("clean", "flipmine"):
                name = f"{aug}_s{seed}_{m}"
                mp = D02D / f"aug_{aug}_s{seed}" / m
                out[name] = eval_dir(mp)
                print(name, out[name], flush=True)
    with open(D02D / "D02_AUG_EVAL.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", D02D / "D02_AUG_EVAL.json")


if __name__ == "__main__":
    main()
