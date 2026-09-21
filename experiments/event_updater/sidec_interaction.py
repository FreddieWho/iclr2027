#!/usr/bin/env python3
"""Side C: second-order interaction diagnostic (frozen clean, eval only).

On N04-style emergent quartets: D_AB f = f(x+A+B)-f(x+A)-f(x+B)+f(x) on
logits + same quantity on hidden z (norm). Compare composition-correct vs
composition-miss (both atomics correct). 3 seeds = 3 mining seeds (model
fixed; seeds vary quartet sampling). No stable deficit -> SECOND_ORDER_KILL.
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
from paths import oracle_at, atomic_edits  # noqa: E402
from common import load_model, preprocess  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
OUT = ROOT / "artifacts" / "event_updater"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n_parents", type=int, default=256)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    d = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    X = d["positions"].astype(np.float32)
    model, stats = load_model(ART / "r04b_s11" / "clean")
    model.eval()

    def fwd(xs):
        xs = np.asarray(xs, dtype=np.float32)
        with torch.no_grad():
            lgs, zs = [], []
            for s in range(0, len(xs), 256):
                lg, z = model(preprocess(xs[s:s + 256], stats), return_feat=True)
                lgs.append(lg.numpy())
                zs.append(z.numpy())
        return np.concatenate(lgs), np.concatenate(zs)

    dlogit_hit, dlogit_miss, dhid_hit, dhid_miss = [], [], [], []
    n_q = 0
    for pi in rng.permutation(len(X))[:a.n_parents]:
        x = X[pi].astype(float)
        try:
            y0, _, _ = oracle_at(x)
        except ValueError:
            continue
        atomic = []
        for e, _ in atomic_edits(x, rng):
            try:
                y, _, _ = oracle_at(x + e)
            except ValueError:
                continue
            atomic.append((e, y))
        for i in range(len(atomic)):
            for j in range(i + 1, len(atomic)):
                (ea, ya), (eb, yb) = atomic[i], atomic[j]
                if ya != y0 or yb != y0:
                    continue
                try:
                    yc, _, _ = oracle_at(x + ea + eb)
                except ValueError:
                    continue
                if yc == y0:
                    continue
                xs = np.stack([x, x + ea, x + eb, x + ea + eb])
                lg, z = fwd(xs)
                pa, pb, pc = (lg[1] > 0), (lg[2] > 0), (lg[3] > 0)
                if (pa != ya) or (pb != yb):
                    continue
                n_q += 1
                dlo = abs((lg[3] - lg[1]) - (lg[2] - lg[0]))
                dhi = float(np.linalg.norm((z[3] - z[1]) - (z[2] - z[0])))
                if pc == yc:
                    dlogit_hit.append(dlo)
                    dhid_hit.append(dhi)
                else:
                    dlogit_miss.append(dlo)
                    dhid_miss.append(dhi)
    res = {"seed": a.seed, "n_quartets": n_q}
    for nm, h, m in (("dlogit", dlogit_hit, dlogit_miss),
                     ("dhid", dhid_hit, dhid_miss)):
        h, m = np.array(h), np.array(m)
        res[nm] = {"n_hit": len(h), "n_miss": len(m),
                   "med_hit": round(float(np.median(h)), 5) if len(h) else None,
                   "med_miss": round(float(np.median(m)), 5) if len(m) else None}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"sidec_s{a.seed}.json", "w"), indent=1)
    print(f"SAW sideC s{a.seed}: {json.dumps(res)}")


if __name__ == "__main__":
    main()
