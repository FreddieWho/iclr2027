#!/usr/bin/env python3
"""WP-B eval: unified metrics + denominator triple-report per transfer model.

Per model: static, atomic A/B, both-frac, comp miss on (1) own conditional
denom, (2) baseline-fixed paired subset (r04b_s11/clean atomics correct),
(3) common-correct (own AND baseline correct); matched-single miss;
preserve FA (stay-pair sign-change rate on dev stays). Filter by --target.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "event_updater"))
torch.set_num_threads(4)
from coord_mlp import CoordMLP  # noqa: E402
from common import load_model, preprocess  # noqa: E402
from paths import oracle_at, atomic_edits  # noqa: E402
from r02_search import candidates_for_scene  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
TR = ROOT / "artifacts" / "novelty_round3" / "transferable_repair"
EU = ROOT / "artifacts" / "event_updater"


def pred_of(model, mu, sd, xs):
    xs = np.asarray(xs, dtype=np.float32)
    with torch.no_grad():
        out = []
        for s in range(0, len(xs), 256):
            F = (xs[s:s + 256].reshape(-1, 8) - mu) / sd
            out.append(model(torch.from_numpy(F.astype(np.float32))).numpy())
    lg = np.concatenate(out).reshape(-1)
    return lg, (lg > 0).astype(int)


def first_flip(x, frng):
    for item in candidates_for_scene(x, frng, n_random=32):
        e = np.asarray(item[0], dtype=float)
        try:
            y0, _, _ = oracle_at(x)
            y1, m1, _ = oracle_at(x + e)
        except ValueError:
            continue
        if y1 != y0 and m1 >= 0.03:
            return e, y1
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--out", type=Path, default=TR / "eval_results")
    a = p.parse_args()
    d = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    X = d["positions"].astype(np.float32)
    y_all = np.array([oracle_at(x)[0] for x in X])
    base, bstats = load_model(ART / "r04b_s11" / "clean")
    base.eval()

    def pred_base(xs):
        xs = np.asarray(xs, dtype=np.float32)
        with torch.no_grad():
            out = []
            for s in range(0, len(xs), 256):
                out.append(base(preprocess(xs[s:s + 256], bstats)).numpy())
        lg = np.concatenate(out)
        return (lg > 0).astype(int)

    dv = np.load(EU / "events_dev.npz")
    stay0, stay1 = dv["x0"][dv["u"] == 0], dv["x1"][dv["u"] == 0]
    res = {}
    files = sorted(TR.glob(f"transfer_*_{a.target}_s*.pt"))
    for f in files:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        m = CoordMLP(ck["hidden"], ck["feat"], ck.get("in_dim", 8))
        m.load_state_dict(ck["state"])
        m.eval()
        mu = np.asarray(ck["mu"], dtype=np.float32).reshape(-1)
        sd = np.asarray(ck["sd"], dtype=np.float32).reshape(-1)
        lg, pl = pred_of(m, mu, sd, X)
        static = float((pl != y_all).mean())
        # transition miss
        miss = n = 0
        frng = np.random.default_rng(4)
        for pi in range(len(X)):
            x = X[pi].astype(float)
            e, y1 = first_flip(x, frng)
            if e is None:
                continue
            n += 1
            miss += (pred_of(m, mu, sd, (x + e)[None])[1][0] != y1)
        trans = miss / n
        # composition triple denominators
        own_m = own_n = base_m = base_n = com_m = com_n = 0
        aA = aB = both = 0
        tot_ab = 0
        erng = np.random.default_rng(4)
        for pi in np.random.default_rng(4).permutation(len(X))[:128]:
            x = X[pi].astype(float)
            try:
                y0, _, _ = oracle_at(x)
            except ValueError:
                continue
            atomic = []
            for e, _ in atomic_edits(x, erng):
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
                    tot_ab += 1
                    pa = pred_of(m, mu, sd, (x + ea)[None])[1][0]
                    pb = pred_of(m, mu, sd, (x + eb)[None])[1][0]
                    ba = pred_base((x + ea)[None])[0] == ya
                    bb = pred_base((x + eb)[None])[0] == yb
                    aA += (pa == ya)
                    aB += (pb == yb)
                    both += (pa == ya and pb == yb)
                    pc = pred_of(m, mu, sd, (x + ea + eb)[None])[1][0] != yc
                    if pa == ya and pb == yb:
                        own_n += 1
                        own_m += pc
                    if ba and bb:
                        base_n += 1
                        base_m += pc
                    if pa == ya and pb == yb and ba and bb:
                        com_n += 1
                        com_m += pc
        # preserve FA on dev stays
        lg0, _ = pred_of(m, mu, sd, stay0)
        lg1, _ = pred_of(m, mu, sd, stay1)
        fa = float((((lg0 > 0) != (lg1 > 0))).mean())
        # matched-single miss (displacement-matched flips)
        msm = msn = 0
        for pi in range(len(X)):
            x = X[pi].astype(float)
            e, y1 = first_flip(x, np.random.default_rng(8))
            if e is None:
                continue
            msn += 1
            msm += (pred_of(m, mu, sd, (x + e)[None])[1][0] != y1)
        key = f.stem.replace("transfer_", "")
        res[key] = {"static": round(static, 4), "trans": round(trans, 4),
                    "atomicA": round(aA / max(1, tot_ab), 4),
                    "atomicB": round(aB / max(1, tot_ab), 4),
                    "both": round(both / max(1, tot_ab), 4),
                    "comp_own": round(own_m / own_n, 4) if own_n else None,
                    "comp_own_n": own_n,
                    "comp_basefixed": round(base_m / base_n, 4) if base_n else None,
                    "comp_basefixed_n": base_n,
                    "comp_common": round(com_m / com_n, 4) if com_n else None,
                    "comp_common_n": com_n,
                    "matched_single": round(msm / msn, 4) if msn else None,
                    "preserve_FA": round(fa, 4)}
        print(f"SAW {key}: {res[key]}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / f"RESULTS_{a.target}.json", "w"), indent=1)
    print(f"DONE eval {a.target}")


if __name__ == "__main__":
    main()
