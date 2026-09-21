#!/usr/bin/env python3
"""Radius pilot eval: WP4-style unified metrics + R-shift + stay FA per arm/seed.

Metrics on eval_202 (dev): static_err, trans_miss (first flip), comp_cond
(N04-style, 128 parents), action_invalid lam=2 (unified scenes), stay_FA
(P(flip pred) on dev stay pairs), R medians (r_m miss/hit on 100 parents).
Checkpoints: artifacts/radius_loss/radius_{arm}_s{seed}.pt (own mu/sd).
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
sys.path.insert(0, str(ROOT / "experiments" / "event_updater"))
torch.set_num_threads(4)
from paths import action_library, oracle_at, atomic_edits  # noqa: E402
from common import load_model, preprocess  # noqa: E402
from coord_mlp import CoordMLP  # noqa: E402
from r02_search import candidates_for_scene  # noqa: E402
from sided_radius import bisect_radius  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
RL = ROOT / "artifacts" / "radius_loss"
EU = ROOT / "artifacts" / "event_updater"


def load_arm(arm, seed):
    ck = torch.load(RL / f"radius_{arm}_s{seed}.pt", map_location="cpu",
                    weights_only=False)
    m = CoordMLP(ck["hidden"], ck["feat"], ck.get("in_dim", 8))
    m.load_state_dict(ck["state"])
    m.eval()
    return m, {"mu": ck["mu"], "sd": ck["sd"]}


def pred(m, s, xs):
    xs = np.asarray(xs, dtype=np.float32)
    with torch.no_grad():
        out = []
        for i in range(0, len(xs), 256):
            out.append(m(preprocess(xs[i:i + 256], s)).numpy())
    lg = np.concatenate(out)
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
    p.add_argument("--out", type=Path, default=RL / "pilot_results")
    a = p.parse_args()
    d = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    X = d["positions"].astype(np.float32)
    y_all = np.array([oracle_at(x)[0] for x in X])
    lib = action_library()
    dv = np.load(EU / "events_dev.npz")
    stay = dv["x0"][dv["u"] == 0]
    cross = [i for i in np.random.default_rng(6).permutation(len(X))[:512]
             if oracle_at(X[i])[0] == 1][:256]
    table = {}
    for arm in ("clean", "flipcov", "radius"):
        for seed in (11, 23, 47):
            m, s = load_arm(arm, seed)
            lg, _ = pred(m, s, X)
            static_err = float(((lg > 0).astype(int) != y_all).mean())
            miss = n = 0
            frng = np.random.default_rng(4)
            for pi in range(len(X)):
                x = X[pi].astype(float)
                e, y1 = first_flip(x, frng)
                if e is None:
                    continue
                n += 1
                miss += (pred(m, s, (x + e)[None])[1][0] != y1)
            trans = miss / n
            cn = ce = 0
            erng = np.random.default_rng(4)
            for pi in np.random.default_rng(4).permutation(len(X))[:128]:
                x = X[pi].astype(float)
                try:
                    y0, _, _ = oracle_at(x)
                except ValueError:
                    continue
                res = []
                for e, _ in atomic_edits(x, erng):
                    try:
                        y, _, _ = oracle_at(x + e)
                    except ValueError:
                        continue
                    res.append((e, y))
                for i in range(len(res)):
                    for j in range(i + 1, len(res)):
                        (ea, ya), (eb, yb) = res[i], res[j]
                        if ya != y0 or yb != y0:
                            continue
                        try:
                            yc, _, _ = oracle_at(x + ea + eb)
                        except ValueError:
                            continue
                        if yc == y0:
                            continue
                        pa = pred(m, s, (x + ea)[None])[1][0]
                        pb = pred(m, s, (x + eb)[None])[1][0]
                        if pa == ya and pb == yb:
                            cn += 1
                            ce += (pred(m, s, (x + ea + eb)[None])[1][0] != yc)
            comp = ce / cn if cn else None
            inv = tot = 0
            for pi in cross:
                x = X[pi].astype(float)
                cands = []
                for act_ in lib:
                    try:
                        y, _, _ = oracle_at(x + act_["edit"])
                    except ValueError:
                        continue
                    cands.append((act_, y))
                feas = [c for c in cands if c[1] == 0]
                if not feas:
                    continue
                xs = np.stack([(x + c[0]["edit"]).astype(np.float32) for c in cands])
                lgg, _ = pred(m, s, xs)
                prob0 = 1 / (1 + np.exp(lgg))
                scores = prob0 - 2.0 * np.array([c[0]["cost"] for c in cands])
                tot += 1
                inv += (cands[int(np.argmax(scores))][1] != 0)
            act_inv = inv / tot if tot else None
            lgstay, _ = pred(m, s, stay)
            stay_fa = float((lgstay > 0).mean())  # P(predict flip-side?) -> refined below
            # stay FA proper: fraction of stay pairs with sign change across pair
            lg0, _ = pred(m, s, dv["x0"][dv["u"] == 0])
            lg1, _ = pred(m, s, dv["x1"][dv["u"] == 0])
            stay_fa = float((((lg0 > 0) != (lg1 > 0))).mean())
            # R medians on 100-parent subset
            rm_hit, rm_miss = [], []
            rrng = np.random.default_rng(9)
            for pi in rrng.permutation(len(X))[:100]:
                x = X[pi].astype(float)
                try:
                    y0, _, _ = oracle_at(x)
                except ValueError:
                    continue
                dirs = []
                for item in candidates_for_scene(x, rrng, n_random=8):
                    e = np.asarray(item[0], dtype=float)
                    nn = np.linalg.norm(e)
                    if nn > 1e-9:
                        dirs.append(e / nn)
                    if len(dirs) >= 12:
                        break
                with torch.no_grad():
                    ml = lambda xx: float(m(preprocess(xx.astype(np.float32)[None], s)).detach().numpy()[0])
                rm = [bisect_radius(x, int(ml(x) > 0), dd, False, m, s) for dd in dirs]
                rm = [r for r in rm if r is not None]
                if not rm:
                    continue
                e, y1 = first_flip(x, np.random.default_rng(10))
                if e is None:
                    continue
                (rm_miss if (ml(x + e) > 0) != y1 else rm_hit).append(min(rm))
            key = f"{arm}_s{seed}"
            table[key] = {"static": round(static_err, 4),
                          "trans": round(trans, 4), "trans_n": n,
                          "comp": round(comp, 4) if comp else None, "comp_n": cn,
                          "action": round(act_inv, 4) if act_inv else None,
                          "stay_fa": round(stay_fa, 4),
                          "rm_hit_med": round(float(np.median(rm_hit)), 4) if rm_hit else None,
                          "rm_miss_med": round(float(np.median(rm_miss)), 4) if rm_miss else None}
            print(f"SAW {key}: {table[key]}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(table, open(a.out / "RESULTS.json", "w"), indent=1)
    print("DONE radius pilot eval")


if __name__ == "__main__":
    main()
