#!/usr/bin/env python3
"""WP-A eval: state gate + unseen composition + A6 representation diagnostic.

A6: mine state pairs (x1,x2) with same oracle label where the SAME edit e
gives different outcomes; compare ||h(x1)-h(x2)|| across A0/A1/A2 encoders.
Behavioral gain (A5) and separation gain (A6) must co-occur for a claim.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "event_updater"))
torch.set_num_threads(4)
from paths import oracle_at, atomic_edits  # noqa: E402
from r02_search import candidates_for_scene  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
OUT = ROOT / "artifacts" / "novelty_round3" / "intervention_sufficiency"
EU = ROOT / "artifacts" / "event_updater"


def load_arm(arm, seed):
    ck = torch.load(OUT / f"interv_{arm}_s{seed}.pt", map_location="cpu",
                    weights_only=False)
    enc = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
    feat = nn.Linear(64, 32)
    cls = nn.Linear(32, 1)
    enc.load_state_dict(ck["enc"])
    feat.load_state_dict(ck["feat"])
    cls.load_state_dict(ck["cls"])
    for m in (enc, feat, cls):
        m.eval()
    mu = np.asarray(ck["mu"], dtype=np.float32).reshape(-1)
    sd = np.asarray(ck["sd"], dtype=np.float32).reshape(-1)
    return enc, feat, cls, mu, sd


def fwd(enc, feat, cls, mu, sd, xs):
    xs = np.asarray(xs, dtype=np.float32)
    with torch.no_grad():
        out_l, out_h = [], []
        for s in range(0, len(xs), 256):
            F = torch.from_numpy(((xs[s:s + 256].reshape(-1, 8) - mu) / sd).astype(np.float32))
            h = enc(F)
            out_h.append(h.numpy())
            out_l.append(cls(feat(h)).numpy())
    return np.concatenate(out_l).reshape(-1), np.concatenate(out_h)


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
    p.add_argument("--out", type=Path, default=OUT / "eval_results")
    a = p.parse_args()
    d = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    X = d["positions"].astype(np.float32)
    y_all = np.array([oracle_at(x)[0] for x in X])
    res = {}
    for arm in ("A0", "A1", "A2"):
        for seed in (11, 23, 47):
            enc, feat, cls, mu, sd = load_arm(arm, seed)
            lg, _ = fwd(enc, feat, cls, mu, sd, X)
            static = float((((lg > 0).astype(int)) != y_all).mean())
            miss = n = 0
            frng = np.random.default_rng(4)
            for pi in range(len(X)):
                x = X[pi].astype(float)
                e, y1 = first_flip(x, frng)
                if e is None:
                    continue
                n += 1
                l2, _ = fwd(enc, feat, cls, mu, sd, (x + e)[None])
                miss += (int(l2[0] > 0) != y1)
            trans = miss / n
            cn = ce = 0
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
                        l_a, _ = fwd(enc, feat, cls, mu, sd, (x + ea)[None])
                        l_b, _ = fwd(enc, feat, cls, mu, sd, (x + eb)[None])
                        if int(l_a[0] > 0) == ya and int(l_b[0] > 0) == yb:
                            cn += 1
                            l_c, _ = fwd(enc, feat, cls, mu, sd, (x + ea + eb)[None])
                            ce += (int(l_c[0] > 0) != yc)
            comp = ce / cn if cn else None
            key = f"{arm}_s{seed}"
            res[key] = {"static": round(static, 4), "trans": round(trans, 4),
                        "comp": round(comp, 4) if comp else None, "comp_n": cn}
            print(f"SAW {key}: {res[key]}", flush=True)
    # A6 representation diagnostic: same-label pairs, same edit, different outcome
    for seed in (11, 23, 47):
        encs = {}
        for arm in ("A0", "A1", "A2"):
            enc, feat, cls, mu, sd = load_arm(arm, seed)
            encs[arm] = (enc, mu, sd)

        def h_of(enc, mu, sd, xs):
            xs = np.asarray(xs, dtype=np.float32)
            with torch.no_grad():
                out = []
                for s in range(0, len(xs), 256):
                    F = torch.from_numpy(((xs[s:s + 256].reshape(-1, 8) - mu) / sd).astype(np.float32))
                    out.append(enc(F).numpy())
            return np.concatenate(out)

        dists = {"A0": [], "A1": [], "A2": []}
        mrng = np.random.default_rng(50 + seed)
        for pi in mrng.permutation(len(X))[:256]:
            x1 = X[pi].astype(float)
            try:
                y1, _, _ = oracle_at(x1)
            except ValueError:
                continue
            for item in candidates_for_scene(x1, mrng, n_random=8):
                e = np.asarray(item[0], dtype=float)
                try:
                    o1, _, _ = oracle_at(x1 + e)
                except ValueError:
                    continue
                found = False
                for pj in mrng.permutation(len(X))[:40]:
                    x2 = X[pj].astype(float)
                    try:
                        y2, _, _ = oracle_at(x2)
                        o2, _, _ = oracle_at(x2 + e)
                    except ValueError:
                        continue
                    if y2 == y1 and o2 != o1:
                        for arm in ("A0", "A1", "A2"):
                            enc, mu, sd = encs[arm]
                            H = h_of(enc, mu, sd, np.stack([x1, x2]))
                            dists[arm].append(float(np.linalg.norm(H[0] - H[1])))
                        found = True
                        break
                if found:
                    break
        res[f"A6_s{seed}"] = {arm: {"n": len(dists[arm]), "median": round(float(np.median(dists[arm])), 4) if dists[arm] else None} for arm in ("A0", "A1", "A2")}
        print(f"SAW A6_s{seed}: {res[f'A6_s{seed}']}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "RESULTS.json", "w"), indent=1)
    print("DONE WP-A behavioral eval (A6 v2 separate)")


if __name__ == "__main__":
    main()
