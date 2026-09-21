#!/usr/bin/env python3
"""WP1.14 aimed pools: mine single-crossing paths, bin by pre-registered I.

Mix of (a) aimed axis sweeps (closest-point axis, N03-style, expect high I)
and (b) random-direction edits (N01-style, expect low I). Keep oracle exactly
one crossing + gradient audit pass. Same row schema as wp1_paths.
Discovery pool first; confirm pool only after pools/matching check out.
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
sys.path.insert(0, str(ROOT / "experiments" / "update_geometry"))
sys.path.insert(0, str(ROOT / "docs" / "iclr2027_discovery_campaign_20260917"))
torch.set_num_threads(4)
from paths import (  # noqa: E402
    oracle_at, scan_linear, locate_oracle_turns, locate_model_turns,
    match_turns, _seg_closest)
from common import load_model, preprocess  # noqa: E402
from wp1_paths import orient_vals, audit_normal  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"


def aimed_edits(x, rng):
    out = []
    q_ab, q_cd = _seg_closest(x[0], x[1], x[2], x[3])
    gap = float(np.linalg.norm(q_ab - q_cd))
    if gap < 1e-6:
        return out
    axis = (q_ab - q_cd) / gap
    for nodes, tgt, nm in (((2, 3), 1.0, "aimCD"), ((0, 1), -1.0, "aimAB")):
        for over in (0.05, 0.10, 0.18, 0.25, 0.45):
            e = np.zeros((4, 2))
            e[list(nodes)] = (gap + over) * axis * tgt
            out.append((e, f"{nm}_o{over}"))
    perp = np.array([-axis[1], axis[0]])
    for nodes, nm in (((2, 3), "grazeCD"), ((0, 1), "grazeAB")):
        for ang in (0.10, 0.20, 0.35, 0.60):
            d = np.cos(ang) * perp + np.sin(ang) * axis
            for L in (0.20, 0.35, 0.50):
                e = np.zeros((4, 2))
                e[list(nodes)] = L * d
                out.append((e, f"{nm}_a{ang}_L{L}"))
    return out


def random_edits(x, rng, n=24):
    out = []
    for k in range(n):
        kind = rng.integers(0, 3)
        e = np.zeros((4, 2))
        if kind == 0:
            node = rng.integers(0, 4)
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0.08, 0.30)
            e[node] = r * np.array([np.cos(ang), np.sin(ang)])
            out.append((e, f"rnd_node{node}"))
        elif kind == 1:
            nodes = (0, 1) if rng.random() < 0.5 else (2, 3)
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0.08, 0.30)
            e[list(nodes)] = r * np.array([np.cos(ang), np.sin(ang)])
            out.append((e, "rnd_shift"))
        else:
            nodes = (0, 1) if rng.random() < 0.5 else (2, 3)
            cen = x[list(nodes)].mean(axis=0)
            ang = rng.uniform(-0.4, 0.4)
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            pts = x[list(nodes)]
            e[list(nodes)] = pts @ R.T - pts
            out.append((e, "rnd_rot"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pool", default="eval_202")
    p.add_argument("--model", type=Path, default=ART / "r04b_s11" / "clean")
    p.add_argument("--n_parents", type=int, default=512)
    p.add_argument("--seed", type=int, default=21)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    d = np.load(ART / "scenes" / a.pool / "scenes.npz")
    X = d["positions"].astype(np.float32)
    model, stats = load_model(a.model)
    model.eval()

    def predict_fn(xs):
        xs = np.asarray(xs, dtype=np.float32)
        with torch.no_grad():
            out = []
            for s in range(0, len(xs), 256):
                out.append(model(preprocess(xs[s:s + 256], stats)).numpy())
        lg = np.concatenate(out)
        return lg, (lg > 0).astype(int)

    rows = []
    order = rng.permutation(len(X))[:a.n_parents]
    for pi in order:
        x = X[pi].astype(float)
        try:
            oracle_at(x)
        except ValueError:
            continue
        for e, fam in aimed_edits(x, rng) + random_edits(x, rng):
            try:
                scan = scan_linear(x, e, predict_fn, n_scan=51)
            except Exception:
                continue
            ot = locate_oracle_turns(x, e, scan)
            if len(ot) != 1:
                continue
            mt = locate_model_turns(x, e, scan, predict_fn)
            matched = match_turns(ot, mt)
            sl = int(scan["oracle_label"][0])
            el = int(scan["oracle_label"][-1])
            lg0 = float(scan["logit"][0])
            t_star = ot[0]["t_star"]
            x_star = x + t_star * e
            s0 = np.sign(orient_vals(x))
            s1 = np.sign(orient_vals(x + e))
            changed = np.nonzero(s0 != s1)[0]
            if len(changed) != 1:
                continue
            I, rel = audit_normal(x_star, int(changed[0]), e)
            if I is None or rel > 1e-3:
                continue
            rows.append({"parent_id": int(pi), "path_type": "aimed",
                         "fam_A": fam, "fam_B": "single",
                         "start_label": sl, "endpoint_label": el,
                         "oracle_n_crossings": 1, "oracle_crossing_t": [t_star],
                         "model_n_crossings": len(mt),
                         "model_crossing_ts": [m["t_theta"] for m in mt],
                         "match": matched,
                         "start_correct": bool((lg0 > 0) == sl),
                         "endpoint_correct": bool((float(scan["logit"][-1]) > 0) == el),
                         "start_logit": lg0,
                         "endpoint_logit": float(scan["logit"][-1]),
                         "edit_norm": float(np.linalg.norm(e)),
                         "start_margin": float(scan["oracle_margin"][0]),
                         "endpoint_margin": float(scan["oracle_margin"][-1]),
                         "incidence": I, "incidence_audit_rel": rel,
                         "flip_orient": int(changed[0])})
    I = np.array([r["incidence"] for r in rows])
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows}, open(a.out / "rows.json", "w"))
    summ = {"n_paths": len(rows),
            "n_tan": int((I <= 0.25).sum()), "n_nor": int((I >= 0.75).sum())}
    json.dump(summ, open(a.out / "summary.json", "w"), indent=1)
    print(f"SAW WP1mine: {summ}")


if __name__ == "__main__":
    main()
