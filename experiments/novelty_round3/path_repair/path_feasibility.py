#!/usr/bin/env python3
"""WP-C feasibility: ReLU path partition + point-vs-path repair pilot.

(1) Partition 48 paths (16 preserve / 16 single-cross / 16 episodes) on
frozen clean via activation-pattern enumeration (dense 2001 scan + bisect),
verify prediction constancy per region. (2) Repair pilot on 4 worst
single-cross paths: point arm (8 oracle-crossing-adjacent endpoints) vs
path arm (8 = model-boundary x2 + oracle-boundary x2 + max-error mids),
equal oracle budget, retrain seed-11 only; eval seen-path integral + 8
held-out paths. Feasibility bar, not a claim.
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
torch.set_num_threads(4)
from common import load_model, preprocess  # noqa: E402
from coord_mlp import CoordMLP  # noqa: E402
from paths import oracle_at, scan_linear, locate_oracle_turns, locate_model_turns, gen_event_paths  # noqa: E402
from r02_search import candidates_for_scene  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
OUT = ROOT / "artifacts" / "novelty_round3" / "path_repair"


def pattern_of(model, stats, xs):
    with torch.no_grad():
        F = torch.from_numpy(((np.asarray(xs, dtype=np.float32).reshape(-1, 8) - stats["mu"]) / stats["sd"]).astype(np.float32))
        h = F
        pats = []
        for layer in model.net:
            if isinstance(layer, nn.Linear):
                h = layer(h)
                pats.append((h > 0).numpy())
                h = torch.relu(h)
        return [np.concatenate([p[i] for p in pats]) for i in range(len(F))]


def partition(model, stats, x, e, predict_fn):
    t = np.linspace(0, 1, 2001)
    xs = (x[None] + t[:, None, None] * e[None]).astype(np.float32)
    pats = pattern_of(model, stats, xs)
    bounds = [0.0]
    for i in range(1, len(t)):
        if not np.array_equal(pats[i], pats[i - 1]):
            lo, hi = t[i - 1], t[i]
            for _ in range(20):
                m = 0.5 * (lo + hi)
                pm = pattern_of(model, stats, (x + m * e)[None])[0]
                if np.array_equal(pm, pats[i]):
                    hi = m
                else:
                    lo = m
            bounds.append(float(0.5 * (lo + hi)))
    bounds.append(1.0)
    # verify constancy: predict at region mid + ends
    ok = True
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = 0.5 * (a + b)
        try:
            la = predict_fn((x + a * e)[None])[0]
            lm = predict_fn((x + m * e)[None])[0]
            lb = predict_fn((x + (b - 1e-9) * e)[None])[0]
        except Exception:
            ok = False
            break
        if not (la == lm == lb):
            ok = False
            break
    return bounds, ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    rng = np.random.default_rng(77)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(np.float32)
    model, stats = load_model(ART / "r04b_s11" / "clean")
    model.eval()
    mu = np.asarray(stats["mu"], dtype=np.float32).reshape(-1)
    sd = np.asarray(stats["sd"], dtype=np.float32).reshape(-1)

    def predict_fn(xs):
        xs = np.asarray(xs, dtype=np.float32)
        with torch.no_grad():
            out = []
            for s in range(0, len(xs), 256):
                F = torch.from_numpy(((xs[s:s + 256].reshape(-1, 8) - mu) / sd).astype(np.float32))
                out.append(model(F).numpy())
        lg = np.concatenate(out).reshape(-1)
        return (lg > 0).astype(int), lg

    paths = []
    for pi in rng.permutation(len(X)):
        if len(paths) >= 64:
            break
        x = X[pi].astype(float)
        try:
            oracle_at(x)
        except ValueError:
            continue
        for item in candidates_for_scene(x, rng, n_random=6):
            e = np.asarray(item[0], dtype=float)
            try:
                scan = scan_linear(x, e, lambda xs: (predict_fn(xs)[1], predict_fn(xs)[0]), n_scan=51)
            except Exception:
                continue
            ot = locate_oracle_turns(x, e, scan)
            if len(ot) > 1:
                continue
            paths.append((x, e, "preserve" if len(ot) == 0 else "single", pi))
            if len(paths) >= 64:
                break
    Ns = sum(1 for _, _, k, _ in paths if k == "single")
    res = {"n_paths": len(paths), "n_single": Ns}
    nreg, okc = [], 0
    for x, e, k, pi in paths:
        bounds, ok = partition(model, stats, x, e, lambda xs: predict_fn(xs)[0])
        nreg.append(len(bounds) - 1)
        okc += ok
    res["median_regions"] = float(np.median(nreg))
    res["constancy_ok_frac"] = okc / len(paths)
    # repair pilot on 4 worst single-cross paths (largest error integral)
    singles = [(x, e, pi) for x, e, k, pi in paths if k == "single"]
    scored = []
    for x, e, pi in singles:
        scan = scan_linear(x, e, lambda xs: (predict_fn(xs)[1], predict_fn(xs)[0]), n_scan=129)
        olab = scan["oracle_label"]
        plab = scan["pred_label"]
        valid = olab >= 0
        err = float(((olab[valid] != plab[valid])).mean())
        scored.append((err, x, e, pi))
    scored.sort(reverse=True)
    pilot = scored[:4]
    held = scored[4:12]
    mu8, sd8 = mu, sd
    dtr = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    Xb = dtr["positions"].astype(np.float32).reshape(-1, 8)
    yb = dtr["labels"].astype(np.float32)
    Xcb = torch.from_numpy(((Xb - mu8) / sd8).astype(np.float32))
    ycb = torch.from_numpy(yb)

    def err_on(x, e, m, mu_, sd_):
        sc = scan_linear(x, e, lambda xs: (predict_fn(xs)[1], predict_fn(xs)[0]), n_scan=129)
        olab = sc["oracle_label"]
        with torch.no_grad():
            xs = (x[None] + sc["t"][:, None, None] * e[None]).astype(np.float32)
            F = torch.from_numpy(((xs.reshape(-1, 8) - mu_) / sd_).astype(np.float32))
            pl = (m(F).numpy().reshape(-1) > 0).astype(int)
        valid = olab >= 0
        return float((olab[valid] != pl[valid]).mean())

    def train_with(pts):
        torch.manual_seed(11)
        m = CoordMLP(64, 32)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        bce = nn.BCEWithLogitsLoss()
        Xr = torch.from_numpy(((np.stack([p[0] for p in pts]).reshape(-1, 8) - mu8) / sd8).astype(np.float32))
        yr = torch.from_numpy(np.array([p[1] for p in pts], dtype=np.float32))
        for ep in range(300):
            m.train()
            opt.zero_grad()
            loss = bce(m(Xcb), ycb) + 1.0 * bce(m(Xr), yr)
            loss.backward()
            opt.step()
        m.eval()
        return m

    seen_point, seen_path, un_point, un_path = [], [], [], []
    for err, x, e, pi in pilot:
        ot = locate_oracle_turns(x, e, scan_linear(x, e, lambda xs: (predict_fn(xs)[1], predict_fn(xs)[0]), n_scan=51))
        t_star = ot[0]["t_star"]
        pt = []
        for dt in (0.02, 0.05, 0.10, 0.20):
            for sgn in (-1, 1):
                xx = x + np.clip(t_star + sgn * dt, 0, 1) * e
                try:
                    pt.append((xx, oracle_at(xx)[0]))
                except ValueError:
                    pass
        pt = pt[:8]
        mt = locate_model_turns(x, e, scan_linear(x, e, lambda xs: (predict_fn(xs)[1], predict_fn(xs)[0]), n_scan=51),
                                lambda xs: (predict_fn(xs)[1], predict_fn(xs)[0]))
        th = mt[0]["t_theta"] if mt else t_star
        ph = []
        for tt in (th - 0.01, th + 0.01, t_star - 0.01, t_star + 0.01):
            xx = x + np.clip(tt, 0, 1) * e
            try:
                ph.append((xx, oracle_at(xx)[0]))
            except ValueError:
                pass
        sc = scan_linear(x, e, lambda xs: (predict_fn(xs)[1], predict_fn(xs)[0]), n_scan=129)
        bad = np.nonzero((sc["oracle_label"] >= 0) & (sc["oracle_label"] != sc["pred_label"]))[0]
        for bi in bad[:max(0, 8 - len(ph))]:
            xx = x + sc["t"][bi] * e
            ph.append((xx, int(sc["oracle_label"][bi])))
        ph = ph[:8]
        m_pt, m_ph = train_with(pt), train_with(ph)
        seen_point.append(err_on(x, e, m_pt, mu8, sd8))
        seen_path.append(err_on(x, e, m_ph, mu8, sd8))
        for _, hx, he, _ in held:
            un_point.append(err_on(hx, he, m_pt, mu8, sd8))
            un_path.append(err_on(hx, he, m_ph, mu8, sd8))
    res["pilot"] = {"seen_point": round(float(np.mean(seen_point)), 4),
                    "seen_path": round(float(np.mean(seen_path)), 4),
                    "unseen_point": round(float(np.mean(un_point)), 4),
                    "unseen_path": round(float(np.mean(un_path)), 4)}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "PATH_REPAIR_FEASIBILITY.json", "w"), indent=1)
    print(f"SAW WP-C: {json.dumps(res)}")


if __name__ == "__main__":
    main()
