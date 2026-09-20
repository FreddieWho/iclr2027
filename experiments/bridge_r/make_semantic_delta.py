#!/usr/bin/env python3
"""Bridge-R semantic-delta M1-M2: matched flip/no-flip single-edit dataset.

Per-split (parent-disjoint by construction: distinct generator seeds).
New seeds 26092001/02/03. Base scenes from make_relational_scenes;
atomic_edits mined for flip/no-flip endpoints (margin>=0.005); canonical
v1a renders (fresh RNG per state, shared pair seed -> locked backgrounds)
for pixel_change; exact base_label strata + standardized-Euclidean NN
matching on [edit_norm, pixel_change, base_margin, endpoint_margin].

Writes artifacts/bridge_r/semantic_delta/{split}.npz + balance JSON.
Holdout never touched.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "docs" / "iclr2027_discovery_campaign_20260917"))
sys.path.insert(0, str(ROOT / "experiments" / "bridge_r"))
from paths import atomic_edits, oracle_at
from core.relations import make_relational_scenes
from render_v1 import render_canonical

ART = ROOT / "artifacts" / "bridge_r" / "semantic_delta"
CFG = {
    "train": {"seed": 26092001, "parents": 15000, "pairs": 2000, "nuis": 400000},
    "dev": {"seed": 26092002, "parents": 4000, "pairs": 500, "nuis": 500000},
    "confirm": {"seed": 26092003, "parents": 4000, "pairs": 500, "nuis": 600000},
}
MARGIN_MIN = 0.005


def pix_change(x0, x1, seed):
    i0 = render_canonical(x0, np.random.default_rng(seed))
    i1 = render_canonical(x1, np.random.default_rng(seed))
    return float(np.abs(i1 - i0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev", "confirm"], required=True)
    a = ap.parse_args()
    cfg = CFG[a.split]
    rng = np.random.default_rng(cfg["seed"])
    X, y, m = make_relational_scenes(cfg["parents"], cfg["seed"], 0.02)
    ids = [f"semdelta_{a.split}_{cfg['seed']}_{i:06d}" for i in range(len(X))]
    flips, keeps = [], []
    for pi in range(len(X)):
        x = X[pi].astype(float)
        y0, m0, _ = oracle_at(x)
        erng = np.random.default_rng(cfg["seed"] + 7919 + pi)
        for e, fam in atomic_edits(x, erng):
            e = np.asarray(e, float)
            try:
                y1, m1, _ = oracle_at(x + e)
            except ValueError:
                continue
            if m1 < MARGIN_MIN:
                continue
            rec = {"x": x, "e": e, "y0": int(y0), "y1": int(y1),
                   "m0": float(m0), "m1": float(m1),
                   "norm": float(np.linalg.norm(e)), "fam": fam,
                   "parent": ids[pi]}
            (flips if y1 != y0 else keeps).append(rec)
    print(f"[{a.split}] candidates: flip={len(flips)} noflip={len(keeps)}", flush=True)
    # pixel change for survivors only
    for k, rec in enumerate(flips + keeps):
        rec["dpix"] = pix_change(rec["x"], rec["x"] + rec["e"], cfg["nuis"] + k)
    # match within base_label strata
    feats = ["norm", "dpix", "m0", "m1"]
    pairs = []
    for y0 in (0, 1):
        F = [r for r in flips if r["y0"] == y0]
        K = [r for r in keeps if r["y0"] == y0]
        if not F or not K:
            continue
        A = np.array([[r[f] for f in feats] for r in F + K])
        mu, sd = A.mean(0), A.std(0) + 1e-12
        Zf = (np.array([[r[f] for f in feats] for r in F]) - mu) / sd
        Zk = (np.array([[r[f] for f in feats] for r in K]) - mu) / sd
        used = np.zeros(len(K), bool)
        nf = min(len(F), len(K))
        for i in range(len(F)):
            d = ((Zk - Zf[i]) ** 2).sum(1)
            d[used] = np.inf
            j = int(np.argmin(d))
            if np.isinf(d[j]):
                break
            used[j] = True
            pairs.append((F[i], K[j]))
    print(f"[{a.split}] matched pairs: {len(pairs)} (target {cfg['pairs']})", flush=True)
    if len(pairs) < cfg["pairs"]:
        raise SystemExit(f"FATAL {a.split}: only {len(pairs)} matched pairs")
    # balance the two classes to target with best-match priority? keep first N
    pairs = pairs[:cfg["pairs"]]
    rows = []
    for f, k in pairs:
        rows.append((f, 1))
        rows.append((k, 0))
    order = np.random.default_rng(cfg["seed"] + 555).permutation(len(rows))
    rows = [rows[i] for i in order]
    ART.mkdir(parents=True, exist_ok=True)
    pack = {
        "x": np.array([r["x"] for r, _ in rows]),
        "e": np.array([r["e"] for r, _ in rows]),
        "flip": np.array([f for _, f in rows], dtype=np.int64),
        "y0": np.array([r["y0"] for r, _ in rows]),
        "y1": np.array([r["y1"] for r, _ in rows]),
        "m0": np.array([r["m0"] for r, _ in rows]),
        "m1": np.array([r["m1"] for r, _ in rows]),
        "edit_norm": np.array([r["norm"] for r, _ in rows]),
        "dpix": np.array([r["dpix"] for r, _ in rows]),
        "parent": np.array([r["parent"] for r, _ in rows]),
        "fam": np.array([r["fam"] for r, _ in rows]),
        "nuis": cfg["nuis"] + np.arange(len(rows)),
    }
    np.savez(ART / f"{a.split}.npz", **pack)
    # balance table
    Y = pack["flip"]
    bal = {"n": len(Y), "n_flip": int(Y.sum()), "n_noflip": int((1 - Y).sum())}
    for v in ["edit_norm", "dpix", "m0", "m1"]:
        a1, a0 = pack[v][Y == 1], pack[v][Y == 0]
        sd = np.sqrt((a1.var() + a0.var()) / 2) + 1e-12
        bal[v] = {"flip_mean": float(a1.mean()), "noflip_mean": float(a0.mean()),
                  "flip_sd": float(a1.std()), "noflip_sd": float(a0.std()),
                  "flip_median": float(np.median(a1)), "noflip_median": float(np.median(a0)),
                  "flip_iqr": [float(np.percentile(a1, 25)), float(np.percentile(a1, 75))],
                  "noflip_iqr": [float(np.percentile(a0, 25)), float(np.percentile(a0, 75))],
                  "SMD": float((a1.mean() - a0.mean()) / sd)}
    for y0 in (0, 1):
        bal[f"base_label_{y0}"] = {"flip": int(((Y == 1) & (pack['y0'] == y0)).sum()),
                                   "noflip": int(((Y == 0) & (pack['y0'] == y0)).sum())}
    famc = {}
    for f in sorted(set(pack["fam"].tolist())):
        famc[str(f)] = {"flip": int(((Y == 1) & (pack["fam"] == f)).sum()),
                        "noflip": int(((Y == 0) & (pack["fam"] == f)).sum())}
    bal["fam"] = famc
    json.dump(bal, open(ART / f"{a.split}_balance.json", "w"), indent=1)
    smds = {v: round(bal[v]["SMD"], 4) for v in ["edit_norm", "dpix", "m0", "m1"]}
    print(f"[{a.split}] SMDs: {smds} base_labels: "
          f"{bal['base_label_0']} {bal['base_label_1']}", flush=True)
    print(f"DONE {a.split}: {len(Y)} rows", flush=True)


if __name__ == "__main__":
    main()
