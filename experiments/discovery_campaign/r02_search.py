#!/usr/bin/env python3
"""R02 round 1: structured candidate search for relational metamers.

Per scene, structured edit families (single-endpoint, coherent-pair,
global, random-norm-matched) are enumerated. Oracle (segment_relation)
decides label flips with margin floor m (excludes boundary-epsilon cases).
For each scene with a valid flip: stealthiest flip = min feature-distance
flip; null = min feature-distance among label-preserving edits of
input-norm >= stealthiest flip's norm. Direct criterion: model prediction
follows oracle or not. Representation distance is secondary.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

PACK = Path(__file__).resolve().parents[2] / "docs" / "iclr2027_discovery_campaign_20260917"
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.relations import segment_relation
from common import load_model, preprocess


def candidates_for_scene(x: np.ndarray, rng: np.random.Generator, n_random: int = 64):
    """Yield (edit_matrix [4,2], family_name). Deterministic grids + random."""
    cands = []
    radii = [0.03, 0.06, 0.10, 0.16]
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    # F1: single endpoint
    for node in range(4):
        for r in radii:
            for d in dirs:
                e = np.zeros((4, 2)); e[node] = r * d
                cands.append((e, "single"))
    # F2: coherent pairs AB / CD
    for nodes in ((0, 1), (2, 3)):
        for r in radii:
            for d in dirs:
                e = np.zeros((4, 2)); e[list(nodes)] = r * d / np.sqrt(2)
                cands.append((e, "pair"))
    # F3: global (translation / rotation / scale about centroid)
    cen = x.mean(axis=0)
    for r in radii:
        for d in dirs:
            e = np.zeros((4, 2)) + r * d  # translation
            cands.append((e, "global_trans"))
    for ang in [0.05, 0.1, 0.2, -0.05, -0.1, -0.2]:
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        e = (x - cen) @ R.T - (x - cen)
        cands.append((e, "global_rot"))
    for s in [0.95, 1.05, 0.9, 1.1]:
        cands.append(((s - 1) * (x - cen), "global_scale"))
    # F4: random dense edits, norms matched to grid range
    for _ in range(n_random):
        e = rng.normal(size=(4, 2))
        e *= rng.choice(radii) * 2 / np.linalg.norm(e)
        cands.append((e, "random"))
    return cands


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--scenes", type=Path, required=True)
    p.add_argument("--slice", type=str, default="")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-margin", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=21)
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)
    d = np.load(a.scenes / "scenes.npz")
    X, y = d["positions"].astype(float), d["labels"].astype(int)
    if a.slice:
        lo, hi = [int(v) for v in a.slice.split(":")]
        X, y = X[lo:hi], y[lo:hi]
    model, stats = load_model(a.model)
    rng = np.random.default_rng(a.seed)
    model.eval()
    with torch.no_grad():
        Z0 = model(preprocess(X, stats), return_feat=True)[1].numpy()
        P0 = (torch.sigmoid(model(preprocess(X, stats))).numpy() > .5).astype(int)

    results, fam_flip_counts, fam_totals = [], {}, {}
    for i in range(len(X)):
        cands = candidates_for_scene(X[i], rng)
        flips, preserves = [], []
        for e, fam in cands:
            fam_totals[fam] = fam_totals.get(fam, 0) + 1
            xp = X[i] + e
            try:
                r = segment_relation(xp)
            except ValueError:
                continue
            if r["ambiguous"] or r["margin"] < a.min_margin:
                continue
            with torch.no_grad():
                zin = preprocess(xp[None], stats)
                z = model(zin, return_feat=True)[1].numpy()[0]
                pred = int((torch.sigmoid(model(zin)).numpy()[0] > .5))
            nd = float(np.linalg.norm(e))
            fd = float(np.linalg.norm(z - Z0[i]))
            rec = {"fam": fam, "input_norm": nd, "feat_dist": fd,
                   "oracle": r["label"], "pred": pred, "margin": r["margin"]}
            if r["label"] != int(y[i]):
                flips.append(rec); fam_flip_counts[fam] = fam_flip_counts.get(fam, 0) + 1
            else:
                preserves.append(rec)
        if not flips:
            results.append({"scene": i, "orig": int(y[i]), "has_flip": False})
            continue
        stealth = min(flips, key=lambda r: r["feat_dist"])
        null_pool = [r for r in preserves if r["input_norm"] >= stealth["input_norm"]]
        null_fd = min([r["feat_dist"] for r in null_pool]) if null_pool else float("nan")
        rand_pool = [r for r in preserves if r["fam"] == "random"
                     and r["input_norm"] >= stealth["input_norm"]]
        rand_fd = min([r["feat_dist"] for r in rand_pool]) if rand_pool else float("nan")
        results.append({"scene": i, "orig": int(y[i]), "has_flip": True,
                        "n_flips": len(flips),
                        "stealth": stealth,
                        "null_feat_dist": null_fd, "random_feat_dist": rand_fd,
                        "model_misses_stealth": bool(stealth["pred"] == int(y[i])),
                        "stealthier_than_null": bool(stealth["feat_dist"] < null_fd)
                        if null_pool else None})
    n = len(results)
    has = [r for r in results if r["has_flip"]]
    miss = [r for r in has if r["model_misses_stealth"]]
    summary = {
        "n_scenes": n, "n_with_flip": len(has),
        "frac_with_flip": len(has) / n,
        "n_model_miss_given_flip": len(miss),
        "miss_rate_given_flip": len(miss) / len(has) if has else float("nan"),
        "unconditional_miss_rate": len(miss) / n,
        "stealth_feat_median": float(np.median([r["stealth"]["feat_dist"] for r in has])) if has else None,
        "null_feat_median": float(np.nanmedian([r["null_feat_dist"] for r in has])) if has else None,
        "stealth_input_median": float(np.median([r["stealth"]["input_norm"] for r in has])) if has else None,
        "miss_stealthier_than_null_frac": float(np.nanmean(
            [1.0 if r["stealthier_than_null"] else 0.0 for r in has if r["stealthier_than_null"] is not None])) if has else None,
        "flip_rate_by_family": {f: fam_flip_counts.get(f, 0) / c for f, c in fam_totals.items()},
        "model": str(a.model), "scenes": str(a.scenes), "slice": a.slice,
        "min_margin": a.min_margin}
    (a.output / "R02_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Sample-level detail (compact): one row per scene with flip.
    import csv
    with open(a.output / "metamers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "orig", "fam", "input_norm", "feat_dist",
                                          "null_feat_dist", "oracle", "pred", "margin", "model_miss"])
        w.writeheader()
        for r in has:
            s = r["stealth"]
            w.writerow({"scene": r["scene"], "orig": r["orig"], "fam": s["fam"],
                        "input_norm": round(s["input_norm"], 4), "feat_dist": round(s["feat_dist"], 4),
                        "null_feat_dist": round(r["null_feat_dist"], 4), "oracle": s["oracle"],
                        "pred": s["pred"], "margin": round(s["margin"], 4),
                        "model_miss": r["model_misses_stealth"]})
    np.savez_compressed(a.output / "exemplars.npz",
                        X=X[np.array([r["scene"] for r in has])] if has else np.zeros((0, 4, 2)),
                        y=y[np.array([r["scene"] for r in has])] if has else np.zeros(0))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
