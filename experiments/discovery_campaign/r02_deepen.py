#!/usr/bin/env python3
"""R1: R02 coordinate deepening — targeted family/intensity tuning, same honesty.

Configs (same oracle + same model, honest denominators throughout):
  A baseline : standard grids (radii .03/.06/.10/.16), pairs AB/CD, n_random=64
  B pair-heavy: +cross pairs AC/AD/BC/BD, radii +0.24, n_random=64
  C random-heavy: standard grids, n_random=256
Each cfg reports FULL-set metrics plus BUDGET-MATCHED subset metrics
(random 298 candidates/scene, seed-fixed) so win-rate tuning can't hide
behind bigger budgets. Pick winner by miss|flip, confirm on fresh scenes.
"""
from __future__ import annotations
import argparse
import csv
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

BASE_RADII = [0.03, 0.06, 0.10, 0.16]
EXT_RADII = [0.03, 0.06, 0.10, 0.16, 0.24]
BASE_PAIRS = ((0, 1), (2, 3))
XTRA_PAIRS = ((0, 2), (0, 3), (1, 2), (1, 3))
ANGLES = np.linspace(0, 2 * np.pi, 8, endpoint=False)
DIRS = np.stack([np.cos(ANGLES), np.sin(ANGLES)], axis=1)
BUDGET = 298  # baseline candidates/scene


def build_candidates(x, rng, cfg):
    cands = []
    radii = EXT_RADII if cfg == "B" else BASE_RADII
    pairs = BASE_PAIRS + XTRA_PAIRS if cfg == "B" else BASE_PAIRS
    n_rand = 256 if cfg == "C" else 64
    for node in range(4):
        for r in radii:
            for d in DIRS:
                e = np.zeros((4, 2)); e[node] = r * d
                cands.append((e, "single"))
    for nodes in pairs:
        for r in radii:
            for d in DIRS:
                e = np.zeros((4, 2)); e[list(nodes)] = r * d / np.sqrt(2)
                cands.append((e, "pair"))
    cen = x.mean(axis=0)
    for r in radii:
        for d in DIRS:
            e = np.zeros((4, 2)) + r * d
            cands.append((e, "global_trans"))
    for ang in [0.05, 0.1, 0.2, -0.05, -0.1, -0.2]:
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        e = (x - cen) @ R.T - (x - cen)
        cands.append((e, "global_rot"))
    for s in [0.95, 1.05, 0.9, 1.1]:
        cands.append(((s - 1) * (x - cen), "global_scale"))
    for _ in range(n_rand):
        e = rng.normal(size=(4, 2))
        e *= rng.choice(radii) * 2 / np.linalg.norm(e)
        cands.append((e, "random"))
    return cands


def eval_pool(model, stats, X, y, cand_lists, min_margin):
    """cand_lists: list per scene of (edit, fam). Returns summary + rows."""
    results, fam_f, fam_t = [], {}, {}
    for i in range(len(X)):
        flips, preserves = [], []
        for e, fam in cand_lists[i]:
            fam_t[fam] = fam_t.get(fam, 0) + 1
            try:
                r = segment_relation(X[i] + e)
            except ValueError:
                continue
            if r["ambiguous"] or r["margin"] < min_margin:
                continue
            with torch.no_grad():
                zin = preprocess((X[i] + e)[None], stats)
                z = model(zin, return_feat=True)[1].numpy()[0]
                pred = int((torch.sigmoid(model(zin))).numpy()[0] > .5)
            rec = {"fam": fam, "input_norm": float(np.linalg.norm(e)),
                   "feat_dist": float(np.linalg.norm(z - Z0[i])),
                   "oracle": r["label"], "pred": pred, "margin": r["margin"]}
            if r["label"] != int(y[i]):
                flips.append(rec); fam_f[fam] = fam_f.get(fam, 0) + 1
            else:
                preserves.append(rec)
        if not flips:
            results.append({"scene": i, "orig": int(y[i]), "has_flip": False})
            continue
        stealth = min(flips, key=lambda r: r["feat_dist"])
        null_pool = [r for r in preserves if r["input_norm"] >= stealth["input_norm"]]
        null_fd = min([r["feat_dist"] for r in null_pool]) if null_pool else float("nan")
        results.append({"scene": i, "orig": int(y[i]), "has_flip": True,
                        "n_flips": len(flips), "stealth": stealth,
                        "null_feat_dist": null_fd,
                        "model_misses_stealth": bool(stealth["pred"] == int(y[i]))})
    n = len(results)
    has = [r for r in results if r["has_flip"]]
    miss = [r for r in has if r["model_misses_stealth"]]
    return {
        "n_scenes": n, "n_with_flip": len(has),
        "frac_with_flip": len(has) / n,
        "n_model_miss_given_flip": len(miss),
        "miss_rate_given_flip": len(miss) / len(has) if has else float("nan"),
        "unconditional_miss_rate": len(miss) / n,
        "stealth_feat_median": float(np.median([r["stealth"]["feat_dist"] for r in has])) if has else None,
        "stealth_input_median": float(np.median([r["stealth"]["input_norm"] for r in has])) if has else None,
        "stealth_margin_median": float(np.median([r["stealth"]["margin"] for r in has])) if has else None,
        "flip_rate_by_family": {f: fam_f.get(f, 0) / c for f, c in fam_t.items()},
    }, results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--scenes", type=Path, required=True)
    p.add_argument("--slice", type=str, default="")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cfgs", nargs="+", default=["A", "B", "C"])
    p.add_argument("--min-margin", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--dump-csv", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)
    global Z0
    d = np.load(a.scenes / "scenes.npz")
    X, y = d["positions"].astype(float), d["labels"].astype(int)
    if a.slice:
        lo, hi = [int(v) for v in a.slice.split(":")]
        X, y = X[lo:hi], y[lo:hi]
    model, stats = load_model(a.model)
    model.eval()
    with torch.no_grad():
        Z0 = model(preprocess(X, stats), return_feat=True)[1].numpy()
    out = {"model": str(a.model), "scenes": str(a.scenes), "slice": a.slice,
           "min_margin": a.min_margin, "seed": a.seed, "budget_cap": BUDGET,
           "configs": {}}
    for cfg in a.cfgs:
        rng = np.random.default_rng(a.seed)
        cands = [build_candidates(X[i], rng, cfg) for i in range(len(X))]
        full, rows = eval_pool(model, stats, X, y, cands, a.min_margin)
        full["n_candidates_per_scene"] = len(cands[0])
        # budget-matched subset (seed-fixed per scene)
        sub = []
        for i in range(len(X)):
            r2 = np.random.default_rng(1000 + i)
            k = min(BUDGET, len(cands[i]))
            sub.append([cands[i][j] for j in r2.choice(len(cands[i]), k, replace=False)])
        matched, _ = eval_pool(model, stats, X, y, sub, a.min_margin)
        out["configs"][cfg] = {"full": full, "budget_matched": matched}
        print(f"cfg {cfg}: n_cand={len(cands[0])} full miss|flip={full['miss_rate_given_flip']:.3f} "
              f"uncond={full['unconditional_miss_rate']:.3f} | matched miss|flip={matched['miss_rate_given_flip']:.3f} "
              f"uncond={matched['unconditional_miss_rate']:.3f}", flush=True)
        if a.dump_csv and cfg == a.cfgs[0]:
            pass
    (a.output / "R02_deepen.json").write_text(json.dumps(out, indent=2) + "\n")
    print("saved", a.output, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
