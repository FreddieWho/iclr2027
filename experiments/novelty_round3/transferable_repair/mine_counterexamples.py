#!/usr/bin/env python3
"""WP-B mining: source-model counterexamples (missing/false update), K=256 frozen.

Scans train_101 parents x candidates_for_scene; oracle labels every candidate
(query count recorded). Dedup: per-parent cap 4, family strata, preserve/flip
1:1, pairwise edit-distance >= 0.05. Writes per-source repair set npz.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(4)
from paths import oracle_at  # noqa: E402
from common import load_model, preprocess  # noqa: E402
from r02_search import candidates_for_scene  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
OUT = ROOT / "artifacts" / "novelty_round3" / "transferable_repair"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--K", type=int, default=256)
    p.add_argument("--seed", type=int, default=301)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(np.float32)
    model, stats = load_model(ART / a.source)
    model.eval()

    def pred(xs):
        xs = np.asarray(xs, dtype=np.float32)
        with torch.no_grad():
            out = []
            for s in range(0, len(xs), 256):
                out.append(model(preprocess(xs[s:s + 256], stats)).numpy())
        lg = np.concatenate(out)
        return lg, (lg > 0).astype(int)

    miss, false = [], []
    n_queries = 0
    per_parent = {}
    for pi in rng.permutation(len(X)):
        if len(miss) >= a.K // 2 and len(false) >= a.K // 2:
            break
        x = X[pi].astype(float)
        try:
            y0, _, _ = oracle_at(x)
        except ValueError:
            continue
        p0 = pred(x[None])[1][0]
        for item in candidates_for_scene(x, rng, n_random=16):
            e = np.asarray(item[0], dtype=float)
            fam = item[1]
            try:
                y1, m1, _ = oracle_at(x + e)
            except ValueError:
                continue
            if m1 < 0.005:
                continue
            n_queries += 1
            p1 = pred((x + e)[None])[1][0]
            if per_parent.get(pi, 0) >= 4:
                continue
            if y1 != y0 and p1 != y1 and len(miss) < a.K // 2:
                miss.append((pi, e, fam, y0, y1, p0, p1))
                per_parent[pi] = per_parent.get(pi, 0) + 1
            elif y1 == y0 and p1 != p0 and len(false) < a.K // 2:
                false.append((pi, e, fam, y0, y1, p0, p1))
                per_parent[pi] = per_parent.get(pi, 0) + 1
    # geometry dedup within each kind
    def dedup(rows):
        kept = []
        for r in rows:
            if all(np.linalg.norm(r[1] - k[1]) >= 0.05 for k in kept):
                kept.append(r)
        return kept
    miss, false = dedup(miss), dedup(false)
    rows = ([(*r, "missing") for r in miss] + [(*r, "false") for r in false])
    rng.shuffle(rows)
    rows = rows[:a.K]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"repairset_{a.tag}.npz",
             scene=np.array([r[0] for r in rows]),
             edit=np.array([r[1] for r in rows]),
             y0=np.array([r[3] for r in rows]),
             y1=np.array([r[4] for r in rows]),
             kind=np.array([r[7] for r in rows]))
    json.dump({"tag": a.tag, "source": a.source, "K": a.K, "kept": len(rows),
               "n_missing": sum(1 for r in rows if r[7] == "missing"),
               "n_false": sum(1 for r in rows if r[7] == "false"),
               "n_queries": n_queries,
               "fams": {f: sum(1 for r in rows if r[2] == f) for f in set(r[2] for r in rows)}},
              open(OUT / f"repairset_{a.tag}.json", "w"), indent=1)
    print(f"SAW repairset_{a.tag}: kept={len(rows)} queries={n_queries}")


if __name__ == "__main__":
    main()
