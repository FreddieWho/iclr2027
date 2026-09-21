#!/usr/bin/env python3
"""G1 data: mine 1:1 stay/flip single transitions from a scene pool.

Stay: y(x+e)==y(x); flip: y(x+e)!=y(x). Saves pairs (x0, x1, u) + parent ids.
Composition NEVER mined here. Deterministic given seed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
from paths import oracle_at  # noqa: E402
from r02_search import candidates_for_scene  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
OUT = ROOT / "artifacts" / "event_updater"


def mine(pool, n_want, seed):
    rng = np.random.default_rng(seed)
    d = np.load(ART / "scenes" / pool / "scenes.npz")
    X = d["positions"].astype(np.float32)
    stay, flip = [], []
    for pi in rng.permutation(len(X)):
        if len(stay) >= n_want and len(flip) >= n_want:
            break
        x = X[pi].astype(float)
        try:
            y0, _, _ = oracle_at(x)
        except ValueError:
            continue
        for item in candidates_for_scene(x, rng, n_random=16):
            e = np.asarray(item[0], dtype=float)
            try:
                y1, m1, _ = oracle_at(x + e)
            except ValueError:
                continue
            if m1 < 0.005:
                continue
            if y1 == y0 and len(stay) < n_want:
                stay.append((x, x + e, pi))
            elif y1 != y0 and len(flip) < n_want:
                flip.append((x, x + e, pi))
            if len(stay) >= n_want and len(flip) >= n_want:
                break
    return stay, flip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n_each", type=int, default=2000)
    p.add_argument("--seed", type=int, default=101)
    a = p.parse_args()
    stay, flip = mine(a.pool, a.n_each, a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    S = np.array([s[0] for s in stay] + [s[0] for s in flip], dtype=np.float32)
    E = np.array([s[1] for s in stay] + [s[1] for s in flip], dtype=np.float32)
    U = np.array([0] * len(stay) + [1] * len(flip), dtype=np.int64)
    P = np.array([s[2] for s in stay] + [s[2] for s in flip], dtype=np.int64)
    assert len(stay) == a.n_each and len(flip) == a.n_each, (len(stay), len(flip))
    np.savez(OUT / f"events_{a.tag}.npz", x0=S, x1=E, u=U, parent=P)
    json.dump({"tag": a.tag, "pool": a.pool, "n_each": a.n_each, "seed": a.seed,
               "n_parents_stay": len(set(s[2] for s in stay)),
               "n_parents_flip": len(set(s[2] for s in flip))},
              open(OUT / f"events_{a.tag}.json", "w"), indent=1)
    print(f"SAW events_{a.tag}: stay={len(stay)} flip={len(flip)}")


if __name__ == "__main__":
    main()
