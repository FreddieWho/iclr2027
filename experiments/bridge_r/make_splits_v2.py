#!/usr/bin/env python3
"""Bridge-R R2: generate fresh deterministic train/dev/holdout v2 quartets.

Fresh generator seeds, disjoint from every existing pool (101/202/303/404/
505/606/707/808/909) and from each other. Parent scenes are generated with
make_relational_scenes; emergent quartets mined with the U01 rule (atomics
preserve with margin>=0.005, combo flips with margin>=0.005). Records oracle
margins and edit norms per quartet for later matched-single control.

Writes artifacts/bridge_r/{quartets_train_v2,quartets_dev_v2,
quartets_holdout_v2}.npz + SPLITS_MANIFEST.json. No model, no rendering here.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "docs" / "iclr2027_discovery_campaign_20260917"))
from paths import atomic_edits, oracle_at
from core.relations import make_relational_scenes

DEFAULT_SEEDS = {"train": 26091801, "dev": 26091802, "holdout": 26091803}
DEFAULT_PARENTS = {"train": 6000, "dev": 1800, "holdout": 2600}
MIN_QUARTETS = {"train": 2000, "dev": 500, "holdout": 512}
MAX_QUARTETS = {"train": 100000, "dev": 100000, "holdout": 1024}
ATOMIC_MARGIN = 0.005


def mine(X, ids, rng):
    out = []
    for pi in range(len(X)):
        x = X[pi].astype(float)
        try:
            y0, m0, _ = oracle_at(x)
        except ValueError:
            continue
        res = []
        for e, fam in atomic_edits(x, rng):
            try:
                y, m, _ = oracle_at(x + e)
            except ValueError:
                continue
            if m < ATOMIC_MARGIN:
                continue
            res.append((np.asarray(e, float), y, m, fam))
        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                (ea, ya, ma, fa), (eb, yb, mb, fb) = res[i], res[j]
                if ya != y0 or yb != y0:
                    continue
                try:
                    yc, mc, _ = oracle_at(x + ea + eb)
                except ValueError:
                    continue
                if yc != y0 and mc >= ATOMIC_MARGIN:
                    out.append({
                        "x": x, "ea": ea, "eb": eb,
                        "y0": y0, "ya": ya, "yb": yb, "yc": yc,
                        "m0": m0, "ma": ma, "mb": mb, "mc": mc,
                        "n_ea": float(np.linalg.norm(ea)),
                        "n_eb": float(np.linalg.norm(eb)),
                        "n_eab": float(np.linalg.norm(ea + eb)),
                        "fam_a": fa, "fam_b": fb,
                        "parent_id": str(ids[pi]),
                    })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_train_parents", type=int, default=DEFAULT_PARENTS["train"])
    p.add_argument("--n_dev_parents", type=int, default=DEFAULT_PARENTS["dev"])
    p.add_argument("--n_holdout_parents", type=int, default=DEFAULT_PARENTS["holdout"])
    p.add_argument("--seed_train", type=int, default=DEFAULT_SEEDS["train"])
    p.add_argument("--seed_dev", type=int, default=DEFAULT_SEEDS["dev"])
    p.add_argument("--seed_holdout", type=int, default=DEFAULT_SEEDS["holdout"])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"generator": "make_relational_scenes", "min_margin": 0.02,
                "atomic_margin": ATOMIC_MARGIN, "splits": {}}
    for split, n_par, gseed in (("train", a.n_train_parents, a.seed_train),
                                ("dev", a.n_dev_parents, a.seed_dev),
                                ("holdout", a.n_holdout_parents, a.seed_holdout)):
        X, y, m = make_relational_scenes(n_par, gseed, 0.02)
        ids = np.array([f"bridge_r_{split}_{gseed}_{i:06d}" for i in range(n_par)])
        Q = mine(X, ids, np.random.default_rng(gseed + 7919))
        if len(Q) < MIN_QUARTETS[split]:
            raise SystemExit(f"FATAL {split}: {len(Q)} < {MIN_QUARTETS[split]}; "
                             f"raise parent count")
        Q = Q[:MAX_QUARTETS[split]]
        pack = {f"q{i}": np.stack([q["x"].ravel(), q["ea"].ravel(), q["eb"].ravel()])
                for i, q in enumerate(Q)}
        pack["meta"] = np.array([[q["y0"], q["ya"], q["yb"], q["yc"]] for q in Q])
        pack["margins"] = np.array([[q["m0"], q["ma"], q["mb"], q["mc"]] for q in Q])
        pack["norms"] = np.array([[q["n_ea"], q["n_eb"], q["n_eab"]] for q in Q])
        pack["parent_id"] = np.array([q["parent_id"] for q in Q])
        pack["fam"] = np.array([[q["fam_a"], q["fam_b"]] for q in Q])
        np.savez(a.out / f"quartets_{split}_v2.npz", **pack)
        manifest["splits"][split] = {
            "generator_seed": gseed, "mine_rng_seed": gseed + 7919,
            "n_parents": n_par, "n_quartets": len(Q),
            "margin_med": [float(np.median(pack["margins"][:, k])) for k in range(4)],
        }
        print(f"SAW {split}: parents={n_par} quartets={len(Q)}", flush=True)
    json.dump(manifest, open(a.out / "SPLITS_MANIFEST.json", "w"), indent=1)
    print("DONE R2")


if __name__ == "__main__":
    main()
