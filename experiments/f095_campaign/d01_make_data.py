#!/usr/bin/env python3
"""D01 nested data banks: N (=train_101, 512 scenes) ⊂ 4N (2048) ⊂ 16N (8192).

4N = train_101 + 1536 fresh scenes; 16N = 4N + 6144 fresh scenes.
Fresh scenes use make_relational_scenes with fixed seeds and the same
min_margin=0.02; class balance is maintained per chunk. All parent ids,
seeds and SHAs go into the manifest. Fresh chunks are appended (order
fixed by RNG), so N/4N/16N are strict nesting prefixes.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "docs" / "iclr2027_discovery_campaign_20260917"
sys.path.insert(0, str(PACK))
from core.relations import make_relational_scenes  # noqa: E402

BASE = ROOT / "artifacts" / "discovery_campaign" / "scenes" / "train_101"
OUT = ROOT / "artifacts" / "f095_campaign" / "D01"


def load_base():
    d = np.load(BASE / "scenes.npz")
    return (d["positions"], d["labels"], d["oracle_margin"],
            d["parent_scene_id"])


def fresh(n, seed, tag):
    pos, lab, mar = make_relational_scenes(n, seed=seed, min_margin=0.02)
    ids = np.array([f"{tag}_{i:06d}" for i in range(n)])
    return pos, lab, mar, ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    bx, by, bm, bid = load_base()
    assert len(bx) == 512, len(bx)
    chunks = [("base(train_101,seed101)", bx, by, bm, bid)]
    for n, seed, tag in ((1536, 401, "d01fresh401"), (6144, 1601, "d01fresh1601")):
        x, y, m, i = fresh(n, seed, tag)
        chunks.append((f"{tag}(seed{seed})", x, y, m, i))
    banks = {}
    banks["N"] = (0, 512)
    banks["4N"] = (0, 2048)
    banks["16N"] = (0, 8192)
    X = np.concatenate([c[1] for c in chunks])
    Y = np.concatenate([c[2] for c in chunks])
    M = np.concatenate([c[3] for c in chunks])
    I = np.concatenate([c[4] for c in chunks])
    assert len(X) == 8192
    a.out.mkdir(parents=True, exist_ok=True)
    for name, (lo, hi) in banks.items():
        d = a.out / f"scenes_{name}"
        d.mkdir(exist_ok=True)
        if (d / "scenes.npz").exists():
            raise FileExistsError(f"refusing to overwrite {d}")
        np.savez_compressed(d / "scenes.npz", positions=X[lo:hi], labels=Y[lo:hi],
                            oracle_margin=M[lo:hi], parent_scene_id=I[lo:hi])
    manifest = {
        "nesting": "N ⊂ 4N ⊂ 16N prefixes of one 8192-scene bank",
        "chunks": [{"tag": c[0], "n": len(c[1]),
                    "pos_frac": round(float(c[2].mean()), 4)} for c in chunks],
        "banks": {k: {"n": hi - lo, "sha256": hashlib.sha256(
            (a.out / f"scenes_{k}" / "scenes.npz").read_bytes()).hexdigest()}
            for k, (lo, hi) in banks.items()},
        "base_sha256": hashlib.sha256((BASE / "scenes.npz").read_bytes()).hexdigest(),
        "min_margin": 0.02,
    }
    (a.out / "data_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
