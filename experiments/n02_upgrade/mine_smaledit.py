#!/usr/bin/env python3
"""Mine a small-edit-dominant test stratum for N02 upgrade (CPU only).

Same miner / renderer contract as sampling_gpu_prepare.generate(), but sweeps a
fresh parent pool and keeps only small-edit quartets (smaller constituent edit
<=2 native64 pixels), cap 4/parent, target >=60 quartets / >=30 parents.
Selection is input-geometry filtering (response-blind), declared as filtered stratum.
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for d in ['experiments/f095_campaign', 'experiments/next6_ef0f7a3',
          'experiments/last15h', 'experiments/last15h/shared',
          'experiments/e1a933_review']:
    sys.path.insert(0, str(ROOT / d))
from paths import atomic_edits, oracle_at
from core.relations import make_relational_scenes
from sampling_gpu_protocol import RADIUS, canonical_coords, digest

TARGET_Q = 60
TARGET_P = 30
PARENT_CHUNK = 512
MAX_CHUNKS = 8
SEED_BASE = 962101


def can(x):
    return canonical_coords(np.asarray(x).reshape(-1, 4, 2)).reshape(-1, 8).astype(float)


def parenthash(x):
    return digest(can(x))


def in_fov(x):
    return bool(np.max(np.abs(x)) <= 1 - RADIUS)


SMALL_RADII = [0.03, 0.05, 0.07]
TRIALS_PER_PARENT = 4


def mine_small_quartets(x, rng):
    """Same quartet logic as mine_quartets but with small radii and more trials.
    Identical oracle/margin/qualification rules; only the proposal is biased
    toward small displacements (response-blind, input-geometry selection)."""
    try:
        y0, _, _ = oracle_at(x)
    except ValueError:
        return
    res = []
    for _ in range(TRIALS_PER_PARENT):
        for r in SMALL_RADII:
            for e, _ in atomic_edits(x, rng, radius=r):
                try:
                    y, m, _ = oracle_at(x + e)
                except ValueError:
                    continue
                if m < 0.005:
                    continue
                res.append((e, y))
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            (ea, ya), (eb, yb) = res[i], res[j]
            if ya != y0 or yb != y0:
                continue
            try:
                yc, mc, _ = oracle_at(x + ea + eb)
            except ValueError:
                continue
            if yc != y0 and mc >= 0.005:
                yield (x, ea, eb, y0, ya, yb, yc)


def main():
    out = ROOT / 'artifacts/n02_upgrade'
    out.mkdir(parents=True, exist_ok=True)
    qcoords, qy, qp_global, qbg = [], [], [], []
    all_parents = []
    counts, n_small = {}, 0
    chunk = 0
    while (len(qcoords) < TARGET_Q or len(counts) < TARGET_P) and chunk < MAX_CHUNKS:
        seed = SEED_BASE + chunk
        pool = np.asarray(make_relational_scenes(PARENT_CHUNK * 4, seed=seed)[0])
        xs = np.asarray([x for x in pool if in_fov(x)][:PARENT_CHUNK])
        assert len(xs) == PARENT_CHUNK, (seed, len(xs))
        base = len(all_parents)
        all_parents.extend(xs)
        lookup = {parenthash(px): i for i, px in enumerate(xs)}
        rng = np.random.default_rng(963004 + chunk)
        for xp in xs:
            for x, ea, eb, y0, ya, yb, yab in mine_small_quartets(xp, rng):
                states = np.stack([x, x + ea, x + eb, x + ea + eb])
                if not in_fov(states):
                    continue
                small = min(np.linalg.norm(ea, axis=-1).max(),
                            np.linalg.norm(eb, axis=-1).max()) * 32 <= 2
                if not small:
                    continue
                # parent index within this chunk
                pi = base + lookup[parenthash(x)]
                if counts.get(pi, 0) >= 4:
                    continue
                counts[pi] = counts.get(pi, 0) + 1
                qcoords.append(states)
                qy.append([y0, ya, yb, yab])
                qp_global.append(pi)
                bg = .5 + np.random.default_rng(963005 + pi).uniform(-.05, .05)
                qbg.append([bg] * 4)
        chunk += 1
        print(f'chunk {chunk}: quartets={len(qcoords)} parents={len(counts)}', flush=True)
    assert len(qcoords) >= TARGET_Q and len(counts) >= TARGET_P, (len(qcoords), len(counts))
    data = dict(quartet_coords=np.asarray(qcoords),
                quartet_labels=np.asarray(qy),
                quartet_parents=np.asarray(qp_global),
                quartet_bg=np.asarray(qbg, dtype=np.float32),
                parent_coords=np.asarray(all_parents))
    np.savez_compressed(out / 'smaledit_bank.npz', **data)
    # Overlap: vs N02 frozen splits + exposed banks, 1e-6 canonical.
    fresh = can(np.asarray(all_parents))
    n02 = np.load(ROOT / 'artifacts/e1a933_review/N02_gpu_bundle/data.npz')
    checks = {}
    for key in ['train_coords', 'dev_coords', 'test_coords', 'quartet_coords']:
        old = can(n02[key].reshape(-1, 4, 2))
        from scipy.spatial import cKDTree as KD
        dist, _ = KD(old).query(fresh, p=np.inf)
        checks['n02_' + key] = dict(min_parent_linf=float(dist.min()),
                                    matches_within_1e6=int((dist <= 1e-6).sum()))
        assert dist.min() > 1e-6, (key, dist.min())
    meta = dict(seeds=list(range(SEED_BASE, SEED_BASE + chunk)),
                chunks=chunk, rows=len(qcoords), parents=len(counts),
                coords_sha256=digest(data['quartet_coords']),
                overlap_checks=checks,
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                      cwd=ROOT, text=True).strip(),
                created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (out / 'smaledit_manifest.json').write_text(json.dumps(meta, indent=2))
    print('BANK_FROZEN', out, 'quartets', len(qcoords), 'parents', len(counts), flush=True)


if __name__ == '__main__':
    main()
