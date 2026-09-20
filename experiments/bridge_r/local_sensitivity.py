#!/usr/bin/env python3
"""Bridge-R Task 2 (Am03): local patch-response sensitivity (no classifier).

For each dev quartet and edit (base->A, base->B, base->AB):
  d_i = 1 - cos(z_base_i, z_edit_i) per patch (primary; normalized L2 stored
  for audit only).
Changed/unchanged patch mask from canonical 512 renders (pixel diff minus
image median, block-averaged to 14 bins; threshold 0.02 frozen in Am03
BEFORE any feature-sensitivity result). Mask sanity guard aborts before
features are touched if degenerate.

Outputs local_sensitivity_<model>.json with D_changed/D_unchanged/R_local
per edit + quartet-level raw. Holdout never loaded.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "bridge_r"))
from render_v1 import render_canonical, CANON

ART = ROOT / "artifacts" / "bridge_r"
PGRID = ART / "patchgrid"
OUT = ART / "spatial"
NUISANCE_BASE = {"train": 100000, "dev": 200000}
PIXEL_DIFF_THRESH = 0.02  # frozen in Am03
EPS = 1e-8
EDITS = ["A", "B", "AB"]


def block14(m):
    """Mean-pool a 512x512 map into 14x14 (approximate patch correspondence)."""
    rows = np.array_split(m, 14, axis=0)
    return np.array([[b.mean() for b in np.array_split(r, 14, axis=1)]
                     for r in rows])


def changed_mask(img_base, img_edit):
    diff = np.abs(img_base.astype(np.float64) - img_edit.astype(np.float64)).mean(axis=2)
    res = np.abs(diff - np.median(diff))
    return block14(res) > PIXEL_DIFF_THRESH, block14(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["dinov3s", "dinov3b", "dinov3l"],
                    required=True)
    a = ap.parse_args()
    bb = a.backbone
    qd = np.load(ART / "quartets_dev_v2.npz")
    nq = len(qd["meta"])

    # 1) masks from renders only (no features involved)
    masks, fracs = {}, {}
    for e in EDITS:
        m = np.zeros((nq, 14, 14), bool)
        fr = np.zeros(nq)
        for qi in range(nq):
            x, ea, eb = qd[f"q{qi}"].reshape(3, 4, 2)
            states = {"base": x, "A": x + ea, "B": x + eb, "AB": x + ea + eb}
            seed = NUISANCE_BASE["dev"] + int(qi)
            # Fresh RNG per state: identical to extract_r.py feature inputs
            # (backgrounds locked across the four states).
            imgs = {k: render_canonical(states[k], np.random.default_rng(seed))
                    for k in ("base", "A", "B", "AB")}
            mm, _ = changed_mask(imgs["base"], imgs[e])
            m[qi] = mm
            fr[qi] = mm.mean()
        masks[e], fracs[e] = m, fr
        print(f"MASK {bb}/{e}: median_changed_frac={np.median(fr):.3f} "
              f"mean={fr.mean():.3f} allzero={(fr == 0).sum()}/{nq}", flush=True)
    med = np.median(np.concatenate([fracs[e] for e in EDITS]))
    if not (0 < med < 0.5):
        raise SystemExit(f"FATAL: degenerate change masks (median frac={med:.3f}); "
                         f"aborting BEFORE features (see Am03).")

    # 2) features: per-patch cosine distance base->edit
    files = sorted(glob.glob(str(PGRID / f"patchgrid_{bb}_dev.shard*.npz")))
    assert files, "no patchgrid shards"
    parts = [np.load(f) for f in files]
    P = np.concatenate([pt["patches"] for pt in parts]).astype(np.float64)
    kinds = np.concatenate([pt["kinds"] for pt in parts])
    qid = np.concatenate([pt["qid"] for pt in parts]).astype(int)
    order = np.argsort(qid, kind="stable")
    P, kinds, qid = P[order], kinds[order], qid[order]
    assert len(np.unique(qid)) == nq
    G = P.reshape(nq, 4, 196, -1)  # quartet, state(base/A/B/AB), patch, C
    K = kinds.reshape(nq, 4)
    assert (K == np.tile(["base", "A", "B", "AB"], (nq, 1))).all()

    res = {"backbone": bb, "n_dev_quartets": nq,
           "metric_primary": "d_cos = 1 - cosine",
           "metric_secondary": "normalized L2 (audit only)",
           "pixel_diff_threshold": PIXEL_DIFF_THRESH,
           "median_changed_frac": float(med),
           "edits": {}, "quartet_raw": {}}
    nb = np.linalg.norm(G[:, 0], axis=2, keepdims=True)
    for j, e in enumerate(EDITS, start=1):
        Ge = G[:, j]
        ne = np.linalg.norm(Ge, axis=2, keepdims=True)
        cos = (G[:, 0] * Ge).sum(axis=2) / np.maximum(nb * ne, EPS).squeeze(-1)
        dcos = 1.0 - np.clip(cos, -1, 1)
        l2n = np.linalg.norm(G[:, 0] - Ge, axis=2) / (
            nb.squeeze(-1) + ne.squeeze(-1) + EPS)
        Dc = np.array([dcos[i][masks[e][i].reshape(-1)].mean()
                       if masks[e][i].any() else float("nan") for i in range(nq)])
        Du = np.array([dcos[i][(~masks[e][i]).reshape(-1)].mean()
                       if (~masks[e][i]).any() else float("nan") for i in range(nq)])
        ok = ~(np.isnan(Dc) | np.isnan(Du))
        R = Dc[ok] / (Du[ok] + EPS)
        res["edits"][e] = {
            "D_changed": float(np.mean(Dc[ok])), "D_unchanged": float(np.mean(Du[ok])),
            "R_local": float(np.mean(R)), "n_ok": int(ok.sum()),
            "L2n_changed": float(np.mean(
                np.array([l2n[i][masks[e][i].reshape(-1)].mean() for i in range(nq) if masks[e][i].any()]))),
            "L2n_unchanged": float(np.mean(
                np.array([l2n[i][(~masks[e][i]).reshape(-1)].mean() for i in range(nq) if (~masks[e][i]).any()]))),
        }
        res["quartet_raw"][e] = {"D_changed": Dc.tolist(), "D_unchanged": Du.tolist()}
        print(f"SENS {bb}/{e}: D_ch={res['edits'][e]['D_changed']:.4f} "
              f"D_un={res['edits'][e]['D_unchanged']:.4f} "
              f"R={res['edits'][e]['R_local']:.3f} n={ok.sum()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"local_sensitivity_{bb}.json", "w"), indent=1)
    print("SAW sensitivity", bb)


if __name__ == "__main__":
    main()
