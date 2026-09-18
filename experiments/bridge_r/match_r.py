#!/usr/bin/env python3
"""Bridge-R R6: matched-single control construction (geometry only, no model).

For each holdout composition quartet, finds a 1:1 single flip
y(x+es) != y(x) matched on standardized [|e|, endpoint margin, pixel-change].
Rule frozen pre-lock; tolerance tiers T1->T2->T3 applied in fixed order.
Called ONLY inside the one-shot holdout runner (holdout geometry is sealed).

Pixel change D_pix = mean L1 between canonical 512 renders of base and
endpoint, rendered with the quartet's own holdout nuisance seed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "bridge_r"))
from paths import oracle_at
from r02_search import candidates_for_scene
from render_v1 import render_canonical

ART = ROOT / "artifacts" / "bridge_r"
NUISANCE_HOLDOUT = 300000
N_CANDIDATES = 64
TIERS = [0.5, 1.0, 1.5]  # max|z| fallback order, frozen
MARGIN_MIN = 0.005


def dpix(a, b):
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


def build(holdout_npz, rng_seed=4242):
    d = np.load(holdout_npz)
    nq = len(d["meta"])
    rng = np.random.default_rng(rng_seed)
    # composition covariates
    comp = []
    for qi in range(nq):
        q = d[f"q{qi}"]
        x, ea, eb = q[0], q[1], q[2]
        seed = NUISANCE_HOLDOUT + qi
        rb = render_canonical(x, np.random.default_rng(seed))
        rab = render_canonical(x + ea + eb, np.random.default_rng(seed))
        mc = float(d["margins"][qi][3])
        comp.append({"qi": qi, "n_e": float(np.linalg.norm(ea + eb)),
                     "m": mc, "dpix": dpix(rb, rab)})
    C = np.array([[c["n_e"], c["m"], c["dpix"]] for c in comp])
    mu, sd = C.mean(0), C.std(0) + 1e-12
    # candidate singles per quartet
    cands = []
    for qi in range(nq):
        q = d[f"q{qi}"]
        x = q[0]
        try:
            y0, _, _ = oracle_at(x)
        except ValueError:
            continue
        seed = NUISANCE_HOLDOUT + qi
        rb = render_canonical(x, np.random.default_rng(seed))
        for e, fam in candidates_for_scene(x, rng, n_random=N_CANDIDATES):
            e = np.asarray(e, float)
            try:
                y1, m1, _ = oracle_at(x + e)
            except ValueError:
                continue
            if y1 == y0 or m1 < MARGIN_MIN:
                continue
            r1 = render_canonical(x + e, np.random.default_rng(seed))
            cands.append({"qi": qi, "e": e, "y1": int(y1), "m": float(m1),
                          "n_e": float(np.linalg.norm(e)),
                          "dpix": dpix(rb, r1)})
        if qi % 200 == 0:
            print(f"MATCH scan {qi}/{nq}", flush=True)
    # 1:1 NN without replacement, tier fallback
    used = set()
    matches, tiers_used = [], []
    order = np.random.default_rng(rng_seed + 1).permutation(len(comp))
    for ci in order:
        zc = (np.array([comp[ci]["n_e"], comp[ci]["m"], comp[ci]["dpix"]]) - mu) / sd
        pool = [c for c in cands if c["qi"] == comp[ci]["qi"] and
                id(c) not in used]
        best, tier = None, None
        for t, tol in enumerate(TIERS):
            scored = []
            for c in pool:
                z = (np.array([c["n_e"], c["m"], c["dpix"]]) - mu) / sd
                dz = float(np.abs(z - zc).max())
                if dz <= tol:
                    scored.append((dz, c))
            if scored:
                scored.sort(key=lambda t_: t_[0])
                best, tier = scored[0][1], t
                break
        if best is None:
            continue
        used.add(id(best))
        matches.append({"qi": comp[ci]["qi"], "single": best, "tier": tier})
        tiers_used.append(tier)
    return comp, matches, {"mu": mu.tolist(), "sd": sd.tolist(), "tiers": TIERS}


def balance_table(comp, matches):
    import io
    matched = {m["qi"]: m["single"] for m in matches}
    rows = []
    for c in comp:
        if c["qi"] not in matched:
            continue
        s = matched[c["qi"]]
        rows.append(([c["n_e"], c["m"], c["dpix"]], [s["n_e"], s["m"], s["dpix"]]))
    A = np.array([r[0] for r in rows])
    B = np.array([r[1] for r in rows])
    out = {"n_comp": len(comp), "n_matched": len(rows)}
    for j, name in enumerate(["edit_norm", "endpoint_margin", "pixel_change"]):
        d = (A[:, j].mean() - B[:, j].mean())
        pooled = np.sqrt((A[:, j].var() + B[:, j].var()) / 2) + 1e-12
        out[name] = {"comp_median": float(np.median(A[:, j])),
                     "single_median": float(np.median(B[:, j])),
                     "comp_iqr": [float(np.percentile(A[:, j], 25)),
                                  float(np.percentile(A[:, j], 75))],
                     "single_iqr": [float(np.percentile(B[:, j], 25)),
                                    float(np.percentile(B[:, j], 75))],
                     "std_diff": float(d / pooled)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    comp, matches, cfg = build(ART / "quartets_holdout_v2.npz")
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / "matched_singles.npz",
             qi=np.array([m["qi"] for m in matches]),
             e=np.stack([m["single"]["e"] for m in matches]),
             y1=np.array([m["single"]["y1"] for m in matches]),
             tier=np.array([m["tier"] for m in matches]))
    json.dump({"matching": cfg, "balance": balance_table(comp, matches)},
              open(a.out / "match_report.json", "w"), indent=1)
    print(f"SAW matched {len(matches)}/{len(comp)}", flush=True)
    print("DONE match")


if __name__ == "__main__":
    main()
