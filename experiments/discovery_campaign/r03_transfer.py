#!/usr/bin/env python3
"""R03 mini-run: source-only greedy mode cover -> frozen target evaluation.

Bank-8 (35 modes), eps fixed 0.1, oracle both-arm validity per (mode,scene).
Source (eval_202[0:256], mlpA): failure matrix S[mode,scene] = 1 if any valid
arm mispredicted. Greedy cover k=1,2,4,8 (frozen thereafter) + seed-fixed
random-mode control at same k.
Target (eval_202[256:512], frozen): union coverage@k (offline-library sense;
2k forwards/scene noted) + uniform-fixed-weight pair risk@k.
Cross-arch: same frozen modes evaluated on mlpB/mlpC (no re-selection).
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
from core.perturbations import (balanced_sign_patterns, antithetic_fields,
                                greedy_mode_cover)
from core.relations import segment_relation
from common import load_model, preprocess


def oracle_batch(pos: np.ndarray, min_margin: float):
    n = len(pos)
    labels = np.full(n, -1)
    for i in range(n):
        try:
            r = segment_relation(pos[i])
        except ValueError:
            continue
        if not r["ambiguous"] and r["margin"] >= min_margin:
            labels[i] = r["label"]
    return labels


def fail_matrix(model, stats, X, y, fields, min_margin):
    """Return (fail_any[M,N] bool, valid_any[M,N] bool, pair_risk[M,N] float)."""
    M = fields.shape[0]
    N = len(X)
    fail_any = np.zeros((M, N), bool)
    valid_any = np.zeros((M, N), bool)
    pair_risk = np.full((M, N), np.nan)
    with torch.no_grad():
        for m in range(M):
            fails, nvalid = [], []
            for arm in range(2):
                pert = X + fields[m, arm]
                lab = oracle_batch(pert, min_margin)
                ok = lab >= 0
                pr = (torch.sigmoid(model(preprocess(pert, stats))).numpy() > .5).astype(int)
                f = np.zeros(N, bool)
                f[ok] = (pr[ok] != lab[ok])
                fails.append(f)
                nvalid.append(ok)
            fail_any[m] = fails[0] | fails[1]
            valid_any[m] = nvalid[0] | nvalid[1]
            both = nvalid[0] & nvalid[1]
            pair_risk[m, both] = ((fails[0] & both).astype(float)[both]
                                  + (fails[1] & both).astype(float)[both]) / 2
    return fail_any, valid_any, pair_risk


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-model", type=Path, required=True)
    p.add_argument("--models", nargs="+", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--src-slice", type=str, default="0:256")
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--tgt-slice", type=str, default="256:512")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--min-margin", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(a.seed + 999)

    def load_slice(d, s):
        z = np.load(d / "scenes.npz")
        if s:
            lo, hi = [int(v) for v in s.split(":")]
            return z["positions"][lo:hi].astype(float), z["labels"][lo:hi].astype(int)
        return z["positions"].astype(float), z["labels"].astype(int)

    Xs, ys = load_slice(a.source, a.src_slice)
    Xt, yt = load_slice(a.target, a.tgt_slice)
    patterns = balanced_sign_patterns((8,), max_patterns=64, seed=a.seed)
    fields = antithetic_fields(patterns, a.eps, np.array([1.0])).reshape(
        len(patterns), 2, 4, 2).astype(np.float32)
    M = len(patterns)
    smodel, sstats = load_model(a.source_model)

    S_src, V_src, P_src = fail_matrix(smodel, sstats, Xs, ys, fields, a.min_margin)
    ks = [1, 2, 4, 8]
    selection, rand_sel = {}, {}
    for k in ks:
        chosen, _ = greedy_mode_cover(S_src, k=min(k, M))
        selection[k] = chosen
        pool = rng.permutation(M)[:min(k, M)].tolist()
        rand_sel[k] = pool
    (a.output / "source_selection.json").write_text(json.dumps(
        {"greedy": {str(k): v for k, v in selection.items()},
         "random_control": {str(k): v for k, v in rand_sel.items()},
         "source": str(a.source), "slice": a.src_slice,
         "source_model": str(a.source_model), "eps": a.eps,
         "coverage_def": "scene covered if ANY valid arm of ANY selected mode fails (offline-library sense; costs <=2k forwards/scene)"}, indent=2) + "\n")
    np.savez_compressed(a.output / "templates.npz", patterns=patterns,
                        greedy_ks=np.array([selection[k] for k in ks], dtype=object))
    np.savez_compressed(a.output / "source_matrix.npz", fail=S_src, valid=V_src,
                        pair_risk=P_src)

    rows = []
    loaded = [(str(m), *load_model(m)) for m in a.models]
    Ft, Vt, Pt = {}, {}, {}
    for name, model, stats in loaded:
        Ft[name], Vt[name], Pt[name] = fail_matrix(model, stats, Xt, yt, fields, a.min_margin)
    Nt = len(Xt)
    for name, _, _ in loaded:
        for k in ks:
            for tag, sel in (("greedy", selection[k]), ("random", rand_sel[k])):
                sel = [s for s in sel if s < M]
                if not sel:
                    continue
                cov = Ft[name][sel].any(axis=0).mean()
                pr = np.nanmean(Pt[name][sel], axis=0)
                rows.append({"model": Path(name).name, "k": k, "selection": tag,
                             "union_coverage": float(cov),
                             "fixed_weight_risk": float(np.nanmean(pr)),
                             "n_target": Nt,
                             "scenes_with_any_valid": int(Vt[name][sel].any(axis=0).sum())})
    with open(a.output / "transfer_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    oracle_evals = (len(Xs) + len(Xt) * len(loaded)) * M * 2
    (a.output / "R03_result.json").write_text(json.dumps(
        {"ks": ks, "eps": a.eps, "n_source": len(Xs), "n_target": Nt,
         "models": [Path(n).name for n, _, _ in loaded],
         "oracle_evals": oracle_evals,
         "notes": ["Selection frozen on source; target is forward-only.",
                   "Union coverage is offline-library coverage, not single-draw failure rate."]}, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if k != "n_target"} for r in rows], indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
