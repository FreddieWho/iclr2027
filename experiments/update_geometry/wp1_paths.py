#!/usr/bin/env python3
"""WP1 discovery: quartet path decomposition + incidence (frozen model, eval only).

For each valid emergent quartet (yA=yB=y0, yAB!=y0) build two composed paths
gamma_A(t)=x+A+tB, gamma_B(t)=x+B+tA; 51-pt coarse scan + 12-iter bisection
for oracle and model crossings; record full trajectory fields + incidence I
at oracle crossings (autograd orient gradient with finite-diff audit).
Writes rows JSON + summary. No training, no selection on outcomes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "docs" / "iclr2027_discovery_campaign_20260917"))
torch.set_num_threads(4)
from paths import (  # noqa: E402
    oracle_at, scan_linear, locate_oracle_turns, locate_model_turns,
    match_turns, atomic_edits)
from common import load_model, preprocess  # noqa: E402
from core.relations import segment_relation  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"


def orient_vals(x):
    a, b, c, d = x
    def o(p, q, r):
        u, v = q - p, r - p
        return u[0] * v[1] - u[1] * v[0]
    return np.array([o(a, b, c), o(a, b, d), o(c, d, a), o(c, d, b)])


def orient_normal(x_star, k):
    """Exact gradient of orient value k at x* via autograd (polynomial)."""
    xt = torch.tensor(np.asarray(x_star, dtype=float), requires_grad=True)
    a, b, c, d = xt[0], xt[1], xt[2], xt[3]
    def o(p, q, r):
        u, v = q - p, r - p
        return u[0] * v[1] - u[1] * v[0]
    vals = torch.stack([o(a, b, c), o(a, b, d), o(c, d, a), o(c, d, b)])
    g = torch.autograd.grad(vals[k], xt)[0].reshape(-1).detach().numpy()
    return g


def audit_normal(x_star, k, v):
    """Finite-diff audit of n_o with two epsilons; returns (I, rel_disagree)."""
    n_auto = orient_normal(x_star, k)
    xs = np.asarray(x_star, dtype=float).reshape(-1)
    vv = np.asarray(v, dtype=float).reshape(-1)
    ds = []
    for eps in (1e-5, 1e-6):
        g = np.zeros_like(xs)
        for j in range(len(xs)):
            dp = np.zeros_like(xs); dp[j] = eps
            fp = orient_vals((xs + dp).reshape(4, 2))[k]
            fm = orient_vals((xs - dp).reshape(4, 2))[k]
            g[j] = (fp - fm) / (2 * eps)
        ds.append(g)
    rel = float(np.linalg.norm(ds[0] - ds[1]) /
                max(1e-12, np.linalg.norm(ds[0])))
    na, nv = np.linalg.norm(n_auto), np.linalg.norm(vv)
    I = float(abs(n_auto @ vv) / (na * nv)) if na > 1e-12 and nv > 1e-12 else None
    return I, rel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pool", default="eval_202")
    p.add_argument("--model", type=Path, default=ART / "r04b_s11" / "clean")
    p.add_argument("--n_parents", type=int, default=512)
    p.add_argument("--seed", type=int, default=11)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    d = np.load(ART / "scenes" / a.pool / "scenes.npz")
    X = d["positions"].astype(np.float32)
    model, stats = load_model(a.model)
    model.eval()

    def predict_fn(xs):
        xs = np.asarray(xs, dtype=np.float32)
        with torch.no_grad():
            out = []
            for s in range(0, len(xs), 256):
                lg = model(preprocess(xs[s:s + 256], stats)).numpy()
                out.append(lg)
        lg = np.concatenate(out)
        return lg, (lg > 0).astype(int)

    rows = []
    n_quart = 0
    order = rng.permutation(len(X))[:a.n_parents]
    for pi in order:
        x = X[pi].astype(float)
        try:
            y0, _, _ = oracle_at(x)
        except ValueError:
            continue
        atomic = []
        for e, fam in atomic_edits(x, rng):
            try:
                y, m, _ = oracle_at(x + e)
            except ValueError:
                continue
            atomic.append((e, fam, y, m))
        for i in range(len(atomic)):
            for j in range(i + 1, len(atomic)):
                ea, fa, ya, ma = atomic[i]
                eb, fb, yb, mb = atomic[j]
                if ya != y0 or yb != y0:
                    continue
                try:
                    yc, mc, _ = oracle_at(x + ea + eb)
                except ValueError:
                    continue
                if yc == y0:
                    continue
                n_quart += 1
                for ptype, (xs_lo, v) in (("A", (x + ea, eb)), ("B", (x + eb, ea))):
                    try:
                        scan = scan_linear(xs_lo, v, predict_fn, n_scan=51)
                    except Exception:
                        continue
                    ot = locate_oracle_turns(xs_lo, v, scan)
                    mt = locate_model_turns(xs_lo, v, scan, predict_fn)
                    matched = match_turns(ot, mt)
                    sl = int(scan["oracle_label"][0])
                    el = int(scan["oracle_label"][-1])
                    lg0 = float(scan["logit"][0])
                    lg1 = float(scan["logit"][-1])
                    row = {"parent_id": int(pi), "path_type": ptype,
                           "fam_A": fa, "fam_B": fb,
                           "start_label": sl, "endpoint_label": el,
                           "oracle_n_crossings": len(ot),
                           "oracle_crossing_t": [o["t_star"] for o in ot],
                           "model_n_crossings": len(mt),
                           "model_crossing_ts": [m["t_theta"] for m in mt],
                           "match": matched,
                           "start_correct": bool((lg0 > 0) == sl),
                           "endpoint_correct": bool((lg1 > 0) == el),
                           "start_logit": lg0, "endpoint_logit": lg1,
                           "edit_norm": float(np.linalg.norm(v)),
                           "start_margin": float(scan["oracle_margin"][0]),
                           "endpoint_margin": float(scan["oracle_margin"][-1]),
                           "incidence": None, "incidence_audit_rel": None,
                           "flip_orient": None}
                    if len(ot) == 1:
                        t_star = ot[0]["t_star"]
                        x_star = xs_lo + t_star * v
                        s0 = np.sign(orient_vals(xs_lo))
                        s1 = np.sign(orient_vals(xs_lo + v))
                        changed = np.nonzero(s0 != s1)[0]
                        if len(changed) == 1:
                            k = int(changed[0])
                            I, rel = audit_normal(x_star, k, v)
                            row.update({"incidence": I,
                                        "incidence_audit_rel": rel,
                                        "flip_orient": k})
                    rows.append(row)
    single = sum(1 for r in rows if r["oracle_n_crossings"] == 1)
    withI = sum(1 for r in rows if r["incidence"] is not None)
    bad_audit = sum(1 for r in rows if r["incidence_audit_rel"] is not None
                    and r["incidence_audit_rel"] > 1e-3)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows}, open(a.out / "rows.json", "w"))
    summ = {"n_parents": int(a.n_parents), "n_quartets": n_quart,
            "n_paths": len(rows), "n_single_cross": single,
            "n_with_incidence": withI, "n_bad_audit": bad_audit}
    json.dump(summ, open(a.out / "summary.json", "w"), indent=1)
    print(f"SAW WP1: {summ}")


if __name__ == "__main__":
    main()
