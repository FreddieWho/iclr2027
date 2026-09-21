#!/usr/bin/env python3
"""WP2 minimal diagnostic (discovery-grade, NO confirm burn).

At oracle single-crossing points x*: Metric A (signed cosine n_o vs model
logit grad), Metric B (offset D to nearest model root along n_o within frozen
r; else NO_LOCAL_MODEL_BOUNDARY), Metric C (true normal alignment where a root
exists). Clean primary; flipmine secondary on IDENTICAL points (paired).
Fresh mining run that saves x* (WP1 rows lack states). Frozen checkpoints only.
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
sys.path.insert(0, str(ROOT / "experiments" / "update_geometry"))
sys.path.insert(0, str(ROOT / "docs" / "iclr2027_discovery_campaign_20260917"))
torch.set_num_threads(4)
from paths import (  # noqa: E402
    oracle_at, scan_linear, locate_oracle_turns, atomic_edits,
    _seg_closest)
from common import load_model, preprocess  # noqa: E402
from wp1_paths import orient_vals, orient_normal  # noqa: E402
from wp1_mine import aimed_edits, random_edits  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
rng = np.random.default_rng(26092203)


def model_grad(model, stats, x):
    mu = torch.tensor(np.asarray(stats["mu"], dtype=np.float32))
    sd = torch.tensor(np.asarray(stats["sd"], dtype=np.float32))
    xt = torch.tensor(np.asarray(x, dtype=float).reshape(-1),
                      dtype=torch.float32, requires_grad=True)
    z = ((xt - mu) / sd).unsqueeze(0)
    h = model.net(z)
    fh = model.feat_head(h) if hasattr(model, "feat_head") else h
    logit = model.cls(fh).squeeze(-1).sum()
    g = torch.autograd.grad(logit, xt)[0].detach().numpy()
    return g


def logit_at(model, stats, x):
    with torch.no_grad():
        lg = model(preprocess(np.asarray(x, dtype=np.float32)[None],
                              stats)).numpy()
    return float(lg.squeeze())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pool", default="eval_202")
    p.add_argument("--n_parents", type=int, default=512)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--n_grid", type=int, default=401)
    a = p.parse_args()
    mrng = np.random.default_rng(a.seed)
    d = np.load(ART / "scenes" / a.pool / "scenes.npz")
    X = d["positions"].astype(np.float32)
    clean, stats = load_model(ART / "r04b_s11" / "clean")
    flip, _ = load_model(ART / "r04b_s11" / "flipmine")
    clean.eval()
    flip.eval()
    assert not stats.get("featurize"), "nonlinear featurize unsupported"
    # frozen radius: 0.25 x median atomic-edit norm
    norms = []
    for pi in mrng.choice(len(X), 100, replace=False):
        for e, _ in atomic_edits(X[pi].astype(float), mrng):
            norms.append(float(np.linalg.norm(e)))
    r = 0.25 * float(np.median(norms))

    def predict_fn(xs, model=clean):
        xs = np.asarray(xs, dtype=np.float32)
        with torch.no_grad():
            out = []
            for s in range(0, len(xs), 256):
                out.append(model(preprocess(xs[s:s + 256], stats)).numpy())
        lg = np.concatenate(out)
        return lg, (lg > 0).astype(int)

    recs = []
    order = mrng.permutation(len(X))[:a.n_parents]
    for pi in order:
        x = X[pi].astype(float)
        try:
            oracle_at(x)
        except ValueError:
            continue
        for e, fam in aimed_edits(x, mrng) + random_edits(x, mrng, n=12):
            try:
                scan = scan_linear(x, e, predict_fn, n_scan=51)
            except Exception:
                continue
            ot = locate_oracle_turns(x, e, scan)
            if len(ot) != 1:
                continue
            t_star = ot[0]["t_star"]
            x_star = x + t_star * e
            s0 = np.sign(orient_vals(x))
            s1 = np.sign(orient_vals(x + e))
            changed = np.nonzero(s0 != s1)[0]
            if len(changed) != 1:
                continue
            n_o = orient_normal(x_star, int(changed[0]))
            # orient sign toward class 1: flip if needed (check s_o side)
            if np.sign(orient_vals(x_star + 1e-7 * n_o.reshape(4, 2))[int(changed[0])]) < 0:
                n_o = -n_o
            nn = np.linalg.norm(n_o)
            if nn < 1e-12:
                continue
            u = n_o / nn
            rec = {"parent_id": int(pi), "fam": fam}
            for mname, model in (("clean", clean), ("flip", flip)):
                g = model_grad(model, stats, x_star)
                ng = np.linalg.norm(g)
                A = float(n_o @ g / (nn * ng)) if ng > 1e-12 else None
                ds = np.linspace(-r, r, a.n_grid)
                lgs = np.array([logit_at(model, stats,
                                         (x_star.reshape(-1) + dd * u).reshape(4, 2))
                                for dd in ds])
                sgn = np.sign(lgs)
                cross = np.nonzero(sgn[:-1] * sgn[1:] < 0)[0]
                if len(cross) == 0:
                    rec[mname] = {"A_local": A, "coverage": False,
                                  "offset": None, "A_boundary": None}
                else:
                    j = cross[np.argmin(np.abs(ds[cross]))]
                    lo, hi = ds[j], ds[j + 1]
                    flo = logit_at(model, stats, (x_star.reshape(-1) + lo * u).reshape(4, 2))
                    for _ in range(20):
                        m = 0.5 * (lo + hi)
                        fm = logit_at(model, stats, (x_star.reshape(-1) + m * u).reshape(4, 2))
                        if (fm > 0) == (flo > 0):
                            lo, flo = m, fm
                        else:
                            hi = m
                    dm = 0.5 * (lo + hi)
                    xm = (x_star.reshape(-1) + dm * u).reshape(4, 2)
                    gm = model_grad(model, stats, xm)
                    ngm = np.linalg.norm(gm)
                    Ab = float(n_o @ gm / (nn * ngm)) if ngm > 1e-12 else None
                    rec[mname] = {"A_local": A, "coverage": True,
                                  "offset": float(abs(dm)), "A_boundary": Ab}
            recs.append(rec)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump({"r": r, "recs": recs}, open(a.out / "rows.json", "w"))

    def summ(mname):
        A = np.array([r[mname]["A_local"] for r in recs
                      if r[mname]["A_local"] is not None])
        cov = np.mean([r[mname]["coverage"] for r in recs])
        off = np.array([r[mname]["offset"] for r in recs
                        if r[mname]["offset"] is not None])
        Ab = np.array([r[mname]["A_boundary"] for r in recs
                       if r[mname]["A_boundary"] is not None])
        return {"n": len(recs), "A_mean": round(float(A.mean()), 4),
                "A_median": round(float(np.median(A)), 4),
                "|A|_mean": round(float(np.abs(A).mean()), 4),
                "coverage": round(float(cov), 4),
                "offset_median": round(float(np.median(off)), 5) if len(off) else None,
                "offset_n": len(off),
                "Ab_mean": round(float(Ab.mean()), 4) if len(Ab) else None,
                "Ab_n": len(Ab)}

    res = {"search_radius": r, "clean": summ("clean"), "flip": summ("flip")}
    # paired deltas (identical points)
    both = [r for r in recs if r["clean"]["A_local"] is not None
            and r["flip"]["A_local"] is not None]
    dA = np.array([r["flip"]["A_local"] - r["clean"]["A_local"] for r in both])
    lo, hi = np.quantile(rng.choice(dA, size=(10000, len(dA))).mean(0),
                         [0.025, 0.975])
    res["paired_dA"] = {"mean": round(float(dA.mean()), 4),
                        "ci": [round(float(lo), 4), round(float(hi), 4)]}
    json.dump(res, open(a.out / "WP2_RESULTS.json", "w"), indent=1)
    print(f"SAW WP2: r={r:.4f} " + json.dumps(res))


if __name__ == "__main__":
    main()
