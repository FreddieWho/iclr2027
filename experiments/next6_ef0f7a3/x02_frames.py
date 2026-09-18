#!/usr/bin/env python3
"""X02 round1: test-time relation frames on FROZEN models. Zero training.

Views (exact index permutations — same coordinate multiset, label-invariant):
 v0 identity; v1 swap A<->B; v2 swap C<->D; v3 both; v4 AB<->CD segments;
 v5..v7 = v4+v1/v2/v3. Crossing label provably unchanged (unordered segments).
Rule fixed a priori: mean probability over 8 views. Truth never picks views.
Controls (same 8-forward budget): single AB-swap view; centered-single view;
8-jitter vote (sigma=0.02, labels recomputed by oracle per jittered scene).
Beds: (1) emergent combos on eval_202 (fresh mine); (2) static eval_202 acc.
Models: r04b_s11/clean (main) + flipmine (reference).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
from paths import atomic_edits, oracle_at
from common import load_model, preprocess

ART = ROOT / "artifacts" / "discovery_campaign"


def views_of(x):
    x = np.asarray(x, dtype=float)
    v0 = x
    v1 = x[[1, 0, 2, 3]]
    v2 = x[[0, 1, 3, 2]]
    v3 = x[[1, 0, 3, 2]]
    v4 = x[[2, 3, 0, 1]]
    return [v0, v1, v2, v3, v4, v4[[1, 0, 2, 3]], v4[[0, 1, 3, 2]],
            v4[[1, 0, 3, 2]]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(202)
    de = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    X = de["positions"].astype(float)
    y = de["labels"].astype(int)
    loaded = {}
    for k, mp in (("clean", ART / "r04b_s11" / "clean"),
                  ("flipmine", ART / "r04b_s11" / "flipmine")):
        m, s = load_model(mp)
        m.eval()
        loaded[k] = (m, s)

    def proba(model_key, xs):
        m, s = loaded[model_key]
        with torch.no_grad():
            lg = m(preprocess(xs.astype(np.float32), s)).numpy()
        return 1 / (1 + np.exp(-lg))

    # bed 1: emergent combos (fresh mine on eval_202)
    combos, atomA, atomB = [], [], []
    for pi in rng.permutation(len(X)):
        x = X[pi]
        try:
            y0, _, _ = oracle_at(x)
        except ValueError:
            continue
        res = []
        for e, _ in atomic_edits(x, rng):
            try:
                yy, mm, _ = oracle_at(x + e)
            except ValueError:
                continue
            if mm < 0.005:
                continue
            res.append((e, yy))
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
                    combos.append((x + ea + eb, yc))
                    atomA.append((x + ea, ya))
                    atomB.append((x + eb, yb))
    Cb = np.stack([c for c, _ in combos]); Cby = np.array([t for _, t in combos])
    Aa = np.stack([c for c, _ in atomA]); Aay = np.array([t for _, t in atomA])
    Ab = np.stack([c for c, _ in atomB]); Aby = np.array([t for _, t in atomB])

    res = {"n_combos": len(combos)}
    for k in loaded:
        # identity baseline
        p_id = proba(k, Cb)
        miss_id = float((((p_id > 0.5).astype(int)) != Cby).mean())
        # 8-frame vote (mean prob)
        pv = np.stack([proba(k, np.stack([v[i] for v in
                                          [views_of(c) for c in Cb]]))
                       for i in range(8)])
        miss_vote = float((((pv.mean(0) > 0.5).astype(int)) != Cby).mean())
        agree = float(((((pv > 0.5).astype(int)) == (p_id > 0.5)).sum(0)
                       == 8).mean())
        # single-frame control (AB-swap only, 1 forward)
        p_s1 = proba(k, np.stack([views_of(c)[1] for c in Cb]))
        miss_s1 = float((((p_s1 > 0.5).astype(int)) != Cby).mean())
        # centered-single control
        cen = np.stack([c - c.reshape(-1, 2).mean(0) for c in Cb])
        p_c = proba(k, cen)
        miss_c = float((((p_c > 0.5).astype(int)) != Cby).mean())
        # 8-jitter vote (honest labels per jittered scene)
        jit = [Cb + rng.normal(0, 0.02, Cb.shape) for _ in range(8)]
        pj = np.stack([proba(k, j) for j in jit])
        jy, valid = [], []
        for j in jit:
            row, ok = [], []
            for w in j:
                try:
                    row.append(oracle_at(w)[0])
                    ok.append(True)
                except ValueError:
                    row.append(0)
                    ok.append(False)
            jy.append(row)
            valid.append(ok)
        jy = np.array(jy)
        valid = np.array(valid)
        pv_j = (pj.mean(0) > 0.5).astype(int)
        yv_j = (jy.mean(0) > 0.5).astype(int)
        keep = valid.all(0)
        miss_j = float(((pv_j[keep] != yv_j[keep])).mean()) if keep.sum() else None
        res[k] = {"combo_miss_id": miss_id, "combo_miss_vote8": miss_vote,
                  "combo_miss_swap1": miss_s1, "combo_miss_centered": miss_c,
                  "combo_miss_jitter8": miss_j, "view_agree8": agree}
        # bed 2: static
        ps_id = proba(k, X)
        err_id = float((((ps_id > 0.5).astype(int)) != y).mean())
        psv = np.stack([proba(k, np.stack([v[i] for v in
                                           [views_of(c) for c in X]]))
                        for i in range(8)])
        err_vote = float((((psv.mean(0) > 0.5).astype(int)) != y).mean())
        res[k].update({"static_err_id": err_id, "static_err_vote8": err_vote})
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    for k in loaded:
        print(f"SAW {k}: combo id={res[k]['combo_miss_id']:.3f} "
              f"vote8={res[k]['combo_miss_vote8']:.3f} "
              f"swap1={res[k]['combo_miss_swap1']:.3f} "
              f"cen={res[k]['combo_miss_centered']:.3f} "
              f"jit8={res[k]['combo_miss_jitter8']:.3f} "
              f"static id/vote={res[k]['static_err_id']:.3f}/{res[k]['static_err_vote8']:.3f} "
              f"agree={res[k]['view_agree8']:.3f} n={len(combos)}", flush=True)
    print("DONE X02")


if __name__ == "__main__":
    main()
