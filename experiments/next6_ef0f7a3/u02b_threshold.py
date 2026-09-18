#!/usr/bin/env python3
"""U02b: dev-chosen threshold rule (zero new forwards, reuses U02 cache).

Rescues the vacuous calib arm: raw-prob threshold tau swept on DEV scenes
(success on feasible scenes, tie-break lower refusal), fixed, then reported
on TEST. Same dev/test split as u02 (identical RNG replay). Rules compared:
score(lam), lexico(tau=0.5), tau-dev. Per model.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
from paths import action_library, oracle_at
from common import load_model, preprocess
import torch

torch.set_num_threads(2)
ART = ROOT / "artifacts" / "discovery_campaign"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=206)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    d = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    X = d["positions"].astype(float)
    lib = action_library()
    costs = np.array([c["cost"] for c in lib])
    cross = [i for i in rng.permutation(len(X))[:512]
             if oracle_at(X[i])[0] == 1][:256]
    c = np.load(ROOT / "artifacts" / "next6_ef0f7a3" / "u02" / "cache.npz")
    frames, feas, scene_of = c["frames"], c["feas"], c["scene_of"]
    uniq = np.array(sorted(set(scene_of)))
    rng.shuffle(uniq)
    dev_scenes = set(uniq[:len(uniq) // 3])
    te_scenes = [s for s in uniq if s not in dev_scenes]
    acost = costs[[next(i for i, act in enumerate(lib)
                         if np.allclose((X[s] + act["edit"]).astype(np.float32), f))
                   for s, f in zip(scene_of, frames)]]

    def rule_stats(P0, scenes, tau):
        suc = tot = exc = ref = 0
        for s in scenes:
            idx = np.nonzero(scene_of == s)[0]
            of = feas[idx]
            if not of.any():
                continue
            tot += 1
            cand = idx[P0[idx] >= tau]
            if len(cand) == 0:
                ref += 1
                continue
            j = cand[int(np.argmin(acost[cand]))]
            if feas[j]:
                suc += 1
                exc += acost[j] - acost[idx][of].min()
        return (suc / tot if tot else None, ref / tot if tot else None,
                (exc / suc) if suc else None, tot)

    res = {}
    for k, mp in (("clean", ART / "r04b_s11" / "clean"),
                  ("flipmine", ART / "r04b_s11" / "flipmine")):
        m, s = load_model(mp)
        m.eval()
        with torch.no_grad():
            lg = m(preprocess(frames, s)).numpy()
        P0 = 1 / (1 + np.exp(lg))
        dev_list = [s for s in uniq if s in dev_scenes]
        best, best_tau = (-1, 2), 0.5
        for tau in np.arange(0.05, 1.0, 0.05):
            su, re, _, _ = rule_stats(P0, dev_list, round(float(tau), 2))
            key = (su if su is not None else -1, -(re if re is not None else 2))
            if key > best:
                best, best_tau = key, round(float(tau), 2)
        te = rule_stats(P0, te_scenes, best_tau)
        lx = rule_stats(P0, te_scenes, 0.5)
        res[k] = {"dev_tau": best_tau, "dev_stats": rule_stats(P0, dev_list, best_tau)[:3],
                  "test_tau_rule": te[:3], "test_lexico": lx[:3], "n_test": te[3]}
        print(f"SAW {k}: dev_tau={best_tau} test tau-rule suc/ref/exc="
              f"{te[0]:.3f}/{te[1]:.3f}/{te[2]:.3f} vs lexico "
              f"{lx[0]:.3f}/{lx[1]:.3f}/{lx[2]:.3f}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print("DONE U02b")


if __name__ == "__main__":
    main()
