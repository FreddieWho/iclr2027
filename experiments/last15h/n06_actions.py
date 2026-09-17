#!/usr/bin/env python3
"""N06 round1: action selection from fixed library, zero training."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
from paths import action_library, oracle_at
from common import load_model, preprocess

ART = ROOT / "artifacts" / "discovery_campaign"
MODELS = {"clean": ART / "r04b_s11" / "clean",
          "flipmine": ART / "r04b_s11" / "flipmine",
          "unifmatched": ART / "r04c_budget" / "unifmatched"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_parents", type=int, default=256)
    p.add_argument("--lam_cost", type=float, default=2.0)
    a = p.parse_args()
    rng = np.random.default_rng(6)
    d = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    X = d["positions"].astype(np.float32)
    lib = action_library()
    loaded = {}
    for k, mp in MODELS.items():
        m, s = load_model(mp)
        m.eval()
        loaded[k] = (m, s)

    cross = [i for i in rng.permutation(len(X))[:a.n_parents * 2]
             if oracle_at(X[i])[0] == 1][:a.n_parents]
    out = {}
    for k, (m, s) in loaded.items():
        inv = suc = feas = tot = 0
        regret = []
        for pi in cross:
            x = X[pi].astype(float)
            cands = []
            for j, act in enumerate(lib):
                try:
                    y, _, _ = oracle_at(x + act["edit"])
                except ValueError:
                    continue
                cands.append((j, act, y))
            feas_oracle = [c for c in cands if c[2] == 0]
            if not feas_oracle:
                continue  # infeasible scene, reported via count below
            feas += 1
            xs = np.stack([(x + c[1]["edit"]).astype(np.float32) for c in cands])
            with torch.no_grad():
                lg = m(preprocess(xs, s)).numpy()
            prob0 = 1 / (1 + np.exp(lg))  # P(target class 0)
            scores = prob0 - a.lam_cost * np.array([c[1]["cost"] for c in cands])
            jstar = int(np.argmax(scores))
            ystar = cands[jstar][2]
            tot += 1
            if ystar != 0:
                inv += 1
            else:
                suc += 1
                best = min(c[1]["cost"] for c in feas_oracle)
                regret.append(cands[jstar][1]["cost"] - best)
        out[k] = {"n_scenes": len(cross), "n_feasible": feas,
                  "invalid_rate": inv / tot if tot else None,
                  "success_rate": suc / tot if tot else None,
                  "mean_regret": float(np.mean(regret)) if regret else None}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out / "result.json", "w"), indent=1)
    for k, v in out.items():
        print(f"SAW {k}: invalid={v['invalid_rate']} success={v['success_rate']} "
              f"regret={v['mean_regret']} feas={v['n_feasible']}/{v['n_scenes']}")
    print("NEXT: if clean invalid high but flipmine low -> boundary fix improves "
          "decisions; then extend to full-path actions with N03 scenes")
    print("CLAIM: close classifiers can give very different advice")


if __name__ == "__main__":
    main()
