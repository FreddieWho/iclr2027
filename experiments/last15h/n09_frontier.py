#!/usr/bin/env python3
"""N09 round1: C0-keep vs C1-miss frontier of existing checkpoints (no training)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
from paths import oracle_at
from common import load_model, preprocess
from r02_search import candidates_for_scene

ART = ROOT / "artifacts" / "discovery_campaign"
MODELS = {"clean": ART / "r04b_s11" / "clean",
          "flipmine": ART / "r04b_s11" / "flipmine",
          "fliprand": ART / "r04b_s11" / "fliprand",
          "unifmatched": ART / "r04c_budget" / "unifmatched"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(9)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(np.float32)
    mine = np.load(ART / "r04c_budget" / "mined.npz")
    meta = mine["meta"]
    # C1: mined flips (recompute oracle labels, keep true changes)
    C1 = []
    for i in range(len(meta)):
        if f"edit_{i}" not in mine:
            break
        pi = int(meta[i, 0])
        x = X[pi].astype(float)
        e = np.asarray(mine[f"edit_{i}"], dtype=float)
        try:
            y0, _, _ = oracle_at(x)
            y1, m1, _ = oracle_at(x + e)
        except ValueError:
            continue
        if y1 != y0 and m1 >= 0.005:
            C1.append((x, e, y1))
    # C0: label-preserving edits, norm-matched to C1 costs
    costs = sorted(np.linalg.norm(e) for _, e, _ in C1)
    C0 = []
    for (x, _, _), c in zip(C1, costs):
        y0, _, _ = oracle_at(x)
        for e, _ in candidates_for_scene(x, rng, n_random=32):
            e = np.asarray(e, dtype=float)
            if abs(np.linalg.norm(e) - c) > 0.02:
                continue
            try:
                y, m, _ = oracle_at(x + e)
            except ValueError:
                continue
            if y == y0 and m >= 0.005:
                C0.append((x, e, y0))
                break
    out = {"n_C1": len(C1), "n_C0": len(C0)}
    # honest-denominator arm: C1/C0 on EVAL parents (oracle-only mining)
    de = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    Xe = de["positions"].astype(float)
    eC1, eC0 = [], []
    for pi in rng.permutation(len(Xe)):
        if len(eC1) >= 400:
            break
        x = Xe[pi]
        try:
            y0, _, _ = oracle_at(x)
        except ValueError:
            continue
        for e, _ in candidates_for_scene(x, rng, n_random=16):
            e = np.asarray(e, dtype=float)
            try:
                y, m, _ = oracle_at(x + e)
            except ValueError:
                continue
            if m < 0.005:
                continue
            if y != y0 and len(eC1) < 400:
                eC1.append((x, e, y))
                break
    ecosts = sorted(np.linalg.norm(e) for _, e, _ in eC1)
    for (x, _, _), c in zip(eC1, ecosts):
        y0, _, _ = oracle_at(x)
        for e, _ in candidates_for_scene(x, rng, n_random=32):
            e = np.asarray(e, dtype=float)
            if abs(np.linalg.norm(e) - c) > 0.02:
                continue
            try:
                y, m, _ = oracle_at(x + e)
            except ValueError:
                continue
            if y == y0 and m >= 0.005:
                eC0.append((x, e, y0))
                break
    out["n_eval_C1"] = len(eC1)
    out["n_eval_C0"] = len(eC0)
    for k, mp in MODELS.items():
        m, s = load_model(mp)
        m.eval()
        def err(pairs):
            xs = np.stack([(x + e).astype(np.float32) for x, e, _ in pairs])
            ys = np.array([y for _, _, y in pairs])
            with torch.no_grad():
                lg = m(preprocess(xs, s)).numpy()
            return float((((lg > 0).astype(int)) != ys).mean())
        # clean error on parents
        xp = np.stack([x.astype(np.float32) for x, _, _ in C1])
        yp = np.array([oracle_at(x)[0] for x, _, _ in C1])
        with torch.no_grad():
            lgp = m(preprocess(xp, s)).numpy()
        out[k] = {"C1_miss": err(C1), "C0_false_alarm": err(C0),
                  "clean_err": float((((lgp > 0).astype(int)) != yp).mean()),
                  "eval_C1_miss": err(eC1) if eC1 else None,
                  "eval_C0_false_alarm": err(eC0) if eC0 else None}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump({k: (v if isinstance(v, dict) else v) for k, v in out.items()},
              open(a.out / "result.json", "w"), indent=1)
    for k in MODELS:
        v = out[k]
        print(f"SAW {k}: C1_miss={v['C1_miss']} C0_FA={v['C0_false_alarm']} "
              f"clean={v['clean_err']} | eval_C1={v['eval_C1_miss']} eval_C0={v['eval_C0_false_alarm']}")
    print("NEXT: if flipmine cut C1_miss at visible C0 cost -> constrained "
          "primal-dual trial on conflict boundaries; if no tradeoff visible -> "
          "use flipmine as-is, drop N09 method")
    print("CLAIM: semantic decomposition of the robustness tradeoff")


if __name__ == "__main__":
    main()
