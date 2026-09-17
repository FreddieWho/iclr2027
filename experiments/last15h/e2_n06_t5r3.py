#!/usr/bin/env python3
"""E2-N06: action selection on T5R3 (WOY valid), frozen vs cover vs cleanonly.

Fixed library of home shifts; goal = nearest other zone (oracle-feasible
scenes only: aimed_home_shift must succeed). Model picks
argmax P(target) - lam_cost*|shift|; oracle evaluates. Mirrors coord N06.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
import t5r3_paths as T

MATCH = "J03WOY"
SHIFTS = [-0.40, -0.25, -0.16, -0.10, -0.05, 0.05, 0.10, 0.16, 0.25, 0.40]


def build_library(x, home_idx):
    lib = []
    for s in SHIFTS:
        e = np.zeros_like(x)
        e[home_idx, 0] = s
        lib.append((e, abs(s), f"homex_{s:+.2f}"))
    for s in (0.10, -0.10):
        e = np.zeros_like(x)
        e[home_idx, 1] = s
        lib.append((e, abs(s), f"homey_{s:+.2f}"))
    lib.append((np.zeros_like(x), 0.0, "noop"))
    return lib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_parents", type=int, default=256)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--lam_costs", type=float, nargs="+", default=[0.0, 2.0])
    p.add_argument("--hard", action="store_true",
                   help="flip/null/noop triplets: flip shift vs equal-magnitude "
                        "away shift vs noop; isolates flip-blindness -> bad advice")
    a = p.parse_args()
    rng = np.random.default_rng(23)
    raw, team, mids, hcx = T.load_views()
    home_idx = (team[0] == 0).nonzero()[0]
    team1 = team[0]
    pool = np.where(mids == MATCH)[0]
    models = {"frozen": T.load_t5r3(T.base_ckpt(a.seed)),
              "cover": T.load_t5r3(T.cover_file(a.seed, "cover")),
              "cleanonly": T.load_t5r3(T.cover_file(a.seed, "cleanonly"))}

    # oracle-feasible scenes + target zone + candidate oracle labels
    tasks = []
    for pi in rng.permutation(pool)[:a.n_parents * 3]:
        if len(tasks) >= a.n_parents:
            break
        x = raw[pi].astype(float)
        r = T.aimed_home_shift(x, home_idx)
        if r is None:
            continue
        _, ztarget = r
        z0 = int(T.zone_of(float(x[home_idx, 0].mean())))
        if ztarget == z0:
            continue
        cands = []
        if a.hard:
            r2 = T.aimed_home_shift(x, home_idx)
            if r2 is None:
                continue
            e_f, zt2 = r2
            if zt2 != ztarget:
                continue
            cf = float(np.abs(e_f[home_idx, 0]).mean())
            e_n = np.zeros_like(x)
            e_n[home_idx, 0] = -np.sign(e_f[home_idx, 0]) * cf
            for e, cost, fam in ((e_f, cf, "flip"), (e_n, cf, "null"),
                                 (np.zeros_like(x), 0.0, "noop")):
                xx = x + e
                z = int(T.zone_of(float(xx[home_idx, 0].mean())))
                cands.append({"edit": e.astype(np.float32), "cost": cost,
                              "fam": fam, "zone": z})
        else:
            for e, cost, fam in build_library(x, home_idx):
                try:
                    xx = x + e
                    z = int(T.zone_of(float(xx[home_idx, 0].mean())))
                except Exception:
                    continue
                cands.append({"edit": e.astype(np.float32), "cost": cost,
                              "fam": fam, "zone": z})
        if not any(c["zone"] == ztarget for c in cands):
            continue  # target unreachable within library
        tasks.append({"parent": int(pi), "z0": z0, "ztarget": int(ztarget),
                      "cands": cands})
    print(f"tasks={len(tasks)}", flush=True)

    res = {"n_tasks": len(tasks), "match": MATCH, "seed": a.seed}
    for name, m in models.items():
        # one batched forward per model over all candidates
        frames, owners = [], []
        for ti, t in enumerate(tasks):
            x = raw[t["parent"]].astype(np.float32)
            for ci, c in enumerate(t["cands"]):
                frames.append(x + c["edit"])
                owners.append((ti, ci))
        probs = T.zone_probs(m, np.stack(frames), np.stack([team1] * len(frames)))
        res[name] = {}
        for lam in a.lam_costs:
            inv = suc = 0
            regret = []
            k = 0
            for ti, t in enumerate(tasks):
                nc = len(t["cands"])
                P = probs[k:k + nc, t["ztarget"]]
                k += nc
                costs = np.array([c["cost"] for c in t["cands"]])
                jstar = int(np.argmax(P - lam * costs))
                zstar = t["cands"][jstar]["zone"]
                if zstar != t["ztarget"]:
                    inv += 1
                else:
                    suc += 1
                    best = min(c["cost"] for c in t["cands"]
                               if c["zone"] == t["ztarget"])
                    regret.append(t["cands"][jstar]["cost"] - best)
            tot = inv + suc
            res[name][f"lam{lam:g}"] = {
                "invalid_rate": inv / tot if tot else None,
                "success_rate": suc / tot if tot else None,
                "mean_regret": float(np.mean(regret)) if regret else None,
                "n": tot}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    for k in models:
        for lam in a.lam_costs:
            v = res[k][f"lam{lam:g}"]
            print(f"SAW {k} lam{lam:g}: invalid={v['invalid_rate']} "
                  f"success={v['success_rate']} regret={v['mean_regret']} (n={v['n']})")
    print("NEXT: if frozen invalid high, cover lower (esp lam2) -> action "
          "consequence generalizes to real data; expand seeds")
    print("CLAIM: accurate zone classifiers still give invalid move advice")


if __name__ == "__main__":
    main()
