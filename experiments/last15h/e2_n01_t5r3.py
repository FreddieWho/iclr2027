#!/usr/bin/env python3
"""E2-N01: single-turn zone paths on T5R3 (WOY valid), frozen vs cover vs cleanonly.

Mirrors coord N01 round1: aimed home-x sweep ending flipped (margin>=0.03),
dense oracle single-turn check, 17-pt model scan, turn match + integral error.
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
from paths import match_turns

MATCH = "J03WOY"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_paths", type=int, default=256)
    p.add_argument("--seed", type=int, default=11)
    a = p.parse_args()
    rng = np.random.default_rng(21)
    raw, team, mids, hcx = T.load_views()
    home_idx = (team[0] == 0).nonzero()[0]
    team1 = team[0]
    pool = np.where(mids == MATCH)[0]
    assert len(pool) >= a.n_paths, f"pool {len(pool)} < {a.n_paths}"

    models = {"frozen": T.load_t5r3(T.base_ckpt(a.seed)),
              "cover": T.load_t5r3(T.cover_file(a.seed, "cover")),
              "cleanonly": T.load_t5r3(T.cover_file(a.seed, "cleanonly"))}
    oturns_of = T.make_oracle_turns(home_idx)

    paths = []
    for pi in rng.permutation(pool)[:a.n_paths * 3]:
        if len(paths) >= a.n_paths:
            break
        x = raw[pi].astype(float)
        r = T.aimed_home_shift(x, home_idx)
        if r is None:
            continue
        e, znew = r
        t = np.linspace(0, 1, 65)
        x_of_t = lambda tt, _x=x, _e=e: _x + tt * _e
        ot = oturns_of(x_of_t, t)
        if len(ot) != 1:
            continue
        paths.append({"parent": int(pi), "edit": e, "t_star": ot[0]["t_star"],
                      "old": ot[0]["old"], "new": ot[0]["new"]})
    print(f"built {len(paths)} single-turn paths", flush=True)

    res = {"n_paths": len(paths), "match": MATCH, "seed": a.seed}
    for name, m in models.items():
        mturns_of = T.make_model_turns(m, team1, home_idx)
        n_miss, errs, integ = 0, [], []
        for pp in paths:
            x = raw[pp["parent"]].astype(float)
            t = np.linspace(0, 1, 17)
            xs = np.stack([x + tt * pp["edit"] for tt in t]).astype(np.float32)
            probs = T.zone_probs(m, xs, np.stack([team1] * len(t)))
            plab = probs.argmax(1)
            olab = np.array([int(T.zone_of(float((x + tt * pp["edit"])[home_idx, 0].mean())))
                             for tt in t])
            integ.append(float((plab != olab).mean()))
            x_of_t = lambda tt, _x=x, _e=pp["edit"]: _x + tt * _e
            mt = mturns_of(x_of_t, t)
            same = [u for u in mt if u["old"] == pp["old"] and u["new"] == pp["new"]]
            if not same:
                n_miss += 1
            else:
                errs.append(abs(same[0]["t_theta"] - pp["t_star"]))
        got = [e for e in errs]
        res[name] = {"turn_miss": n_miss / len(paths) if paths else None,
                     "mean_abs_terr": float(np.mean(got)) if got else None,
                     "mean_integral": float(np.mean(integ)) if integ else None,
                     "n": len(paths)}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    for k in models:
        v = res[k]
        print(f"SAW {k}: turn_miss={v['turn_miss']} mean|terr|={v['mean_abs_terr']} "
              f"integ={v['mean_integral']} (n={v['n']})")
    print("NEXT: if frozen misses like coord-clean (high) and cover cuts it -> "
          "T5R3 turn story holds; then expand seeds")
    print("CLAIM: static zone accuracy does not guarantee turn location on real data")


if __name__ == "__main__":
    main()
