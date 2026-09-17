#!/usr/bin/env python3
"""E2-N03: oscillating zone events on T5R3 (WOY valid).

Waypoints [x, x+e_in, x] with middle flipped (margin>=0.03); controls =
home-y in-and-out verified zone-preserving. Frame-by-frame zone preds.
Mirrors coord N03 round1d metrics.
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
N_PER_LEG = 33


def scan_waypoints(model, wps, team1):
    xs = []
    for k in range(len(wps) - 1):
        s = np.linspace(0, 1, N_PER_LEG)
        xs.append((1 - s)[:, None, None] * wps[k][None] +
                  s[:, None, None] * wps[k + 1][None])
    xs = np.concatenate(xs, axis=0).astype(np.float32)
    probs = T.zone_probs(model, xs, np.stack([team1] * len(xs)))
    return probs.argmax(1), xs


def episodes_of(olab):
    eps, i, n = [], 0, len(olab)
    while i < n:
        if olab[i] == 1 and (i == 0 or olab[i - 1] == 0):
            j = i
            while j + 1 < n and olab[j + 1] == 1:
                j += 1
            if j + 1 < n and olab[j + 1] == 0:
                eps.append((i, j))
            i = j + 1
        else:
            i += 1
    return eps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_want", type=int, default=200)
    p.add_argument("--seed", type=int, default=11)
    a = p.parse_args()
    rng = np.random.default_rng(22)
    raw, team, mids, hcx = T.load_views()
    home_idx = (team[0] == 0).nonzero()[0]
    team1 = team[0]
    pool = np.where(mids == MATCH)[0]
    models = {"frozen": T.load_t5r3(T.base_ckpt(a.seed)),
              "cover": T.load_t5r3(T.cover_file(a.seed, "cover")),
              "cleanonly": T.load_t5r3(T.cover_file(a.seed, "cleanonly"))}

    def oracle_seq(xs):
        return np.array([int(T.zone_of(float(xx[home_idx, 0].mean()))) for xx in xs])

    events, controls = [], []
    for pi in rng.permutation(pool):
        if len(events) >= a.n_want and len(controls) >= a.n_want:
            break
        x = raw[pi].astype(float)
        z0 = int(T.zone_of(float(x[home_idx, 0].mean())))
        if len(events) < a.n_want:
            r = T.aimed_home_shift(x, home_idx)
            if r is not None:
                e, _ = r
                wps = [x, x + e, x]
                xs = []
                for k in range(2):
                    s = np.linspace(0, 1, N_PER_LEG)
                    xs.append((1 - s)[:, None, None] * wps[k][None] +
                              s[:, None, None] * wps[k + 1][None])
                xs = np.concatenate(xs, axis=0)
                ol = oracle_seq(xs)
                changed = (ol != z0).astype(int)
                eps = episodes_of(changed)
                good = [ep for ep in eps if (ep[1] - ep[0] + 1) >= 3]
                if good:
                    events.append({"wps": [w.astype(np.float32) for w in wps],
                                   "episodes": good, "z0": z0, "parent": int(pi)})
        if len(controls) < a.n_want:
            e = np.zeros_like(x)
            e[home_idx, 1] = 0.3
            wps = [x, x + e, x]
            xs = []
            for k in range(2):
                s = np.linspace(0, 1, N_PER_LEG)
                xs.append((1 - s)[:, None, None] * wps[k][None] +
                          s[:, None, None] * wps[k + 1][None])
            xs = np.concatenate(xs, axis=0)
            if (oracle_seq(xs) == z0).all():
                controls.append({"wps": [w.astype(np.float32) for w in wps],
                                 "parent": int(pi)})
    print(f"events={len(events)} controls={len(controls)}", flush=True)

    res = {"n_events": len(events), "n_controls": len(controls),
           "match": MATCH, "seed": a.seed}
    for name, model in models.items():
        res[name] = eval_model(model, events, controls, team1, oracle_seq)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    for name in models:
        v = res[name]
        print(f"SAW {name}: cond_miss={v['cond_path_miss']} full={v['full_path_rate']} "
              f"FA={v['control_FA']} (cond_n={v['cond_paths']})")
def eval_model(model, events, controls, team1, oracle_seq):
    cond_n = cond_miss = full_ok = 0
    entry_err, exit_err = [], []
    for ev in events:
        plab, _ = scan_waypoints(model, ev["wps"], team1)
        # oracle on the same scanned frames
        xs = []
        for k in range(2):
            s = np.linspace(0, 1, N_PER_LEG)
            xs.append((1 - s)[:, None, None] * ev["wps"][k][None] +
                      s[:, None, None] * ev["wps"][k + 1][None])
        xs = np.concatenate(xs, axis=0)
        ol = oracle_seq(xs)
        z0 = ev["z0"]
        if plab[0] != z0:
            continue
        cond_n += 1
        if (plab == ol).all():
            full_ok += 1
        # detected = model leaves z0 somewhere inside an event episode
        detected = any((plab[i0:j0 + 1] != z0).any() for i0, j0 in ev["episodes"])
        cond_miss += (not detected)
        if detected:
            fir = [i for i0, j0 in ev["episodes"] for i in range(i0, j0 + 1)
                   if plab[i] != z0]
            if fir:
                entry_err.append((min(fir) - ev["episodes"][0][0]) / (len(plab) - 1))
                exit_err.append((max(fir) - ev["episodes"][-1][1]) / (len(plab) - 1))
    fa = 0
    for ct in controls:
        plab, _ = scan_waypoints(model, ct["wps"], team1)
        if (plab != plab[0]).any():
            fa += 1
    return {"cond_paths": cond_n,
            "cond_path_miss": cond_miss / cond_n if cond_n else None,
            "full_path_rate": full_ok / len(events) if events else None,
            "mean_abs_entry_err": float(np.mean(np.abs(entry_err))) if entry_err else None,
            "mean_abs_exit_err": float(np.mean(np.abs(exit_err))) if exit_err else None,
            "control_FA": fa / len(controls) if controls else None}


if __name__ == "__main__":
    main()
