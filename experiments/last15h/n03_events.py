#!/usr/bin/env python3
"""N03 round1b: oscillating in-and-out paths (0->1->0 by construction).

Waypoints [x, x+e_in, x]; e_in aims the moving segment through the other
(verified crossing with margin floor). Model never consulted at build time.
Endpoints are the same scene, so 'both endpoints correct' = model right on x.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
import paths
from paths import oracle_at, scan_polyline
from common import load_model, preprocess

ART = ROOT / "artifacts" / "discovery_campaign"
MARGIN_FLOOR = 0.005


def build_oscillating(x, rng):
    """Returns (waypoints, family) or None. Inward leg must reach label 1."""
    q_ab, q_cd = paths._seg_closest(x[0], x[1], x[2], x[3])
    gap = float(np.linalg.norm(q_ab - q_cd))
    if gap < 1e-6:
        return None
    axis = (q_ab - q_cd) / gap
    cands = []
    for nodes, tgt in (((2, 3), 1.0), ((0, 1), -1.0)):
        for over in (0.10, 0.20, 0.35, 0.55):
            e = np.zeros((4, 2))
            e[list(nodes)] = (gap + over) * axis * tgt
            try:
                y, m, _ = oracle_at(x + e)
            except ValueError:
                continue
            if y == 1 and m >= MARGIN_FLOOR:
                cands.append((e, f"osc_{'CD' if nodes == (2, 3) else 'AB'}"))
                break
    if not cands:
        return None
    e, fam = cands[int(rng.integers(len(cands)))]
    return [x, x + e, x], fam


def build_control(x, rng):
    q_ab, q_cd = paths._seg_closest(x[0], x[1], x[2], x[3])
    gap = float(np.linalg.norm(q_ab - q_cd))
    if gap < 1e-6:
        return None
    axis = (q_ab - q_cd) / gap
    perp = np.array([-axis[1], axis[0]])
    e = np.zeros((4, 2))
    e[2, :] = 0.3 * perp
    e[3, :] = 0.3 * perp
    for tt in np.linspace(0, 1, 65):
        for w in (x + tt * e, x + (1 - tt) * e):
            try:
                if oracle_at(w)[0] != 0:
                    return None
            except ValueError:
                return None
    return [x, x + e, x]


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
    p.add_argument("--n_want", type=int, default=256)
    p.add_argument("--pools", nargs="+", default=["eval_202"])
    p.add_argument("--model", type=Path, default=ART / "r04b_s11" / "clean")
    a = p.parse_args()
    rng = np.random.default_rng(3)
    Xs = []
    for pool in a.pools:
        dd = np.load(ART / "scenes" / pool / "scenes.npz")
        Xp = dd["positions"].astype(float)
        Xs += [(pool, i, Xp[i]) for i in range(len(Xp))]
    rng.shuffle(Xs)
    model, stats = load_model(a.model)
    model.eval()

    def predict(xs):
        with torch.no_grad():
            lg = model(preprocess(xs, stats)).numpy()
        return lg, (lg > 0).astype(int)

    events, controls = [], []
    for pool, pi, x in [t for t in Xs]:
        if len(events) >= a.n_want and len(controls) >= a.n_want:
            break
        x = np.asarray(x, dtype=float)
        try:
            if oracle_at(x)[0] != 0:
                continue
        except ValueError:
            continue
        if len(events) < a.n_want:
            r = build_oscillating(x, rng)
            if r is not None:
                wps, fam = r
                sc = scan_polyline([w.astype(np.float32) for w in wps], predict, n_per_leg=65)
                eps = episodes_of(sc["oracle_label"])
                # dwell>=3 eval samples, mid margin floor
                good = [ep for ep in eps
                        if (ep[1] - ep[0] + 1) >= 3
                        and sc["oracle_margin"][ep[0]:ep[1] + 1].min() >= MARGIN_FLOOR]
                if good:
                    events.append({"scan": sc, "episodes": good,
                                   "parent_id": f"{pool}:{pi}",
                                   "family": fam})
        if len(controls) < a.n_want:
            wps = build_control(x, rng)
            if wps is not None:
                sc = scan_polyline([w.astype(np.float32) for w in wps], predict, n_per_leg=65)
                if (sc["oracle_label"] == 0).all():
                    controls.append({"scan": sc,
                                     "parent_id": f"{pool}:{pi}"})
    # metrics, per-episode, endpoints-correct-conditional
    cond_n = cond_miss = det = 0
    entry_err, exit_err = [], []
    full_ok = 0
    per_pool = {}
    for ev in events:
        sc = ev["scan"]
        pool = ev["parent_id"].split(":")[0]
        pp = per_pool.setdefault(pool, {"cond": 0, "miss": 0, "n": 0, "full": 0})
        pp["n"] += 1
        if sc["pred_label"][0] != 0:
            continue  # endpoint (same scene) wrong -> out of conditional denom
        cond_n += 1
        pp["cond"] += 1
        if (sc["pred_label"] == sc["oracle_label"]).all():
            full_ok += 1
            pp["full"] += 1
        for (i0, j0) in ev["episodes"]:
            pmid = sc["pred_label"][i0:j0 + 1].max()
            if pmid == 1:
                det += 1
                pe = int(np.nonzero(sc["pred_label"] == 1)[0][0])
                px = int(np.nonzero(sc["pred_label"] == 1)[0][-1])
                entry_err.append((pe - i0) / (len(sc["t"]) - 1))
                exit_err.append((px - j0) / (len(sc["t"]) - 1))
        # episode-level miss: none of the episodes detected
        detected_any = any(sc["pred_label"][i0:j0 + 1].max() == 1 for i0, j0 in ev["episodes"])
        cond_miss += (not detected_any)
        pp["miss"] += (not detected_any)
    fa = sum(1 for c in controls if c["scan"]["pred_label"].max() == 1)
    res = {"n_events": len(events), "n_controls": len(controls),
           "cond_paths": cond_n,
           "cond_path_miss_rate": (cond_miss / cond_n) if cond_n else None,
           "mean_abs_entry_err": float(np.mean(np.abs(entry_err))) if entry_err else None,
           "mean_abs_exit_err": float(np.mean(np.abs(exit_err))) if exit_err else None,
           "full_path_correct_rate": full_ok / len(events) if events else None,
           "control_false_alarm_rate": fa / len(controls) if controls else None,
           "per_pool": {k: {"n": v["n"], "cond": v["cond"],
                              "cond_miss": (v["miss"] / v["cond"]) if v["cond"] else None,
                              "full_rate": (v["full"] / v["n"]) if v["n"] else None}
                        for k, v in per_pool.items()}}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print(f"SAW: events={len(events)} controls={len(controls)} "
          f"cond_path_miss={res['cond_path_miss_rate']} "
          f"false_alarm={res['control_false_alarm_rate']}")
    print("NEXT: if cond miss high -> entry/exit supervision trial (N01-style "
          "brackets at event edges); if ~0 -> N03 phenomenon absent, drop")
    print("CLAIM: endpoint-correct models can still miss mid-path events")


if __name__ == "__main__":
    main()
