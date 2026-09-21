#!/usr/bin/env python3
"""WP-D: contested-event selectivity with motion-matched controls.

Per contested event, pre-event window covariates (ball speed/accel, player
mean/max displacement, centroid displacement, spread change, ball location,
match time, pre-event entropy/pred). NN-matched non-event controls, SMD<0.10
gate. S = P(turn|event) - P(turn|control), paired bootstrap 10k (unit=pair).
Event-triggered curves -3..+3 for event/control x frozen/cover.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
torch.set_num_threads(4)
import t5r3_paths as T  # noqa: E402

VIEW = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1" / "data_views"
ART = ROOT / "artifacts" / "data_v2" / "idsse" / "canonical"
OUT = ROOT / "artifacts" / "l002_ball"
rng = np.random.default_rng(26092211)

COVS = ["ball_speed", "ball_accel", "p_mean_disp", "p_max_disp",
        "cent_disp", "spread_change", "ball_x", "ball_y", "match_t", "entropy"]


def load_match(match, split, seed):
    import pandas as pd
    X = np.load(VIEW / f"positions_raw_{split}.npy").astype(np.float32)
    Tm = np.load(VIEW / f"team_slots_{split}.npy").astype(np.int64)
    idx = pd.read_parquet(VIEW / f"snapshot_index_{split}.parquet")
    sel = (idx.source_match_id == match).values
    X, Tm = X[sel], Tm[sel]
    idx = idx[sel].reset_index(drop=True)
    ts = idx.timestamp_ms.values.astype(np.int64)
    order = np.argsort(ts)
    X, Tm, idx, ts = X[order], Tm[order], idx.iloc[order], ts[order]
    fr = pd.read_parquet(ART / "sampled_frames" / f"{match}.parquet")
    ball = {}
    for t0, g in fr.groupby("timestamp_ms"):
        b = g[g.entity_type == "ball"]
        if len(b):
            ball[int(t0)] = (float(b.x.values[0]), float(b.y.values[0]))
    bx = np.array([ball.get(int(t), (np.nan, np.nan))[0] for t in ts])
    by = np.array([ball.get(int(t), (np.nan, np.nan))[1] for t in ts])
    models = {"frozen": T.load_t5r3(T.base_ckpt(seed)),
              "cover": T.load_t5r3(ROOT / "artifacts" / "discovery_campaign" /
                                   f"r04b_t5r3_s{seed}" / "cover.pt")}
    probs = {k: T.zone_probs(m, X, Tm) for k, m in models.items()}
    pred = {k: v.argmax(1) for k, v in probs.items()}
    ent = {k: float((-v[i] * np.log(v[i] + 1e-12)).sum()) for k, v in probs.items() for i in range(len(v))}
    ent = {k: np.array([float((-probs[k][i] * np.log(probs[k][i] + 1e-12)).sum()) for i in range(len(ts))]) for k in probs}
    return {"X": X, "ts": ts, "bx": bx, "by": by, "pred": pred, "ent": ent}


def feats_at(d, i):
    """Covariates from pre-event window [i-3, i-1]; None if unavailable."""
    if i < 4:
        return None
    X, ts, bx, by = d["X"], d["ts"], d["bx"], d["by"]
    if not (np.isfinite(bx[i - 3:i + 1]).all() and np.isfinite(by[i - 3:i + 1]).all()):
        return None
    dt = np.diff(ts[i - 3:i + 1]) / 1000.0 + 1e-9
    vx = np.diff(bx[i - 3:i + 1]) / dt
    vy = np.diff(by[i - 3:i + 1]) / dt
    spd = np.sqrt(vx ** 2 + vy ** 2)
    acc = np.abs(np.diff(spd) / dt[1:]).mean() if len(spd) > 1 else 0.0
    disp = np.linalg.norm(np.diff(X[i - 3:i + 1], axis=0), axis=2)
    cent = np.linalg.norm(X[i - 3:i + 1, :, :].mean(1)[1:] - X[i - 3:i + 1, :, :].mean(1)[:-1], axis=1)
    spread = X[i - 3:i + 1].reshape(4, -1).std(1)
    return np.array([spd.mean(), acc, disp.mean(), disp.max(), cent.mean(),
                     abs(spread[-1] - spread[0]), bx[i], by[i],
                     float(ts[i] - ts[0]) / 1e6, d["ent"]["frozen"][i]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=11)
    a = p.parse_args()
    import pandas as pd
    all_pairs = []
    curves = {}
    for match, split in (("J03WOY", "valid"), ("J03WMX", "train")):
        d = load_match(match, split, a.seed)
        ts = d["ts"]
        ev = pd.read_parquet(ART / "events" / f"{match}.parquet")
        cont = ev[ev.event_type.isin(["TacklingGame", "BallClaiming"])]
        ev_idx = set()
        for t0 in cont.timestamp_ms.values:
            i = int(np.abs(ts - int(t0)).argmin())
            if abs(ts[i] - int(t0)) <= 2000:
                ev_idx.add(i)
        ev_idx = sorted(ev_idx)
        # candidate controls: snapshots far from any event
        bad = set()
        for t0 in ev.timestamp_ms.values:
            i = int(np.abs(ts - int(t0)).argmin())
            bad.update(range(max(0, i - 10), min(len(ts), i + 11)))
        cand = [i for i in range(4, len(ts)) if i not in bad]
        F_cand = []
        for i in cand:
            f = feats_at(d, i)
            if f is not None:
                F_cand.append((i, f))
        if not F_cand:
            continue
        C = np.array([f for _, f in F_cand])
        mu, sd = C.mean(0), C.std(0) + 1e-12
        used = set()
        for i in ev_idx:
            f = feats_at(d, i)
            if f is None:
                continue
            z = (f - mu) / sd
            Zn = (C - mu) / sd
            dists = sorted(((float(((Zn[j] - z) ** 2).sum()), j) for j in range(len(F_cand)) if F_cand[j][0] not in used))
            if not dists:
                continue
            used.add(F_cand[dists[0][1]][0])
            all_pairs.append({"match": match, "event_i": i,
                              "control_i": F_cand[dists[0][1]][0],
                              "event_f": f.tolist(),
                              "control_f": F_cand[dists[0][1]][1].tolist()})
        # event-triggered curves
        for kind, idxs in (("event", ev_idx), ("control", sorted(used))):
            for k in ("frozen", "cover"):
                key = f"{match}_{kind}_{k}"
                arr = []
                for tau in range(-3, 4):
                    vals = []
                    for i in idxs:
                        j = i + tau
                        if 1 <= j < len(ts):
                            vals.append(int(d["pred"][k][j] != d["pred"][k][j - 1]))
                    arr.append(round(float(np.mean(vals)), 4) if vals else None)
                curves[key] = arr
    # SMD gate
    smds = {}
    for c, name in enumerate(COVS):
        e = np.array([p["event_f"][c] for p in all_pairs])
        cc = np.array([p["control_f"][c] for p in all_pairs])
        smds[name] = float(abs(e.mean() - cc.mean()) / max(1e-12, np.sqrt((e.var() + cc.var()) / 2)))
    res = {"n_pairs": len(all_pairs),
           "smd": {k: round(v, 4) for k, v in smds.items()},
           "smd_pass": bool(all(v < 0.10 for v in smds.values()))}
    # S needs turn flags: recompute per-match turn arrays (cheap to reload)
    S = {}
    for match, split in (("J03WOY", "valid"), ("J03WMX", "train")):
        d = load_match(match, split, a.seed)
        tn = {k: np.array([0] + [int(d["pred"][k][i] != d["pred"][k][i - 1]) for i in range(1, len(d["ts"]))]) for k in ("frozen", "cover")}
        mp = [p for p in all_pairs if p["match"] == match]
        for k in ("frozen", "cover"):
            de, dc = [], []
            for p in mp:
                i, j = p["event_i"], p["control_i"]
                de.append(int(tn[k][max(0, i - 2):i + 3].max()))
                dc.append(int(tn[k][max(0, j - 2):j + 3].max()))
            de, dc = np.array(de), np.array(dc)
            dd = de - dc
            lo, hi = np.quantile(rng.choice(dd, size=(10000, len(dd))).mean(0), [0.025, 0.975]) if len(dd) else (None, None)
            S[f"{match}_{k}"] = {"n": len(dd), "P_event": round(float(de.mean()), 4) if len(dd) else None,
                                "P_control": round(float(dc.mean()), 4) if len(dd) else None,
                                "S": round(float(dd.mean()), 4) if len(dd) else None,
                                "CI": [round(float(lo), 4), round(float(hi), 4)] if len(dd) else None}
    res["S"] = S
    # turn outcome in +/-2 window (reuse pilot zone preds? recompute cheap flags)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"pairs": all_pairs, "curves": curves}, open(OUT / "matched_pairs.json", "w"))
    json.dump(res, open(OUT / "MATCHED_SUMMARY.json", "w"), indent=1)
    print(f"SAW WP-D match: n_pairs={len(all_pairs)} smd_pass={res['smd_pass']}")
    print(json.dumps(res["smd"], indent=1))


if __name__ == "__main__":
    main()
