#!/usr/bin/env python3
"""L-002 pilot v2: E2 views + events/ball join (frozen + cover, seed 11).

6127 view snapshots (WOY 5702 + WN1 425) with timestamp_ms; join WOY events
(nearest snapshot <=2000ms); ball pos + possession from sampled_frames.
Main: pass outcome-conditioned model turns. S1: possession flips. S2:
contested events. S3: ball speed vs turns in stable windows.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--match", default="J03WOY")
    p.add_argument("--split", default="valid")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "l002_ball")
    a = p.parse_args()
    import pandas as pd
    X = np.load(VIEW / f"positions_raw_{a.split}.npy").astype(np.float32)
    Tm = np.load(VIEW / f"team_slots_{a.split}.npy").astype(np.int64)
    idx = pd.read_parquet(VIEW / f"snapshot_index_{a.split}.parquet")
    wo = (idx.source_match_id == a.match).values
    X, Tm = X[wo], Tm[wo]
    idx = idx[wo].reset_index(drop=True)
    ts = idx.timestamp_ms.values.astype(np.int64)
    order = np.argsort(ts)
    X, Tm, idx = X[order], Tm[order], idx.iloc[order].reset_index(drop=True)
    ts = ts[order]
    models = {"frozen": T.load_t5r3(T.base_ckpt(a.seed)),
              "cover": T.load_t5r3(ROOT / "artifacts" / "discovery_campaign" /
                                    f"r04b_t5r3_s{a.seed}" / "cover.pt")}
    probs = {k: T.zone_probs(m, X, Tm) for k, m in models.items()}
    pred = {k: v.argmax(1) for k, v in probs.items()}
    turn = {k: np.array([0] + [int(pred[k][i] != pred[k][i - 1])
                               for i in range(1, len(pred[k]))]) for k in probs}
    # ball + possession per snapshot ts
    fr = pd.read_parquet(ART / "sampled_frames" / f"{a.match}.parquet")
    ball = {}
    poss = {}
    bstat = {}
    for t0, g in fr.groupby("timestamp_ms"):
        b = g[g.entity_type == "ball"]
        if len(b):
            ball[int(t0)] = (float(b.x.values[0]), float(b.y.values[0]))
            bs = b.ball_status.dropna().values
            if len(bs):
                bstat[int(t0)] = int(bs[0])
        pl = g[g.entity_type == "player"]
        if len(pl):
            mv = pl.ball_possession.dropna().mode().values
            poss[int(t0)] = str(mv[0]) if len(mv) else "?"
    bx = np.array([ball.get(int(t), (np.nan, np.nan))[0] for t in ts])
    by = np.array([ball.get(int(t), (np.nan, np.nan))[1] for t in ts])
    poss_a = np.array([poss.get(int(t), "?") for t in ts])
    ev = pd.read_parquet(ART / "events" / f"{a.match}.parquet").sort_values("timestamp_ms")

    def window_turn(t0, w=2):
        i = int(np.abs(ts - t0).argmin())
        if abs(ts[i] - t0) > 2000:
            return None
        lo, hi = max(1, i - w), min(len(ts) - 1, i + w)
        return {k: int(turn[k][lo:hi + 1].max()) for k in turn}

    res = {"n_snaps": len(ts), "n_ball_hit": int(np.isfinite(bx).sum())}
    passes = ev[ev.event_type.str.contains("Pass", na=False)]
    for oc in ["successfullyCompleted", "unsuccessful"]:
        sub = passes[passes.outcome == oc]
        hits = {k: [] for k in turn}
        n = 0
        for t0 in sub.timestamp_ms.values:
            w = window_turn(int(t0))
            if w is None:
                continue
            n += 1
            for k in turn:
                hits[k].append(w[k])
        res[f"pass_{oc}"] = {"n": n, **{k: round(float(np.mean(hits[k])), 3) if hits[k] else None for k in turn}}
    flip_idx = [i for i in range(1, len(ts)) if bstat.get(int(ts[i]), -1) != bstat.get(int(ts[i - 1]), -2)]
    flip_idx = [i for i in flip_idx if int(ts[i]) in bstat and int(ts[i - 1]) in bstat]
    base_idx = [i for i in range(1, len(ts), 5) if bstat.get(int(ts[i]), -1) == bstat.get(int(ts[i - 1]), -2)]
    base_idx = [i for i in base_idx if int(ts[i]) in bstat]
    for nm, sidx in (("ballstatus_flip", flip_idx), ("ballstatus_stable", base_idx)):
        res[f"S1_{nm}"] = {"n": len(sidx), **{k: round(float(turn[k][sidx].mean()), 3) if len(sidx) else None for k in turn}}
    cont = ev[ev.event_type.isin(["TacklingGame", "BallClaiming"])]
    rng = np.random.default_rng(0)
    base_ts = rng.choice(ts, size=min(len(cont), 500), replace=False)
    base_hits = {k: [] for k in turn}
    base_n = 0
    for t0 in base_ts:
        w = window_turn(int(t0))
        if w is None:
            continue
        base_n += 1
        for k in turn:
            base_hits[k].append(w[k])
    res["S2_baseline"] = {"n": base_n, **{k: round(float(np.mean(base_hits[k])), 3) if base_hits[k] else None for k in turn}}
    hits = {k: [] for k in turn}
    n = 0
    for t0 in cont.timestamp_ms.values:
        w = window_turn(int(t0))
        if w is None:
            continue
        n += 1
        for k in turn:
            hits[k].append(w[k])
    res["S2_contested_presence"] = {"n": n, **{k: round(float(np.mean(hits[k])), 3) if hits[k] else None for k in turn}}
    dt = np.diff(ts) / 1000.0 + 1e-9
    spd = np.array([0.0] + list(np.sqrt(np.diff(bx) ** 2 + np.diff(by) ** 2) / dt))
    spd = np.where(np.isfinite(spd), spd, 0.0)
    stable = np.array([i for i in range(1, len(poss_a)) if poss_a[i] == poss_a[i - 1]])
    if len(stable):
        qs = np.quantile(spd[stable], [0.25, 0.5, 0.75])
        for b in range(4):
            sel = stable[(spd[stable] >= (qs[b - 1] if b else -1)) & (spd[stable] <= (qs[b] if b < 3 else 1e18))]
            res[f"S3_speedQ{b}"] = {"n": len(sel), **{k: round(float(turn[k][sel].mean()), 3) if len(sel) else None for k in turn}}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / f"pilot_{a.match}.json", "w"), indent=1)
    print(f"SAW L002 {a.match}: " + json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main()
