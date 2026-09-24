#!/usr/bin/env python3
"""D06 tracking probe (read-only analysis, no training).

Loads T5R3 valid views, reports shape/match inventory, slot-team
continuity, per-slot short-window displacement distribution (motion-cap
input), and runs the channel oracle on sampled frames with placeholder
fixed roles (p=home slot0, r=home slot1, D=all away slots) to show the
oracle executes on real data and to record the margin scale.
Role semantics ( passer/receiver from ball/events) is a separate design
step; match usage (WOY etc.) is recorded here for the D07 audit.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
from d06_oracle import channel_margin  # noqa: E402

VIEW = ROOT / "artifacts/phase3/task_semantic_repair_v1/data_views"


def load_views(split="valid"):
    import pandas as pd
    raw = np.load(VIEW / f"positions_raw_{split}.npy").astype(np.float32)
    team = np.load(VIEW / f"team_slots_{split}.npy").astype(np.int64)
    index = pd.read_parquet(VIEW / f"snapshot_index_{split}.parquet")
    mids = index["source_match_id"].astype(str).to_numpy()
    return raw, team, mids

OUT = ROOT / "artifacts" / "f095_campaign" / "D06"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="valid")
    p.add_argument("--frames", type=int, default=2000)
    a = p.parse_args()
    raw, team, mids = load_views(split=a.split)
    n, s, _ = raw.shape
    matches, counts = np.unique(mids, return_counts=True)
    # slot-team continuity across adjacent frames (same match)
    same_match = mids[1:] == mids[:-1]
    team_same = (team[1:] == team[:-1]).all(axis=1)
    cont = float(team_same[same_match].mean()) if same_match.any() else float("nan")
    # displacement per adjacent same-match frame
    d = np.linalg.norm(raw[1:] - raw[:-1], axis=-1)[same_match]
    step = {"median": round(float(np.median(d)), 4),
            "p90": round(float(np.quantile(d, 0.90)), 4),
            "p99": round(float(np.quantile(d, 0.99)), 4),
            "max": round(float(d.max()), 4)}
    # oracle demo on sampled frames with fixed placeholder roles
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(a.frames, n), replace=False)
    margins = []
    for i in idx:
        t = team[i]
        home = np.nonzero(t == 0)[0]
        away = np.nonzero(t == 1)[0]
        if len(home) < 2 or len(away) < 1:
            continue
        margins.append(channel_margin(raw[i, home[0]], raw[i, home[1]],
                                      raw[i, away], radius=0.5, end_excl=1.0))
    margins = np.array(margins)
    finite = margins[np.isfinite(margins)]
    out = {"split": a.split, "n_frames": n, "n_slots": s,
           "matches": {m: int(c) for m, c in zip(matches.tolist(), counts.tolist())},
           "slot_team_continuity": round(cont, 4), "step_stats": step,
           "oracle_demo": {"n": len(margins),
                           "open_frac": round(float(np.mean(margins >= 0)), 4),
                           "margin_median": round(float(np.median(finite)), 4) if len(finite) else None,
                           "radius": 0.5, "end_excl": 1.0,
                           "roles": "PLACEHOLDER p=home0 r=home1 D=away (semantics TBD)"}}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "D06_PROBE.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
