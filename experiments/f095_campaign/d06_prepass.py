#!/usr/bin/env python3
"""D06b probe: channel margin 1.5s BEFORE the pass (read-only analysis).

Same roles (passer/recipient/opponents from the event), frame at t-1.5s:
is the channel more contested before the pass is played? Reports margin
distribution shift + paired-edit feasibility at the pre-pass frame.
Decides whether the D06b (pre-pass pressure) pivot is viable.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
from d06_oracle import channel_label  # noqa: E402

EV = ROOT / "artifacts" / "data_v2/idsse/canonical/events"
FR = ROOT / "artifacts" / "data_v2/idsse/canonical/frames"
OUT = ROOT / "artifacts" / "f095_campaign" / "D06"


def snap_at(fr, t):
    dt = np.abs(fr["timestamp_ms"].to_numpy() - t)
    if dt.min() > 500:
        return None
    return fr[dt == dt.min()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--match", default="J03WOY")
    p.add_argument("--dt-ms", type=float, default=1500)
    p.add_argument("--radius", type=float, default=0.5)
    p.add_argument("--end-excl", type=float, default=1.0)
    p.add_argument("--cap", type=float, default=0.15)
    p.add_argument("--tries", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    ev = pd.read_parquet(EV / f"{a.match}.parquet")
    fr = pd.read_parquet(FR / f"{a.match}.parquet")
    passes = ev[(ev["event_type"] == "Play:Pass") & ev["recipient_id"].notna()]
    rec = {"match": a.match, "dt_ms": a.dt_ms, "n_pass": len(passes), "used": 0,
           "margins_t": [], "margins_pre": [], "paired_pre": 0}
    for _, e in passes.iterrows():
        t = float(e["timestamp_ms"])
        rows_t = snap_at(fr, t)
        rows_pre = snap_at(fr, t - a.dt_ms)
        if rows_t is None or rows_pre is None:
            continue
        ok = True
        sides = []
        for rows in (rows_t, rows_pre):
            by_pid = {r["person_id"]: (float(r["x"]), float(r["y"]), str(r["team_id"]))
                      for _, r in rows.iterrows()}
            if e["player_id"] not in by_pid or e["recipient_id"] not in by_pid:
                ok = False
                break
            px, py, pteam = by_pid[e["player_id"]]
            rx, ry, _ = by_pid[e["recipient_id"]]
            opp = np.array([[x, y] for _, (x, y, t_) in by_pid.items() if t_ != pteam])
            if len(opp) == 0:
                ok = False
                break
            sides.append(([px, py], [rx, ry], opp))
        if not ok:
            continue
        rec["used"] += 1
        mt = channel_label(*sides[0], a.radius, a.end_excl)
        mp = channel_label(*sides[1], a.radius, a.end_excl)
        rec["margins_t"].append(mt["margin"] if np.isfinite(mt["margin"]) else 99.0)
        rec["margins_pre"].append(mp["margin"] if np.isfinite(mp["margin"]) else 99.0)
        # paired feasibility at the pre-pass frame
        keep = flip = False
        base_lab = mp["label"]
        _, r_pre, opp_pre = sides[1]
        cands = [("r", np.array(r_pre))] + [("d", o) for o in opp_pre[:6]]
        for kind, pos in cands:
            for _ in range(a.tries // len(cands)):
                v = rng.normal(size=2)
                v *= a.cap * rng.random() / (np.linalg.norm(v) + 1e-9)
                if kind == "r":
                    lab = channel_label(sides[1][0], pos + v, opp_pre,
                                        a.radius, a.end_excl)["label"]
                else:
                    oo = opp_pre.copy()
                    oo[np.argmin(np.linalg.norm(opp_pre - pos, axis=1))] = pos + v
                    lab = channel_label(sides[1][0], r_pre, oo,
                                        a.radius, a.end_excl)["label"]
                keep |= lab == base_lab
                flip |= lab != base_lab
                if keep and flip:
                    break
            if keep and flip:
                break
        rec["paired_pre"] += keep and flip
    mt = np.array(rec["margins_t"])
    mp = np.array(rec["margins_pre"])
    out = {"match": a.match, "dt_ms": a.dt_ms, "used": rec["used"],
           "margin_med_t": round(float(np.median(mt)), 4),
           "margin_med_pre": round(float(np.median(mp)), 4),
           "frac_open_pre": round(float(np.mean(mp >= 0)), 4),
           "paired_rate_pre": round(rec["paired_pre"] / rec["used"], 4),
           "paired_rate_pre_contested": None}
    narrow = mp < 1.0
    out["n_contested_pre"] = int(narrow.sum())
    out["note"] = "per-event paired flags recomputed in follow-up if contested pool viable"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"D06_PREPASS_{a.match}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
