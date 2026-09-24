#!/usr/bin/env python3
"""D06 role feasibility on real pass events (read-only analysis, no training).

For Play:Pass events with a recipient: take the nearest frame (same game
section), read p=passer / r=recipient / D=opponent positions, compute the
geometric channel margin (radius/end_excl from scene scale). Then test
paired-edit feasibility: random small displacements (<= motion cap from
the D06 probe p99) of r or the nearest defender that KEEP vs FLIP the
channel state. Reports the E-constructible rate: the fraction of real
passes admitting a paired keep+flip edit pair. No training, no labels
beyond geometry; natural outcome column is reported, never used as oracle.
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

EV = ROOT / "artifacts/data_v2/idsse/canonical/events"
FR = ROOT / "artifacts/data_v2/idsse/canonical/frames"
OUT = ROOT / "artifacts" / "f095_campaign" / "D06"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--match", default="J03WOY")
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
    fr = fr.sort_values("timestamp_ms").reset_index(drop=True)
    frame_times = fr["timestamp_ms"].to_numpy()
    rec = {"match": a.match, "n_pass": len(passes), "used": 0, "no_frame": 0,
           "no_person": 0, "margins": [], "paired": 0, "keep_only": 0,
           "flip_only": 0, "neither": 0,
           "outcome_of_used": {"successfullyCompleted": 0, "unsuccessful": 0}}
    for _, e in passes.iterrows():
        j = int(np.argmin(np.abs(frame_times - e["timestamp_ms"])))
        if abs(float(frame_times[j]) - float(e["timestamp_ms"])) > 500:
            rec["no_frame"] += 1
            continue
        t = float(e["timestamp_ms"])
        dt = np.abs(fr["timestamp_ms"].to_numpy() - t)
        if dt.min() > 500:
            rec["no_frame"] += 1
            continue
        rows = fr[dt == dt.min()]
        by_pid = {r["person_id"]: (float(r["x"]), float(r["y"]), str(r["team_id"]))
                  for _, r in rows.iterrows()}
        if e["player_id"] not in by_pid or e["recipient_id"] not in by_pid:
            rec["no_person"] += 1
            continue
        px, py, pteam = by_pid[e["player_id"]]
        rx, ry, _ = by_pid[e["recipient_id"]]
        opp = np.array([[x, y] for _, (x, y, t_) in by_pid.items() if t_ != pteam])
        if len(opp) == 0:
            rec["no_person"] += 1
            continue
        base = channel_label([px, py], [rx, ry], opp, a.radius, a.end_excl)
        rec["used"] += 1
        rec["margins"].append(base["margin"] if np.isfinite(base["margin"]) else 99.0)
        rec["outcome_of_used"][e["outcome"]] = rec["outcome_of_used"].get(e["outcome"], 0) + 1
        keep = flip = False
        base_lab = base["label"]
        cands = [("r", np.array([rx, ry]))] + \
                [("d", o) for o in opp[:6]]
        for kind, pos in cands:
            for _ in range(a.tries // len(cands)):
                v = rng.normal(size=2)
                v *= a.cap * rng.random() / (np.linalg.norm(v) + 1e-9)
                if kind == "r":
                    lab = channel_label([px, py], pos + v, opp, a.radius, a.end_excl)["label"]
                else:
                    oo = opp.copy()
                    oo[np.argmin(np.linalg.norm(opp - pos, axis=1))] = pos + v
                    lab = channel_label([px, py], [rx, ry], oo, a.radius, a.end_excl)["label"]
                if lab == base_lab:
                    keep = True
                else:
                    flip = True
                if keep and flip:
                    break
            if keep and flip:
                break
        rec["paired"] += keep and flip
        rec["keep_only"] += keep and not flip
        rec["flip_only"] += flip and not keep
        rec["neither"] += (not keep) and (not flip)
        rec.setdefault("event_rows", []).append({
            "outcome": e["outcome"],
            "margin": base["margin"] if np.isfinite(base["margin"]) else 99.0,
            "paired": int(keep and flip),
        })
    m = np.array(rec["margins"])
    rec["margin_median"] = round(float(np.median(m)), 4) if len(m) else None
    rec["margin_frac_open"] = round(float(np.mean(m >= 0)), 4) if len(m) else None
    rec["paired_rate"] = round(rec["paired"] / rec["used"], 4) if rec["used"] else None
    per = pd.DataFrame(rec.pop("event_rows"))
    per.to_csv(OUT / f"D06_ROLES_{a.match}_events.csv", index=False)
    strat = per.groupby("outcome").agg(n=("paired", "size"),
                                         paired_rate=("paired", "mean"),
                                         margin_med=("margin", "median"))
    rec["by_outcome"] = {k: {"n": int(v["n"]),
                               "paired_rate": round(float(v["paired_rate"]), 4),
                               "margin_med": round(float(v["margin_med"]), 4)}
                            for k, v in strat.iterrows()}
    narrow = per[per["margin"] < 1.0]
    rec["contested"] = {"n": int(len(narrow)),
                          "paired_rate": round(float(narrow["paired"].mean()), 4)
                          if len(narrow) else None}
    del rec["margins"]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"D06_ROLES_{a.match}.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
