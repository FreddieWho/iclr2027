#!/usr/bin/env python3
"""WP-D exploratory sensitivity: motion-model residual check.

Fit P(turn | motion) on non-event windows (logistic on ball_speed,
p_mean_disp, p_max_disp, cent_disp); predict on contested-event windows;
observed minus predicted = motion-unexplained excess. Labeled EXPLORATORY:
does not override the MATCHING_NOT_FEASIBLE primary verdict.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "l002_ball"))
from matched_selectivity import load_match  # noqa: E402

ART = ROOT / "artifacts" / "data_v2" / "idsse" / "canonical"


def main():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    res = {}
    for match, split in (("J03WOY", "valid"), ("J03WMX", "train")):
        d = load_match(match, split, 11)
        ts = d["ts"]
        ev = pd.read_parquet(ART / "events" / f"{match}.parquet")
        cont = ev[ev.event_type.isin(["TacklingGame", "BallClaiming"])]
        ev_idx = set()
        for t0 in cont.timestamp_ms.values:
            i = int(np.abs(ts - int(t0)).argmin())
            if abs(ts[i] - int(t0)) <= 2000:
                ev_idx.add(i)
        from matched_selectivity import feats_at
        Xe, ye = [], []
        Xu, yu = [], []
        tn_f = np.array([0] + [int(d["pred"]["frozen"][i] != d["pred"]["frozen"][i - 1]) for i in range(1, len(ts))])
        for i in range(4, len(ts)):
            f = feats_at(d, i)
            if f is None:
                continue
            j = max(0, i - 2)
            t = int(tn_f[j:i + 3].max())
            (Xe, ye) if i in ev_idx else (Xu, yu)
            if i in ev_idx:
                Xe.append(f[:4])
                ye.append(t)
            else:
                Xu.append(f[:4])
                yu.append(t)
        Xe, ye, Xu, yu = map(np.array, (Xe, ye, Xu, yu))
        lr = LogisticRegression().fit(Xu, yu)
        pred_e = lr.predict_proba(Xe)[:, 1]
        obs = ye.mean()
        exp = pred_e.mean()
        lo, hi = np.quantile(np.random.default_rng(0).choice(
            ye - pred_e, size=(10000, len(ye))).mean(0), [0.025, 0.975])
        res[match] = {"n_event": len(ye), "n_nonevent": len(yu),
                      "obs": round(float(obs), 4), "motion_pred": round(float(exp), 4),
                      "excess": round(float(obs - exp), 4),
                      "excess_ci": [round(float(lo), 4), round(float(hi), 4)]}
    json.dump(res, open(ROOT / "artifacts" / "l002_ball" / "MOTION_RESIDUAL.json", "w"), indent=1)
    print(f"SAW residual: {json.dumps(res, indent=1)}")


if __name__ == "__main__":
    main()
