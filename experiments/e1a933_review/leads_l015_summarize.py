"""Summarize L-015/L-006 rows. Reads SUMMARY.json only; does not retrain."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main():
    src = Path(sys.argv[1])
    d = json.loads(src.read_text())
    rows = d["rows"]
    arms = []
    for arm in ("raw", "permaug", "setmlp", "relfeat"):
        sub = [r for r in rows if r["arm"] == arm]
        j = np.array([r["J"] for r in sub])
        swing = np.array([r["J_swing"] for r in sub])
        flip = np.array([r["pred_flip_frac"] for r in sub])
        arms.append({
            "arm": arm, "n_seeds": len(sub),
            "J_mean": float(j.mean()), "J_sd": float(j.std(ddof=1)),
            "J_min": float(j.min()), "J_max": float(j.max()),
            "swing_mean": float(swing.mean()), "swing_max": float(swing.max()),
            "pred_flip_mean": float(flip.mean()), "pred_flip_max": float(flip.max()),
            "seconds_mean": float(np.mean([r["seconds"] for r in sub])),
        })
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["arm"]] = r["J"]
    contrasts = {}
    for name in ("permaug", "setmlp", "relfeat"):
        delta = np.array([by_seed[s][name] - by_seed[s]["raw"] for s in sorted(by_seed)])
        contrasts[name] = {
            "delta_mean": float(delta.mean()),
            "delta_sd": float(delta.std(ddof=1)),
            "n_positive": int((delta > 0).sum()),
            "n_ge_12pp": int((delta >= 0.12).sum()),
            "min": float(delta.min()), "max": float(delta.max()),
            "per_seed": {str(s): float(by_seed[s][name] - by_seed[s]["raw"]) for s in sorted(by_seed)},
        }
    out = {"arms": arms, "vs_raw": contrasts, "meta": d["meta"]}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
