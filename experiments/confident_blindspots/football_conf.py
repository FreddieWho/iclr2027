#!/usr/bin/env python3
"""P1a: football confidence-vs-miss replication (frozen + cover, seed 11).

Per match: oracle zone turns between consecutive view snapshots; start
confidence (max softmax) vs model transition miss within +2 window;
boundary-distance control (centroid distance to nearest zone bound).
Zero training, dev views only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
torch.set_num_threads(4)
import t5r3_paths as T  # noqa: E402

VIEW = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1" / "data_views"
OUT = ROOT / "artifacts" / "confident_blindspots"


def run_match(match, split, seed, models):
    X = np.load(VIEW / f"positions_raw_{split}.npy").astype(np.float32)
    Tm = np.load(VIEW / f"team_slots_{split}.npy").astype(np.int64)
    idx = pd.read_parquet(VIEW / f"snapshot_index_{split}.parquet")
    sel = (idx.source_match_id == match).values
    X, Tm, idx = X[sel], Tm[sel], idx[sel].reset_index(drop=True)
    ts = idx.timestamp_ms.values
    order = np.argsort(ts)
    X, Tm, idx, ts = X[order], Tm[order], idx.iloc[order], ts[order]
    zl = idx.home_field_zone.astype(str).map(
        {"defense_third": 0, "middle_third": 1, "attack_third": 2}).values
    out = {}
    for k, m in models.items():
        P = T.zone_probs(m, X, Tm)
        pred = P.argmax(1)
        conf = P.max(1)
        hcx = (X * (Tm == 0)[..., None]).sum(1)[:, 0] / (Tm == 0).sum(1)
        B = T.BOUNDS
        miss, cf, dd = [], [], []
        for i in range(1, len(pred)):
            if zl[i] == zl[i - 1]:
                continue
            win = pred[i:i + 3] if i + 3 <= len(pred) else pred[i:]
            updated = any(int(win[j]) != int(pred[i - 1]) for j in range(len(win)))
            miss.append(0 if updated else 1)
            cf.append(conf[i - 1])
            dd.append(min(abs(hcx[i - 1] - b) for b in B))
        miss, cf, dd = map(np.array, (miss, cf, dd))
        q = np.quantile(cf, [0.5])
        near = dd <= np.median(dd)
        hi, lo = cf >= np.median(cf), cf < np.median(cf)
        out[k] = {"n": len(miss), "miss": round(float(miss.mean()), 4),
                  "confQ0_miss": round(float(miss[cf <= np.quantile(cf, 0.25)].mean()), 4),
                  "confQ3_miss": round(float(miss[cf >= np.quantile(cf, 0.75)].mean()), 4),
                  "near_hi": round(float(miss[hi & near].mean()), 4) if (hi & near).sum() else None,
                  "near_lo": round(float(miss[lo & near].mean()), 4) if (lo & near).sum() else None,
                  "far_hi": round(float(miss[hi & ~near].mean()), 4) if (hi & ~near).sum() else None,
                  "far_lo": round(float(miss[lo & ~near].mean()), 4) if (lo & ~near).sum() else None}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    models = {"frozen": T.load_t5r3(T.base_ckpt(a.seed)),
              "cover": T.load_t5r3(ROOT / "artifacts" / "discovery_campaign" /
                                   f"r04b_t5r3_s{a.seed}" / "cover.pt")}
    res = {}
    for match, split in (("J03WOY", "valid"), ("J03WMX", "train"), ("J03WOH", "train")):
        try:
            res[match] = run_match(match, split, a.seed, models)
        except FileNotFoundError as e:
            res[match] = {"error": str(e)}
        print(f"SAW {match}: {json.dumps(res[match])}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "FOOTBALL_CONF.json", "w"), indent=1)


if __name__ == "__main__":
    main()
