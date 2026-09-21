#!/usr/bin/env python3
"""P1-A football rewrite: start-correct + arrival-at-new-class + time windows.

Oracle turn = home_field_zone change between consecutive snapshots with
dt <= 5000ms (else sequence break). Jitter (return within 2 snaps) merged and
counted separately. Start snapshot must satisfy pred == old oracle class;
confidence taken adjacent AND fixed-lead (-2 snaps, excluded+counted if lead
is no longer old class). Success = pred reaches NEW oracle class within
+6000ms. Third-class visits single-columned. Per-event JSON, seed in filename.
Exploration only (train matches included, no held-out claim).
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
OUT = ROOT / "artifacts" / "p123_upgrade" / "p1"
ZMAP = {"defensive_third": 0, "middle_third": 1, "attacking_third": 2}
GAP_MS = 5000
WIN_MS = 6000


def run_match(match, split, seed, models):
    X = np.load(VIEW / f"positions_raw_{split}.npy").astype(np.float32)
    Tm = np.load(VIEW / f"team_slots_{split}.npy").astype(np.int64)
    idx = pd.read_parquet(VIEW / f"snapshot_index_{split}.parquet")
    sel = (idx.source_match_id == match).values
    X, Tm, idx = X[sel], Tm[sel], idx[sel].reset_index(drop=True)
    ts = idx.timestamp_ms.values.astype(np.int64)
    order = np.argsort(ts)
    X, Tm, idx, ts = X[order], Tm[order], idx.iloc[order], ts[order]
    zl = idx.home_field_zone.astype(str).map(ZMAP)
    assert zl.notna().all(), "unmapped zone labels present"
    zl = zl.values.astype(int)
    hcx = (X * (Tm == 0)[..., None]).sum(1)[:, 0] / (Tm == 0).sum(1)
    B = T.BOUNDS
    out = {}
    for k, m in models.items():
        P = T.zone_probs(m, X, Tm)
        pred = P.argmax(1)
        conf = P.max(1)
        ent = (-P * np.log(P + 1e-12)).sum(1)
        rows = []
        n_jitter = n_lead_excl = 0
        for i in range(1, len(pred)):
            if ts[i] - ts[i - 1] > GAP_MS:
                continue
            if zl[i] == zl[i - 1]:
                continue
            # jitter: oracle returns within next 2 snaps?
            j = i
            while j + 1 < len(zl) and ts[j + 1] - ts[j] <= GAP_MS and (j + 1 - i) <= 2:
                if zl[j + 1] == zl[i - 1]:
                    break
                j += 1
            if j + 1 < len(zl) and zl[j + 1] == zl[i - 1] and (j + 1 - i) <= 2:
                n_jitter += 1
                continue
            if pred[i - 1] != zl[i - 1]:
                continue  # start-correct required
            # dwell: new state persists?
            dwell = 1
            while (i + dwell < len(zl) and zl[i + dwell] == zl[i]
                   and ts[i + dwell] - ts[i + dwell - 1] <= GAP_MS):
                dwell += 1
            # fixed-lead confidence
            li = i - 2
            if li < 0 or pred[li] != zl[i - 1] or ts[i - 1] - ts[li] > GAP_MS:
                n_lead_excl += 1
                lead_conf = None
            else:
                lead_conf = float(conf[li])
            # arrival at NEW class within time window
            arrived = False
            third = False
            jj = i
            while jj < len(pred) and ts[jj] - ts[i] <= WIN_MS:
                if pred[jj] == zl[i]:
                    arrived = True
                    break
                if pred[jj] not in (zl[i - 1], zl[i]):
                    third = True
                jj += 1
            rows.append({
                "t_event": int(ts[i]), "old": int(zl[i - 1]), "new": int(zl[i]),
                "conf_adj": float(conf[i - 1]), "conf_lead": lead_conf,
                "entropy": float(ent[i - 1]),
                "dist_bound": float(min(abs(hcx[i - 1] - b) for b in B)),
                "success": bool(arrived), "third": bool(third),
                "dwell": int(dwell), "truncated": bool(jj >= len(pred))})
        out[k] = {"events": rows, "n_jitter": n_jitter, "n_lead_excl": n_lead_excl}
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
    for match, split in (("J03WOY", "valid"), ("J03WN1", "valid"),
                         ("J03WMX", "train"), ("J03WOH", "train"),
                         ("J03WPY", "train"), ("J03WR9", "train")):
        try:
            res[match] = run_match(match, split, a.seed, models)
        except FileNotFoundError as e:
            res[match] = {"error": str(e)}
        nkev = sum(len(v["events"]) for v in res[match].values() if isinstance(v, dict) and "events" in v) if isinstance(res[match], dict) else 0
        print(f"SAW {match}: events={nkev}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / ("p1football_s%d.json" % a.seed), "w"))
    print("DONE p1_football s%d" % a.seed)


if __name__ == "__main__":
    main()
