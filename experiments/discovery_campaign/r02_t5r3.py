#!/usr/bin/env python3
"""R02 on T5R3 (Track6-i): minimal home-team x-translation across thirds boundary.

Oracle EXACT: zone flips iff home_centroid_x crosses +-1/3 (bit-verified).
Flip edit: translate 10 home players toward nearest boundary by
  shift = dist_to_boundary + FLOOR (new oracle margin == FLOOR by construction).
Null edit: same magnitude AWAY from boundary (oracle must stay; verified).
Graph rebuilt per scene with locked knn (deployment-faithful).
Metrics mirror R02: frac flipped, miss|flip, unconditional miss, stealth-vs-null
feat distance, margin distribution. No training here.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "docs" / "iclr2027_discovery_campaign_20260917"
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_t5r3_sanity import TaskModel, knn_adjacency_batch

VIEW = REPO / "artifacts/phase3/task_semantic_repair_v1/data_views"
T5R3 = REPO / "artifacts/phase3/task_semantic_repair_v1/t5r3_sanity_v3"
BOUNDS = np.array([-1.0 / 3, 1.0 / 3])


def zone_of(home_cx):
    return np.digitize(np.asarray(home_cx), BOUNDS).astype(int)


def zone_margin(home_cx):
    return np.minimum(np.abs(home_cx - 1 / 3), np.abs(home_cx + 1 / 3))


def forward_all(model, pos, team, adj, bs=512):
    """Return zone_pred, centroid_pred, pooled z."""
    model.eval()
    zp, cp, zz = [], [], []
    with torch.no_grad():
        for i in range(0, len(pos), bs):
            z = model.encode(torch.from_numpy(pos[i:i + bs]).float(),
                             torch.from_numpy(team[i:i + bs]).long(),
                             torch.from_numpy(adj[i:i + bs]).float(), "team_mean")
            _, zo, ce = model.context(z)
            zp.append(zo.argmax(-1).numpy())
            cp.append(ce.numpy())
            zz.append(z.numpy())
    return np.concatenate(zp), np.concatenate(cp), np.concatenate(zz)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--match", type=str, default="J03WOY")
    p.add_argument("--n-scenes", type=int, default=256)
    p.add_argument("--floor", type=float, default=0.03)
    p.add_argument("--max-shift", type=float, default=0.40)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--split", type=str, default="valid", choices=["valid", "train"])
    p.add_argument("--role", type=str, default="eval",
                   help="eval: metrics; mine: also dump flip scenes for finetuning")
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(a.seed)

    raw = np.load(VIEW / f"positions_raw_{a.split}.npy").astype(np.float32)
    team = np.load(VIEW / f"team_slots_{a.split}.npy").astype(np.int64)
    index = pd.read_parquet(VIEW / f"snapshot_index_{a.split}.parquet")
    adj_stored = np.load(T5R3 / f"adjacency_{a.split}.npy").astype(np.float32)
    mids = index["source_match_id"].astype(str).to_numpy()
    home_cx_all = (raw * (team == 0)[..., None]).sum(1)[:, 0] / (team == 0).sum(1)
    zmap = {"defensive_third": 0, "middle_third": 1, "attacking_third": 2}
    zstored = index["home_field_zone"].astype(str).map(zmap).to_numpy()
    assert (zone_of(home_cx_all) == zstored).all(), "zone oracle mismatch"
    print(f"match {a.match}: pool={int((mids == a.match).sum())}", flush=True)

    pool = np.where(mids == a.match)[0]
    sel = rng.choice(pool, min(a.n_scenes, len(pool)), replace=False)
    X, T = raw[sel], team[sel]
    hcx = home_cx_all[sel]
    z0 = zstored[sel]
    home_idx = np.where(team[0] == 0)[0]

    # nearest boundary + shift plan (vectorized)
    d_lo = hcx + 1 / 3   # dist to lower bound (signed side: cx - (-1/3))
    d_hi = 1 / 3 - hcx   # dist to upper bound
    # distance to nearest boundary and direction toward it
    to_lo = np.abs(d_lo) * 0 + (hcx - (-1 / 3))  # signed cx+1/3
    to_hi = (1 / 3 - hcx)
    use_lo = np.abs(to_lo) <= np.abs(to_hi)
    dist = np.where(use_lo, np.abs(to_lo), np.abs(to_hi))
    direction = np.where(use_lo, -np.sign(to_lo), np.sign(to_hi))
    direction[direction == 0] = 1.0
    shift = direction * (dist + a.floor)
    feasible = np.abs(shift) <= a.max_shift

    Xf = X.copy()
    Xf[:, home_idx, 0] += (shift * feasible)[:, None]
    Xn = X.copy()
    Xn[:, home_idx, 0] -= (shift * feasible)[:, None]
    # zero-control arm: same-magnitude shift applied to AWAY team instead.
    # Oracle zone (home-based) cannot change; any model flip here is spurious
    # sensitivity, reported as a bound — not pooled into miss rates.
    away_idx = np.where(team[0] == 1)[0]
    Xa = X.copy()
    Xa[:, away_idx, 0] += (shift * feasible)[:, None]
    hcx_f = hcx + shift * feasible
    hcx_n = hcx - shift * feasible
    zf = zone_of(hcx_f)
    zn = zone_of(hcx_n)
    valid_flip = feasible & (zf != z0) & (zone_margin(hcx_f) >= a.floor - 1e-9)
    valid_null = feasible & (zn == z0)
    print(f"feasible={int(feasible.sum())}/{len(sel)} flips={int(valid_flip.sum())} "
          f"nulls={int(valid_null.sum())} shift_med={np.median(np.abs(shift[feasible])):.3f}",
          flush=True)

    adj_f = knn_adjacency_batch(Xf)
    adj_n = knn_adjacency_batch(Xn)
    adj_a = knn_adjacency_batch(Xa)
    if a.role == "mine":
        keep = np.where(valid_flip)[0]
        np.savez_compressed(
            a.output / "mined_flips.npz",
            base_idx=sel[keep], shift=shift[keep].astype(np.float32),
            orig_zone=z0[keep].astype(np.int64), new_zone=zf[keep].astype(np.int64),
            home_margin=np.abs(zone_margin(hcx_f[keep])).astype(np.float32),
            match=np.array([a.match] * len(keep)))
        print(f"mined {len(keep)} flips -> mined_flips.npz", flush=True)

    per_model, all_rows = {}, []
    for mpath in a.models:
        model = TaskModel()
        model.load_state_dict(torch.load(mpath, map_location="cpu", weights_only=True),
                              strict=True)
        mname = Path(mpath).stem
        zp_f, _, zfeat_f = forward_all(model, Xf, T, adj_f)
        zp_n, _, zfeat_n = forward_all(model, Xn, T, adj_n)
        zp_a, _, _ = forward_all(model, Xa, T, adj_a)
        zp_c, _, zfeat_c = forward_all(
            model, X, T, adj_stored[sel])
        miss = (zp_f != zf) & valid_flip
        n_flip = int(valid_flip.sum())
        n_miss = int((miss).sum())
        fd_flip = np.linalg.norm(zfeat_f - zfeat_c, axis=1)
        fd_null = np.linalg.norm(zfeat_n - zfeat_c, axis=1)
        ok_null = valid_null & valid_flip
        stealthier = (fd_flip < fd_null) & ok_null
        per_model[mname] = {
            "n_scenes": len(sel), "n_flip": n_flip,
            "frac_flip": round(n_flip / len(sel), 4),
            "n_miss": n_miss,
            "miss_given_flip": round(n_miss / n_flip, 4) if n_flip else None,
            "unconditional_miss": round(n_miss / len(sel), 4),
            "stealth_feat_med": round(float(np.median(fd_flip[valid_flip])), 4) if n_flip else None,
            "null_feat_med": round(float(np.median(fd_null[ok_null])), 4) if ok_null.sum() else None,
            "stealthier_frac": round(float(stealthier.sum() / ok_null.sum()), 4) if ok_null.sum() else None,
            "clean_acc": round(float((zp_c == z0).mean()), 4),
            "away_flip_rate": round(float((zp_a != z0).mean()), 4),
            "margin_med": round(float(np.median(zone_margin(hcx_f[valid_flip]))), 4) if n_flip else None,
        }
        for i in np.where(valid_flip)[0]:
            all_rows.append({"model": mname, "scene": int(sel[i]),
                             "orig": int(z0[i]), "new": int(zf[i]),
                             "shift": round(float(shift[i]), 4),
                             "pred": int(zp_f[i]), "miss": bool(miss[i]),
                             "away_pred": int(zp_a[i]),
                             "feat_dist": round(float(fd_flip[i]), 4),
                             "null_feat_dist": round(float(fd_null[i]), 4)
                             if valid_null[i] else None})
        print(mname, json.dumps(per_model[mname]), flush=True)
    with open(a.output / "flips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    (a.output / "TRACK6_R02_result.json").write_text(json.dumps(
        {"match": a.match, "floor": a.floor, "max_shift": a.max_shift,
         "per_model": per_model,
         "pooled_miss": round(float(np.mean([v["miss_given_flip"] for v in per_model.values()])), 4),
         "note": "stealth defined vs equal-magnitude away-translation null"}, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
