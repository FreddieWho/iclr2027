#!/usr/bin/env python3
"""Shared T5R3 helpers for E2 (real-data turn experiments).

Mirrors the coord path contract on team frames:
- oracle = home-centroid zone thirds (exact, bit-verified in r02_t5r3)
- edits translate the 10 home players (deployment-faithful, graph rebuilt)
- model turns located by bisection on predicted-zone equality (multiclass;
  coord logit-sign bisection does not apply to 3 zones)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_t5r3_sanity import TaskModel, knn_adjacency_batch  # noqa: E402

VIEW = ROOT / "artifacts/phase3/task_semantic_repair_v1/data_views"
BOUNDS = np.array([-1.0 / 3, 1.0 / 3])
BASECKPT_DIR = ROOT / "artifacts/phase3/task_semantic_repair_v1/t5r3_sanity_v3/models"
def base_ckpt(seed):
    return BASECKPT_DIR / f"raw_single_channel_phase_gat_team_mean_seed{seed}.pt"
COVERDIR = ROOT / "artifacts/discovery_campaign"
def cover_file(seed, name):
    return COVERDIR / f"r04c_t5r3_lr3e4_s{seed}" / f"{name}.pt"
FLOOR = 0.03


def zone_of(home_cx):
    return np.digitize(np.asarray(home_cx), BOUNDS).astype(int)


def zone_margin(home_cx):
    h = np.asarray(home_cx, dtype=float)
    return np.minimum(np.abs(h - 1 / 3), np.abs(h + 1 / 3))


def home_cx_of(X, home_idx):
    return X[:, home_idx, 0].mean(1)


def load_views(split="valid"):
    import pandas as pd
    raw = np.load(VIEW / f"positions_raw_{split}.npy").astype(np.float32)
    team = np.load(VIEW / f"team_slots_{split}.npy").astype(np.int64)
    index = pd.read_parquet(VIEW / f"snapshot_index_{split}.parquet")
    mids = index["source_match_id"].astype(str).to_numpy()
    hcx = (raw * (team == 0)[..., None]).sum(1)[:, 0] / (team == 0).sum(1)
    return raw, team, mids, hcx


def load_t5r3(ckpt):
    m = TaskModel()
    m.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True),
                      strict=True)
    m.eval()
    return m


def zone_probs(model, X, T, bs=256):
    """Zone-class probabilities; adjacency rebuilt per chunk (locked knn)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xc = np.ascontiguousarray(X[i:i + bs].astype(np.float32))
            adj = np.ascontiguousarray(
                knn_adjacency_batch(xc).astype(np.float32))
            z = model.encode(torch.from_numpy(xc),
                             torch.from_numpy(T[i:i + bs].astype(np.int64)),
                             torch.from_numpy(adj), "team_mean")
            _, zhp, _ = model.context(z)
            out.append(torch.softmax(zhp, -1).numpy())
    return np.concatenate(out, axis=0)


def zone_pred_1(model, x, team):
    """Single-scene zone prediction (for bisection)."""
    p = zone_probs(model, x[None].astype(np.float32), team[None].astype(np.int64))
    return int(p[0].argmax())


def make_oracle_turns(home_idx):
    def turns(x_of_t, t_grid, n_bisect=14):
        labs = []
        for tt in t_grid:
            xx = np.asarray(x_of_t(tt), dtype=float)
            cx = float(xx[home_idx, 0].mean())
            labs.append(int(zone_of(cx)))
        out = []
        for i in range(len(t_grid) - 1):
            if labs[i] == labs[i + 1]:
                continue
            lo, hi = t_grid[i], t_grid[i + 1]
            tgt = labs[i + 1]
            for _ in range(n_bisect):
                m = 0.5 * (lo + hi)
                xm = np.asarray(x_of_t(m), dtype=float)
                lm = int(zone_of(float(xm[home_idx, 0].mean())))
                if lm == tgt:
                    hi = m
                else:
                    lo = m
            out.append({"t_star": float(0.5 * (lo + hi)),
                        "old": int(labs[i]), "new": int(tgt)})
        return out
    return turns


def make_model_turns(model, team1, home_idx):
    def turns(x_of_t, t_grid, n_bisect=10):
        probs = zone_probs(model,
                           np.stack([np.asarray(x_of_t(tt), dtype=np.float32)
                                     for tt in t_grid]),
                           np.stack([team1] * len(t_grid)))
        plab = probs.argmax(1)
        out = []
        for i in range(len(t_grid) - 1):
            if plab[i] == plab[i + 1]:
                continue
            lo, hi = t_grid[i], t_grid[i + 1]
            tgt = int(plab[i + 1])
            for _ in range(n_bisect):
                m = 0.5 * (lo + hi)
                lm = zone_pred_1(model, np.asarray(x_of_t(m), dtype=np.float32), team1)
                if lm == tgt:
                    hi = m
                else:
                    lo = m
            out.append({"t_theta": float(0.5 * (lo + hi)),
                        "old": int(plab[i]), "new": tgt})
        return out
    return turns


def nearest_boundary_plan(hcx):
    """(dist, direction) toward nearest zone boundary for a home-centroid x."""
    to_lo = hcx + 1 / 3
    to_hi = 1 / 3 - hcx
    if abs(to_lo) <= abs(to_hi):
        return abs(to_lo), (-np.sign(to_lo) or 1.0)
    return abs(to_hi), (np.sign(to_hi) or 1.0)


def aimed_home_shift(x, home_idx, floor=FLOOR, overs=(0.0, 0.05, 0.10, 0.20)):
    """Translate home-x toward/through nearest boundary; first edit whose
    endpoint flips zone with margin>=floor. Returns (edit, new_zone) or None."""
    hcx = float(x[home_idx, 0].mean())
    z0 = int(zone_of(hcx))
    dist, direction = nearest_boundary_plan(hcx)
    for over in overs:
        e = np.zeros_like(x)
        e[home_idx, 0] = direction * (dist + floor + over)
        hcx1 = float((x + e)[home_idx, 0].mean())
        z1 = int(zone_of(hcx1))
        if z1 != z0 and float(zone_margin(hcx1)) >= floor - 1e-9:
            return e, z1
    return None
