#!/usr/bin/env python3
"""R01 on T5R3 real checkpoints (raw_single_channel_team_mean x3 seeds).

Task A (classification): field-zone from home centroid x thirds (oracle EXACT,
verified bit-identical on all 6127 valid scenes). Task B (regression):
team centroids (oracle EXACT, maxerr 0 on clean).
Bank: within-team balanced signs over 20 players (home 10 / away 10, slots
index-stable) => team-centroid displacement EXACTLY 0 => oracle labels for
BOTH arms equal clean labels BY CONSTRUCTION (all-preserving; flipped table
empty by design, stated not hidden).
Graph policy: adjacency REBUILT per arm with locked knn_adjacency_batch
(deployment-faithful); rebuild==stored verified on clean.
Split: source match J03WN1 -> select; target match J03WOY -> evaluate.
Laws: Q_ref / Q_coherent (team-contrast |s.g|, fixed) / Q_source_greedy(k=2)
+ Q_source_dro / Q_whitebox. Selection loss = zone 0/1 on source.
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
from core.perturbations import (balanced_sign_patterns, antithetic_fields,
                                weighted_sign_law, normalize_weights,
                                worst_case_weights, greedy_mode_cover)
from run_t5r3_sanity import TaskModel, knn_adjacency_batch

VIEW = REPO / "artifacts/phase3/task_semantic_repair_v1/data_views"
T5R3 = REPO / "artifacts/phase3/task_semantic_repair_v1/t5r3_sanity_v3"
ZONE_BINS = np.array([-1.0 / 3, 1.0 / 3])  # verified exact on 6127 scenes


def zone_of(home_cx: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(home_cx), ZONE_BINS)  # 0 def / 1 mid / 2 att


def forward_zone_centroid(model, pos, team, adj, bs=512):
    model.eval()
    zpred, cpred = [], []
    with torch.no_grad():
        for i in range(0, len(pos), bs):
            z = model.encode(torch.from_numpy(pos[i:i + bs]).float(),
                             torch.from_numpy(team[i:i + bs]).long(),
                             torch.from_numpy(adj[i:i + bs]).float(), "team_mean")
            _, zo, ce = model.context(z)
            zpred.append(zo.argmax(-1).numpy())
            cpred.append(ce.numpy())
    return np.concatenate(zpred), np.concatenate(cpred)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-per-match", type=int, default=128)
    p.add_argument("--epsilons", nargs="+", type=float, default=[0.05, 0.15])
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(a.seed)

    raw = np.load(VIEW / "positions_raw_valid.npy").astype(np.float32)
    team = np.load(VIEW / "team_slots_valid.npy").astype(np.int64)
    index = pd.read_parquet(VIEW / "snapshot_index_valid.parquet")
    adj_stored = np.load(T5R3 / "adjacency_valid.npy").astype(np.float32)
    mids = index["source_match_id"].astype(str).to_numpy()

    # --- Sanity A: stored zone == thirds rule; rebuild == stored adjacency ---
    home_cx = raw[team == 0].reshape(len(raw), -1, 2)[:, :, 0].mean(axis=1) \
        if False else (raw * (team == 0)[..., None]).sum(1)[:, 0] / (team == 0).sum(1)
    zrec = zone_of(home_cx)
    zmap = {"defensive_third": 0, "middle_third": 1, "attacking_third": 2}
    zstored = index["home_field_zone"].astype(str).map(zmap).to_numpy()
    assert (zrec == zstored).all(), "zone oracle mismatch"
    chk = rng.choice(len(raw), 16, replace=False)
    assert np.abs(knn_adjacency_batch(raw[chk]) - adj_stored[chk]).max() < 1e-5, \
        "adjacency rebuild mismatch (alignment?)"
    print("sanity A passed: zone exact, adjacency rebuild maxerr < 1e-5", flush=True)

    home_idx = np.where(team[0] == 0)[0]
    away_idx = np.where(team[0] == 1)[0]
    assert len(home_idx) == 10 and len(away_idx) == 10
    patterns = balanced_sign_patterns((10, 10), max_patterns=64, seed=a.seed)
    P20 = np.zeros((len(patterns), 20))
    P20[:, home_idx] = patterns[:, :10]
    P20[:, away_idx] = patterns[:, 10:]
    M = len(P20)
    # team-centroid displacement exactly 0 check (per-node amp eps/sqrt(20) symbolic)
    assert np.abs(P20[:, home_idx].sum(1)).max() == 0 and np.abs(P20[:, away_idx].sum(1)).max() == 0
    # Coherent template must survive within-team balance (team-contrast |s.g| is
    # identically 0 there). Use cross-team synchrony: alignment between the two
    # teams' internal sign patterns (fixed index sets, response-blind).
    align = np.abs((P20[:, home_idx] * P20[:, away_idx]).sum(axis=1)) / 10.0
    assert align.max() > 0, "coherent template degenerate"
    w_coh = align / align.sum()
    ev = weighted_sign_law(P20, np.full(M, 1.0 / M))
    assert np.abs(ev["plus_probability"] - 0.5).max() < 1e-12
    print(f"bank: {M} modes, team-centroid displacement exactly 0, marginals exact", flush=True)

    # --- scene sampling: source J03WN1, target J03WOY ---
    src_pool = np.where(mids == "J03WN1")[0]
    tgt_pool = np.where(mids == "J03WOY")[0]
    src_idx = rng.choice(src_pool, a.n_per_match, replace=False)
    tgt_idx = rng.choice(tgt_pool, a.n_per_match, replace=False)
    splits = {"source": (raw[src_idx], team[src_idx], zstored[src_idx], src_idx),
              "target": (raw[tgt_idx], team[tgt_idx], zstored[tgt_idx], tgt_idx)}
    # zone margin per scene (dist of home_cx to thirds boundary)
    for tag in splits:
        hcx = home_cx[splits[tag][3]]
        mg = np.minimum(np.abs(hcx - 1 / 3), np.abs(hcx + 1 / 3))
        print(f"{tag} zone-margin med {np.median(mg):.3f}, frac>0.05 {(mg > 0.05).mean():.2f}",
              flush=True)

    directions = {"vx": np.array([1., 0.]), "vy": np.array([0., 1.])}
    amp_note = "per-player amplitude eps/sqrt(20)"
    law_defs = {"Q_ref": "uniform", "Q_coherent": "cross-team synchrony |<sH,sA>|/10 fixed",
                "Q_source_greedy": "greedy k=2 on SOURCE match zone 0/1",
                "Q_source_dro": "entropic DRO reg=1 on SOURCE match zone 0/1",
                "Q_whitebox": "per-target-scene worst mode (reference)"}
    (a.output / "law_definition.json").write_text(json.dumps(
        {"laws": law_defs, "n_modes": M, "directions": list(directions),
         "epsilons": a.epsilons, "amp": amp_note, "source_match": "J03WN1",
         "target_match": "J03WOY", "n_per_match": a.n_per_match,
         "graph_policy": "adjacency rebuilt per arm (deployment-faithful)",
         "oracle": "zone thirds exact + centroid exact; all arms label-preserving by construction"},
        indent=2) + "\n")

    rows = []
    fwd_count = 0
    for mpath in a.models:
        model = TaskModel()
        sd = torch.load(mpath, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
        mname = Path(mpath).stem
        # clean reference on sampled scenes
        for tag in splits:
            X, T, _, _ = splits[tag]
            adj = adj_stored[splits[tag][3]]
            zp, cp = forward_zone_centroid(model, X, T, adj)
            fwd_count += len(X)
            true_c = np.stack([(X * (T == s)[..., None]).sum(1) / (T == s).sum(1, keepdims=True)
                               for s in (0, 1)], axis=1).reshape(len(X), 4)
            rows.append({"model": mname, "direction": "-", "epsilon": 0.0, "law": "clean",
                         "zone_risk": float((zp != splits[tag][2]).mean()),
                         "centroid_mse": float(((cp - true_c) ** 2).mean()),
                         "split": tag})
        for dname, v in directions.items():
            for eps in a.epsilons:
                F = (P20[:, :, None] * (v / np.linalg.norm(v))[None, None, :]
                     * (eps / np.sqrt(20))).astype(np.float32)  # [M,20,2]
                # source pair losses (zone 0/1) for selection
                Xs, Ts, zys, sidx = splits["source"]
                Xt, Tt, zyt, tidx = splits["target"]
                src_fail = np.zeros((M, 2, len(Xs)))
                for mi in range(M):
                    for arm, sgn in ((0, 1.0), (1, -1.0)):
                        pert = Xs + sgn * F[mi][None]
                        adj = knn_adjacency_batch(pert)
                        zp, _ = forward_zone_centroid(model, pert, Ts, adj)
                        fwd_count += len(Xs)
                        src_fail[mi, arm] = (zp != zys).astype(float)
                pair_src = src_fail.mean(axis=(1, 2))
                gidx, _ = greedy_mode_cover(src_fail.sum(axis=1) > 0, k=min(2, M))
                w_greedy = np.zeros(M)
                w_greedy[gidx] = 1.0 / max(1, len(gidx))
                w_dro = worst_case_weights(pair_src, regularization=1.0)
                w_ref = np.full(M, 1.0 / M)
                # target evaluation (batched per mode-arm, shared across laws)
                tgt_zone = np.zeros((M, 2, len(Xt)))
                tgt_cmse = np.zeros((M, 2, len(Xt)))
                true_ct = np.stack([(Xt * (Tt == s)[..., None]).sum(1) / (Tt == s).sum(1, keepdims=True)
                                    for s in (0, 1)], axis=1).reshape(len(Xt), 4)
                wb = np.zeros(len(Xt))
                for mi in range(M):
                    arm_bad = np.zeros(len(Xt))
                    for arm, sgn in ((0, 1.0), (1, -1.0)):
                        pert = Xt + sgn * F[mi][None]
                        adj = knn_adjacency_batch(pert)
                        zp, cp = forward_zone_centroid(model, pert, Tt, adj)
                        fwd_count += len(Xt)
                        tgt_zone[mi, arm] = (zp != zyt).astype(float)
                        tgt_cmse[mi, arm] = ((cp - true_ct) ** 2).mean(axis=1)
                        arm_bad = np.maximum(arm_bad, tgt_zone[mi, arm])
                    wb = np.maximum(wb, arm_bad)
                for lname, w in (("Q_ref", w_ref), ("Q_coherent", w_coh),
                                 ("Q_source_greedy", w_greedy), ("Q_source_dro", w_dro)):
                    w = normalize_weights(np.asarray(w, float), M)
                    rows.append({"model": mname, "direction": dname, "epsilon": eps,
                                 "law": lname,
                                 "zone_risk": float((w[:, None, None] * tgt_zone).sum(0).mean()),
                                 "centroid_mse": float((w[:, None, None] * tgt_cmse).sum(0).mean()),
                                 "split": "target"})
                rows.append({"model": mname, "direction": dname, "epsilon": eps,
                             "law": "Q_whitebox", "zone_risk": float(wb.mean()),
                             "centroid_mse": float("nan"), "split": "target"})
    with open(a.output / "risk_by_config.csv", "w", newline="") as f:
        import csv as csvm
        w = csvm.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (a.output / "R01_T5R3_result.json").write_text(json.dumps(
        {"route_id": "R01", "run_id": "t5r3_valid_zone_centroid",
         "n_source": a.n_per_match, "n_target": a.n_per_match,
         "model_forward_calls": fwd_count, "status": "see card",
         "artifacts": ["risk_by_config.csv", "law_definition.json"]}, indent=2) + "\n")
    print("forwards:", fwd_count, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
