#!/usr/bin/env python3
"""Track6-ii: coverage fine-tune of T5R3 on mined zone flips (source match).

Data: mined_flips.npz from r02_t5r3.py --role mine (base scenes + x-shifts).
Flip scenes reconstructed exactly; oracle zone/centroid recomputed (phase
labels original — spatial shift cannot change time-half).
Methods: cover (clean + flips, oracle labels) vs cleanonly (same cleans).
Loss: zone CE + centroid MSE + phase CE(=anchor). Full TaskModel, Adam 1e-4.
Eval: (a) r02_t5r3.py afterwards for follow-rate; (b) clean zone F1 +
centroid MAE here on held-out target scenes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "docs" / "iclr2027_discovery_campaign_20260917"
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(REPO / "scripts"))
from run_t5r3_sanity import TaskModel, knn_adjacency_batch

VIEW = REPO / "artifacts/phase3/task_semantic_repair_v1/data_views"
T5R3 = REPO / "artifacts/phase3/task_semantic_repair_v1/t5r3_sanity_v3"
BOUNDS = np.array([-1.0 / 3, 1.0 / 3])
PHASE = {"firstHalf": 0, "secondHalf": 1}


def zone_of(hcx):
    return np.digitize(np.asarray(hcx), BOUNDS).astype(int)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mine", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--eval-match", type=str, default="J03WOY")
    p.add_argument("--eval-n", type=int, default=512)
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed + 1)

    raw = np.load(VIEW / "positions_raw_valid.npy").astype(np.float32)
    team = np.load(VIEW / "team_slots_valid.npy").astype(np.int64)
    index = pd.read_parquet(VIEW / "snapshot_index_valid.parquet")
    adj_stored = np.load(T5R3 / "adjacency_valid.npy").astype(np.float32)
    mids = index["source_match_id"].astype(str).to_numpy()
    ph = index["phase_label"].astype(str).map(PHASE).to_numpy(np.int64)
    home_idx = np.where(team[0] == 0)[0]
    T0 = team[:1]

    mz = np.load(a.mine / "mined_flips.npz")
    base, shift = mz["base_idx"], mz["shift"].astype(np.float32)
    Xf = raw[base].copy()
    Xf[:, home_idx, 0] += shift[:, None]
    hcx_f = (Xf * (team[base] == 0)[..., None]).sum(1)[:, 0] / 10.0
    zf = zone_of(hcx_f)
    assert (zf == mz["new_zone"]).all(), "flip oracle mismatch"
    cf = np.stack([(Xf * (team[base] == s)[..., None]).sum(1) / 10.0 for s in (0, 1)],
                  axis=1).reshape(len(base), 4)
    Xc, Tc = raw[base], team[base]
    hcx_c = (Xc * (Tc == 0)[..., None]).sum(1)[:, 0] / 10.0
    zc = zone_of(hcx_c)
    cc = np.stack([(Xc * (Tc == s)[..., None]).sum(1) / 10.0 for s in (0, 1)],
                  axis=1).reshape(len(base), 4)
    adj_f = knn_adjacency_batch(Xf)
    adj_c = adj_stored[base]
    print(f"flips={len(base)} clean-pairs={len(base)}", flush=True)

    def T(x):
        return torch.from_numpy(np.asarray(x))
    D = {"xf": T(Xf).float(), "tf": T(team[base]).long(), "af": T(adj_f).float(),
         "zf": T(zf).long(), "cf": T(cf).float(), "pf": T(ph[base]).long(),
         "xc": T(Xc).float(), "tc": T(Tc).long(), "ac": T(adj_c).float(),
         "zc": T(zc).long(), "cc": T(cc).float(), "pc": T(ph[base]).long()}

    def run_loss(model, keys):
        z = model.encode(D[keys[0]], D[keys[1]], D[keys[2]], "team_mean")
        php, zhp, chp = model.context(z)
        return (F.cross_entropy(zhp, D[keys[3]]) +
                F.mse_loss(chp, D[keys[4]]) +
                F.cross_entropy(php, D[keys[5]]))

    results = {}
    for name in ("cover", "cleanonly"):
        model = TaskModel()
        model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True),
                              strict=True)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        for ep in range(a.epochs):
            model.train()
            opt.zero_grad()
            loss = run_loss(model, ("xc", "tc", "ac", "zc", "cc", "pc"))
            if name == "cover":
                loss = loss + a.lam * run_loss(model, ("xf", "tf", "af", "zf", "cf", "pf"))
            loss.backward()
            opt.step()
            if (ep + 1) % 5 == 0:
                print(f"{name} ep{ep+1} loss={float(loss.detach()):.4f}", flush=True)
        model.eval()
        torch.save(model.state_dict(), a.output / f"{name}.pt")
        results[name] = {"final_loss": float(loss.detach())}

    # held-out clean eval on eval-match (exclude nothing; r02 eval subset overlap ok for clean anchor)
    pool = np.where(mids == a.eval_match)[0]
    ev = rng.choice(pool, min(a.eval_n, len(pool)), replace=False)
    Xe, Te = raw[ev], team[ev]
    hcx_e = (Xe * (Te == 0)[..., None]).sum(1)[:, 0] / 10.0
    ze = zone_of(hcx_e)
    ce = np.stack([(Xe * (Te == s)[..., None]).sum(1) / 10.0 for s in (0, 1)],
                  axis=1).reshape(len(ev), 4)
    pe = ph[ev]
    with torch.no_grad():
        for name in ("cover", "cleanonly"):
            model = TaskModel()
            model.load_state_dict(torch.load(a.output / f"{name}.pt", map_location="cpu",
                                             weights_only=True), strict=True)
            model.eval()
            z = model.encode(T(Xe).float(), T(Te).long(),
                             T(adj_stored[ev]).float(), "team_mean")
            php, zhp, chp = model.context(z)
            zp = zhp.argmax(-1).numpy()
            f1s = []
            for c in (0, 1, 2):
                tp = int(((zp == c) & (ze == c)).sum())
                fp = int(((zp == c) & (ze != c)).sum())
                fn = int(((zp != c) & (ze == c)).sum())
                f1s.append(2 * tp / (2 * tp + fp + fn + 1e-9))
            results[name].update({
                "clean_acc": round(float((zp == ze).mean()), 4),
                "clean_zone_f1": round(float(np.mean(f1s)), 4),
                "clean_cent_mae": round(float(np.abs(chp.numpy() - ce).mean()), 4),
                "clean_phase_acc": round(float((php.argmax(-1).numpy() == pe).mean()), 4)})
            print(name, results[name], flush=True)
    (a.output / "manifest.json").write_text(json.dumps(
        {"methods": ["cover", "cleanonly"], "epochs": a.epochs, "lr": a.lr, "lam": a.lam,
         "seed": a.seed, "n_flips": len(base), "base_ckpt": str(a.ckpt),
         "results": results,
         "note": "phase labels anchor (spatial shift cannot change time-half)"},
        indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
