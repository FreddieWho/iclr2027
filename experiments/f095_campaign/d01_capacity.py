#!/usr/bin/env python3
"""D01 unified capacity trainer: width x data matrix, one cell per run.

Contract (same as r04b anchor for comparability): mine flips once with
mine_flips(seed=5, cap=12, margin>=0.03); clean BCE + lam * flip BCE;
Adam(lr), epochs; static/single-dev selection only (no AB/CCM/J-driven
stopping). Saves per-method model.pt (r04b-compatible ckpt schema),
loss curves, and a run manifest with params/FLOPs/wall-time/data hashes.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
from coord_mlp import CoordMLP  # noqa: E402
from common import rel_features  # noqa: E402
from r04b_methods import mine_flips  # noqa: E402
from u01_models import TypedPairMLP, SixDistMLP, ConcatMLP, count_params  # noqa: E402
from symmetry import GROUP  # noqa: E402

torch.set_num_threads(4)


def build_model(arch, width, feat):
    if arch == "raw":
        return CoordMLP(width, feat, in_dim=8), {"in_dim": 8}
    if arch == "relfeat":
        return CoordMLP(width, feat, in_dim=10), {"in_dim": 10, "featurize": "rel12"}
    if arch == "typed":
        return TypedPairMLP(width), {"in_dim": 8, "featurize": "scalar8"}
    if arch == "sixdist":
        return SixDistMLP(width, feat), {"in_dim": 6, "featurize": "sixdist"}
    if arch == "concat":
        return ConcatMLP(width, feat), {"in_dim": 8, "featurize": "scalar8"}
    raise ValueError(arch)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True, help="scenes dir with scenes.npz")
    p.add_argument("--output", type=Path, required=True, help="new run dir")
    p.add_argument("--arch", default="raw",
                   choices=["raw", "relfeat", "typed", "sixdist", "concat"])
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--feat", type=int, default=32)
    p.add_argument("--methods", nargs="+", default=["clean", "flipmine"],
                   help="subset of {clean,flipmine,flip_fx}; flip_fx needs --flip-frac")
    p.add_argument("--flip-frac", type=float, default=0.25)
    p.add_argument("--aug", default="none", choices=["none", "g8", "repeat"],
                   help="g8: per-epoch random legal relabeling (same #forwards); "
                   "repeat: duplicated batch (same info, 2x compute control)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--mine-seed", type=int, default=5)
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    d = np.load(a.train / "scenes.npz")
    X, y = d["positions"].astype(np.float32), d["labels"].astype(np.float32)
    N = len(X)
    data_sha = hashlib.sha256((a.train / "scenes.npz").read_bytes()).hexdigest()
    mu8, sd8 = X.reshape(-1, 8).mean(0), X.reshape(-1, 8).std(0) + 1e-8
    gmu, gsd = float(X.mean()), float(X.std() + 1e-8)  # scalar stats (perm-invariant)
    Fr = rel_features(X)
    mur, sdr = Fr.mean(0), Fr.std(0) + 1e-8
    F6 = Fr[:, :6]
    mu6, sd6 = F6.mean(0), F6.std(0) + 1e-8

    mined = mine_flips(X.astype(float), y.astype(int), seed=a.mine_seed)
    K = len(mined)
    rfrac = np.random.default_rng(a.seed + 999)
    perm = rfrac.permutation(K).tolist()
    frac_idx = sorted(perm[:max(1, int(round(a.flip_frac * K)))])
    print(f"mined {K} flips on {N} scenes; frac({a.flip_frac})={len(frac_idx)}", flush=True)

    def std8(arr):
        return torch.from_numpy(((arr.reshape(-1, 8) - mu8) / sd8).astype(np.float32))

    def sstd8(arr):
        mu = np.full(8, gmu, dtype=np.float32)
        sd = np.full(8, gsd, dtype=np.float32)
        return torch.from_numpy(((arr.reshape(-1, 8) - mu) / sd).astype(np.float32))

    Xc, yc = std8(X), torch.from_numpy(y)
    Xs, _ = sstd8(X), None
    Xr = torch.from_numpy(((Fr - mur) / sdr).astype(np.float32))
    X6 = torch.from_numpy(((F6 - mu6) / sd6).astype(np.float32))
    # raw coordinate copies for per-epoch augmentation (permute raw, then
    # standardize/featurize: permuting standardized features would be wrong
    # for rel12/sixdist and for per-index mu8/sd8)
    X_raw44 = X.reshape(N, 4, 2)
    F_ii = np.array([m["scene"] for m in mined])
    F_ee = np.array([m["edit"] for m in mined])
    F_raw44 = (X[F_ii] + F_ee).reshape(len(mined), 4, 2)
    FR_ii = np.array([mined[k]["scene"] for k in frac_idx])
    FR_ee = np.array([mined[k]["edit"] for k in frac_idx])
    FR_raw44 = (X[FR_ii] + FR_ee).reshape(len(frac_idx), 4, 2)
    aug_rng = np.random.default_rng(a.seed + 4242)
    G8 = np.array(GROUP)

    def enc44(arr44, feat):
        if feat == "raw":
            return std8(arr44)
        if feat == "scalar8":
            return sstd8(arr44)
        if feat == "rel12":
            return torch.from_numpy(((rel_features(arr44) - mur) / sdr).astype(np.float32))
        if feat == "sixdist":
            return torch.from_numpy(
                ((rel_features(arr44)[:, :6] - mu6) / sd6).astype(np.float32))
        raise ValueError(feat)

    def perm44(arr44):
        n = len(arr44)
        g = G8[aug_rng.integers(0, len(G8), size=n)]
        return arr44[np.arange(n)[:, None], g]

    def flip_batch(feat):
        ii = np.array([m["scene"] for m in mined])
        ee = np.array([m["edit"] for m in mined])
        yf = torch.from_numpy(np.array([m["new"] for m in mined], dtype=np.float32))
        if feat == "raw":
            return std8(X[ii] + ee), yf
        if feat == "scalar8":
            return sstd8(X[ii] + ee), yf
        if feat == "rel12":
            return torch.from_numpy(((rel_features(X[ii] + ee) - mur) / sdr).astype(np.float32)), yf
        if feat == "sixdist":
            return torch.from_numpy(
                ((rel_features(X[ii] + ee)[:, :6] - mu6) / sd6).astype(np.float32)), yf
        raise ValueError(feat)

    base_of = {"raw": (Xc, "raw"), "relfeat": (Xr, "rel12"), "typed": (Xs, "scalar8"),
               "sixdist": (X6, "sixdist"), "concat": (Xs, "scalar8")}
    xb, feat = base_of[a.arch]
    Xf_full, yf_full = flip_batch(feat) if any(m != "clean" for m in a.methods) else (None, None)
    Xf_frac, yf_frac = None, None
    if "flip_fx" in a.methods:
        ii = np.array([mined[k]["scene"] for k in frac_idx])
        ee = np.array([mined[k]["edit"] for k in frac_idx])
        yff = torch.from_numpy(np.array([mined[k]["new"] for k in frac_idx],
                                        dtype=np.float32))
        arr = (X[ii] + ee)
        if feat == "raw":
            Xf_frac = std8(arr)
        elif feat == "scalar8":
            Xf_frac = sstd8(arr)
        elif feat == "rel12":
            Xf_frac = torch.from_numpy(((rel_features(arr) - mur) / sdr).astype(np.float32))
        elif feat == "sixdist":
            Xf_frac = torch.from_numpy(
                ((rel_features(arr)[:, :6] - mu6) / sd6).astype(np.float32))
        yf_frac = yff
    bce = nn.BCEWithLogitsLoss(reduction="none")
    models, opt = {}, {}
    for name in a.methods:
        torch.manual_seed(a.seed)
        models[name], _ = build_model(a.arch, a.width, a.feat)
        opt[name] = torch.optim.Adam(models[name].parameters(), lr=a.lr)
    n_params = {n: count_params(m) for n, m in models.items()}
    print(f"params: {n_params}", flush=True)
    hist = {n: [] for n in a.methods}
    for ep in range(a.epochs):
        if a.aug == "none":
            xb_e, Xff_e, Xfx_e = xb, Xf_full, Xf_frac
            yc_e, yff_e, yfx_e = yc, yf_full, yf_frac
        elif a.aug == "repeat":
            xb_e = torch.cat([xb, xb]) if xb is not None else None
            Xff_e = torch.cat([Xf_full, Xf_full]) if Xf_full is not None else None
            Xfx_e = torch.cat([Xf_frac, Xf_frac]) if Xf_frac is not None else None
            yc_e = torch.cat([yc, yc])
            yff_e = torch.cat([yf_full, yf_full]) if yf_full is not None else None
            yfx_e = torch.cat([yf_frac, yf_frac]) if yf_frac is not None else None
        else:  # g8: same-count replacement, fresh random legal relabeling
            xb_e = enc44(perm44(X_raw44), feat)
            yc_e = yc
            Xff_e = enc44(perm44(F_raw44), feat) if Xf_full is not None else None
            yff_e = yf_full
            Xfx_e = enc44(perm44(FR_raw44), feat) if Xf_frac is not None else None
            yfx_e = yf_frac
        for name, model in models.items():
            model.train()
            opt[name].zero_grad()
            lb = bce(model(xb_e), yc_e).mean()
            if name == "clean":
                loss = lb
            elif name == "flip_fx":
                loss = lb + a.lam * bce(model(Xfx_e), yfx_e).mean()
            else:
                loss = lb + a.lam * bce(model(Xff_e), yff_e).mean()
            loss.backward()
            opt[name].step()
            hist[name].append(float(loss.detach()))
        if (ep + 1) % 100 == 0:
            print(f"ep {ep+1}: " + " ".join(f"{n}={hist[n][-1]:.4f}" for n in a.methods),
                  flush=True)
    wall = time.time() - t0
    for name, model in models.items():
        model.eval()
        od = a.output / name
        od.mkdir()
        _, info = build_model(a.arch, a.width, a.feat)
        ck = {"state": model.state_dict(), "hidden": a.width, "feat": a.feat,
              "seed": a.seed, **info}
        if info["in_dim"] == 8 and info.get("featurize") != "scalar8":
            ck.update({"mu": mu8, "sd": sd8})
        elif info.get("featurize") == "scalar8":
            ck.update({"mu": np.full(8, gmu), "sd": np.full(8, gsd)})
        elif info["in_dim"] == 10:
            ck.update({"mu": mur, "sd": sdr})
        else:
            ck.update({"mu": mu6, "sd": sd6})
        torch.save(ck, od / "model.pt")
        np.savez_compressed(od / "training_curve.npz",
                            loss=np.array(hist[name], dtype=np.float32))
    manifest = {"arch": a.arch, "width": a.width, "feat": a.feat,
                "methods": a.methods, "epochs": a.epochs, "lr": a.lr, "lam": a.lam,
                "seed": a.seed, "mine_seed": a.mine_seed, "n_train": N,
                "aug": a.aug,
                "flip_frac": a.flip_frac if "flip_fx" in a.methods else None,
                "n_mined_flips": K, "data_sha256": data_sha,
                "train_dir": str(a.train), "n_params": n_params,
                "wall_seconds": round(wall, 1)}
    (a.output / "run_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("saved", a.output, f"wall={wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
