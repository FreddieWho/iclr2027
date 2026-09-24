#!/usr/bin/env python3
"""D08-six: pct-preserve loss on sixdist encoder (collapse-attribution bonus).

Question: does the sixdist s47 flip-collapse (U02 J .253 vs .439/.426) come
from flip-BCE loss design or from the seed? D08 showed pct matches flipmine
on raw/rel encoders. If six+pct rescues s47 (J >> .253), collapse is
loss-design, not seed fate; if it also collapses, seed noise floor (L-006)
is confirmed for sixdist specifically.

Protocol mirrors d08_fairfight.py pct arm exactly: continuation from the
same-seed sixdist clean checkpoint (U02), preserve = train-correct soft
targets from that checkpoint, same flips (r04b_s11 mined archive), same
budget (300 epochs, Adam 1e-2, lam=1). SixDistMLP w64 (in_dim=6).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
sys.path.insert(0, str(ROOT / "docs" / "f095dfa_review_pack" / "checks"))
from common import rel_features  # noqa: E402
from u10_models import SixDistMLP  # noqa: E402

torch.set_num_threads(4)
ART = ROOT / "artifacts" / "discovery_campaign"
OUT = ROOT / "artifacts" / "f095_campaign" / "D08SIX"
EPOCHS, LR, LAM, LAM_PRES = 300, 1e-2, 1.0, 1.0
SEEDS = (11, 23, 47)


def sha_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X, y = d["positions"].astype(np.float32), d["labels"].astype(np.float32)
    mine = np.load(ART / "r04b_s11" / "mined.npz", allow_pickle=True)
    meta = mine["meta"]
    K = len(meta)
    ii = np.array([int(r[0]) for r in meta])
    ee = np.stack([mine[f"edit_{k}"] for k in range(K)]).astype(np.float32)
    yf = torch.from_numpy(np.array([int(r[1]) for r in meta], dtype=np.float32))
    F6 = rel_features(X)[:, :6]
    mu, sd = F6.mean(0), F6.std(0) + 1e-8
    fe = lambda A: torch.from_numpy(  # noqa: E731
        ((rel_features(A)[:, :6] - mu) / sd).astype(np.float32))
    Xc, yc = fe(X), torch.from_numpy(y)
    Xf = fe((X[ii] + ee))
    summary = {"recipe": {"epochs": EPOCHS, "lr": LR, "lam": LAM,
                          "lam_pres": LAM_PRES, "protocol": "d08 pct on sixdist"},
               "seeds": {}}
    for seed in SEEDS:
        torch.manual_seed(seed)
        ckd = ROOT / "artifacts" / "f095_campaign" / "U02" / f"sixdist_w64_s{seed}" / "clean"
        ck = torch.load(ckd / "model.pt", map_location="cpu", weights_only=False)
        model = SixDistMLP(64, 32)
        model.load_state_dict(ck["state"])
        model.eval()
        with torch.no_grad():
            base_logit = model(Xc).numpy()
        keep = (base_logit > 0).astype(int) == yc.numpy().astype(int)
        Tp = torch.from_numpy(1 / (1 + np.exp(-base_logit))).float()
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        bce = nn.BCEWithLogitsLoss()
        hist = []
        for ep in range(EPOCHS):
            model.train()
            opt.zero_grad()
            loss = (bce(model(Xc), yc) + LAM * bce(model(Xf), yf)
                    + LAM_PRES * bce(model(Xc[keep]), Tp[keep]))
            loss.backward()
            opt.step()
            hist.append(round(float(loss.detach()), 4))
        od = OUT / f"six_pct_s{seed}"
        od.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "hidden": 64, "feat": 32,
                    "in_dim": 6, "featurize": "sixdist", "mu": mu, "sd": sd,
                    "seed": seed, "method": "pct", "continued_from": str(ckd),
                    "hist": hist}, od / "model.pt")
        summary["seeds"][f"s{seed}"] = {
            "ckpt_sha": sha_of(od / "model.pt"),
            "loss_first": hist[0], "loss_last": hist[-1],
            "collapsed": bool(hist[-1] > 0.5)}
        print(f"s{seed} loss {hist[0]} -> {hist[-1]}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "D08SIX_SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote", OUT / "D08SIX_SUMMARY.json")


if __name__ == "__main__":
    main()
