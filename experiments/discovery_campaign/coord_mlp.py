#!/usr/bin/env python3
"""Small coordinate MLPs for the segment-crossing relation task.

Shared workhorse for R01/R02/R05 round 1. CPU-tiny by design.
Saves: state_dict, preprocessing stats, config, metrics.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


class CoordMLP(nn.Module):
    def __init__(self, hidden: int = 64, feat: int = 32, in_dim: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.feat_head = nn.Linear(hidden, feat)
        self.cls = nn.Linear(feat, 1)

    def forward(self, x, return_feat: bool = False):
        h = self.net(x)
        z = self.feat_head(h)
        logit = self.cls(z).squeeze(-1)
        return (logit, z) if return_feat else logit


class MarginMLP(CoordMLP):
    """CoordMLP plus a margin-regression head (auxiliary supervision)."""
    def __init__(self, hidden: int = 64, feat: int = 32, in_dim: int = 8):
        super().__init__(hidden, feat, in_dim)
        import torch.nn as nn
        self.mhead = nn.Linear(feat, 1)

    def forward(self, x, return_feat=False, return_margin=False):
        h = self.net(x)
        z = self.feat_head(h)
        logit = self.cls(z).squeeze(-1)
        if return_margin:
            return logit, z, self.mhead(z).squeeze(-1)
        return (logit, z) if return_feat else logit


def load_split(path: Path):
    d = np.load(path / "scenes.npz")
    return d["positions"].astype(np.float32), d["labels"].astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--feat", type=int, default=32)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--name", type=str, default="mlpA")
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    Xtr, ytr = load_split(a.train)
    Xev, yev = load_split(a.eval)
    mu, sd = Xtr.reshape(-1, 8).mean(0), Xtr.reshape(-1, 8).std(0) + 1e-8
    Xt = torch.from_numpy((Xtr.reshape(-1, 8) - mu) / sd)
    yt = torch.from_numpy(ytr)
    Xe = torch.from_numpy((Xev.reshape(-1, 8) - mu) / sd)
    ye = torch.from_numpy(yev)
    model = CoordMLP(a.hidden, a.feat)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(a.epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        for tag, X, y in (("train", Xt, yt), ("eval", Xe, ye)):
            prob = torch.sigmoid(model(X))
            pred = (prob > 0.5).float()
            acc = float((pred == y).float().mean())
            print(f"{a.name} {tag} acc={acc:.4f}", flush=True)
    a.output.mkdir(parents=True, exist_ok=False)
    torch.save({"state": model.state_dict(), "hidden": a.hidden, "feat": a.feat,
                "in_dim": 8, "mu": mu, "sd": sd, "seed": a.seed}, a.output / "model.pt")
    with torch.no_grad():
        acc_tr = float((((torch.sigmoid(model(Xt)) > .5).float()) == yt).float().mean())
        acc_ev = float((((torch.sigmoid(model(Xe)) > .5).float()) == ye).float().mean())
    (a.output / "metrics.json").write_text(json.dumps(
        {"name": a.name, "hidden": a.hidden, "feat": a.feat, "seed": a.seed,
         "epochs": a.epochs, "lr": a.lr, "train_acc": acc_tr, "eval_acc": acc_ev,
         "train_split": str(a.train), "eval_split": str(a.eval)}, indent=2) + "\n")
    print(f"saved {a.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
