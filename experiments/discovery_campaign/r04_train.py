#!/usr/bin/env python3
"""R04 round 1: dependence-robust training with fixed bank-8 (35 modes).

One GLOBAL pi over modes, updated per batch by entropic DRO on batch pair
losses; applied per scene CONDITIONED on that scene's frozen legal sub-bank
(renormalized; illegal modes excluded, never zero-loss filled). Legality
masks are computed once with the oracle and shared by ALL methods.

Methods (same arch unless --hidden/--feat override, same clean set, same bank,
same perturbed forward budget; all use oracle arm labels):
  minimax : clean + lambda * sum_b pi_b * L_b, pi <- worst_case_weights(batch)
  equal   : clean + lambda * mean_b L_b  (uniform pi)
  adv     : clean + lambda * per-scene worst-arm loss (standard adv baseline)
  unif    : clean + lambda * mean over K=M*2 independent-sign perturbations
            (same per-coordinate amplitude, breaks joint structure; same forwards)
  clean   : clean only (reference, fewer forwards — labeled as such)
lambda=1, reg=1.0, full-batch, fixed epochs. No grid.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn

PACK = Path(__file__).resolve().parents[2] / "docs" / "iclr2027_discovery_campaign_20260917"
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.perturbations import (balanced_sign_patterns, antithetic_fields,
                                normalize_weights, worst_case_weights)
from core.relations import segment_relation
from coord_mlp import CoordMLP


def build_bank(X, patterns, eps, min_margin=0.02):
    F = antithetic_fields(patterns, eps, np.array([1.0])).reshape(len(patterns), 2, 4, 2).astype(np.float32)
    N, M = len(X), len(patterns)
    legal = np.zeros((N, M), bool)
    arm_label = np.full((N, M, 2), -1)
    for i in range(N):
        for m in range(M):
            ok = True
            for arm in range(2):
                try:
                    r = segment_relation(X[i] + F[m, arm])
                except ValueError:
                    ok = False; break
                if r["ambiguous"] or r["margin"] < min_margin:
                    ok = False; break
                arm_label[i, m, arm] = r["label"]
            legal[i, m] = ok
    return F, legal, arm_label


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--reg", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--feat", type=int, default=32)
    p.add_argument("--methods", nargs="+",
                   default=["minimax", "equal", "adv", "clean"],
                   choices=["minimax", "equal", "adv", "unif", "clean"])
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)
    d = np.load(a.train / "scenes.npz")
    X, y = d["positions"].astype(np.float32), d["labels"].astype(np.float32)
    N = len(X)
    mu, sd = X.reshape(-1, 8).mean(0), X.reshape(-1, 8).std(0) + 1e-8
    patterns = balanced_sign_patterns((8,), max_patterns=64, seed=7)
    M = len(patterns)
    F, legal, arm_label = build_bank(X.astype(float), patterns, a.eps)
    torch.manual_seed(a.seed)
    Xc = torch.from_numpy(((X.reshape(-1, 8) - mu) / sd))
    yc = torch.from_numpy(y)
    Xp = torch.from_numpy(((X[:, None, None, :, :] + F[None, :, :, :, :]).reshape(-1, 8) - mu) / sd).reshape(N, M, 2, -1)
    yp = torch.from_numpy(arm_label.astype(np.float32))  # [N,M,2], -1 illegal
    legal_t = torch.from_numpy(legal)
    legal3 = legal_t.unsqueeze(-1)  # [N,M,1] for [N,M,2] broadcast
    # Uniform-sign control: K=M*2 independent perturbations/scene, same
    # per-coordinate amplitude (eps/sqrt(8)), oracle-labeled, same forwards.
    K = M * 2
    urng = np.random.default_rng(a.seed + 1000)
    Us = urng.choice([-1.0, 1.0], size=(N, K, 8)).astype(np.float32) * (a.eps / np.sqrt(8))
    Xf = X.reshape(N, 8).astype(float)
    ulabel = np.full((N, K), -1)
    umask = np.zeros((N, K), bool)
    for i in range(N):
        for k in range(K):
            try:
                r = segment_relation((Xf[i] + Us[i, k]).reshape(4, 2))
            except ValueError:
                continue
            if r["ambiguous"] or r["margin"] < 0.02:
                continue
            ulabel[i, k] = r["label"]
            umask[i, k] = True
    Up = torch.from_numpy((((Xf[:, None, :] + Us).reshape(-1, 8) - mu) / sd).astype(np.float32)).reshape(N, K, -1)
    upl = torch.from_numpy(ulabel.astype(np.float32))
    umk = torch.from_numpy(umask)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    models, opt = {}, {}
    for name in a.methods:
        torch.manual_seed(a.seed)
        models[name] = CoordMLP(a.hidden, a.feat)
        opt[name] = torch.optim.Adam(models[name].parameters(), lr=a.lr)
    hist = {n: [] for n in models}
    for ep in range(a.epochs):
        for name, model in models.items():
            model.train(); opt[name].zero_grad()
            logit_c = model(Xc)
            l_clean = loss_fn(logit_c, yc).mean()
            if name == "clean":
                l_clean.backward(); opt[name].step()
                hist[name].append(float(l_clean.detach()))
                continue
            # Perturbed forwards WITH grad (this is the training signal); a detached
            # copy serves the DRO weight computation (pi is constant w.r.t. theta).
            lp = model(Xp.reshape(-1, 8)).reshape(N, M, 2)
            with torch.no_grad():
                lp_det = lp.detach()
            # pair loss per scene+mode over LEGAL arms only (mean of legal arms)
            arm_ok = torch.from_numpy(arm_label >= 0)  # [N,M,2] per-arm oracle validity
            la = torch.where(arm_ok, loss_fn(lp, torch.clamp(yp, 0, 1)), torch.nan)
            la_det = torch.where(arm_ok, loss_fn(lp_det, torch.clamp(yp, 0, 1)), torch.nan)
            with torch.no_grad():
                arm_cnt = arm_ok.sum(-1).clamp_min(1).float()  # [N,M]
            pair = torch.where(legal_t, torch.nansum(la, -1) / arm_cnt,
                               torch.zeros(N, M))
            if name == "minimax":
                with torch.no_grad():
                    num = torch.where(arm_ok, la_det, torch.tensor(0.0)).sum((0, 2))
                    den = arm_ok.sum((0, 2)).clamp_min(1).float()
                    pi = torch.from_numpy(worst_case_weights(
                        (num / den).numpy(), regularization=a.reg))
                w = pi[None, :] * legal_t.float()
                w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
                l_pert = (w * pair).sum(1).mean()
            elif name == "equal":
                w = legal_t.float()
                w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
                l_pert = (w * pair).sum(1).mean()
            elif name == "adv":
                worst = torch.where(arm_ok, la, torch.tensor(float("-inf"))).amax(-1)
                worst = torch.where(legal_t, worst, torch.zeros_like(worst))
                l_pert = worst.mean()
            elif name == "unif":
                lu = model(Up.reshape(-1, 8)).reshape(N, K)
                lu = torch.where(umk, loss_fn(lu, torch.clamp(upl, 0, 1)), torch.nan)
                ucnt = umk.sum(1).clamp_min(1).float()
                l_pert = (torch.nansum(lu, 1) / ucnt).mean()
            loss = l_clean + a.lam * l_pert
            loss.backward(); opt[name].step()
            hist[name].append(float(loss.detach()))
        if (ep + 1) % 100 == 0:
            print(f"ep {ep+1}: " + " ".join(f"{n}={hist[n][-1]:.4f}" for n in models), flush=True)
    for name, model in models.items():
        model.eval()
        od = a.output / name
        od.mkdir()
        torch.save({"state": model.state_dict(), "hidden": a.hidden, "feat": a.feat,
                    "mu": mu, "sd": sd, "seed": a.seed}, od / "model.pt")
    (a.output / "manifest.json").write_text(json.dumps(
        {"methods": a.methods, "lambda": a.lam, "reg": a.reg,
         "eps": a.eps, "bank": "8 (35 modes)", "epochs": a.epochs, "lr": a.lr,
         "hidden": a.hidden, "feat": a.feat,
         "n_train": N, "legal_coverage_mean": float(legal.mean()),
         "scenes_with_<2_legal_modes": int((legal.sum(1) < 2).sum()),
         "unif_legal_coverage_mean": float(umask.mean()),
         "global_pi": "one vector over 35 modes, per-batch DRO; per-scene conditioned on frozen legal masks",
         "note": "illegal modes excluded, never zero-loss filled; all methods share masks + oracle arm labels"}, indent=2) + "\n")
    print("saved", a.output, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
