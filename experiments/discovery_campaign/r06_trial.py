#!/usr/bin/env python3
"""R06 trial: spring-system re-simulation + one-step predictors.

Physics: 5 particles in 2D, fixed observed spring graph, semi-implicit Euler.
Every counterfactual initial condition is TRULY re-integrated (no window tricks).
Perturbation laws over initial positions share one antithetic bank
(same marginals/energy); Q_source chosen on source initials only.
Models: tiny message-passing one-step predictor vs joint-MLP baseline
(same info budget: positions+velocities+fixed graph observed by both).
Metric: ||Y_pred^d(t) - Y_true^d(t)|| curves per law + model ranking.
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
from core.perturbations import (balanced_sign_patterns, antithetic_fields,
                                normalize_weights, worst_case_weights,
                                greedy_mode_cover)

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (2, 4)]
K_SPRING, REST, DAMP, DT = 1.0, 1.0, 0.05, 0.02


def rollout(p0: np.ndarray, v0: np.ndarray, steps: int) -> np.ndarray:
    """p0,v0 [N,2] -> positions [steps+1,N,2] (includes initial)."""
    p, v = p0.copy(), v0.copy()
    out = [p.copy()]
    E = np.array(EDGES)
    for _ in range(steps):
        d = p[E[:, 0]] - p[E[:, 1]]                      # [E,2]
        L = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        fmag = K_SPRING * (L - REST)                     # [E,1]
        f = np.zeros_like(p)
        np.add.at(f, E[:, 0], -fmag * d / L)
        np.add.at(f, E[:, 1], fmag * d / L)
        v += (f - DAMP * v) * DT
        p += v * DT
        out.append(p.copy())
    return np.array(out)


def sample_initials(n: int, seed: int):
    rng = np.random.default_rng(seed)
    P, V = [], []
    while len(P) < n:
        p = rng.uniform(-2, 2, size=(5, 2))
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        if d[np.triu_indices(5, 1)].min() < 0.6:
            continue
        P.append(p); V.append(rng.normal(0, 0.5, size=(5, 2)))
    return np.array(P), np.array(V)


class MPNN(nn.Module):
    def __init__(self, hidden=32, edges=None):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(4 + 1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.upd = nn.Sequential(nn.Linear(4 + hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))
        e = torch.tensor(list(edges) if edges is not None else EDGES)
        self.register_buffer("send", e[:, 0])
        self.register_buffer("recv", e[:, 1])

    def forward(self, p, v):
        x = torch.cat([p, v], -1)                       # [B,N,4]
        rel = x[:, self.send] - x[:, self.recv]         # [B,E,4]
        dist = rel[..., :2].norm(dim=-1, keepdim=True)  # [B,E,1]
        m = self.msg(torch.cat([rel, dist], -1))        # [B,E,H]
        agg = torch.zeros(p.shape[0], p.shape[1], m.shape[-1])
        agg.index_add_(1, self.recv, m)
        agg.index_add_(1, self.send, m)
        return self.upd(torch.cat([x, agg], -1))        # [B,N,2] delta-v


class JointMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 10))

    def forward(self, p, v):
        b = p.shape[0]
        return self.net(torch.cat([p, v], -1).reshape(b, -1)).reshape(b, 5, 2)


def train_one_step(model, P, V, Traj, epochs=400, lr=3e-3, seed=0, dt=DT):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # one-step pairs: (p_t,v_t) -> (v_{t+1}-v_t); v from finite diff of traj
    vel = (Traj[:, 1:] - Traj[:, :-1]) / dt                   # [n_scenes,T,5,2]
    Xp, Xv, Y = [], [], []
    T = Traj.shape[1] - 1
    for t in range(T - 1):
        Xp.append(Traj[:, t]); Xv.append(vel[:, t]); Y.append(vel[:, t + 1] - vel[:, t])
    Xp = torch.from_numpy(np.concatenate(Xp)).float()
    Xv = torch.from_numpy(np.concatenate(Xv)).float()
    Y = torch.from_numpy(np.concatenate(Y)).float()
    # standardize flattened 20-dim state vectors with train stats
    Zflat = torch.cat([Xp, Xv], -1).reshape(-1, 20)
    mu, sd = Zflat.mean(0), Zflat.std(0) + 1e-8
    def norm(p, v):
        z = (torch.cat([p, v], -1).reshape(-1, 20) - mu) / sd
        return z[:, :10].reshape(-1, 5, 2), z[:, 10:].reshape(-1, 5, 2)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        pn, vn = norm(Xp.reshape(-1, 5, 2), Xv.reshape(-1, 5, 2))
        loss = ((model(pn, vn).reshape(-1, 2) - Y.reshape(-1, 2)) ** 2).mean()
        loss.backward(); opt.step()
    return {"mu": mu.detach().numpy(), "sd": sd.detach().numpy()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-source", type=int, default=50)
    p.add_argument("--n-target", type=int, default=50)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=9)
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)

    Ptr, Vtr = sample_initials(a.n_train, a.seed)
    Pso, Vso = sample_initials(a.n_source, a.seed + 1)
    Pta, Vta = sample_initials(a.n_target, a.seed + 2)
    Ttr = np.array([rollout(p, v, a.steps) for p, v in zip(Ptr, Vtr)])
    # Ttr [n_train, steps+1, 5, 2]
    mpnn, mlp = MPNN(), JointMLP()
    st_mpnn = train_one_step(mpnn, Ptr, Vtr, Ttr, seed=a.seed)
    st_mlp = train_one_step(mlp, Ptr, Vtr, Ttr, seed=a.seed)
    models = {"mpnn": (mpnn, st_mpnn), "mlp": (mlp, st_mlp)}

    patterns = balanced_sign_patterns((4,), max_patterns=64, seed=a.seed)
    v = np.array([1., 0.])
    fields = antithetic_fields(patterns, a.eps, v)[:, :, :4, :]  # [M,2,4,2] on particles 0..3
    M = len(patterns)

    def true_traj(P0, V0):
        return rollout(P0, V0, a.steps)

    def pred_traj(model, stats, P0, V0):
        mu = torch.from_numpy(stats["mu"]).float(); sd = torch.from_numpy(stats["sd"]).float()
        traj = [P0.copy()]
        cur_p, cur_v = P0.copy(), V0.copy()
        pt = torch.from_numpy(cur_p).float(); vt = torch.from_numpy(cur_v).float()
        for _ in range(a.steps):
            z = (torch.cat([pt, vt], -1).reshape(-1, 20) - mu) / sd
            pn, vn = z[:, :10].reshape(1, 5, 2), z[:, 10:].reshape(1, 5, 2)
            with torch.no_grad():
                dv = model(pn, vn).numpy()[0]
            cur_v = cur_v + dv
            cur_p = cur_p + cur_v * DT
            traj.append(cur_p.copy())
        return np.array(traj)

    laws_result = {}
    for tag, Pset, Vset in (("source", Pso, Vso), ("target", Pta, Vta)):
        n = len(Pset)
        err = np.zeros((2, M, 2, n, a.steps + 1))  # [model,mode,arm,scene,t]
        for mi in range(M):
            for arm in range(2):
                for si in range(n):
                    Pc = Pset[si].copy(); Pc[:4] += fields[mi, arm]
                    Yt = true_traj(Pc, Vset[si])
                    for mdi, (mname, (m, st)) in enumerate(models.items()):
                        Yp = pred_traj(m, st, Pc, Vset[si])
                        err[mdi, mi, arm, si] = np.linalg.norm(Yp - Yt, axis=(1, 2))
        laws_result[tag] = err
    # Q_source from SOURCE pair-errors (mean over arms,scenes,time), applied to TARGET.
    pair_src = laws_result["source"].mean(axis=(2, 3, 4))  # [model,M]
    rep = {}
    for mdi, mname in enumerate(["mpnn", "mlp"]):
        w_dro = worst_case_weights(pair_src[mdi], regularization=1.0)
        succ = np.zeros((M, a.n_source), bool)
        arm_err = laws_result["source"][mdi].mean(axis=(1, 3))  # [M,scenes]
        thr = np.median(arm_err)
        succ = arm_err > thr
        gidx, _ = greedy_mode_cover(succ, k=min(2, M))
        w_greedy = np.zeros(M); w_greedy[gidx] = 1.0 / max(1, len(gidx))
        w_ref = np.full(M, 1.0 / M)
        tgt = laws_result["target"][mdi]  # [M,2,n,T]
        for lname, w in (("Q_ref", w_ref), ("Q_source_dro", w_dro), ("Q_source_greedy", w_greedy)):
            w = normalize_weights(w, M)
            # weighted mean over modes of arm-mean error, then mean over scenes
            curve = (w[:, None, None, None] * tgt).sum(0).mean(1).mean(0)
            rep[f"{mname}/{lname}"] = {"curve": [float(x) for x in curve],
                                       "t20": float(curve[-1]), "weights": [float(x) for x in w]}
        # whitebox per-scene worst mode
        worst = tgt.mean(axis=1).max(axis=0)  # [scenes,T]
        rep[f"{mname}/Q_whitebox"] = {"curve": [float(x) for x in worst.mean(0)],
                                      "t20": float(worst[:, -1].mean())}
        # clean (unperturbed) rollout error on target
        ce = []
        for si in range(a.n_target):
            Yt = true_traj(Pta[si], Vta[si])
            m, st = models[mname]
            Yp = pred_traj(m, st, Pta[si], Vta[si])
            ce.append(np.linalg.norm(Yp - Yt, axis=(1, 2)))
        ce = np.array(ce)
        rep[f"{mname}/clean"] = {"curve": [float(x) for x in ce.mean(0)], "t20": float(ce[:, -1].mean())}
    summary = {"models": ["mpnn", "mlp"], "n_modes": M, "eps": a.eps, "steps": a.steps,
               "edges": EDGES, "dt": DT, "k": K_SPRING, "damping": DAMP,
               "graph": "fixed_and_observed_for_both_models",
               "results": rep,
               "notes": ["All counterfactuals truly re-integrated from perturbed initials.",
                         "Q_source_* chosen on source initials only; curves shown on target initials."]}
    (a.output / "R06_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: round(v["t20"], 4) for k, v in rep.items()}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
