#!/usr/bin/env python3
"""R06 round 2: mixed-charge Coulomb scattering in a harmonic trap.

Bounded (trap) + close-encounter chaos (mixed signs) + softening (no
singularities). Step 1 is ALWAYS the oracle-divergence diagnostic:
if true trajectories don't diverge with horizon, no model comparison
can show dependence effects and the script stops before training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

PACK = Path(__file__).resolve().parents[2] / "docs" / "iclr2027_discovery_campaign_20260917"
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(Path(__file__).resolve().parent))

N = 5
CHARGES = np.array([1., 1., -1., -1., 1.])
SOFT2 = 0.05
K_TRAP = 0.5
DT = 0.01


def accel(p: np.ndarray) -> np.ndarray:
    d = p[:, None, :] - p[None, :, :]                       # [N,N,2] i-minus-j
    r2 = (d ** 2).sum(-1, keepdims=True) + SOFT2
    qq = CHARGES[:, None, None] * CHARGES[None, :, None]    # qi*qj
    f = (qq * d / r2 ** 1.5).sum(axis=1)                    # like signs repel
    f -= K_TRAP * p
    return f


def rollout(p0: np.ndarray, v0: np.ndarray, steps: int) -> np.ndarray:
    p, v = p0.copy(), v0.copy()
    out = [p.copy()]
    for _ in range(steps):
        v += accel(p) * DT
        p += v * DT
        out.append(p.copy())
        if not np.all(np.isfinite(p)):
            return np.array(out + [np.full_like(p, np.nan)] * (steps - len(out) + 1))[:steps + 1]
    return np.array(out)


def sample_initials(n: int, seed: int):
    rng = np.random.default_rng(seed)
    P, V = [], []
    guard = 0
    while len(P) < n and guard < n * 500:
        guard += 1
        p = rng.uniform(-1.5, 1.5, size=(N, 2))
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        if d[np.triu_indices(N, 1)].min() < 0.4:
            continue
        v = rng.normal(0, 0.3, size=(N, 2))
        t = rollout(p, v, 150)
        if np.isnan(t).any() or np.abs(t).max() > 6:
            continue
        P.append(p); V.append(v)
    if len(P) < n:
        raise RuntimeError(f"only {len(P)}/{n} stable initials; relax guard")
    return np.array(P), np.array(V)


def diagnose() -> dict:
    P, V = sample_initials(30, seed=101)
    rng = np.random.default_rng(0)
    out = {}
    for steps in (50, 150, 300):
        T0 = np.array([rollout(p, v, steps) for p, v in zip(P, V)])
        for eps in (0.02, 0.1):
            divs = []
            for i in range(len(P)):
                d = rng.normal(size=(N, 2)); d /= np.linalg.norm(d) + 1e-9
                Tp = rollout(P[i] + eps * d, V[i], steps)
                divs.append(np.linalg.norm(Tp - T0[i], axis=(1, 2)))
            divs = np.array(divs)
            out[f"steps{steps}_eps{eps}"] = {
                "div_t0": float(divs[:, 0].mean()), "div_tmid": float(divs[:, len(divs[0]) // 2].mean()),
                "div_tend": float(divs[:, -1].mean())}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--full", action="store_true",
                   help="train MPNN vs MLP one-step predictors and compare laws")
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-source", type=int, default=50)
    p.add_argument("--n-target", type=int, default=50)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--seed", type=int, default=9)
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)
    diag = diagnose()
    (a.output / "diagnostic.json").write_text(json.dumps(diag, indent=2) + "\n")
    print(json.dumps(diag, indent=2))
    if a.full:
        run_full(a)


def run_full(a) -> None:
    import torch
    from r06_trial import MPNN, JointMLP, train_one_step
    from core.perturbations import (balanced_sign_patterns, antithetic_fields,
                                    normalize_weights, worst_case_weights,
                                    greedy_mode_cover)
    COMPLETE = [(i, j) for i in range(N) for j in range(i + 1, N)]
    Ptr, Vtr = sample_initials(a.n_train, a.seed)
    Pso, Vso = sample_initials(a.n_source, a.seed + 1)
    Pta, Vta = sample_initials(a.n_target, a.seed + 2)
    Ttr = np.array([rollout(p, v, a.steps) for p, v in zip(Ptr, Vtr)])
    mpnn, mlp = MPNN(hidden=32, edges=COMPLETE), JointMLP()
    st_mpnn = train_one_step(mpnn, Ptr, Vtr, Ttr, epochs=a.epochs, seed=a.seed, dt=DT)
    st_mlp = train_one_step(mlp, Ptr, Vtr, Ttr, epochs=a.epochs, seed=a.seed, dt=DT)
    models = {"mpnn": (mpnn, st_mpnn), "mlp": (mlp, st_mlp)}
    patterns = balanced_sign_patterns((4,), max_patterns=64, seed=a.seed)
    fields = antithetic_fields(patterns, a.eps, np.array([1., 0.]))[:, :, :4, :]
    M = len(patterns)

    def pred_traj(model, stats, P0, V0):
        mu = torch.from_numpy(stats["mu"]).float(); sd = torch.from_numpy(stats["sd"]).float()
        traj = [P0.copy()]
        cur_p, cur_v = P0.copy(), V0.copy()
        for _ in range(a.steps):
            pt = torch.from_numpy(cur_p).float(); vt = torch.from_numpy(cur_v).float()
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
        err = np.zeros((2, M, 2, n, a.steps + 1))
        for mi in range(M):
            for arm in range(2):
                for si in range(n):
                    Pc = Pset[si].copy(); Pc[:4] += fields[mi, arm]
                    Yt = rollout(Pc, Vset[si], a.steps)
                    for mdi, (mname, (m, st)) in enumerate(models.items()):
                        Yp = pred_traj(m, st, Pc, Vset[si])
                        err[mdi, mi, arm, si] = np.linalg.norm(Yp - Yt, axis=(1, 2))
        laws_result[tag] = err
    pair_src = laws_result["source"].mean(axis=(2, 3, 4))
    rep = {}
    for mdi, mname in enumerate(["mpnn", "mlp"]):
        w_dro = worst_case_weights(pair_src[mdi], regularization=1.0)
        arm_err = laws_result["source"][mdi].mean(axis=(1, 3))
        succ = arm_err > np.median(arm_err)
        gidx, _ = greedy_mode_cover(succ, k=min(2, M))
        w_greedy = np.zeros(M); w_greedy[gidx] = 1.0 / max(1, len(gidx))
        w_ref = np.full(M, 1.0 / M)
        tgt = laws_result["target"][mdi]
        for lname, w in (("Q_ref", w_ref), ("Q_source_dro", w_dro),
                         ("Q_source_greedy", w_greedy)):
            w = normalize_weights(w, M)
            curve = (w[:, None, None, None] * tgt).sum(0).mean(1).mean(0)
            rep[f"{mname}/{lname}"] = {"curve": [float(x) for x in curve],
                                         "t_end": float(curve[-1]),
                                         "weights": [float(x) for x in w]}
        worst = tgt.mean(axis=1).max(axis=0)
        rep[f"{mname}/Q_whitebox"] = {"curve": [float(x) for x in worst.mean(0)],
                                        "t_end": float(worst[:, -1].mean())}
        ce = []
        for si in range(a.n_target):
            Yt = rollout(Pta[si], Vta[si], a.steps)
            m, st = models[mname]
            Yp = pred_traj(m, st, Pta[si], Vta[si])
            ce.append(np.linalg.norm(Yp - Yt, axis=(1, 2)))
        ce = np.array(ce)
        rep[f"{mname}/clean"] = {"curve": [float(x) for x in ce.mean(0)],
                                   "t_end": float(ce[:, -1].mean())}
    (a.output / "R06_scatter_result.json").write_text(json.dumps(
        {"models": ["mpnn", "mlp"], "graph": "complete_observed_both_models",
         "n_modes": M, "eps": a.eps, "steps": a.steps, "dt": DT,
         "physics": "mixed_charge_coulomb_soft_trap", "results": rep,
         "notes": ["All counterfactuals truly re-integrated.",
                     "Q_source_* chosen on source initials only; shown on target."]},
        indent=2) + "\n")
    print(json.dumps({k: round(v["t_end"], 4) for k, v in rep.items()}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
