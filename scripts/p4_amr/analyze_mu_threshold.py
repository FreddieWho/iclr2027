#!/usr/bin/env python3
"""Mu-threshold analysis (task-mu, ZERO training cost; single-run discipline).

Pre-registered prediction (fixed from Track-H P1 BEFORE the single run below;
no threshold/set-count tuning after peeking):
  routing success per trial increases monotonically with the anchor
  displacement's band-energy concentration C = max_b ||P_b delta||^2/||delta||^2;
  the lowest-concentration bin sits near chance (~0.5); mu/Concentration-alone
  ROC/AUC is significantly above 0.5.

Method: rebuild frozen-operator dev sets deterministically (build_sets is a
pure function of its inputs; N=120 for power, documented as differing from
the report's 40), recompute per-set actual/predicted margins by mirroring
sweep.evaluate_model's per-set loop EXACTLY (frozen code untouched), and add
exact-projector band-energy features. Models: v5 routing x3 seeds + v5cap
CAP x3 seeds (saved checkpoints, eval mode).

Methods note (2026-09-16): sklearn is unimportable in this env (broken scipy
libstdc++ linkage), so the logistic step uses a numpy IRLS ridge implementation
(same model class, same standardization, same clustered bootstrap). Per-set rows
are untouched by the swap.

Outputs: artifacts/phase4_amr/mu_threshold_v1/{rows.json, stats.json,
figure.png, plot_data.json, manifest.json}.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
P4 = ROOT / "artifacts" / "phase4_amr"
OUTPUT = P4 / "mu_threshold_v1"
N_SETS = 120
EPS = 0.25
N_BOOT = 2000


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def band_concentration(v4: Any, adjacency: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Exact-projector band-energy fractions eta_b, concentration C=max, argmax."""
    adj_t = torch.from_numpy(adjacency[None]).float()
    lap = v4.normalized_laplacian(adj_t)
    evals, evecs = torch.linalg.eigh(lap)
    ev = evals[0].numpy()
    eu = evecs[0].numpy()
    etas = []
    for b in range(3):
        lo, hi = v4.BAND_EDGES[b], v4.BAND_EDGES[b + 1]
        m = (ev >= lo) & (ev < hi if b < 2 else ev <= hi)
        e = float(np.linalg.norm((eu[:, m] @ (eu[:, m].T @ delta))) ** 2) if m.any() else 0.0
        etas.append(e)
    etas = np.asarray(etas)
    tot = etas.sum()
    frac = etas / tot if tot > 1e-15 else np.full(3, 1.0 / 3.0)
    return frac, float(frac.max()), int(frac.argmax())


def per_set_rows(model: Any, sets: list[dict[str, Any]], FC: Any, T5R3: Any, v4: Any,
                 method: str, seed: int) -> list[dict[str, Any]]:
    """Mirror of sweep.evaluate_model per-set loop (frozen logic) + eta features."""
    rows = []
    for item in sets:
        base = torch.from_numpy(item["positions"].astype(np.float32))
        team_t = torch.from_numpy(item["teams"].astype(np.int64))
        adj_t = torch.from_numpy(item["adjacency"].astype(np.float32))

        def f_map(flat: torch.Tensor) -> torch.Tensor:
            x = flat.reshape(1, 20, 2)
            xc = x - x.mean(dim=1, keepdim=True)
            z = model.encode(xc, team_t.unsqueeze(0), adj_t.unsqueeze(0), "team_mean")
            return torch.nn.functional.normalize(model.mode_head(z), dim=-1).reshape(-1)

        jac = torch.autograd.functional.jacobian(f_map, base.reshape(-1)).cpu().numpy().astype(np.float64)
        flat_base = base.reshape(-1)
        with torch.no_grad():
            f0 = f_map(flat_base).cpu().numpy().astype(np.float64)
        delta_a = item["anchor_unit"] * EPS
        delta_c = item["control_unit"] * EPS
        if not (FC.boundary_valid(item["positions"], delta_a) and FC.boundary_valid(item["positions"], delta_c)):
            continue
        with torch.no_grad():
            fa = f_map(torch.from_numpy((item["positions"] + delta_a).astype(np.float32)).reshape(-1)).cpu().numpy().astype(np.float64)
            fc_ = f_map(torch.from_numpy((item["positions"] + delta_c).astype(np.float32)).reshape(-1)).cpu().numpy().astype(np.float64)
        if not (np.all(np.isfinite(fa)) and np.all(np.isfinite(fc_))):
            continue
        r_a = float(1.0 - np.dot(f0, fa) / max(np.linalg.norm(f0) * np.linalg.norm(fa), 1e-15))
        r_c = float(1.0 - np.dot(f0, fc_) / max(np.linalg.norm(f0) * np.linalg.norm(fc_), 1e-15))
        da, dc = delta_a.reshape(-1), delta_c.reshape(-1)
        q_a = float(0.5 * np.linalg.norm(jac @ da) ** 2)
        q_c = float(0.5 * np.linalg.norm(jac @ dc) ** 2)
        dq_a = sum(float(0.5 * np.linalg.norm(jac[:, 2 * k:2 * k + 2] @ da[2 * k:2 * k + 2]) ** 2) for k in range(20))
        dq_c = sum(float(0.5 * np.linalg.norm(jac[:, 2 * k:2 * k + 2] @ dc[2 * k:2 * k + 2]) ** 2) for k in range(20))
        if not all(np.isfinite([r_a, r_c, q_a, q_c, dq_a, dq_c])):
            continue
        eta, conc, band = band_concentration(v4, item["adjacency"], delta_a)
        dR, dQ = r_a - r_c, q_a - q_c
        rows.append({"snapshot_id": item["snapshot_id"], "method": method, "seed": seed,
                     "band_argmax": band, "concentration": conc,
                     "eta": [float(x) for x in eta], "support_size": len(item["source"]),
                     "dR": dR, "dQ_full": dQ, "dQ_diag": dq_a - dq_c,
                     "success": bool((dQ > 0) == (dR > 0)),
                     "margin_abs": abs(dR), "baseline_rq_diff": float(item["baseline_rq_diff"])})
    return rows


def roc_auc(y: np.ndarray, s: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    order = np.argsort(s, kind="mergesort")
    ys = y[order]
    n1, n0 = int(ys.sum()), len(ys) - int(ys.sum())
    if n1 == 0 or n0 == 0:
        return float("nan"), np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tps = np.cumsum(ys[::-1])
    fps = np.cumsum((1 - ys)[::-1])
    tpr = np.concatenate([[0.0], tps / n1])
    fpr = np.concatenate([[0.0], fps / n0])
    auc = float(np.trapz(tpr, fpr))  # np.trapz: available on all numpy versions
    return auc, fpr, tpr


def youden_threshold(y: np.ndarray, s: np.ndarray) -> float:
    auc, fpr, tpr = roc_auc(y, s)
    if not np.isfinite(auc):
        return float("nan")
    j = tpr - fpr
    k = int(np.argmax(j[1:])) + 1
    # threshold midway between adjacent sorted unique scores around the elbow
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    return float(ss[max(k - 1, 0)])


def irls_logit(Xs: np.ndarray, y: np.ndarray, l2: float = 1.0,
                iters: int = 200) -> np.ndarray:
    """Ridge-penalized logistic regression via IRLS (numpy-only).

    Replaces sklearn (broken scipy in this env; documented estimator swap -
    same model class, clustered-bootstrap CIs are estimator-agnostic).
    Returns beta with intercept first (intercept unpenalized).
    """
    n, p = Xs.shape
    Xa = np.column_stack([np.ones(n), Xs])
    beta = np.zeros(p + 1)
    for _ in range(iters):
        lin = np.clip(Xa @ beta, -30.0, 30.0)
        pr = 1.0 / (1.0 + np.exp(-lin))
        W = pr * (1.0 - pr) + 1e-12
        grad = Xa.T @ (pr - y)
        grad[1:] += l2 * beta[1:] / n
        H = (Xa.T * W) @ Xa
        H[1:, 1:] += np.eye(p) * (l2 / n)
        step = np.linalg.solve(H + np.eye(p + 1) * 1e-9, grad)
        beta -= step
        if float(np.abs(step).max()) < 1e-10:
            break
    return beta


def main() -> int:
    started = time.perf_counter()
    torch.set_num_threads(1)
    T5R3 = load_module(ROOT / "scripts" / "p3_t5r" / "run_t5r3_sanity.py", "t5r3_mu")
    T5R5 = load_module(ROOT / "scripts" / "p3_t5r" / "run_t5r5_hidden_confirmation.py", "t5r5_mu")
    FC = load_module(ROOT / "scripts" / "p2" / "p2_fracture_controls.py", "fc_mu")
    sweep = load_module(ROOT / "scripts" / "p3_t5r" / "run_p3_epsilon_sweep.py", "sweep_mu")
    sweep.EPSILONS = (0.25,)
    sweep.N_SNAPSHOTS = N_SETS
    v4 = load_module(ROOT / "scripts" / "p4_amr" / "amr_model_v4.py", "v4_mu")
    v5mod = load_module(ROOT / "scripts" / "p4_amr" / "amr_model_v5.py", "v5_mu")
    v5capmod = load_module(ROOT / "scripts" / "p4_amr" / "amr_model_v5cap.py", "v5cap_mu")
    split_lock = json.loads((T5R_ROOT / "split_lock.json").read_text(encoding="utf-8"))
    train = T5R3.load_split("train", split_lock)
    valid = T5R3.load_split("valid", split_lock)
    raw_all = np.concatenate([train.raw, valid.raw]).astype(np.float64)
    team_all = np.concatenate([train.team_slots, valid.team_slots])
    snap_ids = train.snapshot_ids.tolist() + valid.snapshot_ids.tolist()
    sets = sweep.build_sets(T5R3, T5R5, FC, raw_all, team_all, snap_ids)
    print(f"sets rebuilt deterministically: {len(sets)}", flush=True)

    jobs = []
    for seed in (11, 23, 47):
        rec = json.loads((P4 / "m1_v5" / f"record_seed{seed}.json").read_text(encoding="utf-8"))
        jobs.append(("v5", seed, v5mod.AMRModelV5, rec["checkpoint"]["path"]))
        rec = json.loads((P4 / "m1_v5cap" / f"record_seed{seed}.json").read_text(encoding="utf-8"))
        jobs.append(("v5cap", seed, v5capmod.AMRModelV5CAP, rec["checkpoint"]["path"]))
    rows: list[dict[str, Any]] = []
    for method, seed, cls, ckpt in jobs:
        model = cls()
        model.load_state_dict(torch.load(ROOT / ckpt, map_location="cpu", weights_only=True))
        model.eval()
        got = per_set_rows(model, sets, FC, T5R3, v4, method, seed)
        rows.extend(got)
        print(f"{method} seed={seed}: {len(got)} rows", flush=True)
    write_json(OUTPUT / "rows.json", {"n_sets": len(sets), "rows": rows})

    out: dict[str, Any] = {}
    for pool in ("v5", "v5cap", "all"):
        sub = [r for r in rows if pool == "all" or r["method"] == pool]
        y = np.asarray([r["success"] for r in sub], dtype=float)
        C = np.asarray([r["concentration"] for r in sub])
        band = np.asarray([r["band_argmax"] for r in sub])
        mag = np.asarray([r["margin_abs"] for r in sub])
        base = np.asarray([r["baseline_rq_diff"] for r in sub])
        snap = np.asarray([r["snapshot_id"] for r in sub])
        X = np.column_stack([C, (band == 1).astype(float), (band == 2).astype(float), mag, base])
        mu = X.mean(0)
        sd = X.std(0) + 1e-12
        Xs = (X - mu) / sd
        beta = irls_logit(Xs, y)
        coef_conc = float(beta[1])
        # clustered bootstrap by snapshot
        rng = np.random.default_rng(20260916)
        uniq = np.unique(snap)
        b_coefs, b_auc = [], []
        for _ in range(N_BOOT):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(snap == u) for u in pick])
            try:
                bbeta = irls_logit(Xs[idx], y[idx])
            except Exception:
                continue
            b_coefs.append(float(bbeta[1]))
            a, _, _ = roc_auc(y[idx], C[idx])
            if np.isfinite(a):
                b_auc.append(a)
        b_coefs = np.asarray(b_coefs)
        auc, fpr, tpr = roc_auc(y, C)
        thr = youden_threshold(y, C)
        b_auc = np.asarray(b_auc)
        # binned success
        qs = np.quantile(C, np.linspace(0, 1, 6))
        bins = []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (C >= lo) & (C <= hi if hi == qs[-1] else C < hi)
            if m.sum():
                bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                             "rate": float(y[m].mean())})
        # stratified spearman
        strat = {}
        for b in (0, 1, 2):
            m = band == b
            if m.sum() > 5:
                strat[str(b)] = {"n": int(m.sum()),
                                 "spearman": float(T5R3.spearman(C[m], y[m]))}
        out[pool] = {"n": len(sub), "base_rate": float(y.mean()),
                     "spearman_success_conc": float(T5R3.spearman(C, y)),
                     "logit_std_coef_conc": coef_conc,
                     "logit_std_coef_ci95": [float(np.quantile(b_coefs, 0.025)),
                                             float(np.quantile(b_coefs, 0.975))],
                     "auc": auc, "auc_ci95": [float(np.quantile(b_auc, 0.025)),
                                              float(np.quantile(b_auc, 0.975))],
                     "youden_threshold": thr, "bins": bins, "stratified": strat}
        print(pool, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in out[pool].items() if k != "bins" and k != "stratified"}), flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, pool in zip(axes, ("v5", "v5cap", "all")):
            b = out[pool]["bins"]
            xs = [(r["lo"] + r["hi"]) / 2 for r in b]
            ys = [r["rate"] for r in b]
            ax.plot(xs, ys, "o-")
            ax.axhline(out[pool]["base_rate"], ls="--", lw=1)
            thr = out[pool]["youden_threshold"]
            if np.isfinite(thr):
                ax.axvspan(0, thr, alpha=0.15)
            ax.set_title(f"{pool} (AUC={out[pool]['auc']:.2f})")
            ax.set_xlabel("concentration C")
        axes[0].set_ylabel("routing success rate")
        fig.suptitle("Concentration predicts routability")
        fig.tight_layout()
        fig.savefig(OUTPUT / "figure.png", dpi=150)
        plt.close(fig)
        print("figure saved", flush=True)
    except Exception as e:
        print(f"figure skipped: {e}", flush=True)
    write_json(OUTPUT / "stats.json", out)
    write_json(OUTPUT / "manifest.json",
               {"status": "MU_THRESHOLD_ANALYSIS_COMPLETE",
                "n_sets": len(sets), "n_rows": len(rows),
                "elapsed_seconds": round(time.perf_counter() - started, 1)})
    print(json.dumps({"status": "MU_THRESHOLD_ANALYSIS_COMPLETE", "rows": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
