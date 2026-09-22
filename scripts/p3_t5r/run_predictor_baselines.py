#!/usr/bin/env python3
"""P3 predictor baselines: does the Jacobian geometry carry nontrivial info?

Two dev-only analyses (review Part 4, items D3/D4), no training, no
reserved/external reads:

D4 simple predictor baselines — model-free statistics of the
configuration/intervention as predictors of the response difference
(r_a - r_b) for 2:1 (3 seeds), compared against the frozen full-Jacobian
predictor (~0.56 at eps=0.25):
  - pairwise-distance change difference (the statistic the intrinsic task
    itself uses, computed directly from coordinates)
  - radius-of-gyration change difference
  - spectral Rayleigh baseline (frozen, recomputed here for reference)

D3 k-sensitivity — the spectral baseline features depend on the kNN graph
(k=4 frozen); recompute the baseline predictor with k in {2,4,6,8} on the
same intervention sets and check the ranking does not flip.

Outputs: artifacts/phase3/predictor_baselines_v1/{summary.json,
manifest.json, SHA256SUMS}.
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
LOCK_PATH = T5R_ROOT / "candidate_lock.json"
OUTPUT = ROOT / "artifacts" / "phase3" / "predictor_baselines_v1"
EPS = 0.25
K_VALUES = (2, 4, 6, 8)
N_SNAPSHOTS = 40


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pairwise_mean_distance(positions: np.ndarray) -> float:
    diff = positions[None, :, :] - positions[:, None, :]
    dist = np.linalg.norm(diff, axis=-1)
    n = len(positions)
    return float(dist.sum() / (n * (n - 1)))


def radius_of_gyration(positions: np.ndarray) -> float:
    centered = positions - positions.mean(axis=0, keepdims=True)
    return float(np.sqrt((centered ** 2).sum(axis=-1).mean()))


def spectral_rq(adjacency: np.ndarray, field: np.ndarray) -> float:
    """Rayleigh quotient of a 2D node field on the graph Laplacian (L kron I_2)."""
    degree = adjacency.sum(axis=1)
    degree[degree == 0] = 1.0
    lap = np.diag(degree) - adjacency
    num = 0.0
    den = 0.0
    for c in range(field.shape[1]):
        f = field[:, c]
        num += float(f @ lap @ f)
        den += float(f @ f)
    return num / max(den, 1e-15)


def main() -> int:
    started = time.perf_counter()
    if OUTPUT.exists():
        print(f"REFUSED: output exists: {OUTPUT}", file=sys.stderr)
        return 2
    torch.set_num_threads(1)
    T5R3 = load_module(ROOT / "scripts" / "p3_t5r" / "run_t5r3_sanity.py", "t5r3_for_baselines")
    T5R5 = load_module(ROOT / "scripts" / "p3_t5r" / "run_t5r5_hidden_confirmation.py", "t5r5_for_baselines")
    FC = load_module(ROOT / "scripts" / "p2" / "p2_fracture_controls.py", "fc_for_baselines")
    sweep = load_module(ROOT / "scripts" / "p3_t5r" / "run_p3_epsilon_sweep.py", "sweep_for_baselines")
    sweep.EPSILONS = (EPS,)
    sweep.N_SNAPSHOTS = N_SNAPSHOTS
    split_lock = json.loads((T5R_ROOT / "split_lock.json").read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    train = T5R3.load_split("train", split_lock)
    valid = T5R3.load_split("valid", split_lock)
    raw = np.concatenate([train.raw, valid.raw]).astype(np.float64)
    team = np.concatenate([train.team_slots, valid.team_slots])
    snap_ids = train.snapshot_ids.tolist() + valid.snapshot_ids.tolist()
    sets = sweep.build_sets(T5R3, T5R5, FC, raw, team, snap_ids)
    print(f"intervention sets: {len(sets)}", flush=True)

    # simple per-arm predictors (model-free)
    simple: dict[str, list[float]] = {"pairwise_dist": [], "gyration": [], "spectral_rq_k4": []}
    k_rq: dict[int, list[float]] = {k: [] for k in K_VALUES}
    for item in sets:
        pos = item["positions"]
        da = item["anchor_unit"] * EPS
        dc = item["control_unit"] * EPS
        pa, pc = pos + da, pos + dc
        simple["pairwise_dist"].append(
            (pairwise_mean_distance(pa) - pairwise_mean_distance(pos))
            - (pairwise_mean_distance(pc) - pairwise_mean_distance(pos)))
        simple["gyration"].append(
            (radius_of_gyration(pa) - radius_of_gyration(pos))
            - (radius_of_gyration(pc) - radius_of_gyration(pos)))
        simple["spectral_rq_k4"].append(
            spectral_rq(item["adjacency"], da) - spectral_rq(item["adjacency"], dc))
        for k in K_VALUES:
            adj_k = T5R3.knn_adjacency_batch(pos[None].astype(np.float32), k=k)[0].astype(np.float64)
            k_rq[k].append(spectral_rq(adj_k, da) - spectral_rq(adj_k, dc))

    results: dict[str, Any] = {"simple_predictors": {}, "k_sensitivity": {}}
    for seed in ("11", "23", "47"):
        model = T5R3.TaskModel()
        model.load_state_dict(torch.load(
            ROOT / lock["model"]["checkpoints"][seed]["path"], map_location="cpu", weights_only=True))
        model.eval()
        delta_r: list[float] = []
        for item in sets:
            base = torch.from_numpy(item["positions"].astype(np.float32))
            team_t = torch.from_numpy(item["teams"].astype(np.int64))
            adj_t = torch.from_numpy(item["adjacency"].astype(np.float32))

            def f_map(flat: torch.Tensor) -> torch.Tensor:
                x = flat.reshape(1, 20, 2)
                xc = x - x.mean(dim=1, keepdim=True)
                z = model.encode(xc, team_t.unsqueeze(0), adj_t.unsqueeze(0), "team_mean")
                return torch.nn.functional.normalize(model.mode_head(z), dim=-1).reshape(-1)

            with torch.no_grad():
                f0 = f_map(base.reshape(-1)).cpu().numpy().astype(np.float64)
                fa = f_map(torch.from_numpy((item["positions"] + item["anchor_unit"] * EPS).astype(np.float32)).reshape(-1)).cpu().numpy().astype(np.float64)
                fc_ = f_map(torch.from_numpy((item["positions"] + item["control_unit"] * EPS).astype(np.float32)).reshape(-1)).cpu().numpy().astype(np.float64)
            r_a = float(1.0 - np.dot(f0, fa) / max(np.linalg.norm(f0) * np.linalg.norm(fa), 1e-15))
            r_c = float(1.0 - np.dot(f0, fc_) / max(np.linalg.norm(f0) * np.linalg.norm(fc_), 1e-15))
            delta_r.append(r_a - r_c)
        dr = np.asarray(delta_r)
        std = float(np.std(dr))
        seed_out: dict[str, Any] = {"response_std": std}
        for name, values in simple.items():
            seed_out[name] = float(T5R3.spearman(np.asarray(values), dr)) if std > 1e-12 else float("nan")
        results["simple_predictors"][seed] = seed_out
        print(f"seed {seed}: " + " ".join(f"{k}={seed_out[k]:.3f}" for k in simple), flush=True)
        if seed == "11":
            for k in K_VALUES:
                results["k_sensitivity"][str(k)] = float(T5R3.spearman(np.asarray(k_rq[k]), dr)) if std > 1e-12 else float("nan")

    write_json(OUTPUT / "summary.json", results)
    manifest = {"status": "P3_PREDICTOR_BASELINES_COMPLETE", "data": "IDSSE train+valid dev only",
                "n_sets": len(sets), "epsilon": EPS, "k_values": list(K_VALUES),
                "candidate_lock_sha256": sha256_file(LOCK_PATH),
                "elapsed_seconds": round(time.perf_counter() - started, 1)}
    write_json(OUTPUT / "manifest.json", manifest)
    lines = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256_file(path)}  {path.relative_to(OUTPUT)}")
    (OUTPUT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "k": results["k_sensitivity"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
