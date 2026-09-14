#!/usr/bin/env python3
"""P3 epsilon-sweep: where does the local-geometry prediction break down?

Scientific question (publication review Part 4-A): the core identity
1 - cos(z(x), z(x+delta)) ~= 1/2 delta^T G_f delta is a local Taylor
approximation. At eps -> 0 the prediction is definitional; the scientific
content is (a) how well the ranking prediction holds at finite eps, and
(b) how the quadratic approximation error grows with eps.

Design notes:
- The quadratic predictor's RANKING is invariant to eps (eps^2 scaling is
  monotone), so the sweep measures exactly when the ACTUAL response ranking
  departs from the frozen local prediction.
- Dev set only (train/valid via T5R3.load_split, which cannot load the
  reserved match). No training, no re-selection, no reserved/external reads.
- Intervention sets are built once with the frozen operator logic
  (support 4, draw 0, seed_root 20260905); deltas are stored unit-scaled and
  re-scaled per eps. Boundary validity is re-checked per eps (valid fraction
  is itself reported).
- Untrained controls: same TaskModel class, seeded random init, no
  checkpoint load. Isolates what training contributes to the
  geometry->response relation.

Outputs: artifacts/phase3/epsilon_sweep_v1/{summary.json, manifest.json,
SHA256SUMS}.
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
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
LOCK_PATH = T5R_ROOT / "candidate_lock.json"
SPLIT_LOCK = T5R_ROOT / "split_lock.json"
OUTPUT = ROOT / "artifacts" / "phase3" / "epsilon_sweep_v1"

EPSILONS = (0.0625, 0.125, 0.25, 0.5, 1.0)
N_SNAPSHOTS = 60
MIN_VALID_PER_EPS = 20


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


def build_sets(T5R3: Any, T5R5: Any, FC: Any, raw: np.ndarray, team_all: np.ndarray,
               snapshot_ids: list[str]) -> list[dict[str, Any]]:
    """Frozen operator logic; store unit-scaled anchor/control fields."""
    total = len(raw)
    take = min(N_SNAPSHOTS, total)
    chosen = sorted(set(int(v) for v in np.linspace(0, total - 1, take).round()))
    sets: list[dict[str, Any]] = []
    for snap in chosen:
        positions = raw[snap]
        teams = team_all[snap]
        adjacency = T5R3.knn_adjacency_batch(positions[None].astype(np.float32))[0].astype(np.float64)
        eigenvalues, eigenvectors = T5R5.normalized_laplacian_eig(adjacency)
        rng = np.random.default_rng(FC.stable_seed(T5R5.OPERATOR_ID, snapshot_ids[snap], T5R5.DRAW))
        team_choice = int(rng.integers(0, 2))
        team_nodes = [int(i) for i in np.flatnonzero(teams == team_choice)]
        source = tuple(sorted(int(v) for v in rng.choice(team_nodes, T5R5.SUPPORT_SIZE, replace=False)))
        base = FC.unit_base_displacement_field(20, source, (T5R5.OPERATOR_ID, snapshot_ids[snap], T5R5.DRAW))
        anchor_unit = FC.scale_displacement_field(base, 1.0)
        anchor_features = FC.spectral_topology_features_2d(eigenvalues, eigenvectors, anchor_unit, source, adjacency, T5R5.N_BANDS)
        ranked = []
        for target_array in FC.enumerate_target_supports(teams, source, team_choice):
            target = tuple(int(v) for v in target_array)
            for bijection in FC.enumerate_vector_bijections(source, target):
                control_unit = FC.reassigned_vector_field(base, source, target, bijection, 1.0)
                features = FC.spectral_topology_features_2d(eigenvalues, eigenvectors, control_unit, target, adjacency, T5R5.N_BANDS)
                if features.component_count != anchor_features.component_count:
                    continue
                score = (abs(anchor_features.rq_total - features.rq_total)
                         + float(np.abs(np.asarray(anchor_features.band_power_total) - np.asarray(features.band_power_total)).sum())
                         + abs(anchor_features.induced_density - features.induced_density) / 0.10
                         + abs(anchor_features.cut_weight - features.cut_weight) / 1.0)
                ranked.append((score, target, bijection, control_unit, features))
        if not ranked:
            continue
        ranked.sort(key=lambda item: (item[0], item[1], str(sorted(item[2].items()))))
        _, target, bijection, control_unit, control_features = ranked[0]
        sets.append({"snapshot_id": snapshot_ids[snap], "positions": positions, "teams": teams,
                     "adjacency": adjacency, "anchor_unit": anchor_unit, "control_unit": control_unit,
                     "baseline_rq_diff": abs(float(anchor_features.rq_total) - float(control_features.rq_total)),
                     "source": list(source), "target": list(target)})
    return sets


def evaluate_model(model: Any, sets: list[dict[str, Any]], FC: Any, T5R3: Any) -> dict[str, Any]:
    """One Jacobian per set; per-eps actual responses; quadratic from stored J."""
    per_eps: dict[str, Any] = {f"{eps:g}": {"valid": 0, "delta_r": [], "delta_q_full": [],
                                            "delta_q_diag": [], "actual_anchor": [], "quad_anchor": []}
                               for eps in EPSILONS}
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
        for eps in EPSILONS:
            delta_a = item["anchor_unit"] * eps
            delta_c = item["control_unit"] * eps
            if not (FC.boundary_valid(item["positions"], delta_a) and FC.boundary_valid(item["positions"], delta_c)):
                continue
            with torch.no_grad():
                fa = f_map(torch.from_numpy((item["positions"] + delta_a).astype(np.float32)).reshape(-1)).cpu().numpy().astype(np.float64)
                fc_ = f_map(torch.from_numpy((item["positions"] + delta_c).astype(np.float32)).reshape(-1)).cpu().numpy().astype(np.float64)
            if not (np.all(np.isfinite(fa)) and np.all(np.isfinite(fc_))):
                continue
            r_a = float(1.0 - np.dot(f0, fa) / max(np.linalg.norm(f0) * np.linalg.norm(fa), 1e-15))
            r_c = float(1.0 - np.dot(f0, fc_) / max(np.linalg.norm(f0) * np.linalg.norm(fc_), 1e-15))
            da = delta_a.reshape(-1)
            dc = delta_c.reshape(-1)
            q_a = float(0.5 * np.linalg.norm(jac @ da) ** 2)
            q_c = float(0.5 * np.linalg.norm(jac @ dc) ** 2)
            diag_a = sum(float(0.5 * np.linalg.norm(jac[:, 2 * k:2 * k + 2] @ da[2 * k:2 * k + 2]) ** 2) for k in range(20))
            diag_c = sum(float(0.5 * np.linalg.norm(jac[:, 2 * k:2 * k + 2] @ dc[2 * k:2 * k + 2]) ** 2) for k in range(20))
            if not all(np.isfinite([r_a, r_c, q_a, q_c, diag_a, diag_c])):
                continue
            row = per_eps[f"{eps:g}"]
            row["valid"] += 1
            row["delta_r"].append(r_a - r_c)
            row["delta_q_full"].append(q_a - q_c)
            row["delta_q_diag"].append(diag_a - diag_c)
            row["actual_anchor"].append(r_a)
            row["quad_anchor"].append(q_a)
    out: dict[str, Any] = {}
    for eps in EPSILONS:
        row = per_eps[f"{eps:g}"]
        n = row["valid"]
        entry: dict[str, Any] = {"valid": n}
        if n >= MIN_VALID_PER_EPS:
            dr = np.asarray(row["delta_r"])
            std = float(np.std(dr))
            entry["response_std"] = std
            if std > 1e-12:
                entry["full_spearman"] = float(T5R3.spearman(np.asarray(row["delta_q_full"]), dr))
                entry["diag_spearman"] = float(T5R3.spearman(np.asarray(row["delta_q_diag"]), dr))
                actual = np.asarray(row["actual_anchor"])
                quad = np.asarray(row["quad_anchor"])
                denom = np.maximum(np.abs(actual), 1e-12)
                entry["quad_rel_error_median"] = float(np.median(np.abs(quad - actual) / denom))
                entry["quad_over_actual_median"] = float(np.median(quad / np.maximum(actual, 1e-15)))
                entry["status"] = "OK"
            else:
                entry["status"] = "UNDERDEGRADED"
        else:
            entry["status"] = "UNDERSAMPLED"
        out[f"{eps:g}"] = entry
    return out


def main() -> int:
    started = time.perf_counter()
    if OUTPUT.exists():
        print(f"REFUSED: output exists: {OUTPUT}", file=sys.stderr)
        return 2
    torch.set_num_threads(1)
    T5R3 = load_module(ROOT / "scripts" / "run_t5r3_sanity.py", "t5r3_for_eps_sweep")
    T5R5 = load_module(ROOT / "scripts" / "run_t5r5_hidden_confirmation.py", "t5r5_for_eps_sweep")
    FC = load_module(ROOT / "scripts" / "p2_fracture_controls.py", "fc_for_eps_sweep")
    split_lock = json.loads(SPLIT_LOCK.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    train = T5R3.load_split("train", split_lock)
    valid = T5R3.load_split("valid", split_lock)
    raw = np.concatenate([train.raw, valid.raw]).astype(np.float64)
    team = np.concatenate([train.team_slots, valid.team_slots])
    snapshot_ids = train.snapshot_ids.tolist() + valid.snapshot_ids.tolist()
    print(f"dev pool: {len(raw)} snapshots", flush=True)
    sets = build_sets(T5R3, T5R5, FC, raw, team, snapshot_ids)
    print(f"intervention sets: {len(sets)}", flush=True)
    if len(sets) < 30:
        print("REFUSED: too few intervention sets", file=sys.stderr)
        return 2

    results: dict[str, Any] = {}
    groups = [("update_ratio_2to1", lock["model"]["checkpoints"], "trained"),
              ("fixed_dual_channel_shared_phase_gat", lock["frozen_comparison_set"]["reference"]["checkpoints"], "trained"),
              ("raw_single_channel_phase_gat_team_mean", lock["frozen_comparison_set"]["raw_baseline"]["checkpoints"], "trained")]
    for model_id, checkpoints, status in groups:
        for seed in ("11", "23", "47"):
            model = T5R3.TaskModel()
            model.load_state_dict(torch.load(ROOT / checkpoints[seed]["path"], map_location="cpu", weights_only=True))
            model.eval()
            results[f"{model_id}__seed{seed}__{status}"] = evaluate_model(model, sets, FC, T5R3)
            print(f"done {model_id} seed {seed}", flush=True)
    for seed in ("11", "23", "47"):
        torch.manual_seed(int(FC.stable_seed("untrained_control_v1", "task_model", seed)) % (2**31))
        model = T5R3.TaskModel()
        model.eval()
        results[f"task_model__seed{seed}__untrained"] = evaluate_model(model, sets, FC, T5R3)
        print(f"done untrained seed {seed}", flush=True)

    summary = {"epsilons": list(EPSILONS), "n_sets": len(sets), "min_valid_per_eps": MIN_VALID_PER_EPS,
               "results": results}
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "summary.json", summary)
    manifest = {"status": "P3_EPSILON_SWEEP_COMPLETE", "data": "IDSSE train+valid dev only",
                "models": {"trained": 9, "untrained": 3}, "n_sets": len(sets),
                "candidate_lock_sha256": sha256_file(LOCK_PATH),
                "elapsed_seconds": round(time.perf_counter() - started, 1)}
    write_json(OUTPUT / "manifest.json", manifest)
    lines = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256_file(path)}  {path.relative_to(OUTPUT)}")
    (OUTPUT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "n_sets": len(sets)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
