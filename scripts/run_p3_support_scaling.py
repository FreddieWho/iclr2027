#!/usr/bin/env python3
"""P3 support-size scaling: how does support granularity affect the
geometry->response ranking prediction? (Review Part 4, item D5)

Frozen eps=0.25, frozen operator logic, support sizes {1,2,3,4}.
Dev set only; no training; no reserved/external reads.
Reuses the epsilon-sweep machinery (EPSILONS patched to the single
frozen operating point). Outputs artifacts/phase3/support_scaling_v1/.
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

ROOT = Path(__file__).resolve().parents[1]
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
LOCK_PATH = T5R_ROOT / "candidate_lock.json"
OUTPUT = ROOT / "artifacts" / "phase3" / "support_scaling_v1"
SUPPORT_SIZES = (1, 2, 3, 4)
N_SNAPSHOTS = 30


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


def build_sets_sized(T5R3: Any, T5R5: Any, FC: Any, raw: np.ndarray, team_all: np.ndarray,
                     snapshot_ids: list[str], support_size: int) -> list[dict[str, Any]]:
    total = len(raw)
    take = min(N_SNAPSHOTS, total)
    chosen = sorted(set(int(v) for v in np.linspace(0, total - 1, take).round()))
    sets: list[dict[str, Any]] = []
    for snap in chosen:
        positions = raw[snap]
        teams = team_all[snap]
        adjacency = T5R3.knn_adjacency_batch(positions[None].astype(np.float32))[0].astype(np.float64)
        eigenvalues, eigenvectors = T5R5.normalized_laplacian_eig(adjacency)
        rng = np.random.default_rng(FC.stable_seed(T5R5.OPERATOR_ID, snapshot_ids[snap], T5R5.DRAW, f"support{support_size}"))
        team_choice = int(rng.integers(0, 2))
        team_nodes = [int(i) for i in np.flatnonzero(teams == team_choice)]
        source = tuple(sorted(int(v) for v in rng.choice(team_nodes, support_size, replace=False)))
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


def main() -> int:
    started = time.perf_counter()
    if OUTPUT.exists():
        print(f"REFUSED: output exists: {OUTPUT}", file=sys.stderr)
        return 2
    torch.set_num_threads(1)
    T5R3 = load_module(ROOT / "scripts" / "run_t5r3_sanity.py", "t5r3_for_support_scaling")
    T5R5 = load_module(ROOT / "scripts" / "run_t5r5_hidden_confirmation.py", "t5r5_for_support_scaling")
    FC = load_module(ROOT / "scripts" / "p2_fracture_controls.py", "fc_for_support_scaling")
    sweep = load_module(ROOT / "scripts" / "run_p3_epsilon_sweep.py", "eps_sweep_for_support_scaling")
    sweep.EPSILONS = (0.25,)
    split_lock = json.loads((T5R_ROOT / "split_lock.json").read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    train = T5R3.load_split("train", split_lock)
    valid = T5R3.load_split("valid", split_lock)
    raw = np.concatenate([train.raw, valid.raw]).astype(np.float64)
    team = np.concatenate([train.team_slots, valid.team_slots])
    snapshot_ids = train.snapshot_ids.tolist() + valid.snapshot_ids.tolist()

    models: dict[str, Any] = {}
    groups = [("update_ratio_2to1", lock["model"]["checkpoints"]),
              ("fixed_dual_channel_shared_phase_gat", lock["frozen_comparison_set"]["reference"]["checkpoints"])]
    for model_id, checkpoints in groups:
        for seed in ("11", "23", "47"):
            model = T5R3.TaskModel()
            model.load_state_dict(torch.load(ROOT / checkpoints[seed]["path"], map_location="cpu", weights_only=True))
            model.eval()
            models[f"{model_id}__seed{seed}"] = model

    results: dict[str, Any] = {}
    for size in SUPPORT_SIZES:
        sets = build_sets_sized(T5R3, T5R5, FC, raw, team, snapshot_ids, size)
        print(f"support={size}: {len(sets)} sets", flush=True)
        if len(sets) < 20:
            results[str(size)] = {"status": "UNDERSAMPLED", "n_sets": len(sets)}
            continue
        per_model: dict[str, Any] = {}
        for name, model in models.items():
            per_model[name] = sweep.evaluate_model(model, sets, FC, T5R3)
        results[str(size)] = {"n_sets": len(sets), "per_model": per_model}
        print(f"support={size}: done", flush=True)

    summary = {"support_sizes": list(SUPPORT_SIZES), "epsilon": 0.25, "results": results}
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "summary.json", summary)
    manifest = {"status": "P3_SUPPORT_SCALING_COMPLETE", "data": "IDSSE train+valid dev only",
                "models": 6, "support_sizes": list(SUPPORT_SIZES),
                "candidate_lock_sha256": sha256_file(LOCK_PATH),
                "elapsed_seconds": round(time.perf_counter() - started, 1)}
    write_json(OUTPUT / "manifest.json", manifest)
    lines = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256_file(path)}  {path.relative_to(OUTPUT)}")
    (OUTPUT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
