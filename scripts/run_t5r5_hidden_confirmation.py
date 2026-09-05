#!/usr/bin/env python3
"""T5R5 one-time hidden confirmation on IDSSE reserved match J03WQQ.

Runs ONLY after the candidate lock is verified. No retraining, no threshold
or protocol adaptation, exactly one reserved read into a versioned output.

Steps: verify lock+config -> build reserved task views (frozen T5R2 rules) ->
task-semantic confirmation (9 frozen checkpoints + analytic control) ->
actual-intervention confirmation (frozen support-reallocation operator,
Jacobian geometry vs observed response) -> frozen gate evaluation -> manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
T5R3_PATH = ROOT / "scripts" / "run_t5r3_sanity.py"
PREPARE_PATH = ROOT / "scripts" / "prepare_idsse_t5r2.py"
FRACTURE_CONTROLS_PATH = ROOT / "scripts" / "p2_fracture_controls.py"
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
IDSSE_ROOT = ROOT / "artifacts" / "data_v2" / "idsse"
LOCK_PATH = T5R_ROOT / "candidate_lock.json"
HIDDEN_CONFIG_DEFAULT = ROOT / "configs" / "t5r5_hidden_confirmation.yaml"
INTERVENTION_LOCK = T5R_ROOT / "t5r5_intervention_lock.json"
DEFAULT_OUTPUT = T5R_ROOT / "t5r5_reserved_j03wqq_v1"

OPERATOR_ID = "t5r5_support_reallocation_v1"
EPSILON = 0.25
SUPPORT_SIZE = 4
MAX_SNAPSHOTS = 250
SEED_ROOT = 20260905
DRAW = 0
N_BANDS = 6


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_lock_and_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not LOCK_PATH.is_file():
        raise RuntimeError("candidate_lock.json missing")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("status") != "T5R5_CANDIDATE_LOCKED":
        raise RuntimeError("candidate lock status invalid")
    if lock.get("identity", {}).get("selected_candidate") != "update_ratio_2to1":
        raise RuntimeError("locked candidate is not update_ratio_2to1")
    for seed, record in lock.get("model", {}).get("checkpoints", {}).items():
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"selected checkpoint mismatch seed {seed}")
    for group in ("reference", "raw_baseline"):
        for seed, record in lock.get("frozen_comparison_set", {}).get(group, {}).get("checkpoints", {}).items():
            path = ROOT / record["path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"{group} checkpoint mismatch seed {seed}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("candidate_lock_sha256") != sha256_file(LOCK_PATH):
        raise RuntimeError("hidden config differs: candidate lock hash mismatch")
    if config.get("intervention_lock_sha256") != sha256_file(INTERVENTION_LOCK):
        raise RuntimeError("hidden config differs: intervention lock hash mismatch")
    if config.get("intervention", {}).get("epsilon") != EPSILON:
        raise RuntimeError("hidden config differs: epsilon changed")
    intervention = json.loads(INTERVENTION_LOCK.read_text(encoding="utf-8"))
    if intervention.get("status") != "T5R5_INTERVENTION_LOCKED":
        raise RuntimeError("intervention lock status invalid")
    return lock, config, intervention


def normalized_laplacian_eig(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    adjacency = np.asarray(adjacency, dtype=np.float64)
    degree = adjacency.sum(axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalized = np.eye(len(adjacency)) - (inv_sqrt[:, None] * adjacency * inv_sqrt[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    return eigenvalues.astype(np.float64), eigenvectors.astype(np.float64)


VARIANT_SPECS = {
    "update_ratio_2to1": {"context_view": "raw", "intrinsic_view": "centered", "pooling": "team_mean"},
    "fixed_dual_channel_shared_phase_gat": {"context_view": "raw", "intrinsic_view": "centered", "pooling": "team_mean"},
    "raw_single_channel_phase_gat_team_mean": {"context_view": "raw", "intrinsic_view": "raw", "pooling": "team_mean"},
}

def evaluate_tasks(T5R3: Any, lock: dict[str, Any], staging: Path) -> dict[str, Any]:
    raw = np.load(staging / "positions_raw_reserved_holdout.npy").astype(np.float32)
    centered = np.load(staging / "positions_centered_reserved_holdout.npy").astype(np.float32)
    team = np.load(staging / "team_slots_reserved_holdout.npy").astype(np.int64)
    index = pd.read_parquet(staging / "snapshot_index_reserved_holdout.parquet")
    pair_path = staging / "natural_pair_ranking_reserved_holdout.parquet"
    pairs = pd.read_parquet(pair_path) if pair_path.exists() else pd.DataFrame()
    n_pairs = int(len(pairs))
    id_to_index = {str(value): i for i, value in enumerate(index["snapshot_id"].astype(str))}
    if n_pairs:
        pair_query = np.asarray([id_to_index[str(v)] for v in pairs["query_snapshot_id"]], dtype=np.int64)
        pair_positive = np.asarray([id_to_index[str(v)] for v in pairs["positive_snapshot_id"]], dtype=np.int64)
        pair_negative = np.asarray([id_to_index[str(v)] for v in pairs["negative_snapshot_id"]], dtype=np.int64)
        if bool((pair_query < 0).any() or (pair_positive < 0).any() or (pair_negative < 0).any()):
            raise RuntimeError("reserved pair references unknown snapshot")
    else:
        pair_query = pair_positive = pair_negative = np.zeros(0, dtype=np.int64)
    data = T5R3.SplitData(
        name="reserved_holdout", raw=raw, centered=centered, team_slots=team,
        snapshot_ids=index["snapshot_id"].astype(str).to_numpy(),
        match_ids=index["source_match_id"].astype(str).to_numpy(),
        phase=T5R3._label_index(index["phase_label"], ("firstHalf", "secondHalf"), "phase"),
        zone=T5R3._label_index(index["home_field_zone"], ("defensive_third", "middle_third", "attacking_third"), "field zone"),
        centroids=index[["home_centroid_x", "home_centroid_y", "away_centroid_x", "away_centroid_y"]].to_numpy(dtype=np.float32),
        pair_query=pair_query, pair_positive=pair_positive, pair_negative=pair_negative,
        pair_matches=pairs["source_match_id"].astype(str).to_numpy() if n_pairs else np.zeros(0, dtype=str),
        pair_positive_geometry=pairs["positive_internal_geometry_distance"].to_numpy(dtype=np.float32) if n_pairs else np.zeros(0, dtype=np.float32),
        pair_negative_geometry=pairs["negative_internal_geometry_distance"].to_numpy(dtype=np.float32) if n_pairs else np.zeros(0, dtype=np.float32),
    )
    if set(data.match_ids.tolist()) != {"J03WQQ"}:
        raise RuntimeError(f"reserved match set mismatch: {sorted(set(data.match_ids.tolist()))}")
    adjacency = T5R3.knn_adjacency_batch(data.raw)
    np.save(staging / "adjacency_reserved_holdout.npy", adjacency)
    groups = [("update_ratio_2to1", lock["model"]["checkpoints"])]
    groups.append(("fixed_dual_channel_shared_phase_gat", lock["frozen_comparison_set"]["reference"]["checkpoints"]))
    groups.append(("raw_single_channel_phase_gat_team_mean", lock["frozen_comparison_set"]["raw_baseline"]["checkpoints"]))
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for model_id, checkpoints in groups:
            spec = VARIANT_SPECS[model_id]
            for seed in ("11", "23", "47"):
                model = T5R3.TaskModel()
                model.load_state_dict(torch.load(ROOT / checkpoints[seed]["path"], map_location="cpu", weights_only=True))
                model.eval()
                z_ctx = T5R3.encode_split(model, data, adjacency, spec["context_view"], spec["pooling"])
                z_mode = T5R3.encode_split(model, data, adjacency, spec["intrinsic_view"], spec["pooling"])
                context = T5R3.prediction_metrics(data, model, z_ctx)
                translation = T5R3.translation_metrics(model, data, adjacency, {**spec, "name": model_id}, z_ctx, z_mode)
                cross = T5R3.cross_readout_metrics(data, model, z_ctx, z_mode)
                if n_pairs:
                    intrinsic = T5R3.pair_metrics(data, model, z_mode)
                else:
                    intrinsic = {"status": "UNDEFINED_NO_PAIRS", "pair_count": 0}
                records.append({
                    "model_id": model_id, "seed": int(seed),
                    "parameter_count": int(sum(p.numel() for p in model.parameters())),
                    "context": context, "intrinsic": intrinsic, "translation": translation, "cross_readout": cross,
                    "checkpoint_sha256": checkpoints[seed]["sha256"],
                })
                print(f"hidden task {model_id} seed={seed}: zone={context['field_zone_match_grouped_macro_f1']:.4f} "
                      f"cent={context['centroid_match_grouped_mae']:.4f} pairs={n_pairs}", flush=True)
    analytic_block: dict[str, Any] = {"status": "UNDEFINED_NO_PAIRS"} if not n_pairs else T5R3.analytic_pair_metrics(data)
    return {"records": records, "analytic_control": analytic_block, "n_snapshots": int(len(raw)), "n_pairs": n_pairs,
            "adjacency_sha256": sha256_file(staging / "adjacency_reserved_holdout.npy")}


def summarize_tasks(task_block: dict[str, Any]) -> list[dict[str, Any]]:
    summary = []
    for model_id in ("update_ratio_2to1", "fixed_dual_channel_shared_phase_gat", "raw_single_channel_phase_gat_team_mean"):
        rows = [r for r in task_block["records"] if r["model_id"] == model_id]
        entry: dict[str, Any] = {"model_id": model_id, "seed_count": len(rows)}
        for key, section, field in (
            ("match_half_f1", "context", "phase_match_grouped_macro_f1"),
            ("field_zone_f1", "context", "field_zone_match_grouped_macro_f1"),
            ("centroid_mae", "context", "centroid_match_grouped_mae"),
            ("z_mode_translation", "translation", "z_mode_global_translation_response"),
            ("z_ctx_accessibility", "translation", "z_ctx_context_accessibility_response"),
            ("latent_correlation", "cross_readout", "mean_abs_paired_latent_correlation"),
        ):
            values = [float(r[section][field]) for r in rows]
            entry[key + "_mean"] = float(np.mean(values))
            entry[key + "_by_seed"] = values
        if task_block["n_pairs"]:
            for key, section, field in (("pair_accuracy", "intrinsic", "match_grouped_ranking_accuracy"),
                                        ("mrr_at_2", "intrinsic", "match_grouped_mrr_at_2"),
                                        ("geometry_spearman", "intrinsic", "natural_geometry_latent_distance_spearman")):
                values = [float(r[section][field]) for r in rows]
                entry[key + "_mean"] = float(np.mean(values))
                entry[key + "_by_seed"] = values
        else:
            entry["pair_status"] = "UNDEFINED_NO_PAIRS"
        summary.append(entry)
    return summary

def run_intervention(T5R3: Any, FC: Any, lock: dict[str, Any], staging: Path) -> dict[str, Any]:
    raw = np.load(staging / "positions_raw_reserved_holdout.npy").astype(np.float64)
    team_all = np.load(staging / "team_slots_reserved_holdout.npy").astype(np.int64)
    snapshot_ids = pd.read_parquet(staging / "snapshot_index_reserved_holdout.parquet")["snapshot_id"].astype(str).tolist()
    total = len(raw)
    take = min(MAX_SNAPSHOTS, total)
    chosen = sorted(set(int(v) for v in np.linspace(0, total - 1, take).round()))
    print(f"intervention snapshots: {len(chosen)}/{total}", flush=True)
    arm_rows: list[dict[str, Any]] = []
    complete_pairs: list[dict[str, Any]] = []
    incomplete = 0
    for order, snap in enumerate(chosen):
        positions = raw[snap]
        teams = team_all[snap]
        adjacency = T5R3.knn_adjacency_batch(positions[None].astype(np.float32))[0].astype(np.float64)
        eigenvalues, eigenvectors = normalized_laplacian_eig(adjacency)
        rng = np.random.default_rng(FC.stable_seed(OPERATOR_ID, snapshot_ids[snap], DRAW))
        team_choice = int(rng.integers(0, 2))
        team_nodes = [int(i) for i in np.flatnonzero(teams == team_choice)]
        source = tuple(sorted(int(v) for v in rng.choice(team_nodes, SUPPORT_SIZE, replace=False)))
        base = FC.unit_base_displacement_field(20, source, (OPERATOR_ID, snapshot_ids[snap], DRAW))
        anchor = FC.scale_displacement_field(base, EPSILON)
        anchor_valid = FC.boundary_valid(positions, anchor)
        anchor_features = FC.spectral_topology_features_2d(eigenvalues, eigenvectors, anchor, source, adjacency, N_BANDS)
        ranked = []
        for target_array in FC.enumerate_target_supports(teams, source, team_choice):
            target = tuple(int(v) for v in target_array)
            for bijection in FC.enumerate_vector_bijections(source, target):
                control = FC.reassigned_vector_field(base, source, target, bijection, EPSILON)
                if not FC.boundary_valid(positions, control):
                    continue
                features = FC.spectral_topology_features_2d(eigenvalues, eigenvectors, control, target, adjacency, N_BANDS)
                if features.component_count != anchor_features.component_count:
                    continue
                score = (abs(anchor_features.rq_total - features.rq_total)
                         + float(np.abs(np.asarray(anchor_features.band_power_total) - np.asarray(features.band_power_total)).sum())
                         + abs(anchor_features.induced_density - features.induced_density) / 0.10
                         + abs(anchor_features.cut_weight - features.cut_weight) / 1.0)
                ranked.append((score, target, bijection, control, features))
        if anchor_valid and ranked:
            ranked.sort(key=lambda item: (item[0], item[1], str(sorted(item[2].items()))))
            _, target, bijection, control, control_features = ranked[0]
            status = "COMPLETE"
            assignment = {int(k): int(v) for k, v in bijection.items()}
        else:
            target, assignment, control, control_features = None, {}, None, None
            status = "INCOMPLETE"
            incomplete += 1
        arm_rows.append({
            "snapshot_order": order, "snapshot_index": int(snap), "snapshot_id": snapshot_ids[snap],
            "team": team_choice, "source_support": list(source),
            "target_support": None if target is None else list(target),
            "endpoint_assignment": {str(k): int(v) for k, v in assignment.items()},
            "anchor_delta": anchor.tolist(),
            "control_delta": None if control is None else np.asarray(control).tolist(),
            "anchor_valid": bool(anchor_valid),
            "anchor_rq_total": float(anchor_features.rq_total),
            "anchor_band_power_total": [float(v) for v in anchor_features.band_power_total],
            "anchor_cut_weight": float(anchor_features.cut_weight),
            "control_rq_total": None if control_features is None else float(control_features.rq_total),
            "control_cut_weight": None if control_features is None else float(control_features.cut_weight),
            "status": status,
        })
        if status == "COMPLETE":
            complete_pairs.append({
                "snapshot_order": order, "snapshot_index": int(snap), "snapshot_id": snapshot_ids[snap],
                "positions": positions, "teams": teams, "adjacency": adjacency,
                "delta_A": anchor, "delta_B": np.asarray(control),
                "baseline_rq_diff": abs(float(anchor_features.rq_total) - float(control_features.rq_total)),
            })
    write_json(staging / "intervention_arms.json", arm_rows)
    return {"arm_rows": arm_rows, "complete_pairs": complete_pairs, "n_incomplete": incomplete, "n_chosen": len(chosen)}


def evaluate_intervention_models(T5R3: Any, lock: dict[str, Any], staging: Path, built: dict[str, Any]) -> dict[str, Any]:
    complete_pairs = built["complete_pairs"]
    results: dict[str, Any] = {"arms_sha256": sha256_file(staging / "intervention_arms.json")}
    for model_id, group in (("update_ratio_2to1", lock["model"]["checkpoints"]),
                            ("fixed_dual_channel_shared_phase_gat", lock["frozen_comparison_set"]["reference"]["checkpoints"])):
        per_seed: dict[str, Any] = {}
        for seed in ("11", "23", "47"):
            model = T5R3.TaskModel()
            model.load_state_dict(torch.load(ROOT / group[seed]["path"], map_location="cpu", weights_only=True))
            model.eval()
            delta_r: list[float] = []
            delta_q_full: list[float] = []
            delta_q_diag: list[float] = []
            baseline_diff: list[float] = []
            per_pair: list[dict[str, Any]] = []
            for pair in complete_pairs:
                base = torch.from_numpy(pair["positions"].astype(np.float32))
                team_t = torch.from_numpy(pair["teams"].astype(np.int64))
                adj_t = torch.from_numpy(pair["adjacency"].astype(np.float32))

                def f_map(flat: torch.Tensor) -> torch.Tensor:
                    x = flat.reshape(1, 20, 2)
                    xc = x - x.mean(dim=1, keepdim=True)
                    z = model.encode(xc, team_t.unsqueeze(0), adj_t.unsqueeze(0), "team_mean")
                    return F.normalize(model.mode_head(z), dim=-1).reshape(-1)

                with torch.no_grad():
                    f_base = f_map(base.reshape(-1)).cpu().numpy().astype(np.float64)
                    f_a = f_map(torch.from_numpy((pair["positions"] + pair["delta_A"]).astype(np.float32)).reshape(-1)).cpu().numpy().astype(np.float64)
                    f_b = f_map(torch.from_numpy((pair["positions"] + pair["delta_B"]).astype(np.float32)).reshape(-1)).cpu().numpy().astype(np.float64)
                denom_a = max(float(np.linalg.norm(f_base) * np.linalg.norm(f_a)), 1e-15)
                denom_b = max(float(np.linalg.norm(f_base) * np.linalg.norm(f_b)), 1e-15)
                r_a = float(1.0 - np.dot(f_base, f_a) / denom_a)
                r_b = float(1.0 - np.dot(f_base, f_b) / denom_b)
                jac = torch.autograd.functional.jacobian(f_map, base.reshape(-1)).cpu().numpy().astype(np.float64)
                delta_a = pair["delta_A"].reshape(-1)
                delta_b = pair["delta_B"].reshape(-1)
                q_a = float(0.5 * np.linalg.norm(jac @ delta_a) ** 2)
                q_b = float(0.5 * np.linalg.norm(jac @ delta_b) ** 2)
                diag_a = sum(float(0.5 * np.linalg.norm(jac[:, 2 * k:2 * k + 2] @ delta_a[2 * k:2 * k + 2]) ** 2) for k in range(20))
                diag_b = sum(float(0.5 * np.linalg.norm(jac[:, 2 * k:2 * k + 2] @ delta_b[2 * k:2 * k + 2]) ** 2) for k in range(20))
                if not all(np.isfinite([r_a, r_b, q_a, q_b, diag_a, diag_b])):
                    continue
                delta_r.append(r_a - r_b)
                delta_q_full.append(q_a - q_b)
                delta_q_diag.append(diag_a - diag_b)
                baseline_diff.append(pair["baseline_rq_diff"])
                per_pair.append({"snapshot_id": pair["snapshot_id"], "delta_r": r_a - r_b,
                                 "delta_q_full": q_a - q_b, "delta_q_diag": diag_a - diag_b,
                                 "baseline_rq_diff": pair["baseline_rq_diff"]})
            arr_r = np.asarray(delta_r)
            valid_n = int(len(arr_r))
            response_std = float(np.std(arr_r)) if valid_n else 0.0
            if valid_n >= 10 and response_std > 1e-12:
                full_sp = float(T5R3.spearman(np.asarray(delta_q_full), arr_r))
                diag_sp = float(T5R3.spearman(np.asarray(delta_q_diag), arr_r))
                base_sp = float(T5R3.spearman(np.asarray(baseline_diff), arr_r))
                direction = float(np.mean(np.sign(arr_r) == np.sign(np.asarray(delta_q_full))))
                status_m = "OK"
            else:
                full_sp = diag_sp = base_sp = direction = float("nan")
                status_m = "INTERVENTION_UNDERDEGRADED"
            per_seed[seed] = {
                "valid_pairs": valid_n, "response_std": response_std, "status": status_m,
                "full_spearman": full_sp, "diagonal_spearman": diag_sp,
                "spectral_baseline_spearman": base_sp, "direction_accuracy": direction,
                "full_minus_baseline": float(full_sp - base_sp) if np.isfinite(full_sp) and np.isfinite(base_sp) else float("nan"),
                "pairs": per_pair,
            }
            print(f"hidden intervention {model_id} seed={seed}: n={valid_n} full={full_sp:.4f} diag={diag_sp:.4f} base={base_sp:.4f}", flush=True)
        results[model_id] = per_seed
    results["n_sampled_snapshots"] = built["n_chosen"]
    results["n_complete_sets"] = len(complete_pairs)
    results["n_incomplete_sets"] = built["n_incomplete"]
    return results

def decide_gates(task_summary: list[dict[str, Any]], intervention: dict[str, Any], n_pairs: int) -> dict[str, Any]:
    by_id = {entry["model_id"]: entry for entry in task_summary}
    sel = by_id["update_ratio_2to1"]
    ref = by_id["fixed_dual_channel_shared_phase_gat"]
    raw = by_id["raw_single_channel_phase_gat_team_mean"]
    underpowered = n_pairs < 30
    task_checks = {
        "z_mode_translation_below_1e_6": bool(sel["z_mode_translation_mean"] < 1e-6),
        "field_zone_drop_within_0_05": bool(raw["field_zone_f1_mean"] - sel["field_zone_f1_mean"] <= 0.05),
        "centroid_increase_within_0_05": bool(sel["centroid_mae_mean"] - raw["centroid_mae_mean"] <= 0.05),
    }
    if not underpowered and "pair_accuracy_mean" in sel:
        task_checks["pair_within_0_02_of_reference"] = bool(sel["pair_accuracy_mean"] - ref["pair_accuracy_mean"] >= -0.02)
        task_checks["geometry_mean_delta_non_negative"] = bool(sel["geometry_spearman_mean"] - ref["geometry_spearman_mean"] >= 0)
        seeds_nonneg = sum(1 for a, b in zip(sel["geometry_spearman_by_seed"], ref["geometry_spearman_by_seed"]) if a - b >= 0)
        task_checks["geometry_seeds_non_negative_at_least_2"] = bool(seeds_nonneg >= 2)
    else:
        task_checks["pair_and_geometry"] = "UNDERpowered_SKIPPED"
    task_pass = all(v is True for v in task_checks.values())

    def family_stats(model_id: str) -> dict[str, Any]:
        block = intervention[model_id]
        seeds = [block[s] for s in ("11", "23", "47")]
        ok = [s for s in seeds if s["status"] == "OK"]
        full = [s["full_spearman"] for s in ok]
        base = [s["spectral_baseline_spearman"] for s in ok]
        diff = [s["full_minus_baseline"] for s in ok]
        direction = [s["direction_accuracy"] for s in ok]
        diag = [s["diagonal_spearman"] for s in ok]
        return {
            "seeds_ok": len(ok),
            "mean_full": float(np.mean(full)) if full else float("nan"),
            "mean_baseline": float(np.mean(base)) if base else float("nan"),
            "mean_diff": float(np.mean(diff)) if diff else float("nan"),
            "seeds_full_above_baseline": int(sum(1 for f, b in zip(full, base) if f > b)),
            "mean_direction": float(np.mean(direction)) if direction else float("nan"),
            "mean_diag": float(np.mean(diag)) if diag else float("nan"),
        }

    sel_stats = family_stats("update_ratio_2to1")
    ref_stats = family_stats("fixed_dual_channel_shared_phase_gat")

    def strong(stats: dict[str, Any]) -> bool:
        return bool(stats["seeds_ok"] == 3 and stats["mean_full"] >= 0.5
                    and stats["mean_diff"] >= 0.2 and stats["seeds_full_above_baseline"] >= 2)

    def positive(stats: dict[str, Any]) -> bool:
        return bool(stats["seeds_ok"] >= 2 and np.isfinite(stats["mean_full"])
                    and stats["mean_full"] > stats["mean_baseline"] and stats["mean_direction"] > 0.5
                    and stats["mean_full"] >= stats["mean_diag"] - 0.05)

    sel_strong = strong(sel_stats)
    sel_positive = positive(sel_stats)
    ref_positive = positive(ref_stats)
    leakage_ok = bool(sel["latent_correlation_mean"] < 0.5)
    protocol_ok = True
    if task_pass and sel_strong and leakage_ok and protocol_ok:
        verdict = "T5R5_PASS_STRONG"
    elif (task_pass and sel_positive) or (sel_strong and not task_pass) or (underpowered and sel_positive and sel["z_mode_translation_mean"] < 1e-6):
        verdict = "T5R5_PASS_TASK_MECHANISM_MIXED"
    elif (not task_pass) and (sel_positive or ref_positive):
        verdict = "T5R5_FAIL_SELECTED_CANDIDATE_BUT_MECHANISM_SURVIVES"
    else:
        verdict = "T5R5_FAIL_MECHANISM"
    return {
        "task_checks": task_checks, "task_pass": bool(task_pass),
        "pair_underpowered": bool(underpowered),
        "selected_intervention": sel_stats, "reference_intervention": ref_stats,
        "selected_strong": bool(sel_strong), "selected_positive": bool(sel_positive),
        "reference_positive": bool(ref_positive), "leakage_ok": bool(leakage_ok),
        "verdict": verdict,
    }


def write_sha256sums(staging: Path) -> None:
    lines = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            lines.append(f"{h.hexdigest()}  {path.relative_to(staging)}")
    (staging / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_reserved_views(PREPARE: Any, staging: Path) -> dict[str, Any]:
    metadata_map = PREPARE.load_metadata_map(IDSSE_ROOT / "match_metadata.json")
    index, raw, _, _ = PREPARE.build_snapshots(IDSSE_ROOT, staging, metadata_map, "reserved_holdout")
    try:
        pairs, diagnostics = PREPARE.build_natural_pairs(index, raw)
    except ValueError:
        pairs, diagnostics = pd.DataFrame(), {"pairs": 0, "status": "UNDERpowered_NO_TRIPLETS"}
    if len(pairs):
        pairs.to_parquet(staging / "natural_pair_ranking_reserved_holdout.parquet", index=False, compression="zstd")
    index.to_parquet(staging / "context_labels_reserved_holdout.parquet", index=False, compression="zstd")
    files = {}
    for name in ("snapshot_index_reserved_holdout.parquet", "context_labels_reserved_holdout.parquet",
                 "positions_raw_reserved_holdout.npy", "positions_centered_reserved_holdout.npy", "team_slots_reserved_holdout.npy"):
        files[name] = sha256_file(staging / name)
    if len(pairs):
        files["natural_pair_ranking_reserved_holdout.parquet"] = sha256_file(staging / "natural_pair_ranking_reserved_holdout.parquet")
    return {
        "n_snapshots": int(len(index)), "n_pairs": int(len(pairs)),
        "phase_counts": index["phase_label"].value_counts().sort_index().to_dict(),
        "pair_diagnostics": {str(k): int(v) if isinstance(v, (int, np.integer)) else v for k, v in dict(diagnostics).items()},
        "files": files,
        "thresholds": {"positive_internal_max": PREPARE.POSITIVE_INTERNAL_MAX, "positive_centroid_min": PREPARE.POSITIVE_CENTROID_MIN,
                         "hard_negative_centroid_max": PREPARE.HARD_NEGATIVE_CENTROID_MAX, "hard_negative_internal_min": PREPARE.HARD_NEGATIVE_INTERNAL_MIN,
                         "minimum_time_separation_s": PREPARE.MIN_PAIR_TIME_SEPARATION_S, "pair_anchor_stride": PREPARE.PAIR_ANCHOR_STRIDE},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HIDDEN_CONFIG_DEFAULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="read-only re-verification of an existing output")
    return parser.parse_args()


def run_check(output: Path) -> int:
    manifest = json.loads((output / "reserved_manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("status") == "T5R5_HIDDEN_CONFIRMATION_COMPLETE", "manifest status mismatch"
    assert (output / "SHA256SUMS").is_file(), "SHA256SUMS missing"
    task_summary = json.loads((output / "task_summary.json").read_text(encoding="utf-8"))
    intervention = json.loads((output / "intervention_summary.json").read_text(encoding="utf-8"))
    gates = decide_gates(task_summary, intervention, manifest["views"]["n_pairs"])
    assert gates["verdict"] == manifest["verdict"], "verdict mismatch"
    print(json.dumps({"status": "T5R5_CHECK_OK", "verdict": gates["verdict"]}, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if args.check:
        if not output.is_dir():
            print("T5R5_HIDDEN_REFUSED: nothing to check", file=sys.stderr, flush=True)
            return 2
        return run_check(output)
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        print(f"T5R5_HIDDEN_REFUSED: output already exists: {output}", file=sys.stderr, flush=True)
        return 2
    try:
        lock, config, intervention_lock = verify_lock_and_config(args.config.resolve())
    except RuntimeError as exc:
        print(f"T5R5_HIDDEN_REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2
    T5R3 = load_module(T5R3_PATH, "run_t5r3_sanity_for_t5r5")
    PREPARE = load_module(PREPARE_PATH, "prepare_idsse_t5r2_for_t5r5")
    FC = load_module(FRACTURE_CONTROLS_PATH, "p2_fracture_controls_for_t5r5")
    for key, expected in (("POSITIVE_INTERNAL_MAX", 0.28), ("POSITIVE_CENTROID_MIN", 0.35),
                          ("HARD_NEGATIVE_CENTROID_MAX", 0.12), ("HARD_NEGATIVE_INTERNAL_MIN", 0.35),
                          ("MIN_PAIR_TIME_SEPARATION_S", 4.0), ("PAIR_ANCHOR_STRIDE", 5)):
        if float(getattr(PREPARE, key)) != expected:
            print(f"T5R5_HIDDEN_REFUSED: pair threshold changed: {key}", file=sys.stderr, flush=True)
            return 2
    torch.set_num_threads(1)
    started = time.perf_counter()
    temporary.mkdir(parents=True)
    try:
        views = build_reserved_views(PREPARE, temporary)
        write_json(temporary / "views_manifest.json", views)
        task_block = evaluate_tasks(T5R3, lock, temporary)
        write_json(temporary / "task_results.json", task_block["records"])
        write_json(temporary / "task_analytic_control.json", task_block["analytic_control"])
        task_summary = summarize_tasks(task_block)
        write_json(temporary / "task_summary.json", task_summary)
        built = run_intervention(T5R3, FC, lock, temporary)
        intervention_results = evaluate_intervention_models(T5R3, lock, temporary, built)
        write_json(temporary / "intervention_summary.json", intervention_results)
        gates = decide_gates(task_summary, intervention_results, views["n_pairs"])
        write_json(temporary / "gate_evaluation.json", gates)
        manifest = {
            "status": "T5R5_HIDDEN_CONFIRMATION_COMPLETE",
            "phase": "P3_CAUSAL_MECHANISM", "task_lane": "P3-T5R5",
            "reserved_match": "J03WQQ", "reads": 1, "retraining": False,
            "candidate_lock_sha256": sha256_file(LOCK_PATH),
            "intervention_lock_sha256": sha256_file(INTERVENTION_LOCK),
            "hidden_config_sha256": sha256_file(args.config.resolve()),
            "views": views, "n_snapshots": task_block["n_snapshots"], "n_pairs": task_block["n_pairs"],
            "adjacency_sha256": task_block["adjacency_sha256"],
            "verdict": gates["verdict"],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        write_json(temporary / "reserved_manifest.json", manifest)
        write_sha256sums(temporary)
        temporary.replace(output)
        print(json.dumps({"status": manifest["status"], "output": str(output), "verdict": gates["verdict"]}, ensure_ascii=False), flush=True)
        return 0
    except Exception:
        print(f"T5R5 hidden run failed; incomplete staging kept at {temporary}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
