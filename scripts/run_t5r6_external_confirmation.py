#!/usr/bin/env python3
"""T5R6 external confirmation on SoccerTrack-v2 (8 matches).

Reuses frozen T5R5 logic (imported) for task metrics, intervention operator,
Jacobian geometry and baselines. Per-match evaluation with match-macro
primary and pooled-pair secondary aggregation (shadow-lock rule).

Reads data_views_soccertrack/<match> views. No training, no threshold
changes. Verdict rule is frozen in t5r6_confirmation_lock.json BEFORE results.
"""
from __future__ import annotations

import argparse
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
T5R3_PATH = ROOT / "scripts" / "run_t5r3_sanity.py"
T5R5_PATH = ROOT / "scripts" / "run_t5r5_hidden_confirmation.py"
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
VIEW_ROOT = T5R_ROOT / "data_views_soccertrack"
LOCK_PATH = T5R_ROOT / "candidate_lock.json"
T5R6_LOCK = T5R_ROOT / "t5r6_confirmation_lock.json"
DEFAULT_OUTPUT = T5R_ROOT / "t5r6_soccertrack_v1"

MATCHES = ["118575", "118576", "118577", "118578", "128057", "128058", "132831", "132877"]
EXCLUDED_SMOKE = ["117092", "117093"]
MAX_SNAPSHOTS_PER_MATCH = 150


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
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


def load_match_views(T5R3: Any, match: str) -> Any:
    raw = np.load(VIEW_ROOT / f"positions_raw_{match}.npy").astype(np.float32)
    centered = np.load(VIEW_ROOT / f"positions_centered_{match}.npy").astype(np.float32)
    team = np.load(VIEW_ROOT / f"team_slots_{match}.npy").astype(np.int64)
    index = pd.read_parquet(VIEW_ROOT / f"snapshot_index_{match}.parquet")
    pair_path = VIEW_ROOT / f"natural_pair_ranking_{match}.parquet"
    pairs = pd.read_parquet(pair_path) if pair_path.exists() else pd.DataFrame()
    n_pairs = int(len(pairs))
    id_to_index = {str(v): i for i, v in enumerate(index["snapshot_id"].astype(str))}
    if n_pairs:
        pq = np.asarray([id_to_index[str(v)] for v in pairs["query_snapshot_id"]], dtype=np.int64)
        pp = np.asarray([id_to_index[str(v)] for v in pairs["positive_snapshot_id"]], dtype=np.int64)
        pn = np.asarray([id_to_index[str(v)] for v in pairs["negative_snapshot_id"]], dtype=np.int64)
        if bool((pq < 0).any() or (pp < 0).any() or (pn < 0).any()):
            raise RuntimeError(f"{match}: pair references unknown snapshot")
    else:
        pq = pp = pn = np.zeros(0, dtype=np.int64)
    data = T5R3.SplitData(
        name="external", raw=raw, centered=centered, team_slots=team,
        snapshot_ids=index["snapshot_id"].astype(str).to_numpy(),
        match_ids=index["source_match_id"].astype(str).to_numpy(),
        phase=T5R3._label_index(index["phase_label"], ("firstHalf", "secondHalf"), "phase"),
        zone=T5R3._label_index(index["home_field_zone"], ("defensive_third", "middle_third", "attacking_third"), "field zone"),
        centroids=index[["home_centroid_x", "home_centroid_y", "away_centroid_x", "away_centroid_y"]].to_numpy(dtype=np.float32),
        pair_query=pq, pair_positive=pp, pair_negative=pn,
        pair_matches=pairs["source_match_id"].astype(str).to_numpy() if n_pairs else np.zeros(0, dtype=str),
        pair_positive_geometry=pairs["positive_internal_geometry_distance"].to_numpy(dtype=np.float32) if n_pairs else np.zeros(0, dtype=np.float32),
        pair_negative_geometry=pairs["negative_internal_geometry_distance"].to_numpy(dtype=np.float32) if n_pairs else np.zeros(0, dtype=np.float32),
    )
    if set(data.match_ids.tolist()) != {match}:
        raise RuntimeError(f"{match}: match set mismatch")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-snapshots", type=int, default=MAX_SNAPSHOTS_PER_MATCH)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        print(f"REFUSED: output exists: {output}", file=sys.stderr)
        return 2
    t5r6_lock = json.loads(T5R6_LOCK.read_text(encoding="utf-8"))
    if t5r6_lock.get("status") != "T5R6_CONFIRMED_RUN_AUTHORIZED":
        print("REFUSED: T5R6 lock not in authorized state", file=sys.stderr)
        return 2
    candidate_lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    T5R3 = load_module(T5R3_PATH, "run_t5r3_sanity_for_t5r6")
    T5R5 = load_module(T5R5_PATH, "run_t5r5_hidden_for_t5r6")
    FC = load_module(ROOT / "scripts" / "p2_fracture_controls.py", "p2_fracture_controls_for_t5r6")
    torch.set_num_threads(1)
    started = time.perf_counter()
    output.mkdir(parents=True)

    groups = [("update_ratio_2to1", candidate_lock["model"]["checkpoints"])]
    groups.append(("fixed_dual_channel_shared_phase_gat", candidate_lock["frozen_comparison_set"]["reference"]["checkpoints"]))
    groups.append(("raw_single_channel_phase_gat_team_mean", candidate_lock["frozen_comparison_set"]["raw_baseline"]["checkpoints"]))

    task_records: list[dict[str, Any]] = []
    analytic: dict[str, Any] = {}
    intervention_per_match: dict[str, Any] = {}
    for match in MATCHES:
        print(f"=== match {match} ===", flush=True)
        data = load_match_views(T5R3, match)
        adjacency = T5R3.knn_adjacency_batch(data.raw)
        n_pairs = int(len(data.pair_query))
        with torch.inference_mode():
            for model_id, checkpoints in groups:
                spec = T5R5.VARIANT_SPECS[model_id]
                for seed in ("11", "23", "47"):
                    model = T5R3.TaskModel()
                    model.load_state_dict(torch.load(ROOT / checkpoints[seed]["path"], map_location="cpu", weights_only=True))
                    model.eval()
                    z_ctx = T5R3.encode_split(model, data, adjacency, spec["context_view"], spec["pooling"])
                    z_mode = T5R3.encode_split(model, data, adjacency, spec["intrinsic_view"], spec["pooling"])
                    context = T5R3.prediction_metrics(data, model, z_ctx)
                    translation = T5R3.translation_metrics(model, data, adjacency, {**spec, "name": model_id}, z_ctx, z_mode)
                    cross = T5R3.cross_readout_metrics(data, model, z_ctx, z_mode)
                    intrinsic = T5R3.pair_metrics(data, model, z_mode) if n_pairs else {"status": "UNDEFINED_NO_PAIRS", "pair_count": 0}
                    task_records.append({"match": match, "model_id": model_id, "seed": int(seed),
                                         "n_snapshots": int(len(data.raw)), "n_pairs": n_pairs,
                                         "context": context, "intrinsic": intrinsic,
                                         "translation": translation, "cross_readout": cross})
        analytic[match] = T5R3.analytic_pair_metrics(data) if n_pairs else {"status": "UNDEFINED_NO_PAIRS"}
        print(f"{match}: snaps={len(data.raw)} pairs={n_pairs}", flush=True)

        # intervention on this match (same frozen operator, capped snapshots)
        tmpdir = output / f"_staging_{match}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        np.save(tmpdir / "positions_raw_reserved_holdout.npy", data.raw)
        np.save(tmpdir / "positions_centered_reserved_holdout.npy", data.centered)
        np.save(tmpdir / "team_slots_reserved_holdout.npy", data.team_slots)
        snap_ids = data.snapshot_ids.tolist()
        pd.DataFrame({"snapshot_id": snap_ids}).to_parquet(tmpdir / "snapshot_index_reserved_holdout.parquet", index=False)
        result = run_match_intervention(T5R3, T5R5, FC, candidate_lock, tmpdir, args.max_snapshots)
        intervention_per_match[match] = result
        print(f"{match}: intervention {result['n_complete_sets']} complete + {result['n_incomplete_sets']} incomplete", flush=True)

    write_json(output / "task_records.json", task_records)
    write_json(output / "task_analytic_control.json", analytic)
    write_json(output / "intervention_per_match.json", {m: strip_pairs(v) for m, v in intervention_per_match.items()})
    summary = build_summary(task_records, intervention_per_match)
    write_json(output / "summary.json", summary)
    manifest = {
        "status": "T5R6_EXTERNAL_CONFIRMATION_COMPLETE",
        "matches": MATCHES, "excluded_smoke": EXCLUDED_SMOKE,
        "t5r6_lock_sha256": sha256_file(T5R6_LOCK),
        "candidate_lock_sha256": sha256_file(LOCK_PATH),
        "verdict": summary["verdict"],
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    write_json(output / "manifest.json", manifest)
    lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS" and "_staging_" not in path.parts:
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            lines.append(f"{h.hexdigest()}  {path.relative_to(output)}")
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "verdict": summary["verdict"]}), flush=True)
    return 0


def strip_pairs(result: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in result.items() if k not in ("update_ratio_2to1", "fixed_dual_channel_shared_phase_gat")}
    for model_id in ("update_ratio_2to1", "fixed_dual_channel_shared_phase_gat"):
        if model_id in result:
            out[model_id] = {s: ({k: v for k, v in seed.items() if k != "pairs"}) for s, seed in result[model_id].items()}
    return out


def run_match_intervention(T5R3: Any, T5R5: Any, FC: Any, lock: dict[str, Any], staging: Path, cap: int) -> dict[str, Any]:
    """Same frozen operator as T5R5; snapshot cap is the only T5R6 parameter."""
    import pandas as pd
    raw = np.load(staging / "positions_raw_reserved_holdout.npy").astype(np.float64)
    team_all = np.load(staging / "team_slots_reserved_holdout.npy").astype(np.int64)
    snapshot_ids = pd.read_parquet(staging / "snapshot_index_reserved_holdout.parquet")["snapshot_id"].astype(str).tolist()
    total = len(raw)
    take = min(cap, total)
    chosen = sorted(set(int(v) for v in np.linspace(0, total - 1, take).round()))
    arm_rows: list[dict[str, Any]] = []
    complete_pairs: list[dict[str, Any]] = []
    incomplete = 0
    for order, snap in enumerate(chosen):
        positions = raw[snap]
        teams = team_all[snap]
        adjacency = T5R3.knn_adjacency_batch(positions[None].astype(np.float32))[0].astype(np.float64)
        eigenvalues, eigenvectors = T5R5.normalized_laplacian_eig(adjacency)
        rng = np.random.default_rng(FC.stable_seed(T5R5.OPERATOR_ID, snapshot_ids[snap], T5R5.DRAW))
        team_choice = int(rng.integers(0, 2))
        team_nodes = [int(i) for i in np.flatnonzero(teams == team_choice)]
        source = tuple(sorted(int(v) for v in rng.choice(team_nodes, T5R5.SUPPORT_SIZE, replace=False)))
        base = FC.unit_base_displacement_field(20, source, (T5R5.OPERATOR_ID, snapshot_ids[snap], T5R5.DRAW))
        anchor = FC.scale_displacement_field(base, T5R5.EPSILON)
        anchor_valid = FC.boundary_valid(positions, anchor)
        anchor_features = FC.spectral_topology_features_2d(eigenvalues, eigenvectors, anchor, source, adjacency, T5R5.N_BANDS)
        ranked = []
        for target_array in FC.enumerate_target_supports(teams, source, team_choice):
            target = tuple(int(v) for v in target_array)
            for bijection in FC.enumerate_vector_bijections(source, target):
                control = FC.reassigned_vector_field(base, source, target, bijection, T5R5.EPSILON)
                if not FC.boundary_valid(positions, control):
                    continue
                features = FC.spectral_topology_features_2d(eigenvalues, eigenvectors, control, target, adjacency, T5R5.N_BANDS)
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
            status, assignment = "COMPLETE", {int(k): int(v) for k, v in bijection.items()}
        else:
            target, assignment, control, control_features = None, {}, None, None
            status, incomplete = "INCOMPLETE", incomplete + 1
        arm_rows.append({"snapshot_id": snapshot_ids[snap], "team": team_choice,
                         "source_support": list(source),
                         "target_support": None if target is None else list(target),
                         "anchor_delta": anchor.tolist(),
                         "control_delta": None if control is None else np.asarray(control).tolist(),
                         "anchor_valid": bool(anchor_valid),
                         "anchor_rq_total": float(anchor_features.rq_total),
                         "control_rq_total": None if control_features is None else float(control_features.rq_total),
                         "status": status})
        if status == "COMPLETE":
            complete_pairs.append({"snapshot_id": snapshot_ids[snap], "positions": positions, "teams": teams,
                                   "adjacency": adjacency, "delta_A": anchor, "delta_B": np.asarray(control),
                                   "baseline_rq_diff": abs(float(anchor_features.rq_total) - float(control_features.rq_total))})
    results: dict[str, Any] = {}
    for model_id, group in (("update_ratio_2to1", lock["model"]["checkpoints"]),
                            ("fixed_dual_channel_shared_phase_gat", lock["frozen_comparison_set"]["reference"]["checkpoints"])):
        per_seed: dict[str, Any] = {}
        for seed in ("11", "23", "47"):
            model = T5R3.TaskModel()
            model.load_state_dict(torch.load(ROOT / group[seed]["path"], map_location="cpu", weights_only=True))
            model.eval()
            delta_r, delta_q_full, delta_q_diag, baseline_diff, per_pair = [], [], [], [], []
            for pair in complete_pairs:
                base = torch.from_numpy(pair["positions"].astype(np.float32))
                team_t = torch.from_numpy(pair["teams"].astype(np.int64))
                adj_t = torch.from_numpy(pair["adjacency"].astype(np.float32))

                def f_map(flat: torch.Tensor) -> torch.Tensor:
                    x = flat.reshape(1, 20, 2)
                    xc = x - x.mean(dim=1, keepdim=True)
                    z = model.encode(xc, team_t.unsqueeze(0), adj_t.unsqueeze(0), "team_mean")
                    return torch.nn.functional.normalize(model.mode_head(z), dim=-1).reshape(-1)

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
            per_seed[seed] = {"valid_pairs": valid_n, "response_std": response_std, "status": status_m,
                              "full_spearman": full_sp, "diagonal_spearman": diag_sp,
                              "spectral_baseline_spearman": base_sp, "direction_accuracy": direction,
                              "full_minus_baseline": float(full_sp - base_sp) if np.isfinite(full_sp) and np.isfinite(base_sp) else float("nan"),
                              "pairs": per_pair}
        results[model_id] = per_seed
    results["n_sampled_snapshots"] = len(chosen)
    results["n_complete_sets"] = len(complete_pairs)
    results["n_incomplete_sets"] = incomplete
    results["arm_rows"] = arm_rows
    return results


def build_summary(task_records: list[dict[str, Any]], intervention: dict[str, Any]) -> dict[str, Any]:
    per_match_task: dict[str, Any] = {}
    for match in MATCHES:
        rows = [r for r in task_records if r["match"] == match]
        per_match_task[match] = {}
        for model_id in ("update_ratio_2to1", "fixed_dual_channel_shared_phase_gat", "raw_single_channel_phase_gat_team_mean"):
            mrows = [r for r in rows if r["model_id"] == model_id]
            entry: dict[str, Any] = {"n_snapshots": mrows[0]["n_snapshots"], "n_pairs": mrows[0]["n_pairs"]}
            for key, section, field in (("match_half_f1", "context", "phase_match_grouped_macro_f1"),
                                        ("field_zone_f1", "context", "field_zone_match_grouped_macro_f1"),
                                        ("centroid_mae", "context", "centroid_match_grouped_mae"),
                                        ("z_mode_translation", "translation", "z_mode_global_translation_response"),
                                        ("pair_accuracy", "intrinsic", "match_grouped_ranking_accuracy"),
                                        ("mrr_at_2", "intrinsic", "match_grouped_mrr_at_2"),
                                        ("geometry_spearman", "intrinsic", "natural_geometry_latent_distance_spearman")):
                try:
                    values = [float(r[section][field]) for r in mrows]
                except (KeyError, TypeError):
                    values = [float("nan")] * len(mrows)
                entry[key + "_mean"] = float(np.nanmean(values))
                entry[key + "_by_seed"] = values
            per_match_task[match][model_id] = entry
    macro: dict[str, Any] = {}
    for model_id in ("update_ratio_2to1", "fixed_dual_channel_shared_phase_gat", "raw_single_channel_phase_gat_team_mean"):
        macro[model_id] = {}
        for key in ("match_half_f1", "field_zone_f1", "centroid_mae", "z_mode_translation",
                    "pair_accuracy", "mrr_at_2", "geometry_spearman"):
            vals = [per_match_task[m][model_id][key + "_mean"] for m in MATCHES]
            macro[model_id][key + "_macro"] = float(np.nanmean(vals))
            macro[model_id][key + "_per_match"] = vals
    # intervention cross-match roll-up (selected family, full geometry)
    per_match_full, per_match_base, per_match_dir = [], [], []
    for match in MATCHES:
        block = intervention[match]["update_ratio_2to1"]
        seeds = [block[s] for s in ("11", "23", "47") if block[s]["status"] == "OK"]
        if not seeds:
            per_match_full.append(float("nan"))
            per_match_base.append(float("nan"))
            per_match_dir.append(float("nan"))
            continue
        per_match_full.append(float(np.mean([s["full_spearman"] for s in seeds])))
        per_match_base.append(float(np.mean([s["spectral_baseline_spearman"] for s in seeds])))
        per_match_dir.append(float(np.mean([s["direction_accuracy"] for s in seeds])))
    n_ok = int(np.sum([np.isfinite(v) for v in per_match_full]))
    n_full_above = int(np.sum([(f > b) for f, b in zip(per_match_full, per_match_base) if np.isfinite(f) and np.isfinite(b)]))
    verdict = "T5R6_CONFIRMED" if (n_ok == len(MATCHES) and n_full_above == len(MATCHES)
                                   and all(d > 0.5 for d in per_match_dir if np.isfinite(d))) else \
              "T5R6_MIXED" if (n_full_above >= len(MATCHES) // 2 + 1) else "T5R6_NOT_CONFIRMED"
    return {"per_match_task": per_match_task, "macro": macro,
            "intervention_per_match_full": per_match_full,
            "intervention_per_match_baseline": per_match_base,
            "intervention_per_match_direction": per_match_dir,
            "matches_full_above_baseline": n_full_above, "matches_ok": n_ok,
            "verdict": verdict}


if __name__ == "__main__":
    raise SystemExit(main())
