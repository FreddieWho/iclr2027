#!/usr/bin/env python3
"""Run the bounded P3-T5R4 Round 2 update-balance experiment.

Round 2 tests exactly one mechanism axis: shared-encoder task update balance
(``shared_encoder_task_update_balance`` / ``encoder_update_ratio``).  Two
candidates only (``update_ratio_3to1``, ``update_ratio_2to1``), each with a
matched total of 420 optimizer steps per epoch -- the same total as the T5R3
fixed-dual reference (361 context + 59 intrinsic batches).  All loss weights
are 1.0, gradient routing is shared, there is no readout normalization, and the
architecture is the unchanged T5R3 model.

Scheduling: per epoch the blockwise context-then-intrinsic order of Round 1 is
preserved; only the per-task step counts change.  Each task draws batches from
its own independent seed-deterministic shuffled cyclic stream, so an
under-quota task never permanently drops its tail batches and an over-quota
task continues across shuffle cycles with a deterministic new permutation per
cycle.  Every optimizer step still corresponds to exactly one task loss.

Reads only the T5R2 train/valid views.  Never reads reserved (J03WQQ),
external, or legacy heldout data and never promotes a candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
T5R3_PATH = ROOT / "scripts" / "run_t5r3_sanity.py"
ROUND2_CONFIG = ROOT / "configs" / "t5r4_round2.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1" / "t5r4_round2_v1"
LOCK_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
REFERENCE_SUMMARY = LOCK_ROOT / "t5r3_sanity_v3" / "summary.json"
SEMANTIC_ADDENDUM = LOCK_ROOT / "metric_semantics_addendum.json"

EXPECTED_TOTAL_UPDATES = 420
EXPECTED_QUOTAS = {
    "update_ratio_3to1": (315, 105),
    "update_ratio_2to1": (280, 140),
}


def load_t5r3_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_t5r3_sanity_for_t5r4r2", T5R3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load T5R3 runner: {T5R3_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


T5R3 = load_t5r3_module()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("T5R4 Round 2 config must be a mapping")
    if config.get("task_lane") != "P3-T5R4" or config.get("round") != 2:
        raise ValueError("T5R4 Round 2 config identity mismatch")
    if config.get("axis", {}).get("id") != "shared_encoder_task_update_balance":
        raise ValueError("Round 2 axis must be shared_encoder_task_update_balance")
    if config.get("axis", {}).get("total_optimizer_steps_per_epoch") != EXPECTED_TOTAL_UPDATES:
        raise ValueError("Round 2 total step budget must be 420")
    search = config.get("search", {})
    if search.get("one_major_axis_per_candidate") is not True or search.get("seeds") != [11, 23, 47]:
        raise ValueError("Round 2 search contract mismatch")
    data = config.get("data", {})
    if data.get("visible_splits") != ["train", "valid"] or data.get("reserved_loaded") is not False:
        raise ValueError("T5R4 Round 2 data firewall mismatch")
    frozen = config.get("frozen", {})
    for key, expected in (
        ("gradient_routing", "shared"),
        ("context_weight", 1.0),
        ("intrinsic_weight", 1.0),
        ("layernorm", "none"),
        ("pooling", "team_mean"),
        ("epochs", 80),
        ("batch_size", 64),
        ("parameter_count", 141769),
    ):
        if frozen.get(key) != expected:
            raise ValueError(f"Round 2 frozen field mismatch for {key}: {frozen.get(key)!r}")
    candidates = config.get("candidates", [])
    if len(candidates) != 2:
        raise ValueError("Round 2 must run exactly 2 candidates")
    seen: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate.get("id", ""))
        if candidate_id not in EXPECTED_QUOTAS or candidate_id in seen:
            raise ValueError(f"unexpected or duplicated Round 2 candidate: {candidate_id!r}")
        seen.add(candidate_id)
        expected_ctx, expected_int = EXPECTED_QUOTAS[candidate_id]
        if candidate.get("context_updates_per_epoch") != expected_ctx:
            raise ValueError(f"context quota mismatch for {candidate_id}")
        if candidate.get("intrinsic_updates_per_epoch") != expected_int:
            raise ValueError(f"intrinsic quota mismatch for {candidate_id}")
        if candidate.get("total_updates_per_epoch") != EXPECTED_TOTAL_UPDATES:
            raise ValueError(f"total step mismatch for {candidate_id}")
        if candidate.get("axis") != "shared_encoder_task_update_balance":
            raise ValueError(f"axis mismatch for {candidate_id}")
        if candidate.get("context_weight") != 1.0 or candidate.get("intrinsic_weight") != 1.0:
            raise ValueError(f"loss weights must be 1.0 for {candidate_id}")
        if candidate.get("gradient_routing") != "shared":
            raise ValueError(f"gradient routing must be shared for {candidate_id}")
        if candidate.get("head_readout_normalization") != "none":
            raise ValueError(f"readout normalization must be none for {candidate_id}")
    return config


def build_cyclic_stream(pool_size: int, seed_base: int, batches_per_epoch: int, epochs: int) -> np.ndarray:
    """Build a deterministic continuing shuffled stream of batch slots.

    The stream concatenates full shuffles (cycles) of ``arange(pool_size)``;
    cycle ``c`` uses ``default_rng(seed_base + c)``.  Returned shape is
    ``(epochs, batches_per_epoch, 64)`` with every batch full-size so the
    per-epoch optimizer-step count is exact.
    """
    needed = batches_per_epoch * 64 * epochs
    chunks: list[np.ndarray] = []
    gathered = 0
    cycle = 0
    base = np.arange(pool_size, dtype=np.int64)
    while gathered < needed:
        permutation = base.copy()
        np.random.default_rng(seed_base + cycle).shuffle(permutation)
        chunks.append(permutation)
        gathered += pool_size
        cycle += 1
    stream = np.concatenate(chunks, axis=0)[:needed]
    return stream.reshape(epochs, batches_per_epoch, 64)


def train_candidate(
    model: Any,
    spec: dict[str, Any],
    candidate: dict[str, Any],
    train: Any,
    adjacency: np.ndarray,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    T5R3.set_seed(seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    batch = T5R3.tensors(train, adjacency)
    context_quota = int(candidate["context_updates_per_epoch"])
    intrinsic_quota = int(candidate["intrinsic_updates_per_epoch"])
    # Independent streams; same RNG bases as the Round 1 per-epoch shuffles so
    # the first cycle starts from a comparable permutation.  Documented in the
    # protocol; no loss/accuracy/valid-adaptive sampling is involved.
    context_stream = build_cyclic_stream(len(train.raw), seed * 100_000, context_quota, epochs)
    pair_stream = build_cyclic_stream(len(train.pair_query), seed * 200_000, intrinsic_quota, epochs)
    started = time.perf_counter()
    last_loss = {"context": float("nan"), "intrinsic": float("nan"), "total": float("nan")}
    context_steps = 0
    intrinsic_steps = 0
    for epoch in range(epochs):
        model.train()
        context_total = 0.0
        context_seen = 0
        for slot in range(context_quota):
            index = torch.from_numpy(context_stream[epoch, slot])
            positions = batch[spec["context_view"]][index]
            z = model.encode(positions, batch["team"][index], batch["adjacency"][index], spec["pooling"])
            loss = T5R3.context_loss(
                model,
                z,
                {
                    "phase": batch["phase"][index],
                    "zone": batch["zone"][index],
                    "centroids": batch["centroids"][index],
                },
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            context_total += float(loss.detach()) * len(index)
            context_seen += len(index)
            context_steps += 1
        pair_total = 0.0
        pair_seen = 0
        for slot in range(intrinsic_quota):
            index = torch.from_numpy(pair_stream[epoch, slot])
            query_index = batch["pair_query"][index]
            positive_index = batch["pair_positive"][index]
            negative_index = batch["pair_negative"][index]
            all_indices = torch.cat([query_index, positive_index, negative_index], dim=0)
            positions = batch[spec["intrinsic_view"]][all_indices]
            team = batch["team"][all_indices]
            adjacency_batch = batch["adjacency"][all_indices]
            z = model.encode(positions, team, adjacency_batch, spec["pooling"])
            loss = T5R3.pair_loss(model, z, len(index))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            pair_total += float(loss.detach()) * len(index)
            pair_seen += len(index)
            intrinsic_steps += 1
        last_loss = {
            "context": context_total / max(context_seen, 1),
            "intrinsic": pair_total / max(pair_seen, 1),
            "total": context_total / max(context_seen, 1) + pair_total / max(pair_seen, 1),
        }
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"[{candidate['id']}] seed={seed} epoch={epoch + 1}/{epochs} "
                f"context={last_loss['context']:.5f} intrinsic={last_loss['intrinsic']:.5f} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return {
        "epochs": epochs,
        "batch_size": T5R3.BATCH_SIZE,
        "context_weight": 1.0,
        "intrinsic_weight": 1.0,
        "gradient_routing": "shared",
        "head_readout_normalization": "none",
        "context_updates_per_epoch": context_quota,
        "intrinsic_updates_per_epoch": intrinsic_quota,
        "total_updates_per_epoch": context_quota + intrinsic_quota,
        "context_optimizer_steps_total": context_steps,
        "intrinsic_optimizer_steps_total": intrinsic_steps,
        "last_loss": last_loss,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def run_candidate(
    candidate: dict[str, Any],
    train: Any,
    valid: Any,
    train_adjacency: np.ndarray,
    valid_adjacency: np.ndarray,
    output_root: Path,
    seeds: list[int],
    epochs: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    spec = {
        "name": candidate["id"],
        "context_view": "raw",
        "intrinsic_view": "centered",
        "pooling": "team_mean",
    }
    for seed in seeds:
        print(f"=== {candidate['id']} seed={seed} ===", flush=True)
        T5R3.set_seed(seed)
        model = T5R3.TaskModel()
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        training = train_candidate(model, spec, candidate, train, train_adjacency, seed, epochs)
        temporary_checkpoint = output_root / "models" / f"{candidate['id']}_seed{seed}.pt"
        stable_output_root = output_root.with_name(output_root.name.removesuffix(".tmp"))
        stable_checkpoint = stable_output_root / "models" / f"{candidate['id']}_seed{seed}.pt"
        temporary_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), temporary_checkpoint)

        train_context = T5R3.encode_split(model, train, train_adjacency, "raw", "team_mean")
        train_intrinsic = T5R3.encode_split(model, train, train_adjacency, "centered", "team_mean")
        valid_context = T5R3.encode_split(model, valid, valid_adjacency, "raw", "team_mean")
        valid_intrinsic = T5R3.encode_split(model, valid, valid_adjacency, "centered", "team_mean")
        record = {
            "candidate_id": candidate["id"],
            "axis": candidate["axis"],
            "candidate": candidate,
            "seed": seed,
            "architecture": "fixed-graph-message-passing-equivalent",
            "pooling": "team_mean",
            "context_view": "raw",
            "intrinsic_view": "centered",
            "parameter_count": parameter_count,
            "train": {
                "context": T5R3.prediction_metrics(train, model, train_context),
                "intrinsic": T5R3.pair_metrics(train, model, train_intrinsic),
                "translation": T5R3.translation_metrics(
                    model, train, train_adjacency, spec, train_context, train_intrinsic
                ),
            },
            "valid": {
                "context": T5R3.prediction_metrics(valid, model, valid_context),
                "intrinsic": T5R3.pair_metrics(valid, model, valid_intrinsic),
                "translation": T5R3.translation_metrics(
                    model, valid, valid_adjacency, spec, valid_context, valid_intrinsic
                ),
                "cross_readout": T5R3.cross_readout_metrics(valid, model, valid_context, valid_intrinsic),
            },
            "training": training,
            "checkpoint": {
                "path": portable_path(stable_checkpoint),
                "sha256": sha256_file(temporary_checkpoint),
                "bytes": int(temporary_checkpoint.stat().st_size),
            },
        }
        records.append(record)
        print(
            f"completed {candidate['id']} seed={seed}: "
            f"valid_phase_f1={record['valid']['context']['phase_match_grouped_macro_f1']:.4f} "
            f"valid_pair_acc={record['valid']['intrinsic']['match_grouped_ranking_accuracy']:.4f} "
            f"translation={record['valid']['translation']['z_mode_global_translation_response']:.6f}",
            flush=True,
        )
    return records


def aggregate_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for candidate_id in sorted({record["candidate_id"] for record in records}):
        rows = [record for record in records if record["candidate_id"] == candidate_id]

        def values(path: tuple[str, ...]) -> list[float]:
            output: list[float] = []
            for row in rows:
                value: Any = row
                for key in path:
                    value = value[key]
                output.append(float(value))
            return output

        first = rows[0]
        summary.append(
            {
                "candidate_id": candidate_id,
                "axis": first["axis"],
                "candidate": first["candidate"],
                "seed_count": len(rows),
                "parameter_counts": sorted({row["parameter_count"] for row in rows}),
                "valid_phase_f1_mean": float(np.mean(values(("valid", "context", "phase_match_grouped_macro_f1")))),
                "valid_phase_f1_by_seed": values(("valid", "context", "phase_match_grouped_macro_f1")),
                "valid_field_zone_f1_mean": float(np.mean(values(("valid", "context", "field_zone_match_grouped_macro_f1")))),
                "valid_field_zone_f1_by_seed": values(("valid", "context", "field_zone_match_grouped_macro_f1")),
                "valid_centroid_mae_mean": float(np.mean(values(("valid", "context", "centroid_match_grouped_mae")))),
                "valid_centroid_mae_by_seed": values(("valid", "context", "centroid_match_grouped_mae")),
                "valid_pair_accuracy_mean": float(np.mean(values(("valid", "intrinsic", "match_grouped_ranking_accuracy")))),
                "valid_pair_accuracy_by_seed": values(("valid", "intrinsic", "match_grouped_ranking_accuracy")),
                "valid_mrr_at_2_mean": float(np.mean(values(("valid", "intrinsic", "match_grouped_mrr_at_2")))),
                "valid_geometry_relation_spearman_mean": float(np.mean(values(("valid", "intrinsic", "natural_geometry_latent_distance_spearman")))),
                "valid_geometry_relation_spearman_by_seed": values(("valid", "intrinsic", "natural_geometry_latent_distance_spearman")),
                "valid_z_mode_translation_response_mean": float(np.mean(values(("valid", "translation", "z_mode_global_translation_response")))),
                "valid_z_mode_translation_response_by_seed": values(("valid", "translation", "z_mode_global_translation_response")),
                "valid_z_ctx_accessibility_response_mean": float(np.mean(values(("valid", "translation", "z_ctx_context_accessibility_response")))),
            }
        )
    return summary


def build_comparison(summary: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare Round 2 means against reference floors read from the artifact."""
    reference_by_id = {row["variant"]: row for row in reference}
    t5r3 = reference_by_id["fixed_dual_channel_shared_phase_gat"]
    raw = reference_by_id["raw_single_channel_phase_gat_team_mean"]
    ref_pair = float(t5r3["valid_pair_accuracy_mean"])
    ref_geom = float(t5r3["valid_geometry_relation_spearman_mean"])
    floors = {
        "pair_accuracy_floor": ref_pair - 0.01,
        "geometry_spearman_floor": ref_geom - 0.01,
        "z_mode_translation_ceiling": 1e-6,
    }
    entries = []
    for row in summary:
        pair_by_seed = list(map(float, row["valid_pair_accuracy_by_seed"]))
        geom_by_seed = list(map(float, row["valid_geometry_relation_spearman_by_seed"]))
        zmode_by_seed = list(map(float, row["valid_z_mode_translation_response_by_seed"]))
        pair_ok = [value >= floors["pair_accuracy_floor"] for value in pair_by_seed]
        geom_ok = [value >= floors["geometry_spearman_floor"] for value in geom_by_seed]
        zmode_ok = [value < floors["z_mode_translation_ceiling"] for value in zmode_by_seed]
        entries.append(
            {
                "candidate_id": row["candidate_id"],
                "valid_pair_accuracy_mean": row["valid_pair_accuracy_mean"],
                "valid_pair_accuracy_by_seed": pair_by_seed,
                "valid_geometry_relation_spearman_mean": row["valid_geometry_relation_spearman_mean"],
                "valid_geometry_relation_spearman_by_seed": geom_by_seed,
                "valid_z_mode_translation_response_mean": row["valid_z_mode_translation_response_mean"],
                "delta_vs_reference": {
                    "pair_accuracy": row["valid_pair_accuracy_mean"] - ref_pair,
                    "geometry_spearman": row["valid_geometry_relation_spearman_mean"] - ref_geom,
                },
                "intrinsic_floor_check_means": {
                    "pair_accuracy_mean_above_floor": row["valid_pair_accuracy_mean"] >= floors["pair_accuracy_floor"],
                    "geometry_spearman_mean_above_floor": row["valid_geometry_relation_spearman_mean"] >= floors["geometry_spearman_floor"],
                    "z_mode_translation_mean_below_ceiling": row["valid_z_mode_translation_response_mean"] < floors["z_mode_translation_ceiling"],
                },
                "seeds_holding_both_intrinsic_floors": sum(
                    1 for ok_pair, ok_geom in zip(pair_ok, geom_ok) if ok_pair and ok_geom
                ),
                "at_least_2_of_3_seeds_hold_floors": sum(
                    1 for ok_pair, ok_geom in zip(pair_ok, geom_ok) if ok_pair and ok_geom
                )
                >= 2,
                "context_noninferiority_vs_raw_means": {
                    "phase_f1_drop": float(raw["valid_phase_f1_mean"]) - float(row["valid_phase_f1_mean"]),
                    "field_zone_f1_drop": float(raw["valid_field_zone_f1_mean"]) - float(row["valid_field_zone_f1_mean"]),
                    "centroid_mae_increase": float(row["valid_centroid_mae_mean"]) - float(raw["valid_centroid_mae_mean"]),
                    "margin": 0.05,
                },
            }
        )
    return {
        "reference_variant": "fixed_dual_channel_shared_phase_gat",
        "reference_valid_means": {
            "pair_accuracy": ref_pair,
            "geometry_latent_spearman": ref_geom,
            "z_mode_translation_response": float(t5r3["valid_z_mode_translation_response_mean"]),
        },
        "raw_baseline_valid_means": {
            "phase_f1": float(raw["valid_phase_f1_mean"]),
            "field_zone_f1": float(raw["valid_field_zone_f1_mean"]),
            "centroid_mae": float(raw["valid_centroid_mae_mean"]),
        },
        "floors_read_from_reference_artifact": floors,
        "candidates": entries,
        "note": "Means-level advisory comparison only; promotion requires the frozen seed/match-level stability rules in the Round 2 report.",
    }


def write_sha256sums(output_root: Path) -> Path:
    lines: list[str] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256_file(path)}  {path.relative_to(output_root)}")
    sums_path = output_root / "SHA256SUMS"
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sums_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROUND2_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--smoke", action="store_true", help="allow a reduced-epoch wiring smoke")
    parser.add_argument("--only-candidate", action="append", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs != 80 and not args.smoke:
        raise ValueError("formal T5R4 Round 2 runs must use 80 epochs; use --smoke for a reduced wiring check")
    config_path = args.config.resolve()
    config = load_config(config_path)
    candidates = list(config["candidates"])
    if args.only_candidate:
        wanted = set(args.only_candidate)
        candidates = [candidate for candidate in candidates if candidate["id"] in wanted]
        if set(args.only_candidate) != {candidate["id"] for candidate in candidates}:
            raise ValueError("unknown --only-candidate")
    if not candidates:
        raise ValueError("no candidates selected")

    output = args.output.resolve()
    temporary_output = output.with_name(output.name + ".tmp")
    if output.exists() or temporary_output.exists():
        raise FileExistsError(f"refusing to overwrite existing T5R4 output: {output}")

    split_lock, baseline_lock = T5R3.load_lock()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    train = T5R3.load_split("train", split_lock)
    valid = T5R3.load_split("valid", split_lock)
    train_adjacency = T5R3.knn_adjacency_batch(train.raw)
    valid_adjacency = T5R3.knn_adjacency_batch(valid.raw)
    if not SEMANTIC_ADDENDUM.is_file():
        raise FileNotFoundError(f"missing semantic addendum: {SEMANTIC_ADDENDUM}")
    temporary_output.mkdir(parents=True)
    np.save(temporary_output / "adjacency_train.npy", train_adjacency)
    np.save(temporary_output / "adjacency_valid.npy", valid_adjacency)
    shutil.copy(SEMANTIC_ADDENDUM, temporary_output / "semantic_alias_snapshot.json")
    started = time.perf_counter()
    try:
        protocol: dict[str, Any] = {
            "status": "T5R4_ROUND2_RUNNING",
            "phase": "P3_CAUSAL_MECHANISM",
            "task_lane": "P3-T5R4",
            "round": 2,
            "final_round": True,
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256_file(config_path),
            "t5r2_lock_status": baseline_lock["status"],
            "t5r2_lock_sha256": sha256_file(LOCK_ROOT / "baseline_and_metric_lock.json"),
            "t5r3_reference_summary": str(REFERENCE_SUMMARY.relative_to(ROOT)),
            "t5r3_reference_summary_sha256": sha256_file(REFERENCE_SUMMARY),
            "semantic_addendum": portable_path(SEMANTIC_ADDENDUM),
            "semantic_addendum_sha256": sha256_file(SEMANTIC_ADDENDUM),
            "official_revision": T5R3.EXPECTED_OFFICIAL_REVISION,
            "visible_splits": ["train", "valid"],
            "reserved_holdout": {"match_ids": ["J03WQQ"], "loaded": False, "task_arrays_written": False},
            "external_results_used": False,
            "old_heldout_used": False,
            "candidate_selection_performed": False,
            "candidate_lock_created": False,
            "P4_release": False,
            "seeds": config["search"]["seeds"],
            "torch_threads": torch.get_num_threads(),
            "axis": config["axis"],
            "frozen": config["frozen"],
            "scheduling": config["scheduling"],
            "decision_rules": config["decision_rules"],
            "candidates": candidates,
            "update_counts_per_epoch": {
                candidate["id"]: {
                    "context": candidate["context_updates_per_epoch"],
                    "intrinsic": candidate["intrinsic_updates_per_epoch"],
                    "total": candidate["total_updates_per_epoch"],
                }
                for candidate in candidates
            },
            "input_shapes": {
                "train_snapshots": int(len(train.raw)),
                "valid_snapshots": int(len(valid.raw)),
                "train_pairs": int(len(train.pair_query)),
                "valid_pairs": int(len(valid.pair_query)),
            },
            "adjacency_sha256": {
                "train": sha256_file(temporary_output / "adjacency_train.npy"),
                "valid": sha256_file(temporary_output / "adjacency_valid.npy"),
            },
        }
        write_json(temporary_output / "protocol.json", protocol)
        records: list[dict[str, Any]] = []
        for candidate in candidates:
            records.extend(
                run_candidate(
                    candidate,
                    train,
                    valid,
                    train_adjacency,
                    valid_adjacency,
                    temporary_output,
                    list(config["search"]["seeds"]),
                    args.epochs,
                )
            )
        summary = aggregate_results(records)
        write_json(temporary_output / "model_results.json", records)
        write_json(temporary_output / "summary.json", summary)
        reference = json.loads(REFERENCE_SUMMARY.read_text(encoding="utf-8"))
        write_json(temporary_output / "reference_t5r3_summary.json", reference)
        write_json(temporary_output / "comparison_vs_t5r3.json", build_comparison(summary, reference))
        manifest: dict[str, Any] = {
            "status": "T5R4_ROUND2_COMPLETE",
            "phase": "P3_CAUSAL_MECHANISM",
            "task_lane": "P3-T5R4",
            "round": 2,
            "final_round": True,
            "protocol": "protocol.json",
            "results": "model_results.json",
            "summary": "summary.json",
            "reference_t5r3_summary": "reference_t5r3_summary.json",
            "comparison_vs_t5r3": "comparison_vs_t5r3.json",
            "semantic_alias_snapshot": "semantic_alias_snapshot.json",
            "candidate_count": len(candidates),
            "model_count": len(records),
            "seed_count": len(config["search"]["seeds"]),
            "reserved_holdout_loaded": False,
            "external_results_used": False,
            "old_heldout_used": False,
            "candidate_selection_performed": False,
            "candidate_lock_created": False,
            "P4_release": False,
            "decision_status": "BOUNDED_AUTORESEARCH_CLOSED_PENDING_REPORT",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        write_json(temporary_output / "manifest.json", manifest)
        protocol["status"] = manifest["status"]
        protocol["manifest_sha256"] = json_sha256(manifest)
        write_json(temporary_output / "protocol.json", protocol)
        write_sha256sums(temporary_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output.replace(output)
        print(
            json.dumps(
                {"status": manifest["status"], "output": str(output), "summary": summary},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except Exception:
        print(f"T5R4 Round 2 failed; incomplete staging kept at {temporary_output}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
