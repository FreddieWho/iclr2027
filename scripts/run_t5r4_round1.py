#!/usr/bin/env python3
"""Run the bounded P3-T5R4 Round 1 search on the T5R2 train/valid views.

The runner reuses the T5R3 model and metric implementation.  Round 1 changes
only one declared mechanism axis per candidate; it never reads reserved,
external, or legacy heldout data and it does not promote a candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
T5R3_PATH = ROOT / "scripts" / "run_t5r3_sanity.py"
ROUND1_CONFIG = ROOT / "configs" / "t5r4_round1.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1" / "t5r4_round1_v1"
LOCK_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
REFERENCE_SUMMARY = LOCK_ROOT / "t5r3_sanity_v3" / "summary.json"


def load_t5r3_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_t5r3_sanity_for_t5r4", T5R3_PATH)
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
        raise ValueError("T5R4 Round 1 config must be a mapping")
    if config.get("task_lane") != "P3-T5R4":
        raise ValueError("T5R4 config task lane mismatch")
    search = config.get("search", {})
    if search.get("round") != 1 or search.get("one_major_axis_per_candidate") is not True:
        raise ValueError("Round 1 search contract mismatch")
    if search.get("seeds") != [11, 23, 47]:
        raise ValueError("T5R4 seed contract mismatch")
    data = config.get("data", {})
    if data.get("visible_splits") != ["train", "valid"] or data.get("reserved_loaded") is not False:
        raise ValueError("T5R4 data firewall mismatch")
    candidates = config.get("candidates", [])
    if not 1 <= len(candidates) <= int(search.get("max_candidates_per_round", 6)):
        raise ValueError("invalid Round 1 candidate count")
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate entry must be a mapping")
        candidate_id = str(candidate.get("id", ""))
        axis = str(candidate.get("axis", ""))
        if not candidate_id or candidate_id in seen or not axis:
            raise ValueError("candidate IDs and axes must be unique and non-empty")
        seen.add(candidate_id)
    return config


class CandidateTaskModel(T5R3.TaskModel):
    """T5R3 model with parameter-free exploratory readout placement."""

    def __init__(self, head_placement: str) -> None:
        super().__init__()
        if head_placement not in {"identity", "context_layernorm"}:
            raise ValueError(f"unknown head placement: {head_placement}")
        self.head_placement = head_placement

    def context(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.head_placement == "context_layernorm":
            z = F.layer_norm(z, z.shape[-1:])
        return super().context(z)


def train_candidate(
    model: CandidateTaskModel,
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
    context_order_base = np.arange(len(train.raw), dtype=np.int64)
    pair_order_base = np.arange(len(train.pair_query), dtype=np.int64)
    started = time.perf_counter()
    last_loss = {"context": float("nan"), "intrinsic": float("nan"), "total": float("nan")}
    context_weight = float(candidate["context_weight"])
    intrinsic_weight = float(candidate["intrinsic_weight"])
    routing = candidate["gradient_routing"]

    for epoch in range(epochs):
        context_rng = np.random.default_rng(seed * 100_000 + epoch)
        pair_rng = np.random.default_rng(seed * 200_000 + epoch)
        context_order = context_order_base.copy()
        pair_order = pair_order_base.copy()
        context_rng.shuffle(context_order)
        pair_rng.shuffle(pair_order)
        context_total = 0.0
        context_seen = 0
        model.train()

        for indices in T5R3.batches(context_order):
            index = torch.from_numpy(indices)
            positions = batch[spec["context_view"]][index]
            z = model.encode(positions, batch["team"][index], batch["adjacency"][index], spec["pooling"])
            loss = context_weight * T5R3.context_loss(
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
            context_total += float(loss.detach()) * len(indices)
            context_seen += len(indices)

        pair_total = 0.0
        pair_seen = 0
        for indices in T5R3.batches(pair_order):
            index = torch.from_numpy(indices)
            query_index = batch["pair_query"][index]
            positive_index = batch["pair_positive"][index]
            negative_index = batch["pair_negative"][index]
            all_indices = torch.cat([query_index, positive_index, negative_index], dim=0)
            positions = batch[spec["intrinsic_view"]][all_indices]
            team = batch["team"][all_indices]
            adjacency_batch = batch["adjacency"][all_indices]
            z = model.encode(positions, team, adjacency_batch, spec["pooling"])
            routed_z = z.detach() if routing == "mode_head_only" else z
            loss = intrinsic_weight * T5R3.pair_loss(model, routed_z, len(indices))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            pair_total += float(loss.detach()) * len(indices)
            pair_seen += len(indices)

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
        "context_weight": context_weight,
        "intrinsic_weight": intrinsic_weight,
        "gradient_routing": routing,
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
        model = CandidateTaskModel(candidate["head_placement"])
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
                "valid_centroid_mae_mean": float(np.mean(values(("valid", "context", "centroid_match_grouped_mae")))),
                "valid_pair_accuracy_mean": float(np.mean(values(("valid", "intrinsic", "match_grouped_ranking_accuracy")))),
                "valid_mrr_at_2_mean": float(np.mean(values(("valid", "intrinsic", "match_grouped_mrr_at_2")))),
                "valid_geometry_relation_spearman_mean": float(np.mean(values(("valid", "intrinsic", "natural_geometry_latent_distance_spearman")))),
                "valid_z_mode_translation_response_mean": float(np.mean(values(("valid", "translation", "z_mode_global_translation_response")))),
                "valid_z_ctx_accessibility_response_mean": float(np.mean(values(("valid", "translation", "z_ctx_context_accessibility_response")))),
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROUND1_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--smoke", action="store_true", help="allow a one-epoch wiring smoke")
    parser.add_argument("--only-candidate", action="append", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs != 80 and not args.smoke:
        raise ValueError("formal T5R4 Round 1 runs must use 80 epochs; use --smoke for a reduced wiring check")
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
    temporary_output.mkdir(parents=True)
    np.save(temporary_output / "adjacency_train.npy", train_adjacency)
    np.save(temporary_output / "adjacency_valid.npy", valid_adjacency)
    started = time.perf_counter()
    try:
        protocol: dict[str, Any] = {
            "status": "T5R4_ROUND1_RUNNING",
            "phase": "P3_CAUSAL_MECHANISM",
            "task_lane": "P3-T5R4",
            "round": 1,
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256_file(config_path),
            "t5r2_lock_status": baseline_lock["status"],
            "t5r2_lock_sha256": sha256_file(LOCK_ROOT / "baseline_and_metric_lock.json"),
            "t5r3_reference_summary": str(REFERENCE_SUMMARY.relative_to(ROOT)),
            "t5r3_reference_summary_sha256": sha256_file(REFERENCE_SUMMARY),
            "official_revision": T5R3.EXPECTED_OFFICIAL_REVISION,
            "visible_splits": ["train", "valid"],
            "reserved_holdout": {"match_ids": ["J03WQQ"], "loaded": False, "task_arrays_written": False},
            "external_results_used": False,
            "old_heldout_used": False,
            "candidate_selection_performed": False,
            "seeds": config["search"]["seeds"],
            "torch_threads": torch.get_num_threads(),
            "search_policy": {
                "one_major_axis_per_candidate": True,
                "candidate_specific_values_are_exploratory": True,
                "no_automatic_promotion": True,
                "round2_requires_review": True,
            },
            "candidates": candidates,
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
        manifest: dict[str, Any] = {
            "status": "T5R4_ROUND1_COMPLETE",
            "phase": "P3_CAUSAL_MECHANISM",
            "task_lane": "P3-T5R4",
            "round": 1,
            "protocol": "protocol.json",
            "results": "model_results.json",
            "summary": "summary.json",
            "reference_t5r3_summary": "reference_t5r3_summary.json",
            "candidate_count": len(candidates),
            "model_count": len(records),
            "seed_count": len(config["search"]["seeds"]),
            "reserved_holdout_loaded": False,
            "external_results_used": False,
            "old_heldout_used": False,
            "candidate_selection_performed": False,
            "decision_status": "REVIEW_AFTER_ROUND1",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        write_json(temporary_output / "manifest.json", manifest)
        protocol["status"] = manifest["status"]
        protocol["manifest_sha256"] = json_sha256(manifest)
        write_json(temporary_output / "protocol.json", protocol)
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
        print(f"T5R4 Round 1 failed; incomplete staging kept at {temporary_output}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
