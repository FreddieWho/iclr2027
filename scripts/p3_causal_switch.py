#!/usr/bin/env python3
"""P3-T4/T5 single causal switch: unary versus relational pooling.

The switch has no learnable parameters. Both variants use the same frozen
Phase-GAT initialization, data, optimizer, epochs, and parameter count; only
the pooling statistic changes. The prospective intervention set is read after
its response-blind manifest has been frozen.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p3_support_geometry as geometry  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "phase3_support_geometry_v1.yaml"
VARIANT_MODES = ("team_mean", "relational_pairwise")


def load_metadata(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "artifacts" / "phase1" / "canonical_samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def set_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


def phase_classes(root: Path) -> list[str]:
    manifest = json.loads(
        (root / "artifacts" / "phase1" / "point_mainline" / "model_manifest.json").read_text(encoding="utf-8")
    )
    rows = [row for row in manifest if row["model_id"] == "phase_gat_seed11"]
    if len(rows) != 1:
        raise ValueError("phase-gat manifest row is missing")
    return [str(value) for value in rows[0]["phase_classes"]]


def task_tensors(root: Path, arrays: dict[str, np.ndarray], classes: list[str]):
    import torch

    metadata = load_metadata(root)
    class_index = {name: index for index, name in enumerate(classes)}
    labels = np.asarray([class_index.get(row.get("phase_label"), -1) for row in metadata], dtype=np.int64)
    train = np.asarray([row["split"] == "train" and label >= 0 for row, label in zip(metadata, labels)], dtype=bool)
    dev = np.asarray([row["split"] == "dev" and label >= 0 for row, label in zip(metadata, labels)], dtype=bool)
    heldout = np.asarray([row["split"] == "heldout" and label >= 0 for row, label in zip(metadata, labels)], dtype=bool)
    positions = torch.as_tensor(np.asarray(arrays["positions"], dtype=np.float32))
    teams = torch.as_tensor(np.asarray(arrays["team_slots"], dtype=np.int64))
    adjacency = torch.as_tensor(np.asarray(arrays["adjacency"], dtype=np.float32))
    label_tensor = torch.as_tensor(labels)
    return positions, teams, adjacency, label_tensor, train, dev, heldout


def train_variant(
    root: Path,
    arrays: dict[str, np.ndarray],
    classes: list[str],
    seed: int,
    pooling_mode: str,
    epochs: int,
    learning_rate: float,
) -> tuple[geometry.FrozenGeometryModel, dict[str, Any]]:
    import torch

    base = geometry.load_frozen_model(root, f"phase_gat_seed{seed}")
    model = copy.deepcopy(base.model)
    model.train()
    positions, teams, adjacency, labels, train_mask, dev_mask, heldout_mask = task_tensors(root, arrays, classes)
    wrapper = geometry.FrozenGeometryModel(
        model_id=f"phase_gat_{pooling_mode}_seed{seed}",
        architecture="Phase-GAT",
        model=model,
        uses_adjacency=True,
        pooling_mode=pooling_mode,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    train_indices = torch.as_tensor(np.flatnonzero(train_mask), dtype=torch.long)
    if len(train_indices) == 0:
        raise ValueError("no labelled train examples for phase task")
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        z = wrapper.layer_values(positions, teams, adjacency)["pooled_embedding"]
        logits = model.head(z)
        loss = torch.nn.functional.cross_entropy(logits[train_indices], labels[train_indices])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        z = wrapper.layer_values(positions, teams, adjacency)["pooled_embedding"]
        logits = model.head(z)
        predicted = logits.argmax(dim=-1).cpu().numpy()
    from sklearn.metrics import f1_score

    metrics = {
        "variant": pooling_mode,
        "seed": int(seed),
        "pooling_mode": pooling_mode,
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_label_count": int(train_mask.sum()),
        "dev_label_count": int(dev_mask.sum()),
        "heldout_label_count": int(heldout_mask.sum()),
    }
    for split_name, mask in (("train", train_mask), ("dev", dev_mask), ("heldout", heldout_mask)):
        observed = labels.cpu().numpy()[mask]
        pred = predicted[mask]
        if len(observed):
            metrics[f"{split_name}_accuracy"] = float(np.mean(observed == pred))
            metrics[f"{split_name}_macro_f1"] = float(f1_score(observed, pred, average="macro", zero_division=0))
        else:
            metrics[f"{split_name}_accuracy"] = float("nan")
            metrics[f"{split_name}_macro_f1"] = float("nan")
    return wrapper, metrics


def read_prospective_arms(root: Path) -> pd.DataFrame:
    manifest = json.loads(
        (root / "artifacts" / "phase3" / "support_geometry_prospective_v1" / "intervention_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    frames = [pd.read_parquet(root / str(shard["arms_path"])) for shard in manifest["shards"]]
    return pd.concat(frames, ignore_index=True)


def evaluate_variant_responses(
    root: Path,
    wrapper: geometry.FrozenGeometryModel,
    arrays: dict[str, np.ndarray],
    arms: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = np.asarray(arrays["positions"], dtype=np.float32)
    teams = np.asarray(arrays["team_slots"], dtype=np.int64)
    adjacency = np.asarray(arrays["adjacency"], dtype=np.float32)
    valid = arms[arms["valid"].astype(bool)].copy()
    deltas = np.stack([geometry.parse_delta(value) for value in valid["delta"]])
    sample_indices = valid["sample_index"].to_numpy(dtype=int)
    moved = positions[sample_indices] + deltas
    moved_teams = teams[sample_indices]
    moved_adjacency = adjacency[sample_indices]
    baseline = geometry._encode_in_chunks(wrapper, positions, teams, adjacency, batch_size=64)
    encoded = geometry._encode_in_chunks(wrapper, moved, moved_teams, moved_adjacency, batch_size=64)
    base_norm = geometry._l2_normalize_rows(baseline[sample_indices])
    moved_norm = geometry._l2_normalize_rows(encoded)
    distance = 1.0 - np.sum(base_norm * moved_norm, axis=1)
    response = valid.copy()
    response["model_id"] = wrapper.model_id
    response["architecture"] = wrapper.architecture
    response["model_seed"] = int(wrapper.model_id.rsplit("seed", 1)[1])
    response["pooling_mode"] = wrapper.pooling_mode
    response["response_distance"] = distance.astype(np.float32)
    response["normalized_response"] = (
        distance / response["epsilon"].to_numpy(float)
    ).astype(np.float32)
    arm_geometry = geometry.geometry_arm_rows(
        root,
        wrapper,
        arrays,
        response,
        ["pooled_embedding"],
    )
    pair_geometry = geometry.pair_rows_from_arms(arm_geometry)
    pair_geometry["pooling_mode"] = wrapper.pooling_mode
    return arm_geometry, pair_geometry


def context_translation_metrics(
    wrappers: list[geometry.FrozenGeometryModel],
    arrays: dict[str, np.ndarray],
    epsilons: list[float],
) -> pd.DataFrame:
    positions = np.asarray(arrays["positions"], dtype=np.float32)
    teams = np.asarray(arrays["team_slots"], dtype=np.int64)
    adjacency = np.asarray(arrays["adjacency"], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for wrapper in wrappers:
        baseline = geometry._encode_in_chunks(wrapper, positions, teams, adjacency, batch_size=64)
        values = []
        for sample_index in range(len(positions)):
            rng = np.random.default_rng(20260901 + sample_index)
            direction = rng.normal(size=2).astype(np.float32)
            direction /= np.linalg.norm(direction)
            for epsilon in epsilons:
                delta = np.tile(direction * float(epsilon) / np.sqrt(20.0), (20, 1)).astype(np.float32)
                moved = positions[sample_index : sample_index + 1] + delta[None, ...]
                encoded = geometry._encode_in_chunks(
                    wrapper,
                    moved,
                    teams[sample_index : sample_index + 1],
                    adjacency[sample_index : sample_index + 1],
                    batch_size=1,
                )
                distance = 1.0 - float(
                    np.dot(
                        geometry._l2_normalize_rows(baseline[sample_index : sample_index + 1])[0],
                        geometry._l2_normalize_rows(encoded)[0],
                    )
                )
                values.append(distance / float(epsilon))
        rows.append({
            "model_id": wrapper.model_id,
            "seed": int(wrapper.model_id.rsplit("seed", 1)[1]),
            "pooling_mode": wrapper.pooling_mode,
            "n_translations": len(values),
            "mean_normalized_context_response": float(np.mean(values)),
            "median_normalized_context_response": float(np.median(values)),
        })
    return pd.DataFrame.from_records(rows)


def repair_summaries(root: Path, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Add seed-resolved summaries without overwriting the completed switch."""
    output_dir = root / "artifacts" / "phase3" / "selected_causal_switch_v1"
    pair = pd.read_parquet(output_dir / "switch_pair_geometry.parquet")
    response_records = []
    for (model_id, mode), group in pair.groupby(["model_id", "pooling_mode"], sort=True):
        from scipy.stats import spearmanr

        response_records.append({
            "model_id": str(model_id),
            "seed": int(str(model_id).rsplit("seed", 1)[1]),
            "pooling_mode": str(mode),
            "n_pair_rows": int(len(group)),
            "n_matches": int(group["source_match_id"].nunique()),
            "mean_observed_pair_effect": float(group["observed_pair_effect"].mean()),
            "mean_abs_observed_pair_effect": float(group["observed_pair_effect"].abs().mean()),
            "mean_delta_q_full": float(group["delta_q_full"].mean()),
            "mean_abs_delta_q_full": float(group["delta_q_full"].abs().mean()),
            "mean_abs_delta_q_diagonal": float(group["delta_q_diagonal"].abs().mean()),
            "mean_abs_delta_q_off_diagonal": float(group["delta_q_off_diagonal"].abs().mean()),
            "off_diagonal_to_full_abs_ratio": float(
                group["delta_q_off_diagonal"].abs().mean() / max(group["delta_q_full"].abs().mean(), 1e-30)
            ),
            "geometry_response_spearman": float(
                spearmanr(group["observed_pair_effect"], group["delta_q_full"]).statistic
            ),
        })
    response_path = output_dir / "switch_response_by_seed.parquet"
    geometry.write_parquet(response_path, pd.DataFrame.from_records(response_records))
    arrays = geometry.load_arrays(root)
    wrappers = []
    for seed in (11, 23, 47):
        base = geometry.load_frozen_model(root, f"phase_gat_seed{seed}")
        import torch

        for mode in VARIANT_MODES:
            checkpoint = output_dir / "checkpoints" / f"phase_gat_{mode}_seed{seed}.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            base.model.load_state_dict(state, strict=True)
            wrappers.append(geometry.FrozenGeometryModel(
                model_id=f"phase_gat_{mode}_seed{seed}",
                architecture="Phase-GAT",
                model=copy.deepcopy(base.model),
                uses_adjacency=True,
                pooling_mode=mode,
            ))
    context = context_translation_metrics(
        wrappers,
        arrays,
        [float(value) for value in config["geometry"]["p2_response_epsilons"]],
    )
    context_path = output_dir / "switch_context_robustness_by_seed.parquet"
    geometry.write_parquet(context_path, context)
    task = pd.read_parquet(output_dir / "switch_task_metrics.parquet")
    seed_summary = pd.merge(
        pd.DataFrame.from_records(response_records),
        context,
        on=["model_id", "seed", "pooling_mode"],
        how="left",
    ).merge(
        task[["variant", "seed", "pooling_mode", "heldout_accuracy", "heldout_macro_f1"]],
        on=["seed", "pooling_mode"],
        how="left",
    ).drop(columns=["variant"])
    seed_path = output_dir / "switch_seed_summary.parquet"
    geometry.write_parquet(seed_path, seed_summary)
    receipt = {
        "status": "SUMMARY_PROVENANCE_REPAIRED",
        "config_sha256": geometry.file_sha256(config_path),
        "source_pair_geometry_sha256": geometry.file_sha256(output_dir / "switch_pair_geometry.parquet"),
        "outputs": {
            "response_by_seed": geometry.output_record(root, response_path),
            "context_by_seed": geometry.output_record(root, context_path),
            "seed_summary": geometry.output_record(root, seed_path),
        },
    }
    geometry.write_json(output_dir / "summary_repair_receipt.json", receipt)
    geometry.write_sha256sums(root, output_dir, "summary_repair_SHA256SUMS")
    return {
        "status": receipt["status"],
        "response_by_seed": geometry.relative_path(root, response_path),
        "context_by_seed": geometry.relative_path(root, context_path),
        "seed_summary": geometry.relative_path(root, seed_path),
    }


def run_switch(root: Path, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = root / "artifacts" / "phase3" / "selected_causal_switch_v1"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    arrays = geometry.load_arrays(root)
    classes = phase_classes(root)
    t4 = config["t4_causal_switch"]
    if t4.get("selected_route") != "pooling_accessibility":
        raise ValueError("causal switch route is not the selected pooling-accessibility route")
    seeds = [11, 23, 47]
    epochs = int(t4.get("training_epochs", 80))
    learning_rate = float(t4.get("learning_rate", 1e-3))
    wrappers: list[geometry.FrozenGeometryModel] = []
    task_records: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    output_dir.joinpath("checkpoints").mkdir()
    for seed in seeds:
        for mode in VARIANT_MODES:
            wrapper, metrics = train_variant(
                root, arrays, classes, seed, mode, epochs, learning_rate
            )
            wrappers.append(wrapper)
            task_records.append(metrics)
            checkpoint_path = output_dir / "checkpoints" / f"phase_gat_{mode}_seed{seed}.pt"
            import torch

            torch.save(wrapper.model.state_dict(), checkpoint_path)
            checkpoint_records.append({
                "model_id": wrapper.model_id,
                "path": geometry.relative_path(root, checkpoint_path),
                "sha256": geometry.file_sha256(checkpoint_path),
                "parameter_count": metrics["parameter_count"],
                "pooling_mode": mode,
                "seed": seed,
            })
    arms = read_prospective_arms(root)
    arm_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    for wrapper in wrappers:
        arm_frame, pair_frame = evaluate_variant_responses(root, wrapper, arrays, arms)
        arm_frames.append(arm_frame)
        pair_frames.append(pair_frame)
    arm_frame = pd.concat(arm_frames, ignore_index=True)
    pair_frame = pd.concat(pair_frames, ignore_index=True)
    geometry.write_parquet(output_dir / "switch_arm_geometry.parquet", arm_frame)
    geometry.write_parquet(output_dir / "switch_pair_geometry.parquet", pair_frame)
    response_records: list[dict[str, Any]] = []
    for mode, group in pair_frame.groupby("pooling_mode", sort=True):
        from scipy.stats import spearmanr

        rho = float(spearmanr(group["observed_pair_effect"], group["delta_q_full"]).statistic)
        response_records.append({
            "pooling_mode": mode,
            "n_pair_rows": int(len(group)),
            "n_matches": int(group["source_match_id"].nunique()),
            "mean_observed_pair_effect": float(group["observed_pair_effect"].mean()),
            "mean_abs_observed_pair_effect": float(group["observed_pair_effect"].abs().mean()),
            "mean_delta_q_full": float(group["delta_q_full"].mean()),
            "mean_abs_delta_q_full": float(group["delta_q_full"].abs().mean()),
            "mean_abs_delta_q_diagonal": float(group["delta_q_diagonal"].abs().mean()),
            "mean_abs_delta_q_off_diagonal": float(group["delta_q_off_diagonal"].abs().mean()),
            "off_diagonal_to_full_abs_ratio": float(
                group["delta_q_off_diagonal"].abs().mean() / max(group["delta_q_full"].abs().mean(), 1e-30)
            ),
            "geometry_response_spearman": rho,
        })
    response_frame = pd.DataFrame.from_records(response_records)
    context_frame = context_translation_metrics(
        wrappers,
        arrays,
        [float(value) for value in config["geometry"]["p2_response_epsilons"]],
    )
    task_frame = pd.DataFrame.from_records(task_records)
    geometry.write_parquet(output_dir / "switch_response_summary.parquet", response_frame)
    geometry.write_parquet(output_dir / "switch_context_robustness.parquet", context_frame)
    geometry.write_parquet(output_dir / "switch_task_metrics.parquet", task_frame)
    summary = {
        "status": "T4_T5_SINGLE_SWITCH_COMPLETE",
        "task": "P3-T4-T5",
        "route": "pooling_accessibility",
        "architecture_family": "Phase-GAT",
        "variants": list(VARIANT_MODES),
        "seeds": seeds,
        "matched_capacity": True,
        "parameter_counts": sorted({record["parameter_count"] for record in task_records}),
        "prospective_intervention_manifest_sha256": geometry.file_sha256(
            root / "artifacts" / "phase3" / "support_geometry_prospective_v1" / "intervention_manifest.json"
        ),
        "response_summary_path": geometry.relative_path(root, output_dir / "switch_response_summary.parquet"),
        "context_robustness_path": geometry.relative_path(root, output_dir / "switch_context_robustness.parquet"),
        "task_metrics_path": geometry.relative_path(root, output_dir / "switch_task_metrics.parquet"),
        "checkpoint_records": checkpoint_records,
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
        "interpretation_status": "NOT_YET_ROUTED",
    }
    geometry.write_json(output_dir / "switch_summary.json", summary)
    geometry.write_json(output_dir / "switch_receipt.json", {
        "status": summary["status"],
        "checkpoint": "P3-T4-T5",
        "summary_sha256": geometry.file_sha256(output_dir / "switch_summary.json"),
        "prospective_intervention_manifest_sha256": summary["prospective_intervention_manifest_sha256"],
    })
    (output_dir / "known_limitations.md").write_text(
        "This is one pooling-accessibility switch in the Phase-GAT family. "
        "The phase-label heldout set is small and label coverage is reported. "
        "A response change without an objective task improvement is representation shaping, not repair. "
        "No claim is made for all architectures or for AMR until the P3 gate is reviewed.\n",
        encoding="utf-8",
    )
    geometry.write_sha256sums(root, output_dir, "SHA256SUMS")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repair-summaries", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    config_path = args.config.resolve()
    config = geometry.load_config(root, config_path)
    summary = repair_summaries(root, config_path, config) if args.repair_summaries else run_switch(root, config_path, config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
