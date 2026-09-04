#!/usr/bin/env python3
"""Run the locked P3-T5R3 fixed dual-channel sanity experiment.

This runner consumes only the T5R2 train/valid task views.  It does not open
the IDSSE reserved match, SNGAR test data, the old exposed heldout data, or any
external result.  The four neural variants use the same Phase-GAT-equivalent
encoder width/depth and the same task heads; the fixed dual variant changes
only which frozen view is read by each task head.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
LOCK_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
VIEW_ROOT = LOCK_ROOT / "data_views"
DEFAULT_OUTPUT = LOCK_ROOT / "t5r3_sanity_v1"

EXPECTED_LOCK_STATUS = "T5R2_CLOSED_BASELINE_AND_METRIC_LOCK"
EXPECTED_SPLIT_STATUS = "MATCH_SPLIT_FROZEN_BEFORE_T5R2_TASK_CONSTRUCTION"
EXPECTED_OFFICIAL_REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"
SEEDS = (11, 23, 47)
BATCH_SIZE = 64
LATENT_DIM = 128
HIDDEN_DIM = 128
MESSAGE_PASSING_LAYERS = 2
K_NEIGHBORS = 4
PAIR_MARGIN = 0.20
GLOBAL_TRANSLATION = np.asarray([0.25, -0.15], dtype=np.float32)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260904

MATCH_SPLITS = {
    "J03WMX": "train",
    "J03WOH": "train",
    "J03WPY": "train",
    "J03WR9": "train",
    "J03WN1": "valid",
    "J03WOY": "valid",
    "J03WQQ": "reserved_holdout",
}

VARIANTS = {
    "raw_single_channel_phase_gat_team_mean": {
        "context_view": "raw",
        "intrinsic_view": "raw",
        "pooling": "team_mean",
        "dual": False,
    },
    "centered_single_channel_phase_gat_team_mean": {
        "context_view": "centered",
        "intrinsic_view": "centered",
        "pooling": "team_mean",
        "dual": False,
    },
    "raw_relational_pooling_phase_gat": {
        "context_view": "raw",
        "intrinsic_view": "raw",
        "pooling": "relational_pairwise",
        "dual": False,
    },
    "fixed_dual_channel_shared_phase_gat": {
        "context_view": "raw",
        "intrinsic_view": "centered",
        "pooling": "team_mean",
        "dual": True,
    },
}


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def center_positions(positions: np.ndarray) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float32)
    return values - values.mean(axis=1, keepdims=True)


def knn_adjacency_batch(positions: np.ndarray, k: int = K_NEIGHBORS) -> np.ndarray:
    """Reproduce the locked P0 weighted symmetric kNN-4 construction."""
    values = np.asarray(positions, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (20, 2):
        raise ValueError(f"expected (N,20,2) positions, got {values.shape}")
    adjacency = np.zeros((len(values), 20, 20), dtype=np.float32)
    for sample_index, position in enumerate(values):
        diff = position[:, None, :] - position[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32))
        nonzero = distances[distances > 0]
        scale = float(np.median(nonzero)) if nonzero.size else 1.0
        for node_index in range(20):
            neighbours = np.argsort(distances[node_index], kind="mergesort")[1 : k + 1]
            for neighbour in neighbours:
                weight = math.exp(-float(distances[node_index, neighbour] ** 2) / max(scale**2, 1e-12))
                adjacency[sample_index, node_index, neighbour] = max(
                    adjacency[sample_index, node_index, neighbour], weight
                )
                adjacency[sample_index, neighbour, node_index] = max(
                    adjacency[sample_index, neighbour, node_index], weight
                )
    return adjacency


@dataclass
class SplitData:
    name: str
    raw: np.ndarray
    centered: np.ndarray
    team_slots: np.ndarray
    snapshot_ids: np.ndarray
    match_ids: np.ndarray
    phase: np.ndarray
    zone: np.ndarray
    centroids: np.ndarray
    pair_query: np.ndarray
    pair_positive: np.ndarray
    pair_negative: np.ndarray
    pair_matches: np.ndarray
    pair_positive_geometry: np.ndarray
    pair_negative_geometry: np.ndarray


def _label_index(values: Iterable[str], labels: tuple[str, ...], name: str) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(labels)}
    result = np.asarray([mapping.get(str(value), -1) for value in values], dtype=np.int64)
    if np.any(result < 0):
        raise ValueError(f"unknown {name} label")
    return result


def load_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    split_path = LOCK_ROOT / "split_lock.json"
    baseline_path = LOCK_ROOT / "baseline_and_metric_lock.json"
    split_lock = json.loads(split_path.read_text(encoding="utf-8"))
    baseline_lock = json.loads(baseline_path.read_text(encoding="utf-8"))
    if split_lock.get("status") != EXPECTED_SPLIT_STATUS:
        raise ValueError("T5R2 split lock status mismatch")
    if baseline_lock.get("status") != EXPECTED_LOCK_STATUS:
        raise ValueError("T5R2 baseline lock status mismatch")
    if baseline_lock.get("dataset", {}).get("official_revision") != EXPECTED_OFFICIAL_REVISION:
        raise ValueError("official IDSSE revision mismatch")
    training_lock = baseline_lock.get("training_lock", {})
    expected_training = {
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0,
        "epochs": 80,
        "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "message_passing_layers": MESSAGE_PASSING_LAYERS,
        "augmentation": "none",
        "early_stopping": False,
        "class_weighting": False,
    }
    for key, expected in expected_training.items():
        if training_lock.get(key) != expected:
            raise ValueError(f"T5R2 training lock mismatch for {key}: {training_lock.get(key)!r}")
    firewall = baseline_lock.get("firewall", {})
    if firewall.get("model_results_used") is not False:
        raise ValueError("T5R2 lock reports model results were used")
    if firewall.get("reserved_holdout_task_arrays_written") is not False:
        raise ValueError("reserved task arrays were written")
    if firewall.get("reserved_holdout_match_ids") != ["J03WQQ"]:
        raise ValueError("reserved match firewall mismatch")
    return split_lock, baseline_lock


def load_split(name: str, split_lock: dict[str, Any]) -> SplitData:
    raw_path = VIEW_ROOT / f"positions_raw_{name}.npy"
    centered_path = VIEW_ROOT / f"positions_centered_{name}.npy"
    team_path = VIEW_ROOT / f"team_slots_{name}.npy"
    index_path = VIEW_ROOT / f"snapshot_index_{name}.parquet"
    pair_path = VIEW_ROOT / f"natural_pair_ranking_{name}.parquet"
    raw = np.load(raw_path).astype(np.float32, copy=False)
    centered = np.load(centered_path).astype(np.float32, copy=False)
    team_slots = np.load(team_path).astype(np.int64, copy=False)
    index = pd.read_parquet(index_path)
    pairs = pd.read_parquet(pair_path)
    if raw.shape != (len(index), 20, 2) or centered.shape != raw.shape or team_slots.shape != (len(index), 20):
        raise ValueError(f"{name} array/index shape mismatch")
    if not np.allclose(centered, center_positions(raw), atol=1e-6):
        raise ValueError(f"{name} centered view is not global-centroid centered")
    if index["snapshot_id"].duplicated().any():
        raise ValueError(f"{name} snapshot IDs are not unique")
    expected_matches = {match_id for match_id, split in MATCH_SPLITS.items() if split == name}
    observed_matches = set(index["source_match_id"].astype(str))
    if observed_matches != expected_matches:
        raise ValueError(f"{name} match set mismatch: {sorted(observed_matches)}")
    if set(index["split"].astype(str)) != {name}:
        raise ValueError(f"{name} contains another split")
    id_to_index = {str(value): index for index, value in enumerate(index["snapshot_id"].astype(str))}
    pair_columns = ("query_snapshot_id", "positive_snapshot_id", "negative_snapshot_id")
    pair_indices = []
    for column in pair_columns:
        values = [id_to_index.get(str(value), -1) for value in pairs[column]]
        if any(value < 0 for value in values):
            raise ValueError(f"{name} pair references an unknown snapshot")
        pair_indices.append(np.asarray(values, dtype=np.int64))
    if set(pairs["split"].astype(str)) != {name}:
        raise ValueError(f"{name} pair table contains another split")
    if not set(pairs["source_match_id"].astype(str)).issubset(expected_matches):
        raise ValueError(f"{name} pair table contains a reserved or foreign match")
    if name not in {"train", "valid"}:
        raise ValueError("only train and valid may be loaded by T5R3")
    return SplitData(
        name=name,
        raw=raw,
        centered=centered,
        team_slots=team_slots,
        snapshot_ids=index["snapshot_id"].astype(str).to_numpy(),
        match_ids=index["source_match_id"].astype(str).to_numpy(),
        phase=_label_index(index["phase_label"], ("firstHalf", "secondHalf"), "phase"),
        zone=_label_index(index["home_field_zone"], ("defensive_third", "middle_third", "attacking_third"), "field zone"),
        centroids=index[["home_centroid_x", "home_centroid_y", "away_centroid_x", "away_centroid_y"]]
        .to_numpy(dtype=np.float32),
        pair_query=pair_indices[0],
        pair_positive=pair_indices[1],
        pair_negative=pair_indices[2],
        pair_matches=pairs["source_match_id"].astype(str).to_numpy(),
        pair_positive_geometry=pairs["positive_internal_geometry_distance"].to_numpy(dtype=np.float32),
        pair_negative_geometry=pairs["negative_internal_geometry_distance"].to_numpy(dtype=np.float32),
    )


def team_pool(hidden: torch.Tensor, team_slots: torch.Tensor) -> torch.Tensor:
    pools = []
    for team_slot in (0, 1):
        mask = (team_slots == team_slot).float().unsqueeze(-1)
        pools.append((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0))
    return torch.cat(pools, dim=-1)


def relational_pool(hidden: torch.Tensor, team_slots: torch.Tensor) -> torch.Tensor:
    pools = []
    for team_slot in (0, 1):
        mask = (team_slots == team_slot).float().unsqueeze(-1)
        count = mask.sum(dim=1)
        summed = (hidden * mask).sum(dim=1)
        mean = summed / count.clamp_min(1.0)
        pair_sum = 0.5 * (summed.square() - (hidden.square() * mask).sum(dim=1))
        pair_count = 0.5 * count * (count - 1.0)
        pair_mean = pair_sum / pair_count.clamp_min(1.0)
        pools.append(torch.where(pair_count > 0.0, pair_mean, mean))
    return torch.cat(pools, dim=-1)


class GraphAttentionLayer(nn.Module):
    """Fixed weighted graph message-passing equivalent for the locked encoder."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        weights = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        neighbours = torch.bmm(weights, self.value(hidden))
        return torch.relu(self.self_proj(hidden) + self.neigh_proj(neighbours))


class GraphEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(4, HIDDEN_DIM)
        self.layers = nn.ModuleList([GraphAttentionLayer() for _ in range(MESSAGE_PASSING_LAYERS)])

    def forward(self, positions: torch.Tensor, team_slots: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        features = torch.cat([positions, F.one_hot(team_slots.long(), 2).float()], dim=-1)
        hidden = torch.relu(self.in_proj(features))
        for layer in self.layers:
            hidden = layer(hidden, adjacency)
        return hidden


class TaskModel(nn.Module):
    """Locked-width fixed-graph encoder with task heads and matched capacity."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = GraphEncoder()
        self.pool_projection = nn.Sequential(nn.Linear(2 * HIDDEN_DIM, LATENT_DIM), nn.ReLU())
        self.phase_head = nn.Linear(LATENT_DIM, 2)
        self.zone_head = nn.Linear(LATENT_DIM, 3)
        self.centroid_head = nn.Linear(LATENT_DIM, 4)
        self.mode_head = nn.Sequential(nn.Linear(LATENT_DIM, 64), nn.ReLU())

    def encode(
        self,
        positions: torch.Tensor,
        team_slots: torch.Tensor,
        adjacency: torch.Tensor,
        pooling: str,
    ) -> torch.Tensor:
        hidden = self.encoder(positions, team_slots, adjacency)
        if pooling == "team_mean":
            pooled = team_pool(hidden, team_slots)
        elif pooling == "relational_pairwise":
            pooled = relational_pool(hidden, team_slots)
        else:
            raise ValueError(f"unknown pooling mode: {pooling}")
        return self.pool_projection(pooled)

    def context(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.phase_head(z), self.zone_head(z), self.centroid_head(z)

    def mode_embedding(self, z: torch.Tensor) -> torch.Tensor:
        return self.mode_head(z)


def batches(order: np.ndarray, batch_size: int = BATCH_SIZE) -> Iterable[np.ndarray]:
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def tensors(data: SplitData, adjacency: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        "raw": torch.from_numpy(data.raw),
        "centered": torch.from_numpy(data.centered),
        "team": torch.from_numpy(data.team_slots),
        "adjacency": torch.from_numpy(adjacency),
        "phase": torch.from_numpy(data.phase),
        "zone": torch.from_numpy(data.zone),
        "centroids": torch.from_numpy(data.centroids),
        "pair_query": torch.from_numpy(data.pair_query),
        "pair_positive": torch.from_numpy(data.pair_positive),
        "pair_negative": torch.from_numpy(data.pair_negative),
    }


def context_loss(
    model: TaskModel,
    z: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    phase_logits, zone_logits, centroid = model.context(z)
    return (
        F.cross_entropy(phase_logits, batch["phase"])
        + F.cross_entropy(zone_logits, batch["zone"])
        + F.mse_loss(centroid, batch["centroids"])
    )


def pair_loss(
    model: TaskModel,
    z: torch.Tensor,
    query_count: int,
) -> torch.Tensor:
    query_z, positive_z, negative_z = z.split(query_count, dim=0)
    query_mode = F.normalize(model.mode_embedding(query_z), dim=-1)
    positive_mode = F.normalize(model.mode_embedding(positive_z), dim=-1)
    negative_mode = F.normalize(model.mode_embedding(negative_z), dim=-1)
    positive_score = (query_mode * positive_mode).sum(dim=-1)
    negative_score = (query_mode * negative_mode).sum(dim=-1)
    return F.relu(PAIR_MARGIN - positive_score + negative_score).mean()


def train_model(
    model: TaskModel,
    spec: dict[str, Any],
    train: SplitData,
    adjacency: np.ndarray,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    set_seed(seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    batch = tensors(train, adjacency)
    context_order_base = np.arange(len(train.raw), dtype=np.int64)
    pair_order_base = np.arange(len(train.pair_query), dtype=np.int64)
    started = time.perf_counter()
    last_loss = {"context": float("nan"), "intrinsic": float("nan"), "total": float("nan")}
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
        for indices in batches(context_order):
            index = torch.from_numpy(indices)
            positions = batch[spec["context_view"]][index]
            z = model.encode(positions, batch["team"][index], batch["adjacency"][index], spec["pooling"])
            loss = context_loss(
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
        for indices in batches(pair_order):
            index = torch.from_numpy(indices)
            query_index = batch["pair_query"][index]
            positive_index = batch["pair_positive"][index]
            negative_index = batch["pair_negative"][index]
            all_indices = torch.cat([query_index, positive_index, negative_index], dim=0)
            positions = batch[spec["intrinsic_view"]][all_indices]
            team = batch["team"][all_indices]
            adjacency_batch = batch["adjacency"][all_indices]
            z = model.encode(positions, team, adjacency_batch, spec["pooling"])
            loss = pair_loss(model, z, len(indices))
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
                f"[{spec['name']}] seed={seed} epoch={epoch + 1}/{epochs} "
                f"context={last_loss['context']:.5f} intrinsic={last_loss['intrinsic']:.5f} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return {
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "last_loss": last_loss,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def encode_split(
    model: TaskModel,
    data: SplitData,
    adjacency: np.ndarray,
    view: str,
    pooling: str,
) -> np.ndarray:
    batch = tensors(data, adjacency)
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(data.raw), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(data.raw))
            positions = batch[view][start:end]
            z = model.encode(positions, batch["team"][start:end], batch["adjacency"][start:end], pooling)
            outputs.append(z.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def f1_for_labels(observed: np.ndarray, predicted: np.ndarray, n_classes: int) -> float:
    scores = []
    for label in range(n_classes):
        true_positive = np.sum((observed == label) & (predicted == label))
        false_positive = np.sum((observed != label) & (predicted == label))
        false_negative = np.sum((observed == label) & (predicted != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2.0 * true_positive) / denominator)
    return float(np.mean(scores))


def match_mean(values: np.ndarray, match_ids: np.ndarray) -> float:
    per_match = match_values(values, match_ids)
    return float(np.mean(per_match)) if len(per_match) else float("nan")


def match_values(values: np.ndarray, match_ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [float(np.mean(values[match_ids == match])) for match in sorted(set(match_ids))],
        dtype=np.float64,
    )


def match_bootstrap_ci(values: np.ndarray, match_ids: np.ndarray, salt: str) -> dict[str, Any]:
    per_match = match_values(values, match_ids)
    if not len(per_match):
        return {"match_count": 0, "mean": float("nan"), "lower_2_5": float("nan"), "upper_97_5": float("nan")}
    seed = BOOTSTRAP_SEED + sum((index + 1) * ord(char) for index, char in enumerate(salt))
    rng = np.random.default_rng(seed)
    sampled = rng.choice(per_match, size=(BOOTSTRAP_REPLICATES, len(per_match)), replace=True).mean(axis=1)
    return {
        "match_count": int(len(per_match)),
        "mean": float(np.mean(per_match)),
        "lower_2_5": float(np.quantile(sampled, 0.025)),
        "upper_97_5": float(np.quantile(sampled, 0.975)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }


def grouped_macro_f1(observed: np.ndarray, predicted: np.ndarray, match_ids: np.ndarray, n_classes: int) -> float:
    per_match = [
        f1_for_labels(observed[match_ids == match], predicted[match_ids == match], n_classes)
        for match in sorted(set(match_ids))
    ]
    return float(np.mean(per_match)) if per_match else float("nan")


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if len(values_a) < 2 or np.std(values_a) <= 1e-12 or np.std(values_b) <= 1e-12:
        return float("nan")
    def ordinal_rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        return ranks
    rank_a = ordinal_rank(np.asarray(values_a, dtype=np.float64))
    rank_b = ordinal_rank(np.asarray(values_b, dtype=np.float64))
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def prediction_metrics(data: SplitData, model: TaskModel, z_context: np.ndarray) -> dict[str, Any]:
    with torch.inference_mode():
        phase_logits, zone_logits, centroid = model.context(torch.from_numpy(z_context))
    phase_pred = phase_logits.argmax(dim=-1).cpu().numpy()
    zone_pred = zone_logits.argmax(dim=-1).cpu().numpy()
    centroid_pred = centroid.cpu().numpy()
    centroid_error = np.abs(centroid_pred - data.centroids)
    phase_correct = (phase_pred == data.phase).astype(np.float32)
    zone_correct = (zone_pred == data.zone).astype(np.float32)
    phase_f1_values = np.asarray(
        [
            f1_for_labels(data.phase[data.match_ids == match], phase_pred[data.match_ids == match], 2)
            for match in sorted(set(data.match_ids))
        ],
        dtype=np.float64,
    )
    zone_f1_values = np.asarray(
        [
            f1_for_labels(data.zone[data.match_ids == match], zone_pred[data.match_ids == match], 3)
            for match in sorted(set(data.match_ids))
        ],
        dtype=np.float64,
    )
    return {
        "phase_match_grouped_macro_f1": grouped_macro_f1(data.phase, phase_pred, data.match_ids, 2),
        "phase_match_grouped_accuracy": match_mean(phase_correct, data.match_ids),
        "field_zone_match_grouped_macro_f1": grouped_macro_f1(data.zone, zone_pred, data.match_ids, 3),
        "field_zone_match_grouped_accuracy": match_mean(zone_correct, data.match_ids),
        "centroid_match_grouped_mae": match_mean(np.mean(centroid_error, axis=1), data.match_ids),
        "centroid_global_mae": float(np.mean(centroid_error)),
        "match_level_bootstrap_ci": {
            "phase_macro_f1": match_bootstrap_ci(phase_f1_values, np.asarray(sorted(set(data.match_ids))), "phase_macro_f1"),
            "field_zone_macro_f1": match_bootstrap_ci(zone_f1_values, np.asarray(sorted(set(data.match_ids))), "field_zone_macro_f1"),
            "phase_accuracy": match_bootstrap_ci(phase_correct, data.match_ids, "phase_accuracy"),
            "field_zone_accuracy": match_bootstrap_ci(zone_correct, data.match_ids, "field_zone_accuracy"),
            "centroid_mae": match_bootstrap_ci(np.mean(centroid_error, axis=1), data.match_ids, "centroid_mae"),
        },
        "sample_count": int(len(data.raw)),
        "match_count": int(len(set(data.match_ids))),
    }


def pair_scores(data: SplitData, model: TaskModel, z: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        mode = F.normalize(model.mode_embedding(torch.from_numpy(z)), dim=-1).cpu().numpy()
    query = mode[data.pair_query]
    positive = mode[data.pair_positive]
    negative = mode[data.pair_negative]
    return np.sum(query * positive, axis=1) - np.sum(query * negative, axis=1)


def pair_metrics(data: SplitData, model: TaskModel, z: np.ndarray) -> dict[str, Any]:
    margins = pair_scores(data, model, z)
    correct = margins > 0
    reciprocal = np.where(correct, 1.0, 0.5)
    latent_distance = np.concatenate(
        [
            np.linalg.norm(z[data.pair_query] - z[data.pair_positive], axis=1),
            np.linalg.norm(z[data.pair_query] - z[data.pair_negative], axis=1),
        ]
    )
    geometry_distance = np.concatenate([data.pair_positive_geometry, data.pair_negative_geometry])
    return {
        "match_grouped_ranking_accuracy": match_mean(correct.astype(np.float32), data.pair_matches),
        "match_grouped_mrr_at_2": match_mean(reciprocal, data.pair_matches),
        "hard_negative_ranking_accuracy": match_mean(correct.astype(np.float32), data.pair_matches),
        "natural_geometry_latent_distance_spearman": spearman(geometry_distance, latent_distance),
        "geometry_response_spearman": spearman(geometry_distance, latent_distance),
        "geometry_response_spearman_note": "natural-pair geometry-to-latent-distance proxy; no IDSSE intervention response is available",
        "match_level_bootstrap_ci": {
            "ranking_accuracy": match_bootstrap_ci(correct.astype(np.float32), data.pair_matches, "ranking_accuracy"),
            "mrr_at_2": match_bootstrap_ci(reciprocal, data.pair_matches, "mrr_at_2"),
        },
        "pair_count": int(len(margins)),
        "correct_pair_count": int(np.sum(correct)),
        "margin_mean": float(np.mean(margins)),
    }


def mean_cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second_norm = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    return float(np.mean(1.0 - np.sum(first_norm * second_norm, axis=1)))


def translation_metrics(
    model: TaskModel,
    data: SplitData,
    adjacency: np.ndarray,
    spec: dict[str, Any],
    z_context: np.ndarray,
    z_intrinsic: np.ndarray,
) -> dict[str, Any]:
    shifted_raw = data.raw + GLOBAL_TRANSLATION[None, None, :]
    shifted_centered = center_positions(shifted_raw)
    context_shifted = shifted_raw if spec["context_view"] == "raw" else shifted_centered
    intrinsic_shifted = shifted_raw if spec["intrinsic_view"] == "raw" else shifted_centered
    shifted_context = encode_array(model, context_shifted, data.team_slots, adjacency, spec["pooling"])
    shifted_intrinsic = encode_array(model, intrinsic_shifted, data.team_slots, adjacency, spec["pooling"])
    return {
        "translation_vector_normalized_pitch": GLOBAL_TRANSLATION.tolist(),
        "z_ctx_context_accessibility_response": mean_cosine_distance(z_context, shifted_context),
        "z_mode_global_translation_response": mean_cosine_distance(z_intrinsic, shifted_intrinsic),
        "centered_input_recomputed_after_translation": spec["intrinsic_view"] == "centered",
    }


def encode_array(
    model: TaskModel,
    positions: np.ndarray,
    team_slots: np.ndarray,
    adjacency: np.ndarray,
    pooling: str,
) -> np.ndarray:
    model.eval()
    position_tensor = torch.from_numpy(np.asarray(positions, dtype=np.float32))
    team_tensor = torch.from_numpy(np.asarray(team_slots, dtype=np.int64))
    adjacency_tensor = torch.from_numpy(np.asarray(adjacency, dtype=np.float32))
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(position_tensor), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(position_tensor))
            outputs.append(
                model.encode(
                    position_tensor[start:end],
                    team_tensor[start:end],
                    adjacency_tensor[start:end],
                    pooling,
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(outputs, axis=0)


def cross_readout_metrics(
    data: SplitData,
    model: TaskModel,
    z_context: np.ndarray,
    z_intrinsic: np.ndarray,
) -> dict[str, Any]:
    context_on_intrinsic = prediction_metrics(data, model, z_intrinsic)
    context_on_context = prediction_metrics(data, model, z_context)
    intrinsic_on_context = pair_metrics(data, model, z_context)
    intrinsic_on_intrinsic = pair_metrics(data, model, z_intrinsic)
    centered_context = z_context - z_context.mean(axis=0, keepdims=True)
    centered_intrinsic = z_intrinsic - z_intrinsic.mean(axis=0, keepdims=True)
    covariance = np.mean(centered_context * centered_intrinsic, axis=0)
    correlation = covariance / np.maximum(centered_context.std(axis=0) * centered_intrinsic.std(axis=0), 1e-12)
    return {
        "context_head_on_z_ctx": context_on_context,
        "context_head_on_z_mode": context_on_intrinsic,
        "mode_head_on_z_ctx": intrinsic_on_context,
        "mode_head_on_z_mode": intrinsic_on_intrinsic,
        "mean_abs_paired_latent_correlation": float(np.mean(np.abs(correlation))),
    }


def analytic_pair_metrics(data: SplitData) -> dict[str, Any]:
    query = data.raw[data.pair_query]
    positive = data.raw[data.pair_positive]
    negative = data.raw[data.pair_negative]
    query_centered = center_positions(query)
    positive_centered = center_positions(positive)
    negative_centered = center_positions(negative)
    raw_positive_score = -np.sqrt(np.mean((query - positive) ** 2, axis=(1, 2)))
    raw_negative_score = -np.sqrt(np.mean((query - negative) ** 2, axis=(1, 2)))
    procrustes_positive_score = -np.sqrt(np.mean((query_centered - positive_centered) ** 2, axis=(1, 2)))
    procrustes_negative_score = -np.sqrt(np.mean((query_centered - negative_centered) ** 2, axis=(1, 2)))
    raw_correct = raw_positive_score > raw_negative_score
    procrustes_correct = procrustes_positive_score > procrustes_negative_score
    return {
        "raw_coordinate_match_grouped_ranking_accuracy": match_mean(raw_correct.astype(np.float32), data.pair_matches),
        "procrustes_match_grouped_ranking_accuracy": match_mean(procrustes_correct.astype(np.float32), data.pair_matches),
        "raw_coordinate_pair_count": int(len(raw_correct)),
        "procrustes_pair_count": int(len(procrustes_correct)),
        "status": "analytic_control_only_no_context_head",
    }


def run_variant(
    name: str,
    spec_base: dict[str, Any],
    train: SplitData,
    valid: SplitData,
    train_adjacency: np.ndarray,
    valid_adjacency: np.ndarray,
    output_root: Path,
    epochs: int,
) -> list[dict[str, Any]]:
    records = []
    for seed in SEEDS:
        spec = dict(spec_base)
        spec["name"] = name
        print(f"=== {name} seed={seed} ===", flush=True)
        set_seed(seed)
        model = TaskModel()
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        train_record = train_model(model, spec, train, train_adjacency, seed, epochs)
        checkpoint = output_root / "models" / f"{name}_seed{seed}.pt"
        stable_output_root = output_root.with_name(output_root.name.removesuffix(".tmp"))
        stable_checkpoint = stable_output_root / "models" / f"{name}_seed{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
        train_context = encode_split(model, train, train_adjacency, spec["context_view"], spec["pooling"])
        train_intrinsic = encode_split(model, train, train_adjacency, spec["intrinsic_view"], spec["pooling"])
        valid_context = encode_split(model, valid, valid_adjacency, spec["context_view"], spec["pooling"])
        valid_intrinsic = encode_split(model, valid, valid_adjacency, spec["intrinsic_view"], spec["pooling"])
        record = {
            "variant": name,
            "seed": seed,
            "architecture": "fixed-graph-message-passing-equivalent",
            "pooling": spec["pooling"],
            "context_view": spec["context_view"],
            "intrinsic_view": spec["intrinsic_view"],
            "parameter_count": parameter_count,
            "train": {
                "context": prediction_metrics(train, model, train_context),
                "intrinsic": pair_metrics(train, model, train_intrinsic),
                "translation": translation_metrics(
                    model, train, train_adjacency, spec, train_context, train_intrinsic
                ),
            },
            "valid": {
                "context": prediction_metrics(valid, model, valid_context),
                "intrinsic": pair_metrics(valid, model, valid_intrinsic),
                "translation": translation_metrics(
                    model, valid, valid_adjacency, spec, valid_context, valid_intrinsic
                ),
                "cross_readout": cross_readout_metrics(valid, model, valid_context, valid_intrinsic),
            },
            "training": train_record,
            "checkpoint": {
                "path": str(stable_checkpoint.relative_to(ROOT)),
                "sha256": sha256_file(checkpoint),
                "bytes": int(checkpoint.stat().st_size),
            },
        }
        records.append(record)
        print(
            f"completed {name} seed={seed}: "
            f"valid_phase_f1={record['valid']['context']['phase_match_grouped_macro_f1']:.4f} "
            f"valid_pair_acc={record['valid']['intrinsic']['match_grouped_ranking_accuracy']:.4f} "
            f"translation={record['valid']['translation']['z_mode_global_translation_response']:.6f}",
            flush=True,
        )
    return records


def aggregate_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for variant in sorted({record["variant"] for record in records}):
        rows = [record for record in records if record["variant"] == variant]
        def values(path: tuple[str, ...]) -> list[float]:
            output = []
            for row in rows:
                value: Any = row
                for key in path:
                    value = value[key]
                output.append(float(value))
            return output
        summary.append(
            {
                "variant": variant,
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs != 80:
        raise ValueError("T5R3 must use the locked 80 epochs; use tests for reduced smoke runs")
    split_lock, baseline_lock = load_lock()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    output = args.output.resolve()
    temporary_output = output.with_name(output.name + ".tmp")
    if output.exists() or temporary_output.exists():
        raise FileExistsError(f"refusing to overwrite existing T5R3 output: {output}")
    started = time.perf_counter()
    temporary_output.mkdir(parents=True)
    try:
        train = load_split("train", split_lock)
        valid = load_split("valid", split_lock)
        train_adjacency = knn_adjacency_batch(train.raw)
        valid_adjacency = knn_adjacency_batch(valid.raw)
        np.save(temporary_output / "adjacency_train.npy", train_adjacency)
        np.save(temporary_output / "adjacency_valid.npy", valid_adjacency)
        protocol = {
            "status": "T5R3_FIXED_DUAL_CHANNEL_SANITY_RUNNING",
            "phase": "P3_CAUSAL_MECHANISM",
            "task_lane": "P3-T5R3",
            "t5r2_lock_status": baseline_lock["status"],
            "t5r2_lock_path": str((LOCK_ROOT / "baseline_and_metric_lock.json").relative_to(ROOT)),
            "t5r2_lock_sha256": sha256_file(LOCK_ROOT / "baseline_and_metric_lock.json"),
            "official_revision": EXPECTED_OFFICIAL_REVISION,
            "split_lock_status": split_lock["status"],
            "split_map": MATCH_SPLITS,
            "visible_splits": ["train", "valid"],
            "reserved_holdout": {"match_ids": ["J03WQQ"], "loaded": False, "task_arrays_written": False},
            "external_results_used": False,
            "old_heldout_used": False,
            "model_results_used_for_selection": False,
            "seeds": list(SEEDS),
            "torch_threads": torch.get_num_threads(),
            "training_lock": {
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "weight_decay": 0,
                "epochs": 80,
                "batch_size": BATCH_SIZE,
                "latent_dim": LATENT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "message_passing_layers": MESSAGE_PASSING_LAYERS,
                "augmentation": "none",
                "early_stopping": False,
                "class_weighting": False,
            },
            "loss_protocol": {
                "context": "cross_entropy_phase + cross_entropy_field_zone + centroid_MSE",
                "intrinsic": "triplet_hinge_on_normalized_mode_embedding",
                "triplet_margin": PAIR_MARGIN,
                "joint_loss_weighting": "context_loss + intrinsic_loss",
            },
            "graph_protocol": {
                "type": "fixed_symmetric_weighted_knn",
                "k": K_NEIGHBORS,
                "weight": "exp(-distance^2 / median_nonzero_distance^2)",
                "source": "T5R2 baseline representation_contract / P0 knn_adjacency",
                "encoder_implementation": "two fixed-weighted message-passing layers; no learned graph or attention adjacency",
            },
            "translation_protocol": {
                "vector_normalized_pitch": GLOBAL_TRANSLATION.tolist(),
                "response": "1-cosine(z_base,z_after_global_translation)",
                "z_mode_centering_reapplied": True,
            },
            "variants": VARIANTS,
            "input_files": {
                "train_raw": str((VIEW_ROOT / "positions_raw_train.npy").relative_to(ROOT)),
                "train_centered": str((VIEW_ROOT / "positions_centered_train.npy").relative_to(ROOT)),
                "train_team_slots": str((VIEW_ROOT / "team_slots_train.npy").relative_to(ROOT)),
                "train_snapshot_index": str((VIEW_ROOT / "snapshot_index_train.parquet").relative_to(ROOT)),
                "train_pairs": str((VIEW_ROOT / "natural_pair_ranking_train.parquet").relative_to(ROOT)),
                "valid_raw": str((VIEW_ROOT / "positions_raw_valid.npy").relative_to(ROOT)),
                "valid_centered": str((VIEW_ROOT / "positions_centered_valid.npy").relative_to(ROOT)),
                "valid_team_slots": str((VIEW_ROOT / "team_slots_valid.npy").relative_to(ROOT)),
                "valid_snapshot_index": str((VIEW_ROOT / "snapshot_index_valid.parquet").relative_to(ROOT)),
                "valid_pairs": str((VIEW_ROOT / "natural_pair_ranking_valid.parquet").relative_to(ROOT)),
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
        all_records: list[dict[str, Any]] = []
        for variant, spec in VARIANTS.items():
            all_records.extend(
                run_variant(
                    variant,
                    spec,
                    train,
                    valid,
                    train_adjacency,
                    valid_adjacency,
                    temporary_output,
                    args.epochs,
                )
            )
        analytic = {"variant": "raw_coordinate_procrustes", "valid": analytic_pair_metrics(valid), "train": analytic_pair_metrics(train)}
        write_json(temporary_output / "model_results.json", all_records)
        write_json(temporary_output / "analytic_control.json", analytic)
        summary = aggregate_results(all_records)
        write_json(temporary_output / "summary.json", summary)
        manifest = {
            "status": "T5R3_FIXED_DUAL_CHANNEL_SANITY_COMPLETE",
            "phase": "P3_CAUSAL_MECHANISM",
            "task_lane": "P3-T5R3",
            "protocol": "protocol.json",
            "results": "model_results.json",
            "summary": "summary.json",
            "analytic_control": "analytic_control.json",
            "model_count": len(all_records),
            "variant_count": len(VARIANTS),
            "seed_count": len(SEEDS),
            "reserved_holdout_loaded": False,
            "external_results_used": False,
            "old_heldout_used": False,
            "candidate_selection_performed": False,
            "status_boundary": "T5R3 complete; T5R4 bounded autoresearch not started",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        write_json(temporary_output / "manifest.json", manifest)
        protocol["status"] = manifest["status"]
        protocol["manifest_sha256"] = json_sha256(manifest)
        write_json(temporary_output / "protocol.json", protocol)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output.replace(output)
        print(json.dumps({"status": manifest["status"], "output": str(output), "summary": summary}, ensure_ascii=False), flush=True)
        return 0
    except Exception:
        print(f"T5R3 run failed; incomplete staging kept at {temporary_output}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
