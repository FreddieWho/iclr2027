#!/usr/bin/env python3
"""P3 support-conditioned local geometry.

The module deliberately keeps the prediction surface small: the frozen P1
encoder is restored through the existing P2 adapter, and geometry is computed
from normalized layer outputs with autograd.  It does not train a predictor,
modify a checkpoint, or use response values to generate interventions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = [
    f"{family}_seed{seed}"
    for family in ("deepsets_ae", "gat_ae", "phase_gat")
    for seed in (11, 23, 47)
]
P2_RUN = ROOT / "artifacts" / "phase2" / "p2_fracture_continuity_v1"
DEFAULT_CONFIG = ROOT / "configs" / "phase3_support_geometry_v1.yaml"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def relative_path(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def write_json(path: Path, value: Any, *, allow_existing: bool = False) -> None:
    if path.exists() and not allow_existing:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def output_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(root, path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def write_sha256sums(root: Path, directory: Path, filename: str = "SHA256SUMS") -> Path:
    path = directory / filename
    records = []
    for child in sorted(directory.rglob("*")):
        if child.is_file() and child.name != filename:
            records.append(f"{file_sha256(child)}  {child.relative_to(directory)}")
    path.write_text("\n".join(records) + "\n", encoding="utf-8")
    return path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("P3 config must be a mapping")
    if config.get("phase") != "P3_CAUSAL_MECHANISM":
        raise ValueError("P3 config phase mismatch")
    if config.get("provenance", {}).get("git_head_at_document_migration") != "f44be1e9526796a76b58ca2bd44dc562ab7841a3":
        raise ValueError("unexpected frozen HEAD in P3 config")
    source_lock = config.get("provenance", {}).get("source_locks", {}).get("p3_implementation", {})
    implementation = root / str(source_lock.get("path", ""))
    expected = str(source_lock.get("sha256", ""))
    if not implementation.exists() or expected in {"", "NOT_CREATED_AT_DOCUMENT_MIGRATION"}:
        raise ValueError("P3 implementation source lock is not populated")
    observed = file_sha256(implementation)
    if observed != expected:
        raise ValueError(f"P3 implementation SHA256 mismatch: expected {expected}, observed {observed}")
    return config


def load_arrays(root: Path) -> dict[str, np.ndarray]:
    path = root / "artifacts" / "phase1" / "canonical_samples.npz"
    with np.load(path) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _team_pool(h, team_slots):
    import torch

    pools = []
    for team_slot in (0, 1):
        mask = (team_slots == team_slot).float().unsqueeze(-1)
        pools.append((h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0))
    return torch.cat(pools, dim=-1)


def _relational_pool(h, team_slots):
    import torch

    pools = []
    for team_slot in (0, 1):
        mask = (team_slots == team_slot).float().unsqueeze(-1)
        count = mask.sum(dim=1)
        summed = (h * mask).sum(dim=1)
        mean = summed / count.clamp_min(1.0)
        pair_sum = 0.5 * (summed.square() - (h.square() * mask).sum(dim=1))
        pair_count = 0.5 * count * (count - 1.0)
        pair_mean = pair_sum / pair_count.clamp_min(1.0)
        pools.append(torch.where(pair_count > 0.0, pair_mean, mean))
    return torch.cat(pools, dim=-1)


@dataclass
class FrozenGeometryModel:
    model_id: str
    architecture: str
    model: Any
    uses_adjacency: bool
    pooling_mode: str = "team_mean"

    @property
    def available_layers(self) -> tuple[str, ...]:
        if self.architecture == "DeepSets-AE":
            return ("input_node", "pooling_pre", "pooled_embedding")
        return (
            "input_node",
            "message_passing_1",
            "message_passing_2",
            "pooling_pre",
            "pooled_embedding",
        )

    def layer_values(self, positions, team_slots, adjacency=None) -> dict[str, Any]:
        import torch

        if positions.ndim == 2:
            positions = positions.unsqueeze(0)
        if team_slots.ndim == 1:
            team_slots = team_slots.unsqueeze(0)
        features = torch.cat(
            [positions, torch.nn.functional.one_hot(team_slots.long(), 2).float()],
            dim=-1,
        )
        if self.architecture == "DeepSets-AE":
            node = self.model.node(features)
            pooled_pre = _team_pool(node, team_slots)
            pooled = self.model.project(pooled_pre)
            return {
                "input_node": node,
                "pooling_pre": pooled_pre,
                "pooled_embedding": pooled,
            }
        if adjacency is None:
            raise ValueError(f"{self.model_id} requires the frozen baseline adjacency")
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0)
        encoder = self.model.encoder
        h = torch.relu(encoder.in_proj(features))
        outputs: dict[str, Any] = {"input_node": h}
        for index, layer in enumerate(encoder.layers, start=1):
            h = layer(h, adjacency)
            outputs[f"message_passing_{index}"] = h
        pooled_pre = _team_pool(h, team_slots) if self.pooling_mode == "team_mean" else _relational_pool(h, team_slots)
        outputs["pooling_pre"] = pooled_pre
        outputs["pooled_embedding"] = encoder.project(pooled_pre)
        return outputs

    def encode_numpy(
        self,
        positions: np.ndarray,
        team_slots: np.ndarray,
        adjacency: np.ndarray | None,
    ) -> np.ndarray:
        import torch

        coordinates = np.asarray(positions, dtype=np.float32)
        teams = np.asarray(team_slots, dtype=np.int64)
        graphs = None if adjacency is None else np.asarray(adjacency, dtype=np.float32)
        with torch.inference_mode():
            pos = torch.from_numpy(coordinates)
            team = torch.from_numpy(teams)
            graph = None if graphs is None else torch.from_numpy(graphs)
            return self.layer_values(pos, team, graph)["pooled_embedding"].detach().cpu().numpy()


def load_frozen_model(root: Path, model_id: str) -> FrozenGeometryModel:
    adapter_path = root / "scripts" / "p2_point_model_adapter.py"
    adapter_module = _load_module(adapter_path, f"p2_adapter_for_p3_{file_sha256(adapter_path)[:12]}")
    adapter, receipt = adapter_module.load_point_adapter(root, model_id, batch_size=64)
    return FrozenGeometryModel(
        model_id=model_id,
        architecture=str(receipt["architecture"]),
        model=adapter.model,
        uses_adjacency=bool(adapter.uses_adjacency),
    )


def load_baseline_expected(root: Path, model_id: str) -> np.ndarray:
    manifest_path = root / "artifacts" / "phase1" / "point_mainline" / "embedding_manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    rows = manifest[
        (manifest["model_id"] == model_id)
        & (manifest["embedding_kind"] == "baseline")
    ].sort_values("sample_index")
    if len(rows) != 250 or rows["path"].nunique() != 1:
        raise ValueError(f"unexpected baseline manifest rows for {model_id}: {len(rows)}")
    path = root / str(rows.iloc[0]["path"])
    expected = np.asarray(np.load(path), dtype=np.float32)
    if expected.shape != (250, 128):
        raise ValueError(f"unexpected baseline embedding shape for {model_id}: {expected.shape}")
    return expected


def normalized_flat(values) -> Any:
    import torch

    flat = values.reshape(-1)
    norm = torch.linalg.vector_norm(flat)
    return flat / norm.clamp_min(torch.finfo(flat.dtype).eps)


def layer_function(
    frozen: FrozenGeometryModel,
    teams: Any,
    adjacency: Any,
    layer: str,
):
    def function(positions):
        values = frozen.layer_values(
            positions.unsqueeze(0),
            teams.unsqueeze(0),
            None if adjacency is None else adjacency.unsqueeze(0),
        )[layer][0]
        return normalized_flat(values)

    return function


def jacobian_for_layer(
    frozen: FrozenGeometryModel,
    positions: np.ndarray,
    team_slots: np.ndarray,
    adjacency: np.ndarray | None,
    layer: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    x = torch.as_tensor(np.asarray(positions, dtype=np.float32)).clone().requires_grad_(True)
    teams = torch.as_tensor(np.asarray(team_slots, dtype=np.int64))
    graph = None if adjacency is None else torch.as_tensor(np.asarray(adjacency, dtype=np.float32))
    fn = layer_function(frozen, teams, graph, layer)
    with torch.enable_grad():
        output = fn(x)
        jacobian = torch.autograd.functional.jacobian(fn, x, create_graph=False, vectorize=True)
    return (
        output.detach().cpu().numpy().astype(np.float32),
        jacobian.detach().cpu().numpy().reshape(output.numel(), -1).astype(np.float32),
    )


def jvp_for_layer(
    frozen: FrozenGeometryModel,
    positions: np.ndarray,
    team_slots: np.ndarray,
    adjacency: np.ndarray | None,
    layer: str,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    x = torch.as_tensor(np.asarray(positions, dtype=np.float32)).clone().requires_grad_(True)
    tangent = torch.as_tensor(np.asarray(direction, dtype=np.float32))
    teams = torch.as_tensor(np.asarray(team_slots, dtype=np.int64))
    graph = None if adjacency is None else torch.as_tensor(np.asarray(adjacency, dtype=np.float32))
    fn = layer_function(frozen, teams, graph, layer)
    with torch.enable_grad():
        value, derivative = torch.autograd.functional.jvp(fn, x, tangent, create_graph=False)
    return value.detach().cpu().numpy(), derivative.detach().cpu().numpy()


def normalized_layer_numpy(
    frozen: FrozenGeometryModel,
    positions: np.ndarray,
    team_slots: np.ndarray,
    adjacency: np.ndarray | None,
    layer: str,
) -> np.ndarray:
    import torch

    with torch.no_grad():
        values = frozen.layer_values(
            torch.as_tensor(np.asarray(positions, dtype=np.float32)).unsqueeze(0),
            torch.as_tensor(np.asarray(team_slots, dtype=np.int64)).unsqueeze(0),
            None if adjacency is None else torch.as_tensor(np.asarray(adjacency, dtype=np.float32)).unsqueeze(0),
        )[layer][0]
        return normalized_flat(values).cpu().numpy()


def finite_difference(
    frozen: FrozenGeometryModel,
    positions: np.ndarray,
    team_slots: np.ndarray,
    adjacency: np.ndarray | None,
    layer: str,
    direction: np.ndarray,
    step: float,
) -> np.ndarray:
    plus = normalized_layer_numpy(frozen, positions + step * direction, team_slots, adjacency, layer)
    minus = normalized_layer_numpy(frozen, positions - step * direction, team_slots, adjacency, layer)
    return (plus - minus) / (2.0 * step)


def block_decomposition(
    jacobian: np.ndarray,
    delta: np.ndarray,
    team_slots: np.ndarray,
    adjacency: np.ndarray,
    support_indices: Iterable[int] = (),
) -> dict[str, float]:
    """Return quadratic block contributions with explicit additive identities."""
    n_nodes = int(delta.shape[0])
    blocks = np.asarray(jacobian, dtype=np.float64).reshape(-1, n_nodes, 2)
    displacement = np.asarray(delta, dtype=np.float64)
    node_vectors = np.einsum("dni,ni->nd", blocks, displacement)
    full = 0.5 * float(np.sum(np.sum(node_vectors, axis=0) ** 2))
    diagonal = 0.5 * float(np.sum(node_vectors**2))
    team_within = 0.0
    team_between = 0.0
    graph_edge = 0.0
    graph_non_edge = 0.0
    for left in range(n_nodes):
        for right in range(left + 1, n_nodes):
            pair = float(np.dot(node_vectors[left], node_vectors[right]))
            if int(team_slots[left]) == int(team_slots[right]):
                team_within += pair
            else:
                team_between += pair
            if float(adjacency[left, right]) > 0.0 or float(adjacency[right, left]) > 0.0:
                graph_edge += pair
            else:
                graph_non_edge += pair
    support = {int(index) for index in support_indices if 0 <= int(index) < n_nodes}
    inside = 0.0
    for left in range(n_nodes):
        if left not in support:
            continue
        inside += 0.5 * float(np.dot(node_vectors[left], node_vectors[left]))
        for right in range(left + 1, n_nodes):
            if right in support:
                inside += float(np.dot(node_vectors[left], node_vectors[right]))
    off_diagonal = full - diagonal
    return {
        "q_full": full,
        "q_diagonal": diagonal,
        "q_off_diagonal": off_diagonal,
        "q_team_within": team_within,
        "q_team_between": team_between,
        "q_support_inside": inside,
        "q_support_outside": full - inside,
        "q_graph_edge": graph_edge,
        "q_graph_non_edge": graph_non_edge,
        "identity_full_minus_diag_off": full - diagonal - off_diagonal,
        "identity_off_minus_team": off_diagonal - team_within - team_between,
        "identity_off_minus_graph": off_diagonal - graph_edge - graph_non_edge,
        "identity_full_minus_support": full - inside - (full - inside),
    }


def parse_json_array(value: Any, *, default: Sequence[int] = ()) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.asarray(default, dtype=int)
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return np.asarray(parsed, dtype=int)


def parse_delta(value: Any) -> np.ndarray | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    parsed = json.loads(value) if isinstance(value, str) else value
    delta = np.asarray(parsed, dtype=np.float32)
    if delta.shape != (20, 2) or not np.isfinite(delta).all():
        raise ValueError(f"invalid P2 delta shape/value: {delta.shape}")
    return delta


def parse_band_power(value: Any) -> list[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return [float("nan")] * 6
    parsed = json.loads(value) if isinstance(value, str) else value
    values = np.asarray(parsed, dtype=float).reshape(-1)
    if len(values) != 6:
        raise ValueError(f"expected six band powers, found {len(values)}")
    return [float(item) for item in values]


def response_shards(root: Path, model_id: str) -> list[Path]:
    manifest_path = P2_RUN / "response" / "response_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [
        root / str(shard["response_path"])
        for shard in manifest["shards"]
        if str(shard.get("model_id")) == model_id
    ]
    if len(paths) != 10:
        raise ValueError(f"expected ten P2 response shards for {model_id}, found {len(paths)}")
    return sorted(paths)


def read_response_rows(root: Path, model_id: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    frames = []
    for path in response_shards(root, model_id):
        # Read a single file explicitly; the parent directory contains hive
        # partition fields that can conflict with the file schema.
        frames.append(pq.ParquetFile(path).read().to_pandas())
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) == 0:
        raise ValueError(f"empty P2 response for {model_id}")
    return frame


def geometry_arm_rows(
    root: Path,
    frozen: FrozenGeometryModel,
    arrays: Mapping[str, np.ndarray],
    response: pd.DataFrame,
    layers: Sequence[str],
) -> pd.DataFrame:
    positions = np.asarray(arrays["positions"], dtype=np.float32)
    teams = np.asarray(arrays["team_slots"], dtype=np.int64)
    adjacency = np.asarray(arrays["adjacency"], dtype=np.float32)
    available = set(frozen.available_layers)
    missing = [layer for layer in layers if layer not in available]
    if missing:
        raise ValueError(f"{frozen.model_id} does not expose layers {missing}")
    records: list[dict[str, Any]] = []
    for sample_index, group in response.groupby("sample_index", sort=True):
        sample_index = int(sample_index)
        jacobians: dict[str, np.ndarray] = {}
        for layer in layers:
            _, jacobians[layer] = jacobian_for_layer(
                frozen,
                positions[sample_index],
                teams[sample_index],
                None if not frozen.uses_adjacency else adjacency[sample_index],
                layer,
            )
        for row in group.to_dict(orient="records"):
            if not bool(row.get("valid", False)):
                continue
            delta = parse_delta(row.get("delta"))
            if delta is None:
                continue
            support = parse_json_array(row.get("source_support_indices"))
            output = {
                "model_id": frozen.model_id,
                "sample_index": sample_index,
                "matched_set_id": str(row["matched_set_id"]),
                "intervention_id": str(row["intervention_id"]),
                "source_match_id": str(row["source_match_id"]),
                "source_sample_id": str(row["source_sample_id"]),
                "split": str(row["split"]),
                "graph_id": str(row["graph_id"]),
                "coalition_id": str(row["coalition_id"]),
                "team_slot": int(row["team_slot"]),
                "role_group": str(row["role_group"]),
                "epsilon": float(row["epsilon"]),
                "fracture_draw": int(row["fracture_draw"]),
                "arm_kind": str(row["arm_kind"]),
                "control_family": str(row["control_family"]),
                "matched_set_status": str(row["matched_set_status"]),
                "response": float(row["normalized_response"]),
                "raw_response": float(row["response_distance"]),
                "rayleigh_quotient_normalized": float(row["rayleigh_quotient_normalized"]),
                "support_size": int(row["support_size"]),
                "frequency_residual": float(row["frequency_residual"]),
                "topology_residual": float(row["topology_residual"]),
                "source_support_indices": json.dumps(support.tolist(), separators=(",", ":")),
                "target_support_indices": str(row.get("target_support_indices")),
                "delta_norm": float(np.linalg.norm(delta)),
            }
            output.update({f"band_power_{i}": value for i, value in enumerate(parse_band_power(row.get("band_power")))})
            for layer in layers:
                blocks = block_decomposition(
                    jacobians[layer],
                    delta,
                    teams[sample_index],
                    adjacency[sample_index],
                    support,
                )
                prefix = "" if layer == "pooled_embedding" else f"{layer}__"
                output.update({f"{prefix}{name}": value for name, value in blocks.items()})
            records.append(output)
    return pd.DataFrame.from_records(records)


def pair_rows_from_arms(arms: pd.DataFrame) -> pd.DataFrame:
    if arms.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    q_columns = [
        "q_full",
        "q_diagonal",
        "q_off_diagonal",
        "q_team_within",
        "q_team_between",
        "q_support_inside",
        "q_support_outside",
        "q_graph_edge",
        "q_graph_non_edge",
    ]
    for matched_set_id, group in arms.groupby("matched_set_id", sort=True):
        if str(group["matched_set_status"].iloc[0]) != "COMPLETE":
            continue
        anchors = group[group["arm_kind"] == "anchor"]
        controls = group[group["arm_kind"] == "control"]
        if len(anchors) != 1 or len(controls) != 1:
            continue
        anchor = anchors.iloc[0]
        control = controls.iloc[0]
        record: dict[str, Any] = {
            "model_id": str(anchor["model_id"]),
            "matched_set_id": str(matched_set_id),
            "source_match_id": str(anchor["source_match_id"]),
            "source_sample_id": str(anchor["source_sample_id"]),
            "sample_index": int(anchor["sample_index"]),
            "split": str(anchor["split"]),
            "graph_id": str(anchor["graph_id"]),
            "coalition_id": str(anchor["coalition_id"]),
            "team_slot": int(anchor["team_slot"]),
            "role_group": str(anchor["role_group"]),
            "epsilon": float(anchor["epsilon"]),
            "fracture_draw": int(anchor["fracture_draw"]),
            "support_size": int(anchor["support_size"]),
            "rayleigh_quotient_normalized": float(anchor["rayleigh_quotient_normalized"]),
            "frequency_residual": float(control["frequency_residual"]),
            "topology_residual": float(control["topology_residual"]),
            "observed_pair_effect": float(anchor["response"] - control["response"]),
            "observed_raw_pair_effect": float(anchor["raw_response"] - control["raw_response"]),
        }
        for i in range(6):
            record[f"band_power_{i}"] = float(anchor[f"band_power_{i}"])
        for column in q_columns:
            if column in anchor.index:
                record[column.replace("q_", "delta_q_")] = float(anchor[column] - control[column])
        for column in anchor.index:
            if "__q_" in str(column):
                record[str(column).replace("__q_", "__delta_q_")] = float(anchor[column] - control[column])
        records.append(record)
    return pd.DataFrame.from_records(records)


def _numeric_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    reference: pd.DataFrame | None = None,
) -> np.ndarray:
    values = frame.loc[:, list(columns)].astype(float).to_numpy()
    reference_values = values if reference is None else reference.loc[:, list(columns)].astype(float).to_numpy()
    if not np.isfinite(values).all():
        medians = np.nanmedian(np.where(np.isfinite(reference_values), reference_values, np.nan), axis=0)
        values = np.where(np.isfinite(values), values, medians)
        values = np.nan_to_num(values, nan=0.0)
    return values


def feature_matrix(
    frame: pd.DataFrame,
    family: str,
    train: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    numeric: list[str]
    categorical: list[str] = []
    if family == "spectral":
        numeric = [f"band_power_{i}" for i in range(6)]
    elif family == "rayleigh":
        numeric = ["rayleigh_quotient_normalized"]
    elif family == "support":
        numeric = ["support_size", "epsilon"]
    elif family == "residual":
        numeric = ["frequency_residual", "topology_residual"]
    elif family == "metadata":
        numeric = ["support_size", "epsilon", "rayleigh_quotient_normalized"]
        categorical = ["graph_id", "role_group"]
    elif family == "baseline":
        numeric = [
            *[f"band_power_{i}" for i in range(6)],
            "rayleigh_quotient_normalized",
            "support_size",
            "epsilon",
            "frequency_residual",
            "topology_residual",
        ]
        categorical = ["graph_id", "role_group"]
    elif family in {"diagonal_geometry", "full_geometry", "off_diagonal_geometry"}:
        numeric = [{"diagonal_geometry": "delta_q_diagonal", "full_geometry": "delta_q_full", "off_diagonal_geometry": "delta_q_off_diagonal"}[family]]
    elif family == "baseline_plus_full_geometry":
        numeric = [
            *[f"band_power_{i}" for i in range(6)],
            "rayleigh_quotient_normalized",
            "support_size",
            "epsilon",
            "frequency_residual",
            "topology_residual",
            "delta_q_full",
        ]
        categorical = ["graph_id", "role_group"]
    elif family == "baseline_plus_diagonal_geometry":
        numeric = [
            *[f"band_power_{i}" for i in range(6)],
            "rayleigh_quotient_normalized",
            "support_size",
            "epsilon",
            "frequency_residual",
            "topology_residual",
            "delta_q_diagonal",
        ]
        categorical = ["graph_id", "role_group"]
    elif family == "baseline_plus_off_diagonal_geometry":
        numeric = [
            *[f"band_power_{i}" for i in range(6)],
            "rayleigh_quotient_normalized",
            "support_size",
            "epsilon",
            "frequency_residual",
            "topology_residual",
            "delta_q_off_diagonal",
        ]
        categorical = ["graph_id", "role_group"]
    else:
        raise ValueError(f"unknown predictor family {family}")
    reference = frame if train is None else train
    parts = [_numeric_features(frame, numeric, reference)]
    for column in categorical:
        levels = sorted(str(value) for value in reference[column].dropna().unique())
        parts.append(np.asarray([[float(str(value) == level) for level in levels] for value in frame[column]], dtype=float))
    values = np.concatenate(parts, axis=1) if parts else np.empty((len(frame), 0))
    if train is None:
        train_values = values
    else:
        train_values = feature_matrix(train, family, train=None)[0]
    return values, train_values


def grouped_cv_predictions(
    frame: pd.DataFrame,
    family: str,
    target: str = "observed_pair_effect",
) -> pd.DataFrame:
    predictions = []
    for held_out in sorted(frame["source_match_id"].astype(str).unique()):
        test = frame[frame["source_match_id"].astype(str) == held_out].copy()
        train = frame[frame["source_match_id"].astype(str) != held_out].copy()
        if train.empty or test.empty:
            continue
        x_train_raw, _ = feature_matrix(train, family)
        x_test_raw, _ = feature_matrix(test, family)
        center = np.nanmedian(x_train_raw, axis=0)
        scale = np.nanstd(x_train_raw, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        x_train = (np.nan_to_num(x_train_raw, nan=0.0) - center) / scale
        x_test = (np.nan_to_num(x_test_raw, nan=0.0) - center) / scale
        x_train = np.column_stack([np.ones(len(x_train)), x_train])
        x_test = np.column_stack([np.ones(len(x_test)), x_test])
        coefficients, *_ = np.linalg.lstsq(x_train, train[target].to_numpy(float), rcond=None)
        test["prediction"] = x_test @ coefficients
        test["predictor_family"] = family
        predictions.append(test[["model_id", "matched_set_id", "source_match_id", target, "prediction", "predictor_family"]])
    if not predictions:
        return pd.DataFrame(columns=["model_id", "matched_set_id", "source_match_id", target, "prediction", "predictor_family"])
    return pd.concat(predictions, ignore_index=True)


def metric_values(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    from scipy.stats import spearmanr

    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    finite = np.isfinite(target) & np.isfinite(prediction)
    target = target[finite]
    prediction = prediction[finite]
    if len(target) == 0:
        return {"n_rows": 0, "spearman": float("nan"), "mae": float("nan"), "r2": float("nan"), "direction_accuracy": float("nan"), "calibration_slope": float("nan"), "calibration_intercept": float("nan")}
    centered = target - np.mean(target)
    residual = target - prediction
    variance = float(np.var(prediction))
    slope = float(np.dot(prediction - np.mean(prediction), centered) / (np.sum((prediction - np.mean(prediction)) ** 2) + 1e-30))
    intercept = float(np.mean(target) - slope * np.mean(prediction))
    rho = float(spearmanr(target, prediction).statistic) if len(np.unique(target)) > 1 and len(np.unique(prediction)) > 1 else float("nan")
    return {
        "n_rows": int(len(target)),
        "spearman": rho,
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual**2) / (np.sum(centered**2) + 1e-30)),
        "direction_accuracy": float(np.mean(np.sign(target) == np.sign(prediction))),
        "calibration_slope": slope if variance > 1e-30 else float("nan"),
        "calibration_intercept": intercept,
    }


def bootstrap_match_metric_delta(
    predictions: pd.DataFrame,
    full_family: str,
    baseline_family: str,
    metric: str,
    seed: int = 20260901,
    n_bootstrap: int = 500,
) -> tuple[float, float]:
    full = predictions[predictions["predictor_family"] == full_family]
    baseline = predictions[predictions["predictor_family"] == baseline_family]
    # Collapse rows to source-match summaries before resampling. This keeps
    # frame/pair/draw rows from becoming pseudo-replicates in the bootstrap.
    full = full.groupby("source_match_id", as_index=False).agg(
        observed_pair_effect=("observed_pair_effect", "mean"),
        prediction=("prediction", "mean"),
    )
    baseline = baseline.groupby("source_match_id", as_index=False).agg(
        observed_pair_effect=("observed_pair_effect", "mean"),
        prediction=("prediction", "mean"),
    )
    full_by_match = {str(row["source_match_id"]): row for row in full.to_dict(orient="records")}
    baseline_by_match = {str(row["source_match_id"]): row for row in baseline.to_dict(orient="records")}
    common = sorted(set(full_by_match) & set(baseline_by_match))
    if not common:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(common, size=len(common), replace=True)
        f_target = np.asarray([full_by_match[str(match)]["observed_pair_effect"] for match in sampled])
        f_prediction = np.asarray([full_by_match[str(match)]["prediction"] for match in sampled])
        b_target = np.asarray([baseline_by_match[str(match)]["observed_pair_effect"] for match in sampled])
        b_prediction = np.asarray([baseline_by_match[str(match)]["prediction"] for match in sampled])
        f_metric = metric_values(f_target, f_prediction)[metric]
        b_metric = metric_values(b_target, b_prediction)[metric]
        if np.isfinite(f_metric) and np.isfinite(b_metric):
            deltas.append(float(f_metric - b_metric))
    if not deltas:
        return float("nan"), float("nan")
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def finalize_retrospective(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Finish a geometry run whose large parquet tables already exist."""
    started = time.perf_counter()
    output_dir = root / "artifacts" / "phase3" / "support_geometry_v1"
    pair_frame = pd.read_parquet(output_dir / "pair_geometry.parquet")
    prediction_frame = pd.read_parquet(output_dir / "pair_predictions.parquet")
    families = [
        "spectral",
        "rayleigh",
        "support",
        "residual",
        "metadata",
        "baseline",
        "diagonal_geometry",
        "full_geometry",
        "off_diagonal_geometry",
        "baseline_plus_full_geometry",
        "baseline_plus_diagonal_geometry",
        "baseline_plus_off_diagonal_geometry",
    ]
    metric_records: list[dict[str, Any]] = []
    for model_id, model_predictions in prediction_frame.groupby("model_id", sort=True):
        model_frame = pair_frame[pair_frame["model_id"] == model_id]
        for family in families:
            predictions = model_predictions[model_predictions["predictor_family"] == family]
            metrics = metric_values(predictions["observed_pair_effect"], predictions["prediction"])
            metric_records.append({
                "target": "observed_pair_effect",
                "scope": "model_id",
                "model_id": str(model_id),
                "architecture": str(model_frame["model_id"].iloc[0]).split("_seed")[0],
                "predictor_family": family,
                "grouping_key": "source_match_id",
                "n_matches": int(predictions["source_match_id"].nunique()),
                **metrics,
            })
    metrics_frame = pd.DataFrame.from_records(metric_records)
    metrics_path = output_dir / "prediction_metrics.parquet"
    if not metrics_path.exists():
        write_parquet(metrics_path, metrics_frame)
    baseline_family = "baseline"
    bootstrap_records: list[dict[str, Any]] = []
    for model_id, model_predictions in prediction_frame.groupby("model_id", sort=True):
        for family in ("full_geometry", "diagonal_geometry", "off_diagonal_geometry", "baseline_plus_full_geometry"):
            for metric in ("spearman", "mae", "r2", "direction_accuracy"):
                low, high = bootstrap_match_metric_delta(
                    model_predictions,
                    family,
                    baseline_family,
                    metric,
                    seed=20260901,
                )
                bootstrap_records.append({
                    "model_id": str(model_id),
                    "geometry_family": family,
                    "baseline_family": baseline_family,
                    "metric": metric,
                    "bootstrap_unit": "source_match_id_mean",
                    "n_bootstrap": 500,
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                })
    bootstrap_path = output_dir / "match_bootstrap_comparisons.parquet"
    if not bootstrap_path.exists():
        write_parquet(bootstrap_path, pd.DataFrame.from_records(bootstrap_records))
    summary = {
        "status": "T1_RETROSPECTIVE_COMPLETE",
        "task": "P3-T1",
        "config_sha256": file_sha256(config_path),
        "geometry_layer": "pooled_embedding",
        "n_arm_rows": int(len(pd.read_parquet(output_dir / "arm_geometry.parquet"))),
        "n_pair_rows": int(len(pair_frame)),
        "n_models": int(pair_frame["model_id"].nunique()),
        "n_matches": int(pair_frame["source_match_id"].nunique()),
        "grouping_key": "source_match_id",
        "predictor_families": families,
        "metrics_path": relative_path(root, metrics_path),
        "bootstrap_path": relative_path(root, bootstrap_path),
        "bootstrap_unit": "source_match_id_mean",
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
        "interpretation_status": "NOT_YET_ROUTED",
    }
    summary_path = output_dir / "retrospective_summary.json"
    if not summary_path.exists():
        write_json(summary_path, summary)
    receipt_path = output_dir / "retrospective_receipt.json"
    if not receipt_path.exists():
        write_json(receipt_path, {
            "status": summary["status"],
            "checkpoint": "P3-T1",
            "run_id": config["run_id"],
            "summary_sha256": file_sha256(summary_path),
            "input_p2_response_manifest_sha256": file_sha256(P2_RUN / "response" / "response_manifest.json"),
            "input_p2_matching_manifest_sha256": file_sha256(P2_RUN / "matching" / "matching_manifest.json"),
        })
    sums_path = output_dir / "retrospective_SHA256SUMS"
    if not sums_path.exists():
        write_sha256sums(root, output_dir, "retrospective_SHA256SUMS")
    return summary


def predictor_fit(frame: pd.DataFrame, family: str, target: str = "observed_pair_effect") -> dict[str, Any]:
    raw, _ = feature_matrix(frame, family)
    center = np.nanmedian(raw, axis=0)
    scale = np.nanstd(raw, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    design = np.column_stack([np.ones(len(raw)), (np.nan_to_num(raw, nan=0.0) - center) / scale])
    coefficients, *_ = np.linalg.lstsq(design, frame[target].to_numpy(float), rcond=None)
    return {
        "family": family,
        "target": target,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "n_rows": int(len(frame)),
        "n_matches": int(frame["source_match_id"].nunique()),
    }


def predictor_apply(
    frame: pd.DataFrame,
    family: str,
    fit: Mapping[str, Any],
    reference: pd.DataFrame,
) -> np.ndarray:
    raw, _ = feature_matrix(frame, family, train=reference)
    center = np.asarray(fit["center"], dtype=float)
    scale = np.asarray(fit["scale"], dtype=float)
    coefficients = np.asarray(fit["coefficients"], dtype=float)
    design = np.column_stack([np.ones(len(raw)), (np.nan_to_num(raw, nan=0.0) - center) / scale])
    if design.shape[1] != len(coefficients):
        raise ValueError(f"fixed predictor shape changed for {family}: {design.shape[1]} vs {len(coefficients)}")
    return design @ coefficients


def freeze_predictors(root: Path, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    output_dir = root / "artifacts" / "phase3" / "support_geometry_v1"
    pair_path = output_dir / "pair_geometry.parquet"
    if not pair_path.exists():
        raise FileNotFoundError("T1 pair geometry is required before freezing predictors")
    lock_path = output_dir / "t1_prediction_lock.json"
    if lock_path.exists():
        raise FileExistsError(f"refusing to overwrite {lock_path}")
    frame = pd.read_parquet(pair_path)
    families = ["baseline", "full_geometry", "diagonal_geometry", "off_diagonal_geometry"]
    fits: dict[str, dict[str, Any]] = {}
    for model_id, model_frame in frame.groupby("model_id", sort=True):
        fits[str(model_id)] = {
            family: predictor_fit(model_frame, family)
            for family in families
        }
    payload = {
        "status": "T1_PREDICTORS_FROZEN_BEFORE_PROSPECTIVE",
        "task": "P3-T1-to-T2-freeze",
        "config_sha256": file_sha256(config_path),
        "retrospective_pair_geometry_sha256": file_sha256(pair_path),
        "target": "observed_pair_effect",
        "layer": "pooled_embedding",
        "metrics": ["spearman", "mae", "r2", "direction_accuracy", "calibration"],
        "grouping_key": "source_match_id",
        "families": families,
        "fits": fits,
        "response_blind_prospective_required": True,
        "no_layer_or_threshold_selection_after_prospective_response": True,
    }
    write_json(lock_path, payload)
    return payload


def load_p2_fracture_module(root: Path):
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / "run_phase2_fracture_formal.py"
    return _load_module(path, f"p2_fracture_for_p3_{file_sha256(path)[:12]}")


def run_prospective_generate(root: Path, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Generate new legal reallocations without reading any response artifact."""
    started = time.perf_counter()
    lock_path = root / "artifacts" / "phase3" / "support_geometry_v1" / "t1_prediction_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("freeze T1 predictors before prospective generation")
    output_root = root / "artifacts" / "phase3"
    final_dir = output_root / "support_geometry_prospective_v1"
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite {final_dir}")
    temporary_dir = Path(__import__("tempfile").mkdtemp(prefix=".support-geometry-prospective-v1-", dir=str(output_root)))
    p2_config_path = root / "configs" / "phase2_fracture_continuity_v1.yaml"
    p2_config = yaml.safe_load(p2_config_path.read_text(encoding="utf-8"))
    p2_config["run_id"] = "p3_support_geometry_prospective_v1"
    p2_config.setdefault("fracture", {})["seed_root"] = str(config["t2_prospective"]["new_intervention_seed"])
    p2_config["fracture"]["draws"] = list(config["t2_prospective"].get("draws", [config["t2_prospective"]["new_intervention_seed"]]))
    p2_module = load_p2_fracture_module(root)
    samples = p2_module.full_samples(root)
    by_match: dict[str, list[Any]] = {}
    for sample in samples:
        by_match.setdefault(str(sample.match_id), []).append(sample)
    shard_records: list[dict[str, Any]] = []
    total_sets = total_arms = complete_sets = valid_arms = 0
    for match_id in sorted(by_match):
        arms: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for sample in sorted(by_match[match_id], key=lambda item: (int(item.frame), str(item.sample_id))):
            views, _ = p2_module.graph_views(sample, p2_config["fracture"]["graphs"])
            for graph in views:
                for draw in p2_config["fracture"]["draws"]:
                    for epsilon in p2_config["fracture"]["energies"]:
                        generated, diagnostic_payload = p2_module.build_sets_for_sample(
                            sample, graph.graph_id, graph, p2_config, int(draw), float(epsilon)
                        )
                        arms.extend(generated)
                        diagnostics.extend(diagnostic_payload["diagnostics"])
        arms_path = temporary_dir / "matching" / "shards" / f"match_id={match_id}" / "arms.parquet"
        diagnostics_path = temporary_dir / "matching" / "shards" / f"match_id={match_id}" / "diagnostics.parquet"
        write_parquet(arms_path, pd.DataFrame.from_records(arms))
        write_parquet(diagnostics_path, pd.DataFrame.from_records(diagnostics))
        n_sets = len(diagnostics)
        n_complete = sum(bool(row.get("complete", False)) for row in diagnostics)
        n_valid = sum(bool(row.get("valid", False)) for row in arms)
        total_sets += n_sets
        complete_sets += n_complete
        total_arms += len(arms)
        valid_arms += n_valid
        shard_records.append({
            "match_id": str(match_id),
            "arms_path": relative_path(root, root / relative_path(temporary_dir, arms_path)),
            "diagnostics_path": relative_path(root, root / relative_path(temporary_dir, diagnostics_path)),
            "n_sets": n_sets,
            "n_complete_sets": n_complete,
            "n_arms": len(arms),
            "n_valid_arms": n_valid,
            "arms_sha256": file_sha256(arms_path),
            "diagnostics_sha256": file_sha256(diagnostics_path),
        })
    # The records currently carry paths relative to the temporary directory;
    # replace that root with the final versioned artifact root.
    for record in shard_records:
        record["arms_path"] = str(Path("artifacts/phase3/support_geometry_prospective_v1") / Path(record["arms_path"]))
        record["diagnostics_path"] = str(Path("artifacts/phase3/support_geometry_prospective_v1") / Path(record["diagnostics_path"]))
    manifest = {
        "status": "PROSPECTIVE_INTERVENTIONS_GENERATED_RESPONSE_BLIND",
        "checkpoint": "P3-T2-DESIGN",
        "run_id": config["run_id"],
        "config_sha256": file_sha256(config_path),
        "t1_prediction_lock_sha256": file_sha256(lock_path),
        "p2_operator_id": p2_config["fracture"]["operator_id"],
        "new_intervention_seed": int(config["t2_prospective"]["new_intervention_seed"]),
        "response_blind": True,
        "response_paths_read": [],
        "same_assets": True,
        "n_samples": len(samples),
        "n_matches": len(by_match),
        "n_sets": total_sets,
        "n_complete_sets": complete_sets,
        "n_arms": total_arms,
        "n_valid_arms": valid_arms,
        "shards": shard_records,
        "source_p2_config": output_record(root, p2_config_path),
        "source_p2_generator": output_record(root, root / "scripts" / "run_phase2_fracture_formal.py"),
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    }
    write_json(temporary_dir / "execution_contract.yaml", {
        "task": "P3-T2 prospective intervention generation",
        "response_blind": True,
        "design_locked_before_response": True,
        "new_intervention_seed": manifest["new_intervention_seed"],
        "operator": manifest["p2_operator_id"],
        "response_paths_read": [],
    })
    write_json(temporary_dir / "source_input_provenance.json", {
        "p3_config": output_record(root, config_path),
        "t1_prediction_lock": output_record(root, lock_path),
        "canonical_samples": output_record(root, root / "artifacts" / "phase1" / "canonical_samples.npz"),
        "p2_config": output_record(root, p2_config_path),
        "p2_generator": output_record(root, root / "scripts" / "run_phase2_fracture_formal.py"),
    })
    write_json(temporary_dir / "intervention_manifest.json", manifest)
    write_json(temporary_dir / "intervention_receipt.json", {
        "status": manifest["status"],
        "checkpoint": manifest["checkpoint"],
        "manifest_sha256": file_sha256(temporary_dir / "intervention_manifest.json"),
        "response_blind": True,
        "response_paths_read": [],
    })
    (temporary_dir / "known_limitations.md").write_text(
        "This artifact freezes prospective interventions before any prospective response is read. "
        "It is a design/generation artifact, not evidence that the geometry prediction succeeds.\n",
        encoding="utf-8",
    )
    write_sha256sums(root, temporary_dir, "SHA256SUMS")
    temporary_dir.replace(final_dir)
    return manifest


def _encode_in_chunks(
    frozen: FrozenGeometryModel,
    positions: np.ndarray,
    teams: np.ndarray,
    adjacency: np.ndarray | None,
    batch_size: int = 64,
) -> np.ndarray:
    parts = []
    for start in range(0, len(positions), batch_size):
        end = start + batch_size
        parts.append(frozen.encode_numpy(positions[start:end], teams[start:end], None if adjacency is None else adjacency[start:end]))
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 128), dtype=np.float32)


def _l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-15)


def run_prospective_response(root: Path, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = root / "artifacts" / "phase3" / "support_geometry_prospective_v1"
    manifest_path = run_dir / "intervention_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("generate prospective intervention manifest first")
    response_root = run_dir / "response"
    if response_root.exists():
        raise FileExistsError(f"refusing to overwrite prospective response directory {response_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = load_arrays(root)
    positions = np.asarray(arrays["positions"], dtype=np.float32)
    teams = np.asarray(arrays["team_slots"], dtype=np.int64)
    adjacency = np.asarray(arrays["adjacency"], dtype=np.float32)
    model_records: list[dict[str, Any]] = []
    total_rows = 0
    for model_id in MODEL_IDS:
        frozen = load_frozen_model(root, model_id)
        base = _encode_in_chunks(frozen, positions, teams, adjacency if frozen.uses_adjacency else None)
        expected = load_baseline_expected(root, model_id)
        if not np.allclose(base, expected, atol=5e-7, rtol=1e-6):
            raise AssertionError(f"prospective baseline parity failed for {model_id}")
        for shard in manifest["shards"]:
            arms_path = root / str(shard["arms_path"])
            arms = pd.read_parquet(arms_path)
            valid = arms[arms["valid"].astype(bool)].copy()
            if valid.empty:
                continue
            deltas = np.stack([parse_delta(value) for value in valid["delta"]])
            sample_indices = valid["sample_index"].to_numpy(dtype=int)
            moved = positions[sample_indices] + deltas
            moved_teams = teams[sample_indices]
            moved_adj = adjacency[sample_indices] if frozen.uses_adjacency else None
            encoded = _encode_in_chunks(frozen, moved, moved_teams, moved_adj)
            base_norm = _l2_normalize_rows(base[sample_indices])
            moved_norm = _l2_normalize_rows(encoded)
            distance = 1.0 - np.sum(base_norm * moved_norm, axis=1)
            result = valid.copy()
            result["model_id"] = model_id
            result["architecture"] = frozen.architecture
            result["model_seed"] = int(str(model_id).rsplit("seed", 1)[1])
            result["response_distance"] = distance.astype(np.float32)
            result["normalized_response"] = (
                distance / result["epsilon"].to_numpy(float)
            ).astype(np.float32)
            out_path = response_root / "shards" / f"model_id={model_id}" / f"match_id={shard['match_id']}" / "response.parquet"
            write_parquet(out_path, result)
            receipt_path = out_path.parent / "receipt.json"
            write_json(receipt_path, {
                "status": "PROSPECTIVE_RESPONSE_COMPLETE",
                "model_id": model_id,
                "match_id": str(shard["match_id"]),
                "response_blind_at_generation": True,
                "input_intervention_manifest_sha256": file_sha256(manifest_path),
                "response_path": relative_path(root, out_path),
                "response_sha256": file_sha256(out_path),
                "n_response_rows": int(len(result)),
            })
            total_rows += len(result)
            model_records.append({
                "model_id": model_id,
                "match_id": str(shard["match_id"]),
                "response_path": relative_path(root, out_path),
                "receipt_path": relative_path(root, receipt_path),
                "response_sha256": file_sha256(out_path),
                "receipt_sha256": file_sha256(receipt_path),
                "n_response_rows": int(len(result)),
            })
    response_manifest = {
        "status": "PROSPECTIVE_RESPONSE_COMPLETE",
        "checkpoint": "P3-T2-RESPONSE",
        "run_id": config["run_id"],
        "config_sha256": file_sha256(config_path),
        "intervention_manifest_sha256": file_sha256(manifest_path),
        "response_blind_at_generation": True,
        "n_models": len(MODEL_IDS),
        "n_shards": len(model_records),
        "n_response_rows": total_rows,
        "baseline_parity": "PASS",
        "shards": model_records,
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    }
    write_json(response_root / "response_manifest.json", response_manifest)
    write_json(response_root / "response_receipt.json", {
        "status": response_manifest["status"],
        "response_manifest_sha256": file_sha256(response_root / "response_manifest.json"),
        "intervention_manifest_sha256": file_sha256(manifest_path),
    })
    write_sha256sums(root, response_root, "SHA256SUMS")
    return response_manifest


def run_prospective_evaluate(root: Path, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = root / "artifacts" / "phase3" / "support_geometry_prospective_v1"
    manifest = json.loads((run_dir / "intervention_manifest.json").read_text(encoding="utf-8"))
    response_manifest = json.loads((run_dir / "response" / "response_manifest.json").read_text(encoding="utf-8"))
    lock_path = root / "artifacts" / "phase3" / "support_geometry_v1" / "t1_prediction_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    arrays = load_arrays(root)
    all_arms: list[pd.DataFrame] = []
    all_pairs: list[pd.DataFrame] = []
    for model_id in MODEL_IDS:
        frozen = load_frozen_model(root, model_id)
        paths = [
            root / str(row["response_path"])
            for row in response_manifest["shards"]
            if row["model_id"] == model_id
        ]
        response = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        arms = geometry_arm_rows(root, frozen, arrays, response, ["pooled_embedding"])
        pairs = pair_rows_from_arms(arms)
        all_arms.append(arms)
        all_pairs.append(pairs)
    arm_frame = pd.concat(all_arms, ignore_index=True)
    pair_frame = pd.concat(all_pairs, ignore_index=True)
    write_parquet(run_dir / "prospective_arm_geometry.parquet", arm_frame)
    write_parquet(run_dir / "prospective_pair_geometry.parquet", pair_frame)
    prediction_rows: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for model_id, prospective in pair_frame.groupby("model_id", sort=True):
        retrospective = pd.read_parquet(
            root / "artifacts" / "phase3" / "support_geometry_v1" / "pair_geometry.parquet"
        )
        retrospective = retrospective[retrospective["model_id"] == model_id]
        for family in lock["families"]:
            fit = lock["fits"][str(model_id)][family]
            prediction = predictor_apply(prospective, family, fit, retrospective)
            result = prospective[["model_id", "matched_set_id", "source_match_id", "observed_pair_effect"]].copy()
            result["prediction"] = prediction
            result["predictor_family"] = family
            prediction_rows.append(result)
            metric = metric_values(result["observed_pair_effect"], result["prediction"])
            metrics.append({
                "scope": "prospective_model_id",
                "model_id": str(model_id),
                "predictor_family": family,
                "n_matches": int(result["source_match_id"].nunique()),
                **metric,
            })
    predictions = pd.concat(prediction_rows, ignore_index=True)
    write_parquet(run_dir / "prospective_pair_predictions.parquet", predictions)
    metrics_frame = pd.DataFrame.from_records(metrics)
    write_parquet(run_dir / "prospective_prediction_metrics.parquet", metrics_frame)
    summary = {
        "status": "T2_PROSPECTIVE_COMPLETE",
        "task": "P3-T2",
        "config_sha256": file_sha256(config_path),
        "intervention_manifest_sha256": file_sha256(run_dir / "intervention_manifest.json"),
        "response_manifest_sha256": file_sha256(run_dir / "response" / "response_manifest.json"),
        "t1_prediction_lock_sha256": file_sha256(lock_path),
        "n_arm_rows": int(len(arm_frame)),
        "n_pair_rows": int(len(pair_frame)),
        "n_models": int(pair_frame["model_id"].nunique()),
        "n_matches": int(pair_frame["source_match_id"].nunique()),
        "grouping_key": "source_match_id",
        "metrics_path": relative_path(root, run_dir / "prospective_prediction_metrics.parquet"),
        "response_blind_generation": True,
        "layer_or_formula_changed_after_response": False,
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    }
    write_json(run_dir / "prospective_summary.json", summary)
    write_json(run_dir / "prospective_receipt.json", {
        "status": summary["status"],
        "summary_sha256": file_sha256(run_dir / "prospective_summary.json"),
        "intervention_manifest_sha256": file_sha256(run_dir / "intervention_manifest.json"),
        "response_manifest_sha256": file_sha256(run_dir / "response" / "response_manifest.json"),
    })
    write_sha256sums(root, run_dir, "prospective_SHA256SUMS")
    return summary


def layerwise_sample_indices(root: Path, config: Mapping[str, Any]) -> list[int]:
    metadata = []
    path = root / "artifacts" / "phase1" / "canonical_samples.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            metadata.append({
                "sample_index": int(value["sample_index"]),
                "match_id": str(value["match_id"]),
                "frame": int(value["frame"]),
                "sample_id": str(value["sample_id"]),
            })
    count = int(config["t3_localization"].get("layerwise_sample_count", 50))
    per_match = max(1, count // max(1, len({item["match_id"] for item in metadata})))
    selected = []
    for match_id in sorted({item["match_id"] for item in metadata}):
        rows = sorted(
            (item for item in metadata if item["match_id"] == match_id),
            key=lambda item: (item["frame"], item["sample_id"]),
        )
        selected.extend(item["sample_index"] for item in rows[:per_match])
    if len(selected) != count:
        raise ValueError(f"layerwise selection expected {count}, found {len(selected)}")
    return selected


def run_layerwise(root: Path, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = root / "artifacts" / "phase3" / "support_geometry_v1"
    if not (output_dir / "pair_geometry.parquet").exists():
        raise FileNotFoundError("T1 pair geometry is required before T3")
    arrays = load_arrays(root)
    selected_indices = layerwise_sample_indices(root, config)
    all_arms: list[pd.DataFrame] = []
    all_pairs: list[pd.DataFrame] = []
    rank_records: list[dict[str, Any]] = []
    block_records: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        print(f"[P3-T3] computing layers for {model_id}", flush=True)
        frozen = load_frozen_model(root, model_id)
        response = read_response_rows(root, model_id)
        response = response[response["sample_index"].isin(selected_indices)].copy()
        arms = geometry_arm_rows(root, frozen, arrays, response, list(frozen.available_layers))
        pairs = pair_rows_from_arms(arms)
        if arms.empty or pairs.empty:
            raise ValueError(f"no layerwise rows for {model_id}")
        all_arms.append(arms)
        all_pairs.append(pairs)
        for sample_index in selected_indices:
            for layer in frozen.available_layers:
                _, jacobian = jacobian_for_layer(
                    frozen,
                    arrays["positions"][sample_index],
                    arrays["team_slots"][sample_index],
                    arrays["adjacency"][sample_index] if frozen.uses_adjacency else None,
                    layer,
                )
                singular = np.linalg.svd(jacobian, compute_uv=False)
                energy = singular**2
                probabilities = energy / max(float(energy.sum()), 1e-30)
                entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-30))))
                effective_rank = float(np.exp(entropy))
                rank_records.append({
                    "model_id": model_id,
                    "source_sample_index": int(sample_index),
                    "layer": layer,
                    "jacobian_output_dim": int(jacobian.shape[0]),
                    "input_dim": int(jacobian.shape[1]),
                    "rank_99pct": int(np.searchsorted(np.cumsum(np.sort(energy)[::-1]) / max(float(energy.sum()), 1e-30), 0.99) + 1),
                    "effective_rank": effective_rank,
                    "spectral_norm": float(singular[0]) if len(singular) else 0.0,
                })
        for layer in frozen.available_layers:
            prefix = "" if layer == "pooled_embedding" else f"{layer}__"
            full_col = f"{prefix}q_full"
            diag_col = f"{prefix}q_diagonal"
            off_col = f"{prefix}q_off_diagonal"
            selected = pairs[[full_col.replace("q_", "delta_q_"), diag_col.replace("q_", "delta_q_"), off_col.replace("q_", "delta_q_")]].to_numpy(float)
            full_values, diag_values, off_values = selected.T
            block_records.append({
                "model_id": model_id,
                "layer": layer,
                "n_pairs": int(len(pairs)),
                "mean_delta_q_full": float(np.mean(full_values)),
                "mean_delta_q_diagonal": float(np.mean(diag_values)),
                "mean_delta_q_off_diagonal": float(np.mean(off_values)),
                "mean_abs_delta_q_full": float(np.mean(np.abs(full_values))),
                "mean_abs_delta_q_diagonal": float(np.mean(np.abs(diag_values))),
                "mean_abs_delta_q_off_diagonal": float(np.mean(np.abs(off_values))),
                "identity_max_abs": float(np.max(np.abs(full_values - diag_values - off_values))),
            })
    arm_frame = pd.concat(all_arms, ignore_index=True)
    pair_frame = pd.concat(all_pairs, ignore_index=True)
    arm_path = output_dir / "layerwise_arm_geometry.parquet"
    pair_path = output_dir / "layerwise_pair_geometry.parquet"
    write_parquet(arm_path, arm_frame)
    write_parquet(pair_path, pair_frame)
    metric_records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for model_id, model_pairs in pair_frame.groupby("model_id", sort=True):
        frozen = load_frozen_model(root, model_id)
        for layer in frozen.available_layers:
            prefix = "" if layer == "pooled_embedding" else f"{layer}__"
            source_column = f"{prefix}delta_q_full"
            temporary = model_pairs.copy()
            temporary["delta_q_full"] = temporary[source_column]
            prediction = grouped_cv_predictions(temporary, "full_geometry")
            prediction["layer"] = layer
            prediction_frames.append(prediction)
            metric = metric_values(prediction["observed_pair_effect"], prediction["prediction"])
            metric_records.append({
                "scope": "layerwise_model_id",
                "model_id": str(model_id),
                "layer": layer,
                "predictor_family": "layer_full_geometry",
                "grouping_key": "source_match_id",
                "n_matches": int(prediction["source_match_id"].nunique()),
                **metric,
            })
    prediction_path = output_dir / "layerwise_predictions.parquet"
    metrics_path = output_dir / "layerwise_prediction_metrics.parquet"
    write_parquet(prediction_path, pd.concat(prediction_frames, ignore_index=True))
    write_parquet(metrics_path, pd.DataFrame.from_records(metric_records))
    rank_path = output_dir / "layerwise_jacobian_rank.parquet"
    block_path = output_dir / "layerwise_block_summary.parquet"
    write_parquet(rank_path, pd.DataFrame.from_records(rank_records))
    write_parquet(block_path, pd.DataFrame.from_records(block_records))
    summary = {
        "status": "T3_LAYERWISE_COMPLETE",
        "task": "P3-T3",
        "config_sha256": file_sha256(config_path),
        "selection_rule": config["t3_localization"]["layerwise_sample_selection"],
        "selected_sample_indices": selected_indices,
        "n_samples": len(selected_indices),
        "n_arm_rows": int(len(arm_frame)),
        "n_pair_rows": int(len(pair_frame)),
        "n_models": int(pair_frame["model_id"].nunique()),
        "n_matches": int(pair_frame["source_match_id"].nunique()),
        "grouping_key": "source_match_id",
        "prediction_metrics_path": relative_path(root, metrics_path),
        "rank_path": relative_path(root, rank_path),
        "block_path": relative_path(root, block_path),
        "interpretation_status": "NOT_YET_ROUTED",
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    }
    write_json(output_dir / "layerwise_summary.json", summary)
    write_json(output_dir / "layerwise_receipt.json", {
        "status": summary["status"],
        "checkpoint": "P3-T3",
        "summary_sha256": file_sha256(output_dir / "layerwise_summary.json"),
        "p2_response_manifest_sha256": file_sha256(P2_RUN / "response" / "response_manifest.json"),
    })
    write_sha256sums(root, output_dir, "layerwise_SHA256SUMS")
    return summary


def run_t0(root: Path, config_path: Path, config: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    arrays = load_arrays(root)
    n_samples = int(arrays["positions"].shape[0])
    if n_samples != 250:
        raise ValueError(f"expected 250 canonical samples, found {n_samples}")
    output_root = root / "artifacts" / "phase3"
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / "support_geometry_v1"
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite T0 artifact directory {final_dir}")
    temporary_dir = Path(__import__("tempfile").mkdtemp(prefix=".support-geometry-v1-", dir=str(output_root)))
    before_hashes = {
        "samples_npz": file_sha256(root / "artifacts" / "phase1" / "canonical_samples.npz"),
        "p1_manifest": file_sha256(root / "artifacts" / "phase1" / "point_mainline" / "model_manifest.json"),
    }
    parity: list[dict[str, Any]] = []
    models: dict[str, FrozenGeometryModel] = {}
    positions = np.asarray(arrays["positions"], dtype=np.float32)
    teams = np.asarray(arrays["team_slots"], dtype=np.int64)
    adjacency = np.asarray(arrays["adjacency"], dtype=np.float32)
    try:
        for model_id in MODEL_IDS:
            frozen = load_frozen_model(root, model_id)
            models[model_id] = frozen
            observed = frozen.encode_numpy(positions, teams, adjacency if frozen.uses_adjacency else None)
            expected = load_baseline_expected(root, model_id)
            difference = np.abs(observed - expected)
            parity.append({
                "model_id": model_id,
                "architecture": frozen.architecture,
                "status": "PASS" if np.allclose(observed, expected, atol=5e-7, rtol=1e-6) else "FAIL",
                "shape": list(observed.shape),
                "max_abs_difference": float(np.max(difference)),
                "mean_abs_difference": float(np.mean(difference)),
                "absolute_tolerance": 5e-7,
                "relative_tolerance": 1e-6,
                "available_layers": list(frozen.available_layers),
            })
            if parity[-1]["status"] != "PASS":
                raise AssertionError(f"baseline parity failed for {model_id}")

        selected_models = ["deepsets_ae_seed11", "gat_ae_seed11"]
        selected_sample_indices = [0] if smoke else [0]
        rng = np.random.default_rng(20260901)
        tangent_checks: list[dict[str, Any]] = []
        second_order_checks: list[dict[str, Any]] = []
        for model_id in selected_models:
            frozen = models[model_id]
            for sample_index in selected_sample_indices:
                direction = rng.normal(size=positions[sample_index].shape).astype(np.float32)
                direction /= np.linalg.norm(direction)
                for layer in frozen.available_layers:
                    baseline, jacobian = jacobian_for_layer(
                        frozen,
                        positions[sample_index],
                        teams[sample_index],
                        adjacency[sample_index] if frozen.uses_adjacency else None,
                        layer,
                    )
                    jvp_value, jvp = jvp_for_layer(
                        frozen,
                        positions[sample_index],
                        teams[sample_index],
                        adjacency[sample_index] if frozen.uses_adjacency else None,
                        layer,
                        direction,
                    )
                    matrix_jvp = jacobian @ direction.reshape(-1)
                    jvp_error = float(np.linalg.norm(jvp - matrix_jvp) / max(np.linalg.norm(jvp), 1e-12))
                    tangent = {
                        "model_id": model_id,
                        "sample_index": sample_index,
                        "layer": layer,
                        "jvp_matrix_relative_error": jvp_error,
                        "jvp_output_norm": float(np.linalg.norm(jvp_value)),
                    }
                    for step in config["geometry"]["finite_difference_steps"]:
                        finite = finite_difference(
                            frozen,
                            positions[sample_index],
                            teams[sample_index],
                            adjacency[sample_index] if frozen.uses_adjacency else None,
                            layer,
                            direction,
                            float(step),
                        )
                        tangent[f"finite_difference_relative_error_h{step:g}"] = float(
                            np.linalg.norm(finite - jvp) / max(np.linalg.norm(jvp), 1e-12)
                        )
                    tangent_checks.append(tangent)
                    for epsilon in config["geometry"]["tangent_epsilons"]:
                        perturbed = normalized_layer_numpy(
                            frozen,
                            positions[sample_index] + float(epsilon) * direction,
                            teams[sample_index],
                            adjacency[sample_index] if frozen.uses_adjacency else None,
                            layer,
                        )
                        actual = float(1.0 - np.dot(baseline, perturbed))
                        tangent_q = float(0.5 * np.sum((float(epsilon) * matrix_jvp) ** 2))
                        second_order_checks.append({
                            "model_id": model_id,
                            "sample_index": sample_index,
                            "layer": layer,
                            "epsilon": float(epsilon),
                            "actual_cosine_distance": actual,
                            "tangent_q": tangent_q,
                            "absolute_error": abs(actual - tangent_q),
                            "relative_error": abs(actual - tangent_q) / max(abs(actual), 1e-15),
                        })
        after_hashes = {
            "samples_npz": file_sha256(root / "artifacts" / "phase1" / "canonical_samples.npz"),
            "p1_manifest": file_sha256(root / "artifacts" / "phase1" / "point_mainline" / "model_manifest.json"),
        }
        if before_hashes != after_hashes:
            raise AssertionError("frozen input changed during T0")
        checkpoint_hashes = {
            model_id: str(config["frozen_assets"]["checkpoint_sha256"][model_id])
            for model_id in MODEL_IDS
        }
        summary = {
            "status": "T0_COMPLETE",
            "task": "P3-T0",
            "config_sha256": file_sha256(config_path),
            "n_samples": n_samples,
            "n_models": len(models),
            "model_ids": MODEL_IDS,
            "checkpoint_sha256": checkpoint_hashes,
            "baseline_parity": parity,
            "tangent_checks": tangent_checks,
            "second_order_checks": second_order_checks,
            "frozen_input_hashes_before": before_hashes,
            "frozen_input_hashes_after": after_hashes,
            "torch": torch.__version__,
            "dtype": "float32",
            "device": "cpu",
            "runtime": {
                "wall_seconds": time.perf_counter() - started,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            },
        }
        write_json(temporary_dir / "t0_summary.json", summary)
        write_json(temporary_dir / "source_input_provenance.json", {
            "config": output_record(root, config_path),
            "canonical_samples": output_record(root, root / "artifacts" / "phase1" / "canonical_samples.npz"),
            "model_manifest": output_record(root, root / "artifacts" / "phase1" / "point_mainline" / "model_manifest.json"),
            "p2_authoritative_root": relative_path(root, P2_RUN),
            "p2_matching_manifest": output_record(root, P2_RUN / "matching" / "matching_manifest.json"),
            "p2_response_manifest": output_record(root, P2_RUN / "response" / "response_manifest.json"),
            "p2_statistics_manifest": output_record(root, P2_RUN / "statistics" / "statistics_manifest.json"),
        })
        write_json(temporary_dir / "t0_receipt.json", {
            "status": "T0_COMPLETE",
            "checkpoint": "P3-T0",
            "run_id": config["run_id"],
            "summary_sha256": file_sha256(temporary_dir / "t0_summary.json"),
            "source_input_provenance_sha256": file_sha256(temporary_dir / "source_input_provenance.json"),
        })
        (temporary_dir / "known_limitations.md").write_text(
            "T0 is an interface and numerical-equivalence smoke only. It does not establish a geometry mechanism, "
            "prospective validity, a causal switch, or task improvement. DeepSets has no message-passing layers; "
            "its layerwise interface therefore reports input_node, pooling_pre, and pooled_embedding only.\n",
            encoding="utf-8",
        )
        write_sha256sums(root, temporary_dir, "t0_SHA256SUMS")
        temporary_dir.replace(final_dir)
        return summary
    except Exception:
        raise


def run_retrospective(root: Path, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    arrays = load_arrays(root)
    output_dir = root / "artifacts" / "phase3" / "support_geometry_v1"
    if not output_dir.exists():
        raise FileNotFoundError("run T0 before retrospective geometry")
    if (output_dir / "pair_geometry.parquet").exists() and (output_dir / "pair_predictions.parquet").exists():
        return finalize_retrospective(root, config_path, config)
    all_arms: list[pd.DataFrame] = []
    all_pairs: list[pd.DataFrame] = []
    layer = "pooled_embedding"
    for model_id in MODEL_IDS:
        print(f"[P3-T1] computing geometry for {model_id}", flush=True)
        frozen = load_frozen_model(root, model_id)
        response = read_response_rows(root, model_id)
        arms = geometry_arm_rows(root, frozen, arrays, response, [layer])
        pairs = pair_rows_from_arms(arms)
        if arms.empty or pairs.empty:
            raise ValueError(f"no geometry rows for {model_id}")
        all_arms.append(arms)
        all_pairs.append(pairs)
        print(f"[P3-T1] {model_id}: {len(arms)} arms, {len(pairs)} pairs", flush=True)
    arm_frame = pd.concat(all_arms, ignore_index=True)
    pair_frame = pd.concat(all_pairs, ignore_index=True)
    write_parquet(output_dir / "arm_geometry.parquet", arm_frame)
    write_parquet(output_dir / "pair_geometry.parquet", pair_frame)
    families = [
        "spectral",
        "rayleigh",
        "support",
        "residual",
        "metadata",
        "baseline",
        "diagonal_geometry",
        "full_geometry",
        "off_diagonal_geometry",
        "baseline_plus_full_geometry",
        "baseline_plus_diagonal_geometry",
        "baseline_plus_off_diagonal_geometry",
    ]
    prediction_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    for model_id, model_frame in pair_frame.groupby("model_id", sort=True):
        for family in families:
            predictions = grouped_cv_predictions(model_frame, family)
            prediction_frames.append(predictions)
            metrics = metric_values(predictions["observed_pair_effect"], predictions["prediction"])
            metric_records.append({
                "target": "observed_pair_effect",
                "scope": "model_id",
                "model_id": str(model_id),
                "architecture": str(model_frame["model_id"].iloc[0]).split("_seed")[0],
                "predictor_family": family,
                "grouping_key": "source_match_id",
                "n_matches": int(predictions["source_match_id"].nunique()),
                **metrics,
            })
    prediction_frame = pd.concat(prediction_frames, ignore_index=True)
    write_parquet(output_dir / "pair_predictions.parquet", prediction_frame)
    metrics_frame = pd.DataFrame.from_records(metric_records)
    baseline_family = "baseline"
    bootstrap_records: list[dict[str, Any]] = []
    for model_id, model_frame in pair_frame.groupby("model_id", sort=True):
        model_predictions = prediction_frame[prediction_frame["model_id"] == model_id]
        for family in ("full_geometry", "diagonal_geometry", "off_diagonal_geometry", "baseline_plus_full_geometry"):
            for metric in ("spearman", "mae", "r2", "direction_accuracy"):
                low, high = bootstrap_match_metric_delta(
                    model_predictions,
                    family,
                    baseline_family,
                    metric,
                    seed=20260901,
                )
                bootstrap_records.append({
                    "model_id": str(model_id),
                    "geometry_family": family,
                    "baseline_family": baseline_family,
                    "metric": metric,
                    "bootstrap_unit": "source_match_id",
                    "n_bootstrap": 500,
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                })
    bootstrap_frame = pd.DataFrame.from_records(bootstrap_records)
    write_parquet(output_dir / "match_bootstrap_comparisons.parquet", bootstrap_frame)
    write_parquet(output_dir / "prediction_metrics.parquet", metrics_frame)
    summary = {
        "status": "T1_RETROSPECTIVE_COMPLETE",
        "task": "P3-T1",
        "config_sha256": file_sha256(config_path),
        "geometry_layer": layer,
        "n_arm_rows": int(len(arm_frame)),
        "n_pair_rows": int(len(pair_frame)),
        "n_models": int(pair_frame["model_id"].nunique()),
        "n_matches": int(pair_frame["source_match_id"].nunique()),
        "grouping_key": "source_match_id",
        "predictor_families": families,
        "metrics_path": relative_path(root, output_dir / "prediction_metrics.parquet"),
        "bootstrap_path": relative_path(root, output_dir / "match_bootstrap_comparisons.parquet"),
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
        "interpretation_status": "NOT_YET_ROUTED",
    }
    write_json(output_dir / "retrospective_summary.json", summary)
    write_json(output_dir / "retrospective_receipt.json", {
        "status": summary["status"],
        "checkpoint": "P3-T1",
        "run_id": config["run_id"],
        "summary_sha256": file_sha256(output_dir / "retrospective_summary.json"),
        "input_p2_response_manifest_sha256": file_sha256(P2_RUN / "response" / "response_manifest.json"),
        "input_p2_matching_manifest_sha256": file_sha256(P2_RUN / "matching" / "matching_manifest.json"),
    })
    write_sha256sums(root, output_dir, "retrospective_SHA256SUMS")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=("t0", "retrospective", "freeze_predictors", "prospective_generate", "prospective_response", "prospective_evaluate", "layerwise"),
        required=True,
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    config_path = args.config.resolve()
    config = load_config(root, config_path)
    if args.mode == "t0":
        summary = run_t0(root, config_path, config, smoke=args.smoke)
    elif args.mode == "retrospective":
        summary = run_retrospective(root, config_path, config)
    elif args.mode == "freeze_predictors":
        summary = freeze_predictors(root, config_path, config)
    elif args.mode == "prospective_generate":
        summary = run_prospective_generate(root, config_path, config)
    elif args.mode == "prospective_response":
        summary = run_prospective_response(root, config_path, config)
    elif args.mode == "prospective_evaluate":
        summary = run_prospective_evaluate(root, config_path, config)
    else:
        summary = run_layerwise(root, config_path, config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
