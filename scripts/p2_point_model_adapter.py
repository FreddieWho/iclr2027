#!/usr/bin/env python3
"""Minimal P2 adapter for the frozen P1 DeepSets encoder."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


MODEL_ID = "deepsets_ae_seed11"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_hash(path: Path, expected: str, label: str) -> str:
    observed = file_sha256(path)
    if observed != str(expected):
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, observed {observed}")
    return observed


def _resolve_inside_root(root: Path, relative_path: str) -> Path:
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact path escapes project root: {relative_path}") from exc
    return resolved


def _load_p1_module(path: Path):
    module_name = f"p1_model_factory_for_p2_{file_sha256(path)[:12]}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load P1 model factory: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class DeepSetsAdapter:
    model: Any
    batch_size: int = 128
    uses_adjacency: bool = False
    architecture: str = "DeepSets-AE"

    def encode(
        self,
        positions: np.ndarray,
        team_slots: np.ndarray,
        adjacency: np.ndarray | None = None,
    ) -> np.ndarray:
        import torch

        coordinates = np.asarray(positions, dtype=np.float32)
        teams = np.asarray(team_slots, dtype=np.int64)
        if coordinates.ndim == 2:
            coordinates = coordinates[None, ...]
        if teams.ndim == 1:
            teams = teams[None, ...]
        if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
            raise ValueError("positions must have shape (batch, nodes, 2)")
        if teams.shape != coordinates.shape[:2]:
            raise ValueError("team_slots must have shape (batch, nodes)")
        if not np.isfinite(coordinates).all() or not np.isin(teams, [0, 1]).all():
            raise ValueError("encoder inputs must be finite with team slots in {0,1}")
        graphs: np.ndarray | None = None
        if self.uses_adjacency:
            if adjacency is None:
                raise ValueError(f"{self.architecture} requires baseline adjacency")
            graphs = np.asarray(adjacency, dtype=np.float32)
            if graphs.ndim == 2:
                graphs = graphs[None, ...]
            expected_shape = (len(coordinates), coordinates.shape[1], coordinates.shape[1])
            if graphs.shape != expected_shape or not np.isfinite(graphs).all():
                raise ValueError(f"adjacency must have shape {expected_shape}")
        parts: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(coordinates), int(self.batch_size)):
                end = start + int(self.batch_size)
                if self.uses_adjacency:
                    assert graphs is not None
                    encoded = self.model.encode(
                        torch.from_numpy(coordinates[start:end]),
                        torch.from_numpy(teams[start:end]),
                        torch.from_numpy(graphs[start:end]),
                    )
                else:
                    encoded = self.model.encode(
                        torch.from_numpy(coordinates[start:end]),
                        torch.from_numpy(teams[start:end]),
                    )
                parts.append(encoded.detach().cpu().numpy())
        result = np.concatenate(parts, axis=0) if parts else np.empty((0, 128), dtype=np.float32)
        if result.ndim != 2 or result.shape[1] != 128 or not np.isfinite(result).all():
            raise ValueError(f"invalid DeepSets embedding output: shape={result.shape}")
        return result


def load_point_adapter(
    root: Path,
    model_id: str,
    batch_size: int = 128,
) -> tuple[DeepSetsAdapter, dict[str, Any]]:
    """Restore one frozen P1 point model through P1's sole model factory."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = root.resolve()
    p1_script = root / "scripts" / "run_phase1_pipeline.py"
    manifest_path = root / "artifacts" / "phase1" / "point_mainline" / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [item for item in manifest if item.get("model_id") == model_id]
    if len(matches) != 1:
        raise ValueError(f"expected one {model_id} manifest row, found {len(matches)}")
    model_spec = matches[0]
    expected_contract = {"latent_dim": 128, "pooling": "team_mean"}
    for key, expected in expected_contract.items():
        if model_spec.get(key) != expected:
            raise ValueError(f"unexpected {model_id} {key}: {model_spec.get(key)!r}")
    checkpoint = _resolve_inside_root(root, str(model_spec["path"]))
    checkpoint_hash = verify_file_hash(checkpoint, str(model_spec["checkpoint_sha256"]), "checkpoint")

    import torch

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    p1_module = _load_p1_module(p1_script)
    DeepSetsAE, GATAE, PhaseGAT = p1_module.build_torch_models()
    with np.load(root / "artifacts" / "phase1" / "canonical_samples.npz") as archive:
        n_nodes = int(archive["positions"].shape[1])
    architecture = str(model_spec.get("architecture"))
    if architecture == "DeepSets-AE":
        model = DeepSetsAE(n_nodes)
        uses_adjacency = False
    elif architecture == "GAT-small-AE":
        if model_spec.get("graph") != "baseline_fixed_weighted_knn4":
            raise ValueError(f"unexpected graph contract for {model_id}")
        model = GATAE(n_nodes)
        uses_adjacency = True
    elif architecture == "Phase-GAT":
        if model_spec.get("graph") != "baseline_fixed_weighted_knn4":
            raise ValueError(f"unexpected graph contract for {model_id}")
        classes = model_spec.get("phase_classes", [])
        if not classes:
            raise ValueError(f"missing phase classes for {model_id}")
        model = PhaseGAT(len(classes))
        uses_adjacency = True
    else:
        raise ValueError(f"unsupported point architecture: {architecture}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    adapter = DeepSetsAdapter(
        model=model,
        batch_size=int(batch_size),
        uses_adjacency=uses_adjacency,
        architecture=architecture,
    )
    receipt = {
        "model_id": model_id,
        "family": model_spec["family"],
        "architecture": model_spec["architecture"],
        "latent_dim": int(model_spec["latent_dim"]),
        "pooling": model_spec["pooling"],
        "graph": model_spec.get("graph"),
        "device": "cpu",
        "batch_size": int(batch_size),
        "eval_mode": not bool(model.training),
        "inference_mode": True,
        "weights_only": True,
        "p1_factory_path": str(p1_script.relative_to(root)),
        "p1_factory_sha256": file_sha256(p1_script),
        "model_manifest_path": str(manifest_path.relative_to(root)),
        "model_manifest_sha256": file_sha256(manifest_path),
        "checkpoint_path": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": checkpoint_hash,
        "torch_version": torch.__version__,
    }
    return adapter, receipt


def load_deepsets_adapter(
    root: Path,
    model_id: str = MODEL_ID,
    batch_size: int = 128,
) -> tuple[DeepSetsAdapter, dict[str, Any]]:
    """Restore the single P2-C DeepSets model and reject model drift."""
    if model_id != MODEL_ID:
        raise ValueError(f"P2-C is bounded to {MODEL_ID}, got {model_id}")
    adapter, receipt = load_point_adapter(root, model_id=model_id, batch_size=batch_size)
    if adapter.architecture != "DeepSets-AE" or adapter.uses_adjacency:
        raise ValueError(f"P2-C expected DeepSets-AE, got {adapter.architecture}")
    return adapter, receipt
