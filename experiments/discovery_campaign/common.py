#!/usr/bin/env python3
"""Shared loader for discovery-campaign coordinate MLPs (CPU)."""
from __future__ import annotations
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coord_mlp import CoordMLP, MarginMLP


def rel_features(X) -> "np.ndarray":
    """Label-free relational features: 6 pairwise distances + 4 centroid radii.
    Crossing boolean deliberately EXCLUDED (it is the label)."""
    import numpy as np
    X = np.asarray(X, dtype=np.float32).reshape(-1, 4, 2)
    i, j = np.triu_indices(4, k=1)
    pw = np.linalg.norm(X[:, i] - X[:, j], axis=-1)  # [N,6]
    cen = X.mean(axis=1, keepdims=True)
    rad = np.linalg.norm(X - cen, axis=-1)           # [N,4]
    return np.concatenate([pw, rad], axis=-1)        # [N,10]


def load_model(model_dir: Path) -> tuple[CoordMLP, dict]:
    ckpt = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    cls = MarginMLP if "mhead.weight" in ckpt["state"] else CoordMLP
    model = cls(ckpt["hidden"], ckpt["feat"], ckpt.get("in_dim", 8))
    model.load_state_dict(ckpt["state"])
    model.eval()
    stats = {"mu": ckpt["mu"], "sd": ckpt["sd"]}
    if ckpt.get("featurize"):
        stats["featurize"] = ckpt["featurize"]
    return model, stats


def preprocess(X, stats) -> torch.Tensor:
    import numpy as np
    if stats.get("featurize") == "rel12":
        F = rel_features(X)
    else:
        F = np.asarray(X, dtype=np.float32).reshape(
            -1, len(np.asarray(stats["mu"]).reshape(-1)))
    d = len(np.asarray(stats["mu"]).reshape(-1))
    F = np.asarray(F, dtype=np.float32).reshape(-1, d)
    return torch.from_numpy((F - stats["mu"]) / stats["sd"])
