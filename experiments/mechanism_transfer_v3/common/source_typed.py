"""Segment-preserving source relation head.

Architecture lineage is vendored, not imported, from:
  experiments/f095_campaign/u01_models.py::TypedPairMLP
  experiments/e832_focus/structure/a_source_compare.py::RepairedRho

Vendoring avoids executing legacy training entrypoints while keeping the exact
segment-role contract: phi is shared per point, psi acts within each segment,
and the nonlinear rho sees only psi(AB)+psi(CD).
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn

SOURCE_LINEAGE = {
    "backbone": "experiments/f095_campaign/u01_models.py::TypedPairMLP",
    "repaired_rho": "experiments/e832_focus/structure/a_source_compare.py::RepairedRho",
}
TYPED_DIMS = {64: (48, 48, 48), 256: (160, 160, 160), 512: (320, 320, 320)}
DEFAULT_WIDTH = 64
RHO_K = 16


class SegmentTypedBackbone(nn.Module):
    """Shared phi/psi backbone; segment grouping is preserved by construction."""

    def __init__(self, width: int = DEFAULT_WIDTH):
        super().__init__()
        if width not in TYPED_DIMS:
            raise ValueError(f"unsupported Typed width: {width}")
        point_width, point_hidden, segment_width = TYPED_DIMS[width]
        self.width = width
        self.phi = nn.Sequential(
            nn.Linear(2, point_width),
            nn.ReLU(),
            nn.Linear(point_width, point_hidden),
            nn.ReLU(),
        )
        self.psi = nn.Sequential(
            nn.Linear(point_hidden, segment_width),
            nn.ReLU(),
            nn.Linear(segment_width, segment_width),
            nn.ReLU(),
        )
        self.segment_width = segment_width

    def segment_state(self, x: torch.Tensor) -> torch.Tensor:
        points = x.reshape(-1, 4, 2)
        hidden = self.phi(points)
        hab = hidden[..., 0, :] + hidden[..., 1, :]
        hcd = hidden[..., 2, :] + hidden[..., 3, :]
        return self.psi(hab) + self.psi(hcd)


class RepairedSegmentRho(nn.Module):
    """Segment-preserving head with nonlinearity after the segment sum."""

    def __init__(self, width: int = DEFAULT_WIDTH, k: int = RHO_K):
        super().__init__()
        if k < 1:
            raise ValueError("rho hidden width must be positive")
        self.backbone = SegmentTypedBackbone(width)
        hidden = self.backbone.segment_width
        self.rho = nn.Sequential(nn.Linear(hidden, k), nn.GELU(), nn.Linear(k, 1))
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rho(self.backbone.segment_state(x)).reshape(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def architecture_digest(model: nn.Module) -> str:
    """Stable architecture digest: class, layer shapes, and widths (not weights)."""
    spec = [type(model).__name__]
    for name, parameter in model.named_parameters():
        spec.append(f"{name}:{tuple(parameter.shape)}")
    return hashlib.sha256("\n".join(spec).encode()).hexdigest()


def model_card(model: nn.Module) -> dict:
    return {
        "class": type(model).__name__,
        "width": getattr(model, "width", getattr(getattr(model, "backbone", None), "width", None)),
        "rho_hidden": getattr(model, "k", None),
        "n_parameters": count_parameters(model),
        "architecture_digest": architecture_digest(model),
        "lineage": SOURCE_LINEAGE,
    }
