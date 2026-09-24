#!/usr/bin/env python3
"""U10 task-adapted models. Raw/sixdist reuse CoordMLP with task in_dim.
TypedT variants keep the task's role structure:
- T1 (V1,V2,V3,Q): phi pooled over vertices, Q on a separate branch.
  Exactly invariant to vertex permutations (scalar-standardized inputs).
- T2 (A,B,C=center): phi(A)+phi(B) pooled, center on separate branch.
  Exactly invariant to A<->B.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
from coord_mlp import CoordMLP  # noqa: E402


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


class TypedTriMLP(nn.Module):
    def __init__(self, pw: int = 48, ph: int = 48, sh: int = 48):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(2, pw), nn.ReLU(),
                                 nn.Linear(pw, ph), nn.ReLU())
        self.psi = nn.Sequential(nn.Linear(2 * ph, sh), nn.ReLU(),
                                 nn.Linear(sh, sh), nn.ReLU())
        self.rho = nn.Linear(sh, 1)

    def forward(self, x):
        p = x.reshape(-1, 4, 2)
        h_tri = self.phi(p[..., :3, :]).sum(dim=-2)
        h_q = self.phi(p[..., 3, :])
        z = self.psi(torch.cat([h_tri, h_q], dim=-1))
        return self.rho(z).reshape(-1)


class TypedDiskMLP(nn.Module):
    def __init__(self, pw: int = 48, ph: int = 48, sh: int = 48):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(2, pw), nn.ReLU(),
                                 nn.Linear(pw, ph), nn.ReLU())
        self.psi = nn.Sequential(nn.Linear(2 * ph, sh), nn.ReLU(),
                                 nn.Linear(sh, sh), nn.ReLU())
        self.rho = nn.Linear(sh, 1)

    def forward(self, x):
        p = x.reshape(-1, 3, 2)
        h_ab = self.phi(p[..., :2, :]).sum(dim=-2)
        h_c = self.phi(p[..., 2, :])
        z = self.psi(torch.cat([h_ab, h_c], dim=-1))
        return self.rho(z).reshape(-1)


def SixDistMLP(width: int = 64, feat: int = 32, in_dim: int = 6) -> nn.Module:
    return CoordMLP(width, feat, in_dim=in_dim)
