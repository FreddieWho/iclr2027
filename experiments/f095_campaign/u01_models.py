#!/usr/bin/env python3
"""U01 model family: typed-pair invariant net + capacity controls.

TypedPairMLP: h_AB = phi(A)+phi(B), h_CD = phi(C)+phi(D),
  z = psi(h_AB)+psi(h_CD), logit = rho(z), phi/psi nonlinear.
  Exactly invariant to the D02 8-element group when inputs are
  scalar-standardized (affine scalar commutes with permutation).
SixDistMLP: plain MLP on 6 pairwise distances (rigid-invariant inputs
  whose column order still carries roles) — sameconstruct comparison.
ConcatMLP: shared per-point encoder, concatenated in fixed A,B,C,D order
  (capacity control: extra MLP depth WITHOUT sharing/pooling).
All take standardized flat inputs: typed/concat [N,8] (scalar8),
sixdist [N,6] (per-index standardized distances).
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


_TYPED_DIMS = {64: (48, 48, 48), 256: (160, 160, 160), 512: (320, 320, 320)}
_CONCAT_C = {64: 6, 256: 16, 512: 24}


class TypedPairMLP(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        pw, ph, sh = _TYPED_DIMS[width]
        self.phi = nn.Sequential(nn.Linear(2, pw), nn.ReLU(),
                                 nn.Linear(pw, ph), nn.ReLU())
        self.psi = nn.Sequential(nn.Linear(ph, sh), nn.ReLU(),
                                 nn.Linear(sh, sh), nn.ReLU())
        self.rho = nn.Linear(sh, 1)

    def forward(self, x):
        p = x.reshape(-1, 4, 2)
        h = self.phi(p)  # [...,4,ph], shared
        h_ab, h_cd = h[..., 0, :] + h[..., 1, :], h[..., 2, :] + h[..., 3, :]
        z = self.psi(h_ab) + self.psi(h_cd)
        return self.rho(z).reshape(-1)


def SixDistMLP(width: int = 64, feat: int = 32) -> nn.Module:
    return CoordMLP(width, feat, in_dim=6)


class ConcatMLP(nn.Module):
    def __init__(self, width: int = 64, feat: int = 32):
        super().__init__()
        c = _CONCAT_C[width]
        self.theta = nn.Sequential(nn.Linear(2, c), nn.ReLU(),
                                   nn.Linear(c, c), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(4 * c, width), nn.ReLU(),
                                  nn.Linear(width, width), nn.ReLU(),
                                  nn.Linear(width, feat),
                                  nn.Linear(feat, 1))

    def forward(self, x):
        p = x.reshape(-1, 4, 2)
        e = self.theta(p).reshape(-1, 4 * self.theta[-2].out_features)
        return self.head(e).reshape(-1)
