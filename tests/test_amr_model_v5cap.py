"""Unit tests for the CAP-decoupled control (R2)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "p4_amr"))
import amr_model_v5cap as cap  # noqa: E402


def test_cap_head_and_tower_inheritance():
    torch.manual_seed(0)
    model = cap.AMRModelV5CAP()
    positions = torch.randn(2, 20, 2)
    team = torch.tensor([[0] * 10 + [1] * 10] * 2)
    adj = (torch.rand(2, 20, 20) > 0.7).float()
    feats = model.band_features(positions, team, adj)
    z = model.mode_embedding(feats)
    assert z.shape == (2, 64)
    pred = model.cap_head(z)
    assert pred.shape == (2, cap.CAP_OUT_DIM)
    ctx_ids, mode_ids = model.tower_params()
    assert len(ctx_ids & mode_ids) == 0
    assert id(model.cap_head.weight) in mode_ids


def test_cap_loss_shape_and_determinism():
    torch.manual_seed(1)
    model = cap.AMRModelV5CAP()
    model.eval()
    d1 = torch.randn(4, 64)
    t1 = torch.randn(4, cap.CAP_OUT_DIM)
    with torch.no_grad():
        l1 = ((model.cap_head(d1) - t1) / 0.25).pow(2).sum(-1).mean()
        l2 = ((model.cap_head(d1) - t1) / 0.25).pow(2).sum(-1).mean()
    assert torch.allclose(l1, l2) and bool(torch.isfinite(l1))
