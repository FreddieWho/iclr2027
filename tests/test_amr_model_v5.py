"""Unit tests for AMR R1 (v5) decoupled dual-tower."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "p4_amr"))
import amr_model_v5 as v5  # noqa: E402


def _inputs():
    torch.manual_seed(0)
    positions = torch.randn(3, 20, 2)
    team = torch.tensor([[0] * 10 + [1] * 10] * 3)
    adj = (torch.rand(3, 20, 20) > 0.7).float()
    adj = torch.triu(adj, diagonal=1)
    return positions, team, adj + adj.transpose(1, 2)


def test_towers_share_no_parameters():
    model = v5.AMRModelV5()
    ctx, mode = model.tower_params()
    assert len(ctx & mode) == 0
    assert len(ctx) > 0 and len(mode) > 0
    assert len(ctx) + len(mode) == len({id(p) for p in model.parameters()})


def test_gradient_insulation_both_directions():
    positions, team, adj = _inputs()
    model = v5.AMRModelV5()
    ctx_ids, mode_ids = model.tower_params()
    by_id = {id(p): p for p in model.parameters()}
    # mode-side loss (pair-style on band feats) must not touch ctx tower
    model.zero_grad()
    feats = model.band_features(positions, team, adj)
    emb = model.mode_embedding(feats)
    (emb.pow(2).sum() + model.eqv_heads[0](feats[:, 0, :]).pow(2).sum()).backward()
    assert all(by_id[i].grad is None for i in ctx_ids)
    assert any(by_id[i].grad is not None and float(by_id[i].grad.abs().sum()) > 0 for i in mode_ids)
    # ctx-side loss must not touch mode tower
    model.zero_grad()
    z = model.context_channel(positions, team, adj)
    model.context(z)[0].pow(2).sum().backward()
    assert all(by_id[i].grad is None for i in mode_ids)
    assert any(by_id[i].grad is not None and float(by_id[i].grad.abs().sum()) > 0 for i in ctx_ids)


def test_shapes_and_eval_interface():
    positions, team, adj = _inputs()
    model = v5.AMRModelV5()
    z_ctx = model.context_channel(positions, team, adj)
    feats = model.band_features(positions, team, adj)
    assert z_ctx.shape == (3, v5.LATENT_DIM)
    assert feats.shape == (3, v5.N_BANDS, v5.BAND_OUT_DIM)
    z = model.encode(positions, team, adj, "team_mean")
    assert z.shape == (3, v5.N_BANDS * v5.BAND_OUT_DIM)
    assert torch.allclose(model.mode_embedding(z), model.mode_embedding(feats), atol=1e-6)
    phase, zone, centroid = model.context(z_ctx)
    assert phase.shape == (3, 2) and zone.shape == (3, 3) and centroid.shape == (3, 4)
