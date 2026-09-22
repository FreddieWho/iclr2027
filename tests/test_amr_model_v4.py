"""Unit tests for the AMR M1-redesign v4 model (exact spectral basis)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "p4_amr"))
import amr_model_v4 as v4  # noqa: E402


def _toy_graph(seed: int = 0) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    adj = (torch.rand((2, 20, 20), generator=rng) > 0.7).float()
    adj = torch.triu(adj, diagonal=1)
    return adj + adj.transpose(1, 2)


def test_exact_projectors_partition_and_idempotence():
    adj = _toy_graph()
    evals, evecs = v4.spectral_factors(adj)
    assert evals.shape == (2, 20) and evecs.shape == (2, 20, 20)
    hidden = torch.randn(2, 20, 8)
    bands = v4.band_index(evals)
    recon = sum(v4.exact_band_project(evecs, bands, b, hidden) for b in range(v4.N_BANDS))
    assert torch.allclose(recon, hidden, atol=1e-5)
    for b in range(v4.N_BANDS):
        once = v4.exact_band_project(evecs, bands, b, hidden)
        twice = v4.exact_band_project(evecs, bands, b, once)
        assert torch.allclose(once, twice, atol=1e-5)


def test_band_orthogonality():
    adj = _toy_graph()
    evals, evecs = v4.spectral_factors(adj)
    hidden = torch.randn(2, 20, 8)
    bands = v4.band_index(evals)
    projs = [v4.exact_band_project(evecs, bands, b, hidden) for b in range(v4.N_BANDS)]
    for a in range(v4.N_BANDS):
        for b in range(a + 1, v4.N_BANDS):
            cross = float((projs[a] * projs[b]).sum(dim=(1, 2)).abs().max())
            assert cross < 1e-4


def test_slepian_properties():
    adj = _toy_graph(1).numpy()[0]
    lap = v4.normalized_laplacian(torch.from_numpy(adj[None]).float())[0].numpy()
    evals, evecs = np.linalg.eigh(lap)
    support = np.zeros(20, dtype=bool)
    support[:4] = True
    rng = np.random.default_rng(5)
    delta, mu = v4.slepian_intervention(evecs, evals, 1, support, 0.25, rng)
    assert 0.0 <= mu <= 1.0 + 1e-9
    assert abs(float(np.linalg.norm(delta)) - 0.25) < 1e-9
    delta2, _ = v4.slepian_intervention(evecs, evals, 1, support, 0.25, np.random.default_rng(5))
    assert np.allclose(delta, delta2)
    full = np.ones(20, dtype=bool)
    _, mu_full = v4.slepian_intervention(evecs, evals, 0, full, 0.25, np.random.default_rng(6))
    assert abs(mu_full - 1.0) < 1e-9


def test_vicreg_collapse_positive():
    torch.manual_seed(0)
    live = torch.randn(32, 64) * 0.5 + 1.0
    var_live, cov_live = v4.vicreg_var_cov(live, 0.4)
    assert float(var_live) < 0.4
    dead = torch.ones(32, 64) * 2.0
    var_dead, _ = v4.vicreg_var_cov(dead, 0.4)
    assert abs(float(var_dead) - 0.4) < 1e-6


def test_model_shapes_and_eval_interface():
    torch.manual_seed(1)
    model = v4.AMRModelV4()
    positions = torch.randn(3, 20, 2)
    team = torch.tensor([[0] * 10 + [1] * 10] * 3)
    adj = (_toy_graph(2)[:1].repeat(3, 1, 1) > 0).float()
    z_ctx = model.context_channel(positions, team, adj)
    feats = model.band_features(positions, team, adj)
    assert z_ctx.shape == (3, v4.LATENT_DIM)
    assert feats.shape == (3, v4.N_BANDS, v4.BAND_OUT_DIM)
    z = model.encode(positions, team, adj, "team_mean")
    assert z.shape == (3, v4.N_BANDS * v4.BAND_OUT_DIM)
    assert torch.allclose(model.mode_embedding(z), model.mode_embedding(feats), atol=1e-6)
    assert model.mode_head(z).shape == (3, v4.MODE_DIM)
    for b in range(v4.N_BANDS):
        assert model.eqv_heads[b](feats[:, b, :]).shape == (3, v4.EQV_TARGET_DIM)


def test_stopgrad_clean_branch_carries_no_gradient():
    torch.manual_seed(3)
    model = v4.AMRModelV4()
    centered = torch.randn(4, 20, 2)
    team = torch.tensor([[0] * 10 + [1] * 10] * 4)
    adj = (_toy_graph(4)[:1].repeat(4, 1, 1) > 0).float()
    feats_clean = model.band_features(centered, team, adj).detach()
    feats_pert = model.band_features(centered + 0.01 * torch.randn_like(centered), team, adj)
    d = feats_pert[:, 0, :] - feats_clean[:, 0, :]
    loss = d.pow(2).sum()
    model.zero_grad()
    loss.backward()
    assert feats_clean.grad_fn is None
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.encoder.parameters())
