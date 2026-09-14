"""Unit tests for the AMR M1 model (P4-N1)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import amr_model as amr  # noqa: E402


def test_band_coefficients_partition_of_unity():
    coeffs = amr.chebyshev_band_coefficients()
    assert coeffs.shape == (amr.N_BANDS, amr.CHEBY_ORDER + 1)
    grid = np.linspace(0.0, 2.0, 400)
    x = grid - 1.0
    total = np.zeros_like(grid)
    for b in range(amr.N_BANDS):
        response = np.zeros_like(grid)
        for r in range(amr.CHEBY_ORDER + 1):
            response += coeffs[b, r] * np.cos(r * np.arccos(np.clip(x, -1.0, 1.0)))
        total += response
    interior = (grid > 0.15) & (grid < 1.85)
    assert np.all(np.abs(total[interior] - 1.0) < 0.15)


def test_apply_chebyshev_identity_and_permutation_equivariance():
    torch.manual_seed(0)
    adj = torch.randint(0, 2, (2, 20, 20)).float()
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.transpose(1, 2)
    lap = amr.normalized_laplacian(adj)
    hidden = torch.randn(2, 20, 8)
    identity_coeffs = torch.zeros(amr.CHEBY_ORDER + 1)
    identity_coeffs[0] = 1.0
    out = amr.apply_chebyshev(lap, hidden, identity_coeffs)
    assert torch.allclose(out, hidden, atol=1e-5)
    perm = torch.randperm(20)
    lap_p = lap[:, perm][:, :, perm]
    out_p = amr.apply_chebyshev(lap_p, hidden[:, perm], identity_coeffs)
    assert torch.allclose(out_p, out[:, perm], atol=1e-5)


def test_normalized_laplacian_psd():
    adj = (torch.rand(1, 12, 12) > 0.5).float()
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.transpose(1, 2)
    lap = amr.normalized_laplacian(adj)[0]
    assert torch.allclose(lap, lap.T, atol=1e-6)
    x = torch.randn(12, 5)
    assert float((x * (lap @ x)).sum()) >= -1e-5


def test_amr_model_shapes():
    torch.manual_seed(1)
    model = amr.AMRModel()
    positions = torch.randn(3, 20, 2)
    team = torch.tensor([[0] * 10 + [1] * 10] * 3)
    adj = (torch.rand(3, 20, 20) > 0.7).float()
    z_ctx = model.context_channel(positions, team, adj)
    feats = model.band_features(positions, team, adj)
    z_mode = model.mode_embedding(feats)
    assert z_ctx.shape == (3, amr.LATENT_DIM)
    assert feats.shape == (3, amr.N_BANDS, amr.BAND_OUT_DIM)
    assert z_mode.shape == (3, amr.MODE_DIM)
    phase, zone, centroid = model.context(z_ctx)
    assert phase.shape == (3, 2) and zone.shape == (3, 3) and centroid.shape == (3, 4)


def test_whitened_intervention_norm_and_determinism():
    adj = (np.random.default_rng(0).random((20, 20)) > 0.7).astype(np.float64)
    adj = np.triu(adj, 1)
    adj = adj + adj.T
    lap = amr.normalized_laplacian(torch.from_numpy(adj[None]).float())[0].numpy()
    coeffs = amr.chebyshev_band_coefficients()
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    d1 = amr.whitened_intervention(lap, coeffs, 2, 0.25, rng1)
    d2 = amr.whitened_intervention(lap, coeffs, 2, 0.25, rng2)
    assert np.allclose(d1, d2)
    assert abs(float(np.linalg.norm(d1)) - 0.25) < 1e-6


def test_jvp_matches_finite_difference():
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.sin(x[0]) + x[1] ** 2, x[0] * x[1]])

    x = torch.tensor([0.7, -1.2])
    delta = torch.tensor([0.3, 0.8])
    out = amr.jvp_vs_finite_difference(fn, x, delta)
    assert out["rel_error"] < 1e-2


def test_route_alpha_guard():
    model = amr.AMRModel()
    model.set_route_alpha(torch.tensor([1.0, 0, 0, 0, 0, 0]))
    assert float(model.route_alpha[0]) == 1.0
    try:
        model.set_route_alpha(torch.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError("shape guard failed")


def test_whitened_intervention_support_mask():
    adj = (np.random.default_rng(1).random((20, 20)) > 0.7).astype(np.float64)
    adj = np.triu(adj, 1)
    adj = adj + adj.T
    lap = amr.normalized_laplacian(torch.from_numpy(adj[None]).float())[0].numpy()
    coeffs = amr.chebyshev_band_coefficients()
    support = np.zeros(20, dtype=bool)
    support[:5] = True
    rng = np.random.default_rng(9)
    d = amr.whitened_intervention(lap, coeffs, 3, 0.25, rng, support=support)
    assert abs(float(np.linalg.norm(d)) - 0.25) < 1e-6
    full = amr.whitened_intervention(lap, coeffs, 3, 0.25, np.random.default_rng(9))
    assert not np.allclose(d, full)


def test_encode_frozen_eval_compatible():
    torch.manual_seed(2)
    model = amr.AMRModel()
    positions = torch.randn(2, 20, 2)
    team = torch.tensor([[0] * 10 + [1] * 10] * 2)
    adj = (torch.rand(2, 20, 20) > 0.7).float()
    z = model.encode(positions, team, adj, "team_mean")
    assert z.shape == (2, amr.N_BANDS * amr.BAND_OUT_DIM)
    emb_from_flat = model.mode_embedding(z)
    emb_from_bands = model.mode_embedding(model.band_features(positions, team, adj))
    assert torch.allclose(emb_from_flat, emb_from_bands, atol=1e-6)
    assert emb_from_flat.shape == (2, amr.MODE_DIM)
    head = model.mode_head(z)
    assert head.shape == (2, amr.MODE_DIM)


def test_eqv_heads_shape():
    model = amr.AMRModel()
    feats = torch.randn(4, amr.N_BANDS, amr.BAND_OUT_DIM)
    for b in range(amr.N_BANDS):
        assert model.eqv_heads[b](feats[:, b, :]).shape == (4, 4)
