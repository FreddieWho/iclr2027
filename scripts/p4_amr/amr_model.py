"""AMR-Fixed (M1) model: spectral band-routed dual-channel representation.

Implements the M1 level of docs/02_METHOD_SPEC_AMR.md for the sports
explicit-graph domain:

- encoder: the frozen-width GraphEncoder backbone (matched capacity with the
  T5R TaskModel family; comparable param count reported, not padded);
- context channel: team_mean pool on the RAW view + context heads
  (phase/zone/centroid) — preserves absolute deployment info (spec 4.1);
- mode channel: per-band Chebyshev filtering P_b(L)H of the centered-view
  node features, per-band pooling, concatenated z_mode (spec 4.2);
- B=6 band-pass filters on the normalized Laplacian, Chebyshev order R=5,
  equal-width bands with cosine-tapered edges (spec 3);
- spectral-whitened intervention sampling (spec 5.2);
- fixed routing gates alpha[b] (M1: NOT learned; set from the config lock),
  L_route = sum_b alpha_b L_inv^b + (1-alpha_b) L_eqv^b (spec 7);
- JVP vs central finite-difference validation helper (spec 4.3).

Protocol status: P4-N1 implementation. The routing assignment and loss
weights are frozen in the P4-N2 config lock BEFORE any AMR training.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "p3_t5r"))  # T5R3 encoder sibling era dir
from run_t5r3_sanity import GraphEncoder, team_pool  # noqa: E402

HIDDEN_DIM = 128
LATENT_DIM = 128
N_BANDS = 6
CHEBY_ORDER = 5
CHEBY_NODES = 64
BAND_OUT_DIM = 32
MODE_DIM = 64


def normalized_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    """Symmetric normalized Laplacian L = I - D^-1/2 A D^-1/2 (batched)."""
    degree = adjacency.sum(dim=-1).clamp_min(1.0)
    inv_sqrt = degree.pow(-0.5)
    norm = adjacency * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)
    eye = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype)
    return eye - norm


def band_window(values: np.ndarray, lo: float, hi: float, taper: float = 0.15) -> np.ndarray:
    """Cosine-tapered band window on [0, 2]."""
    out = np.zeros_like(values)
    width = hi - lo
    t = taper * width
    rise = (values >= lo - t) & (values < lo + t)
    if lo <= 0.0:
        rise = (values >= lo) & (values < lo + t)
    fall = (values > hi - t) & (values <= hi + t)
    if hi >= 2.0:
        fall = (values > hi - t) & (values <= hi)
    out[(values >= lo + t) & (values <= hi - t)] = 1.0
    if rise.any():
        x = (values[rise] - (lo - t)) / (2 * t)
        out[rise] = 0.5 - 0.5 * np.cos(np.pi * x)
    if fall.any():
        x = (values[fall] - (hi - t)) / (2 * t)
        out[fall] = 0.5 + 0.5 * np.cos(np.pi * x)
    return out


def chebyshev_band_coefficients(n_bands: int = N_BANDS, order: int = CHEBY_ORDER,
                                n_nodes: int = CHEBY_NODES) -> np.ndarray:
    """Chebyshev coefficients a[b, r] approximating B equal-width band windows.

    Spectrum of the normalized Laplacian is [0, 2]; Chebyshev domain is
    [-1, 1] via lambda = x + 1. Coefficients use discrete orthogonality at
    Chebyshev nodes (deterministic).
    """
    m = np.arange(1, n_nodes + 1)
    x = np.cos(np.pi * (m - 0.5) / n_nodes)
    lam = x + 1.0
    edges = np.linspace(0.0, 2.0, n_bands + 1)
    coeffs = np.zeros((n_bands, order + 1), dtype=np.float64)
    for b in range(n_bands):
        h = band_window(lam, edges[b], edges[b + 1])
        for r in range(order + 1):
            t_r = np.cos(r * np.arccos(np.clip(x, -1.0, 1.0)))
            scale = 1.0 if r == 0 else 2.0
            coeffs[b, r] = scale / n_nodes * float(np.sum(h * t_r))
    return coeffs


def apply_chebyshev(laplacian: torch.Tensor, hidden: torch.Tensor,
                    coeffs: torch.Tensor) -> torch.Tensor:
    """P(L)H = sum_r a_r T_r(L~)H with L~ = L - I (spectrum mapped to [-1,1]).

    laplacian: (batch, n, n); hidden: (batch, n, d); coeffs: (order+1,)
    returns (batch, n, d).
    """
    shifted = laplacian - torch.eye(laplacian.shape[-1], device=laplacian.device, dtype=laplacian.dtype)
    t_prev = hidden
    out = coeffs[0] * t_prev
    if coeffs.shape[0] > 1:
        t_curr = torch.bmm(shifted, hidden)
        out = out + coeffs[1] * t_curr
        for r in range(2, coeffs.shape[0]):
            t_next = 2.0 * torch.bmm(shifted, t_curr) - t_prev
            out = out + coeffs[r] * t_next
            t_prev, t_curr = t_curr, t_next
    return out


class AMRModel(nn.Module):
    """M1 dual-channel spectral band-routed model (gates fixed, not learned)."""

    def __init__(self, n_bands: int = N_BANDS, cheby_order: int = CHEBY_ORDER) -> None:
        super().__init__()
        self.n_bands = n_bands
        coeffs = chebyshev_band_coefficients(n_bands, cheby_order)
        self.register_buffer("band_coeffs", torch.from_numpy(coeffs).float())
        self.register_buffer("route_alpha", torch.zeros(n_bands))
        self.encoder = GraphEncoder()
        self.pool_projection = nn.Sequential(nn.Linear(2 * HIDDEN_DIM, LATENT_DIM), nn.ReLU())
        self.phase_head = nn.Linear(LATENT_DIM, 2)
        self.zone_head = nn.Linear(LATENT_DIM, 3)
        self.centroid_head = nn.Linear(LATENT_DIM, 4)
        self.band_projections = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * HIDDEN_DIM, BAND_OUT_DIM), nn.ReLU()) for _ in range(n_bands)])
        self.mode_head = nn.Sequential(nn.Linear(n_bands * BAND_OUT_DIM, MODE_DIM), nn.ReLU())
        self.eqv_heads = nn.ModuleList([nn.Linear(BAND_OUT_DIM, 4) for _ in range(n_bands)])

    def set_route_alpha(self, alpha: torch.Tensor) -> None:
        """M1: gates are SET from the config lock, never learned."""
        if alpha.shape != (self.n_bands,):
            raise ValueError("route alpha must have shape (n_bands,)")
        self.route_alpha.copy_(alpha)

    def context_channel(self, positions: torch.Tensor, team_slots: torch.Tensor,
                        adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(positions, team_slots, adjacency)
        return self.pool_projection(team_pool(hidden, team_slots))

    def band_features(self, centered: torch.Tensor, team_slots: torch.Tensor,
                      adjacency: torch.Tensor) -> torch.Tensor:
        """Per-band pooled features r_b: (batch, n_bands, BAND_OUT_DIM)."""
        hidden = self.encoder(centered, team_slots, adjacency)
        laplacian = normalized_laplacian(adjacency)
        bands = []
        for b in range(self.n_bands):
            filtered = apply_chebyshev(laplacian, hidden, self.band_coeffs[b])
            pooled = team_pool(filtered, team_slots)
            bands.append(self.band_projections[b](pooled))
        return torch.stack(bands, dim=1)

    def mode_embedding(self, band_feats: torch.Tensor) -> torch.Tensor:
        if band_feats.dim() == 2:
            return self.mode_head(band_feats)
        return self.mode_head(band_feats.reshape(band_feats.shape[0], -1))

    def encode(self, positions: torch.Tensor, team_slots: torch.Tensor,
               adjacency: torch.Tensor, pooling: str) -> torch.Tensor:
        """Frozen-eval-compatible interface: returns flattened band features.

        `positions` is the mode-channel view (callers pass the centered view,
        exactly as the frozen T5R5/epsilon-sweep evaluation does internally).
        Only team_mean pooling is supported; the flattened output feeds
        mode_head directly, so frozen evaluate_model/pair_metrics code paths
        work unchanged on this model.
        """
        if pooling != "team_mean":
            raise ValueError("AMRModel.encode supports team_mean only")
        return self.band_features(positions, team_slots, adjacency).reshape(positions.shape[0], -1)

    def context(self, z_ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.phase_head(z_ctx), self.zone_head(z_ctx), self.centroid_head(z_ctx)


def whitened_intervention(laplacian: np.ndarray, coeffs: np.ndarray, band: int,
                          epsilon: float, rng: np.random.Generator,
                          support: np.ndarray | None = None) -> np.ndarray:
    """delta = eps * P_b(L) xi / ||P_b(L) xi||_F (spec 5.2), numpy for samplers.

    support: optional boolean node mask — the noise is restricted to the
    support BEFORE spectral filtering (support-conditioned whitening for the
    M1 band x support sampling protocol).
    """
    n = laplacian.shape[0]
    xi = rng.standard_normal((n, 2))
    if support is not None:
        xi = xi * support[:, None]
    lap_t = torch.from_numpy(laplacian[None]).float()
    xi_t = torch.from_numpy(xi[None]).float()
    coeff_t = torch.from_numpy(coeffs[band]).float()
    with torch.no_grad():
        filtered = apply_chebyshev(lap_t, xi_t, coeff_t).numpy()[0]
    norm = float(np.linalg.norm(filtered))
    if norm < 1e-12:
        return np.zeros((n, 2), dtype=np.float64)
    return epsilon * filtered / norm


def jvp_vs_finite_difference(model_fn: Any, x: torch.Tensor, delta: torch.Tensor,
                             eps: float = 1e-3) -> dict[str, float]:
    """Validate autograd JVP against central finite differences (spec 4.3).

    model_fn: R^D -> R^d (differentiable, no dropout). Returns directional
    derivative estimates and their relative discrepancy.
    """
    x = x.detach().requires_grad_(True)
    delta = delta.detach()
    _, jvp = torch.autograd.functional.jvp(model_fn, x, delta)
    with torch.no_grad():
        plus = model_fn(x + eps * delta)
        minus = model_fn(x - eps * delta)
        fd = (plus - minus) / (2.0 * eps)
    num = float(torch.norm(jvp - fd))
    den = float(torch.norm(fd)) + 1e-15
    return {"rel_error": num / den, "jvp_norm": float(torch.norm(jvp)), "fd_norm": float(torch.norm(fd))}


def routing_losses(band_feats_before: torch.Tensor, band_feats_after: torch.Tensor,
                   band_targets: torch.Tensor, eqv_heads: nn.ModuleList,
                   route_alpha: torch.Tensor) -> dict[str, torch.Tensor]:
    """L_route = sum_b alpha_b L_inv^b + (1-alpha_b) L_eqv^b (spec 7.1-7.3).

    band_feats_*: (batch, n_bands, BAND_OUT_DIM); band_targets: (batch, n_bands, K)
    true band content c_b = P_b(L) delta flattened; eqv_heads[b]: restricted
    linear head BAND_OUT_DIM -> K (spec 7.2: complexity-limited).
    """
    delta_r = band_feats_after - band_feats_before
    per_band: list[torch.Tensor] = []
    l_inv_total = delta_r.pow(2).sum(dim=-1).mean(dim=0)
    l_eqv_values = []
    for b in range(delta_r.shape[1]):
        pred = eqv_heads[b](delta_r[:, b, :])
        l_eqv_values.append((pred - band_targets[:, b, :]).pow(2).sum(dim=-1).mean())
    l_eqv_total = torch.stack(l_eqv_values)
    route = (route_alpha * l_inv_total + (1.0 - route_alpha) * l_eqv_total).sum()
    return {"route": route, "l_inv_per_band": l_inv_total, "l_eqv_per_band": l_eqv_total,
            "per_band": torch.stack(per_band) if per_band else torch.zeros(0)}
