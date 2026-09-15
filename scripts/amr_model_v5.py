"""AMR R1 (v5): fully decoupled dual-tower (Track D Design-1 lite).

Rationale (SYNTHESIS_5ROUNDS.md, D-20260905-P4-012):
- 15 runs prove hard sharing irreconcilable; E proves conflict is
  directional (magnitude balancers dead end).
- ctx tower (raw view + frozen context loss) and mode tower (centered view
  + v4 machinery) share ZERO parameters: context health becomes
  constructional, and v5 vs v4b isolates EXACTLY one variable
  (encoder shared -> split).
- Deliberately NO gate/bidirectional stitch in R1: the frozen eval
  interface (pair_scores passes band feats only) requires train/eval
  isomorphism; learned sharing deferred to R3+ follow-up if R1 passes.
- Mode side reuses the v4 recipe verbatim: exact 3-band projectors,
  mean+energy pooling, bias-free mode head + affine-free LayerNorm,
  restricted eqv heads, Slepian sampling + normalized direction recovery
  + stop-grad (runner), VICReg-final (runner).

Capacity ~2x trunk is the honest floor for encoding-level conflict
(Track D finding 6); recorded in lock, not apologized for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_t5r3_sanity import GraphEncoder, team_pool  # noqa: E402
from amr_model_v4 import (  # noqa: E402
    BAND_EDGES,
    BAND_OUT_DIM,
    EQV_TARGET_DIM,
    HIDDEN_DIM,
    LATENT_DIM,
    MODE_DIM,
    N_BANDS,
    active_triplet_fraction,
    band_index,
    exact_band_project,
    normalized_laplacian,
    slepian_intervention,
    spectral_factors,
    vicreg_var_cov,
)


class AMRModelV5(nn.Module):
    """Decoupled dual-tower: encoder_ctx serves context only,
    encoder_mode serves the intrinsic/routing pathway only."""

    def __init__(self, n_bands: int = N_BANDS) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.register_buffer("route_alpha", torch.zeros(n_bands))
        self.encoder_ctx = GraphEncoder()
        self.encoder_mode = GraphEncoder()
        self.pool_projection = nn.Sequential(nn.Linear(2 * HIDDEN_DIM, LATENT_DIM), nn.ReLU())
        self.phase_head = nn.Linear(LATENT_DIM, 2)
        self.zone_head = nn.Linear(LATENT_DIM, 3)
        self.centroid_head = nn.Linear(LATENT_DIM, 4)
        self.band_projections = nn.ModuleList(
            [nn.Sequential(nn.Linear(4 * HIDDEN_DIM, BAND_OUT_DIM), nn.ReLU()) for _ in range(n_bands)])
        self.mode_head = nn.Sequential(nn.Linear(n_bands * BAND_OUT_DIM, MODE_DIM, bias=False),
                                        nn.LayerNorm(MODE_DIM, elementwise_affine=False),
                                        nn.ReLU())
        self.eqv_heads = nn.ModuleList([nn.Linear(BAND_OUT_DIM, EQV_TARGET_DIM) for _ in range(n_bands)])

    def set_route_alpha(self, alpha: torch.Tensor) -> None:
        if alpha.shape != (self.n_bands,):
            raise ValueError("route alpha must have shape (n_bands,)")
        self.route_alpha.copy_(alpha)

    def context_channel(self, positions: torch.Tensor, team_slots: torch.Tensor,
                        adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder_ctx(positions, team_slots, adjacency)
        return self.pool_projection(team_pool(hidden, team_slots))

    def band_features(self, centered: torch.Tensor, team_slots: torch.Tensor,
                      adjacency: torch.Tensor,
                      spectral: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        hidden = self.encoder_mode(centered, team_slots, adjacency)
        if spectral is None:
            evals, evecs = spectral_factors(adjacency)
        else:
            evals, evecs = spectral
        bands = []
        for b in range(self.n_bands):
            filtered = exact_band_project(evecs, band_index(evals), b, hidden)
            pooled = torch.cat([team_pool(filtered, team_slots),
                                team_pool(filtered.pow(2), team_slots)], dim=-1)
            bands.append(self.band_projections[b](pooled))
        return torch.stack(bands, dim=1)

    def mode_embedding(self, band_feats: torch.Tensor) -> torch.Tensor:
        if band_feats.dim() == 2:
            return self.mode_head(band_feats)
        return self.mode_head(band_feats.reshape(band_feats.shape[0], -1))

    def encode(self, positions: torch.Tensor, team_slots: torch.Tensor,
               adjacency: torch.Tensor, pooling: str) -> torch.Tensor:
        if pooling != "team_mean":
            raise ValueError("AMRModelV5.encode supports team_mean only")
        return self.band_features(positions, team_slots, adjacency).reshape(positions.shape[0], -1)

    def context(self, z_ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.phase_head(z_ctx), self.zone_head(z_ctx), self.centroid_head(z_ctx)

    def tower_params(self) -> tuple[set[int], set[int]]:
        """id() sets of ctx-tower vs mode-side params (insulation audit)."""
        ctx = {id(p) for p in list(self.encoder_ctx.parameters())
               + list(self.pool_projection.parameters()) + list(self.phase_head.parameters())
               + list(self.zone_head.parameters()) + list(self.centroid_head.parameters())}
        mode = {id(p) for p in self.parameters()} - ctx
        return ctx, mode
