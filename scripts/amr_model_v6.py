"""AMR R3 (v6): momentum-teacher routing rescue on the SHARED trunk (Track F#1).

Rationale: 15 runs prove symmetric routing gradients destroy at least one
channel on a hard-shared trunk. Every durable joint-training success in the
literature (BYOL/BGRL/DINO/data2vec) uses ASYMMETRY — momentum teacher
and/or predictor + stop-grad — which we have never tried. v6 brings the one
untried load-bearing ingredient to the exact regime that kept dying:

- SHARED GraphEncoder returns (the rescue target; decoupled v5 already works).
- Teacher = EMA copy (representation params only: encoder, band_projections,
  mode_head; NOT predictor, NOT task heads), requires_grad_(False) always.
- Student sees perturbed input (Slepian delta), predicts teacher's CLEAN
  band features through a per-band 2-layer predictor MLP (no norms, minimal).
- Loss (normalized direction, kept from v4): ||normalize(pred(r_after)) -
  normalize(sg(r_clean))||^2. VICReg-final kept as safety net.
- Tau cosine 0.99 -> 1.0 over training (F: short schedules need faster
  teachers than DINO's 0.996). Teacher updated after EVERY optimizer step.
- Triple/context losses frozen; Slepian sampling + mu logging identical;
  v4b schedule (warm-start 20 + ramp 20); W_ROUTE=1.0 (judgment middle
  between v4b's 2.0 and F#3's 0.03-0.10; documented, F range is R4 fallback).

Mode channel otherwise = v4 machinery verbatim (exact 3 bands, mean+energy
pooling, bias-free head + affine-free LayerNorm). Frozen-eval interface
preserved (encode/context_channel/mode_embedding operate on the student).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_t5r3_sanity import GraphEncoder, team_pool  # noqa: E402
from amr_model_v4 import (  # noqa: E402, F401  # re-exported for runners
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

TAU_START, TAU_END = 0.99, 1.0
PREDICTOR_HIDDEN = 128

# Representation params mirrored by the teacher (predictor + task heads excluded).
TEACHER_PARAM_PREFIXES = ("encoder.", "band_projections.", "mode_head.0.")
MASK_SIZES = (2, 3)


class AMRModelV6(nn.Module):
    """Shared-trunk student with predictor heads; teacher built externally."""

    def __init__(self, n_bands: int = N_BANDS) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.register_buffer("route_alpha", torch.zeros(n_bands))
        self.encoder = GraphEncoder()
        self.pool_projection = nn.Sequential(nn.Linear(2 * HIDDEN_DIM, LATENT_DIM), nn.ReLU())
        self.phase_head = nn.Linear(LATENT_DIM, 2)
        self.zone_head = nn.Linear(LATENT_DIM, 3)
        self.centroid_head = nn.Linear(LATENT_DIM, 4)
        self.band_projections = nn.ModuleList(
            [nn.Sequential(nn.Linear(4 * HIDDEN_DIM, BAND_OUT_DIM), nn.ReLU()) for _ in range(n_bands)])
        self.mode_head = nn.Sequential(nn.Linear(n_bands * BAND_OUT_DIM, MODE_DIM, bias=False),
                                        nn.LayerNorm(MODE_DIM, elementwise_affine=False),
                                        nn.ReLU())
        self.predictor_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(BAND_OUT_DIM, PREDICTOR_HIDDEN), nn.ReLU(),
                           nn.Linear(PREDICTOR_HIDDEN, BAND_OUT_DIM)) for _ in range(n_bands)])
        # MAE-flavored mask token: substituted for masked players' coordinates
        # in the student view (teacher always sees clean input; excluded from EMA).
        self.mask_token = nn.Parameter(torch.zeros(2))

    def set_route_alpha(self, alpha: torch.Tensor) -> None:
        if alpha.shape != (self.n_bands,):
            raise ValueError("route alpha must have shape (n_bands,)")
        self.route_alpha.copy_(alpha)

    def context_channel(self, positions: torch.Tensor, team_slots: torch.Tensor,
                        adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(positions, team_slots, adjacency)
        return self.pool_projection(team_pool(hidden, team_slots))

    def band_features(self, centered: torch.Tensor, team_slots: torch.Tensor,
                      adjacency: torch.Tensor,
                      spectral: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        hidden = self.encoder(centered, team_slots, adjacency)
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
            raise ValueError("AMRModelV6.encode supports team_mean only")
        return self.band_features(positions, team_slots, adjacency).reshape(positions.shape[0], -1)

    def context(self, z_ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.phase_head(z_ctx), self.zone_head(z_ctx), self.centroid_head(z_ctx)


def build_teacher(student: AMRModelV6) -> AMRModelV6:
    """Deepcopy student; freeze everything; eval mode (deterministic targets)."""
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def update_teacher(teacher: AMRModelV6, student: AMRModelV6, tau: float) -> None:
    """EMA over representation params only; predictor + task heads excluded."""
    with torch.no_grad():
        sdict = dict(student.named_parameters())
        for name, tparam in teacher.named_parameters():
            if name.startswith(TEACHER_PARAM_PREFIXES):
                tparam.data.mul_(tau).add_(sdict[name].data, alpha=1.0 - tau)


def koleo_spread(embeddings: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KoLeo uniform-spread regularizer (DINOv2): -mean(log(min neighbor dist)).

    Applied on L2-normalized embeddings; encourages uniform coverage, complements
    VICReg (which floors variance + kills covariance). Subordinate weight by design.
    """
    zn = torch.nn.functional.normalize(embeddings, dim=-1)
    dist = torch.cdist(zn, zn)
    dist = dist + torch.eye(dist.shape[0], device=dist.device, dtype=dist.dtype) * 1e6
    return -torch.log(dist.min(dim=-1).values + eps).mean()


def tau_schedule(step: int, total_steps: int) -> float:
    """Cosine 0.99 -> 1.0 over training (fast teacher for short schedule)."""
    import math
    frac = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return TAU_END - (TAU_END - TAU_START) * 0.5 * (1.0 + math.cos(math.pi * frac))
