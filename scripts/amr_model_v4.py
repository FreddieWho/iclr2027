"""AMR M1 redesign v4: exact spectral basis + collapse-proof routing.

Implements research routes 1+2 (SYNTHESIS_TOP3.md, D-20260905-P4-007):

Route 1 (exact spectral basis):
- EXACT batched eigendecomposition projectors replace Chebyshev-5
  (Track A+C: at n=20 exact is cheaper and error-free; old band 5
  [1.67,2.0] was empty on 99% of snapshots).
- 3 density-equalized bands frozen from the dev eigenvalue measurement
  (2000 train snapshots): [0,1.0) / [1.0,1.3) / [1.3,2.0].
- Slepian-vector intervention sampling (Track C): per (support S, band b),
  delta is drawn from the span of the top-2 eigenvectors of the
  concentration operator C = P_b S P_b, with concentration mu reported.
  Slepian-matched nulls are eval diagnostics (runner), not training.

Route 2 (final-embedding protection):
- VICReg-style variance hinge + within-mode covariance decorrelation on
  the FINAL 64-dim mode embedding (the layer the triplet actually sees),
  not on intermediates (Track B: intermediate floors are bypassed downstream;
  cross-channel covariance says nothing about within-mode rank).
- Stop-gradient asymmetry on the route branch (Track B/SimSiam): the
  perturbed branch predicts a fixed clean target; the restricted eqv head
  is the predictor analog. Clean branch carries no route gradient.
- Frozen pair_loss already L2-normalizes internally; v4 adds an
  active-triplet-fraction guard (runner early-stop) and a singular-spectrum
  diagnostic (Jing et al.).

Interface: encode() returns flattened band features and mode_head maps
them, so the frozen T5R5/epsilon-sweep evaluation machinery is reusable
verbatim (encode does on-the-fly exact projection; training passes cached
spectral factors for speed — both exact, hence consistent).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_t5r3_sanity import GraphEncoder, team_pool  # noqa: E402

HIDDEN_DIM = 128
LATENT_DIM = 128
N_BANDS = 3
BAND_EDGES = (0.0, 1.0, 1.3, 2.0)
BAND_OUT_DIM = 32
MODE_DIM = 64
EQV_TARGET_DIM = 4  # team_pool over (20, 2) -> (4,)


def normalized_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    degree = adjacency.sum(dim=-1).clamp_min(1.0)
    inv_sqrt = degree.pow(-0.5)
    norm = adjacency * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)
    eye = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype)
    return eye - norm


def spectral_factors(adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact eigendecomposition of the normalized Laplacian (batched).

    Returns (evals, evecs) with ascending eigenvalues. Trivial cost at n=20.
    """
    laplacian = normalized_laplacian(adjacency)
    evals, evecs = torch.linalg.eigh(laplacian)
    return evals, evecs


def band_index(evals: torch.Tensor, edges: tuple[float, ...] = BAND_EDGES) -> torch.Tensor:
    """Integer band id per eigenvalue; last band is closed on the right."""
    idx = torch.bucketize(evals, torch.tensor(edges[1:-1], device=evals.device, dtype=evals.dtype))
    return idx.clamp_max(len(edges) - 2)


def exact_band_project(evecs: torch.Tensor, band_ids: torch.Tensor, band: int,
                       hidden: torch.Tensor) -> torch.Tensor:
    """U_b (U_b^T H): exact band-limited projection of node features."""
    out = torch.zeros_like(hidden)
    for i in range(hidden.shape[0]):
        mask = band_ids[i] == band
        if mask.any():
            ub = evecs[i][:, mask]
            out[i] = ub @ (ub.T @ hidden[i])
    return out


def slepian_intervention(evecs: np.ndarray, evals: np.ndarray, band: int,
                         support: np.ndarray, epsilon: float,
                         rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Slepian-vector intervention for (support S, band b).

    Concentration operator C = P_b S P_b with P_b = U_b U_b^T exact.
    Draws each of the 2 coordinate fields from span{psi_1, psi_2} with
    random Gaussian weights and random signs; delta normalized to epsilon.
    Returns (delta (20, 2), mu_1 top concentration ratio).
    """
    lo, hi = BAND_EDGES[band], BAND_EDGES[band + 1]
    mask = (evals >= lo) & (evals < hi if band < N_BANDS - 1 else evals <= hi)
    if not mask.any():
        return np.zeros((evecs.shape[0], 2)), 0.0
    ub = evecs[:, mask]
    proj = ub @ ub.T
    s = support.astype(np.float64)
    conc = proj @ (s[:, None] * proj)
    mu, psi = np.linalg.eigh(conc)
    if mu[-1] < 1e-8:
        return np.zeros((evecs.shape[0], 2)), 0.0
    vecs = psi[:, -2:] if psi.shape[1] > 1 else np.concatenate([psi[:, -1:], psi[:, -1:]], axis=1)
    mus = mu[-2:] if mu.shape[0] > 1 else np.array([mu[-1], mu[-1]])
    fields = []
    for _ in range(2):
        w = rng.standard_normal(2)
        psi_draw = w[0] * vecs[:, 0] + w[1] * vecs[:, 1]
        norm = float(np.linalg.norm(psi_draw))
        fields.append(np.zeros_like(psi_draw) if norm < 1e-12 else psi_draw / norm)
    delta = np.stack(fields, axis=1)
    norm = float(np.linalg.norm(delta))
    mu_out = float(min(max(mus[0], 0.0), 1.0))
    if norm < 1e-12:
        return np.zeros_like(delta), 0.0
    return (epsilon * delta / norm).astype(np.float64), mu_out


def vicreg_var_cov(embeddings: torch.Tensor, gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg-style variance hinge + within-embedding covariance (final layer).

    embeddings: (batch, dim) raw (unnormalized). Returns (var_loss, cov_loss)
    with mean reductions (scale-free across widths).
    """
    std = embeddings.std(dim=0)
    var_loss = torch.relu(gamma - std).mean()
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / max(embeddings.shape[0] - 1, 1)
    dim = embeddings.shape[1]
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / (dim * (dim - 1))
    return var_loss, cov_loss


class AMRModelV4(nn.Module):
    """M1-redesign dual-channel model on the exact 3-band spectral basis."""

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
        # NOTE (init-scale): band features are ~1e-2 scale, far below a default
        # Linear bias (~1e-1); with bias the bias pattern dominates, all samples
        # map near-identically, LayerNorm normalizes the bias (constant) and the
        # head is dead at init (measured per-dim std 0.005). bias=False lets
        # LayerNorm amplify the signal instead. LayerNorm is affine=False so no
        # learnable parameter can re-collapse the scale.
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
        hidden = self.encoder(positions, team_slots, adjacency)
        return self.pool_projection(team_pool(hidden, team_slots))

    def band_features(self, centered: torch.Tensor, team_slots: torch.Tensor,
                      adjacency: torch.Tensor,
                      spectral: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """Per-band pooled features (batch, n_bands, BAND_OUT_DIM).

        Pooling is mean+energy concatenated per team: team_mean(P_b H)
        preserves signed/DC content, team_mean((P_b H)^2) preserves band
        energy that mean-pooling cancels on oscillatory bands (measured:
        bands 1-2 mean-pooled std ~0.006 vs band 0 ~0.066 at init).
        spectral: optional cached (evals, evecs) aligned to the batch;
        if None, exact factors are computed on the fly (eval path).
        """
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
            raise ValueError("AMRModelV4.encode supports team_mean only")
        return self.band_features(positions, team_slots, adjacency).reshape(positions.shape[0], -1)

    def context(self, z_ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.phase_head(z_ctx), self.zone_head(z_ctx), self.centroid_head(z_ctx)


def active_triplet_fraction(query_mode: torch.Tensor, positive_mode: torch.Tensor,
                            negative_mode: torch.Tensor, margin: float) -> float:
    """Fraction of triplets with positive loss (mining guard; Track B Fix 3)."""
    with torch.no_grad():
        pos = (query_mode * positive_mode).sum(dim=-1)
        neg = (query_mode * negative_mode).sum(dim=-1)
        return float(((margin - pos + neg) > 0).float().mean())
