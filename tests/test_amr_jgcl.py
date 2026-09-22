"""Unit tests for the R2 JGCL-style InfoNCE (single-geometry loss)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "p4_amr"))
import run_amr_m1_jgcl as jgcl  # noqa: E402


def test_infonce_degenerate_singleton():
    torch.manual_seed(0)
    z = torch.randn(1, 16)
    z = z / z.norm(dim=-1, keepdim=True)
    loss = float(jgcl.infonce_loss(z, z.clone(), z.clone(), tau=0.1))
    assert abs(loss) < 1e-2


def test_infonce_matches_naive_loop():
    torch.manual_seed(3)
    B, D, tau = 6, 12, 0.1
    q = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    p = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    v = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    got = float(jgcl.infonce_loss(q, p, v, tau=tau))
    cand = torch.cat([q, p, v], dim=0)
    tot = 0.0
    for i in range(B):
        sc = (q[i] @ cand.T) / tau
        pos = torch.stack([sc[B + i], sc[2 * B + i]])
        denom = torch.cat([sc[:i], sc[i + 1:]])
        tot += -(torch.logsumexp(pos, dim=0) - torch.logsumexp(denom, dim=0)).item()
    assert abs(got - tot / B) < 1e-5


def test_infonce_prefers_aligned_positives():
    torch.manual_seed(1)
    B, D = 8, 16
    q = torch.randn(B, D)
    q = q / q.norm(dim=-1, keepdim=True)
    good_p = q + 0.01 * torch.randn_like(q)
    good_v = q + 0.01 * torch.randn_like(q)
    bad = torch.randn(B, D)
    bad = bad / bad.norm(dim=-1, keepdim=True)
    l_good = float(jgcl.infonce_loss(q, good_p, good_v, tau=0.1))
    l_bad = float(jgcl.infonce_loss(q, bad, bad, tau=0.1))
    assert l_good < l_bad


def test_infonce_finite_and_deterministic():
    torch.manual_seed(2)
    q, p, v = torch.randn(8, 32), torch.randn(8, 32), torch.randn(8, 32)
    l1 = float(jgcl.infonce_loss(q, p, v, tau=0.1))
    l2 = float(jgcl.infonce_loss(q, p, v, tau=0.1))
    assert math.isfinite(l1) and l1 == l2
