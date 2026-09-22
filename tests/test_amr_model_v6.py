"""Unit tests for AMR R3 (v6) momentum-teacher rescue."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "p4_amr"))
import amr_model_v6 as v6  # noqa: E402


def _inputs():
    torch.manual_seed(0)
    positions = torch.randn(3, 20, 2)
    team = torch.tensor([[0] * 10 + [1] * 10] * 3)
    adj = (torch.rand(3, 20, 20) > 0.7).float()
    adj = torch.triu(adj, diagonal=1)
    return positions, team, adj + adj.transpose(1, 2)


def test_teacher_equals_student_at_init_and_frozen():
    torch.manual_seed(0)
    student = v6.AMRModelV6()
    teacher = v6.build_teacher(student)
    for (ns, ps), (nt, pt) in zip(student.named_parameters(), teacher.named_parameters()):
        assert ns == nt and torch.equal(ps, pt)
    assert all(not p.requires_grad for p in teacher.parameters())


def test_ema_moves_representation_only():
    torch.manual_seed(1)
    student = v6.AMRModelV6()
    teacher = v6.build_teacher(student)
    with torch.no_grad():
        for p in student.parameters():
            p.add_(0.1)
    before = {n: p.clone() for n, p in teacher.named_parameters()}
    v6.update_teacher(teacher, student, tau=0.9)
    after = dict(teacher.named_parameters())
    moved, fixed = [], []
    for n in before:
        (moved if not torch.equal(before[n], after[n]) else fixed).append(n)
    assert moved, "nothing moved"
    assert all(n.startswith(v6.TEACHER_PARAM_PREFIXES) for n in moved), moved
    assert any("predictor_heads" in n for n in fixed)
    assert any("phase_head" in n or "zone_head" in n for n in fixed)
    # exact EMA arithmetic on one param
    assert torch.allclose(after[moved[0]], 0.9 * before[moved[0]] + 0.1 * dict(student.named_parameters())[moved[0]])


def test_tau_schedule_endpoints():
    assert abs(v6.tau_schedule(0, 1000) - 0.99) < 1e-9
    assert abs(v6.tau_schedule(1000, 1000) - 1.0) < 1e-9
    mid = v6.tau_schedule(500, 1000)
    assert 0.99 < mid < 1.0


def test_predictor_path_shapes_and_grads():
    positions, team, adj = _inputs()
    student = v6.AMRModelV6()
    feats = student.band_features(positions, team, adj)
    for b in range(3):
        assert student.predictor_heads[b](feats[:, b, :]).shape == (3, v6.BAND_OUT_DIM)
    student.zero_grad()
    loss = student.predictor_heads[1](feats[:, 1, :]).pow(2).sum()
    loss.backward()
    enc_grads = [p.grad for p in student.encoder.parameters()]
    assert any(g is not None and float(g.abs().sum()) > 0 for g in enc_grads)


def test_eval_interface_shapes():
    positions, team, adj = _inputs()
    model = v6.AMRModelV6()
    z_ctx = model.context_channel(positions, team, adj)
    feats = model.band_features(positions, team, adj)
    assert z_ctx.shape == (3, v6.LATENT_DIM)
    assert feats.shape == (3, v6.N_BANDS, v6.BAND_OUT_DIM)
    z = model.encode(positions, team, adj, "team_mean")
    assert z.shape == (3, v6.N_BANDS * v6.BAND_OUT_DIM)
    assert torch.allclose(model.mode_embedding(z), model.mode_embedding(feats), atol=1e-6)


def test_mask_token_excluded_from_teacher_ema():
    torch.manual_seed(2)
    student = v6.AMRModelV6()
    assert student.mask_token.shape == (2,)
    assert not any(n == "mask_token" or n.startswith("mask_token") for n, _ in
                   [(n, p) for n, p in student.named_parameters()
                    if n.startswith(v6.TEACHER_PARAM_PREFIXES)])
    teacher = v6.build_teacher(student)
    with torch.no_grad():
        student.mask_token.add_(1.0)
    v6.update_teacher(teacher, student, tau=0.0)
    assert torch.equal(teacher.mask_token, torch.zeros(2))


def test_masked_forward_differs_and_koleo_sane():
    torch.manual_seed(3)
    model = v6.AMRModelV6()
    centered = torch.randn(4, 20, 2)
    team = torch.tensor([[0] * 10 + [1] * 10] * 4)
    adj = (torch.rand(4, 20, 20) > 0.7).float()
    masked = centered.clone()
    masked[:, :2, :] = model.mask_token
    with torch.no_grad():
        f_clean = model.band_features(centered, team, adj)
        f_mask = model.band_features(masked, team, adj)
    assert not torch.allclose(f_clean, f_mask)
    live = torch.nn.functional.normalize(torch.randn(32, 64), dim=-1)
    k_live = float(v6.koleo_spread(live))
    assert abs(k_live) < 10.0
    dead = torch.nn.functional.normalize(torch.ones(32, 64), dim=-1)
    k_dead = float(v6.koleo_spread(dead))
    assert k_dead > k_live
