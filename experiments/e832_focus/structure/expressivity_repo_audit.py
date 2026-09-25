#!/usr/bin/env python3
"""A0: checkerboard expressivity audit on the repository classes, not the pack mirror.

Imports TypedPairMLP and G8SetMLP from the frozen source files. Does not load
checkpoints, banks, or sealed pools. The mixed-difference identity is the proof.
A short fit is only a demonstration that the identity is not an initialization
accident. The nonlinear-rho wrapper is the first repair's expressivity check on
these four cases, not a claim about the project bank.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "e1a933_review"))

from u01_models import TypedPairMLP  # noqa: E402
from leads_l014_l015_l006 import G8SetMLP  # noqa: E402
from u10_models import TypedDiskMLP, TypedTriMLP  # noqa: E402

OUT = ROOT / "artifacts" / "e832_focus" / "structure" / "EXPRESSIVITY_REPO_AUDIT.json"


def checkerboard():
    s = [np.array([[-0.7, 0.4], [0.7, 0.4]]), np.array([[-0.7, -0.4], [0.7, -0.4]])]
    t = [np.array([[0.1, 0.2], [0.1, 0.6]]), np.array([[0.1, -0.6], [0.1, -0.2]])]
    xs = np.stack([np.concatenate([a, b]) for a in s for b in t]).astype(np.float64)
    return xs, np.array([[1, 0], [0, 1]], dtype=np.int64)


def mixed_residual(logits):
    z = np.asarray(logits, dtype=np.float64).reshape(2, 2)
    return float(z[0, 0] + z[1, 1] - z[0, 1] - z[1, 0])


def threshold_accuracy(logits, ys):
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = ys.reshape(-1)
    best = 0.0
    for t in np.linspace(z.min() - 1.0, z.max() + 1.0, 401):
        best = max(best, float(((z > t) == y).mean()))
    return best


class NonlinearRho(nn.Module):
    """Same segment encoders as TypedPairMLP; only the final rho becomes nonlinear."""

    def __init__(self, width=64, k=16):
        super().__init__()
        self.base = TypedPairMLP(width)
        sh = self.base.rho.in_features
        self.base.rho = nn.Sequential(nn.Linear(sh, k), nn.GELU(), nn.Linear(k, 1))

    def forward(self, x):
        return self.base(x).reshape(-1)


def fit(model, xs, ys, steps=800, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys.reshape(-1), dtype=torch.float64)
    loss_name = "bce_with_logits"
    last = None
    for _ in range(steps):
        opt.zero_grad()
        logit = model(x).reshape(-1)
        last = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
        last.backward()
        opt.step()
    with torch.no_grad():
        z = model(x).detach().cpu().numpy()
    return {
        "steps": steps,
        "lr": lr,
        "loss": loss_name,
        "final_bce": float(last.detach()),
        "mixed_residual": mixed_residual(z),
        "best_threshold_accuracy": threshold_accuracy(z, ys),
        "logits": [float(v) for v in z.reshape(-1)],
    }


def main():
    torch.set_num_threads(1)
    xs, ys = checkerboard()
    x = torch.tensor(xs, dtype=torch.float64)
    residuals = {"TypedPairMLP": [], "G8SetMLP": []}
    for seed in range(10):
        torch.manual_seed(seed)
        for name, model in (("TypedPairMLP", TypedPairMLP(64)), ("G8SetMLP", G8SetMLP(32))):
            model = model.double().eval()
            with torch.no_grad():
                residuals[name].append(mixed_residual(model(x).numpy()))

    torch.manual_seed(0)
    additive_fit = fit(TypedPairMLP(64).double(), xs, ys)
    torch.manual_seed(0)
    g8_fit = fit(G8SetMLP(32).double(), xs, ys)
    torch.manual_seed(0)
    repaired_fit = fit(NonlinearRho(64, 16).double(), xs, ys)

    # Role-concat fusion is a different function class. Record the constructor
    # difference; do not apply the two-segment additive witness to these tasks.
    fusion = {
        "TypedPairMLP": "psi(h_ab)+psi(h_cd), then linear rho",
        "G8SetMLP": "psi(h_ab)+psi(h_cd), then linear rho",
        "TypedTriMLP": "psi(cat(sum_vertices, h_q)), then linear rho",
        "TypedDiskMLP": "psi(cat(sum_ab, h_c)), then linear rho",
    }
    report = {
        "audit_type": "repository-class algebraic witness; no bank, no checkpoint, no sealed pool",
        "basis_commit": "e832887c938948b23e8783719d493f2b9b8c3a17",
        "threads": 1,
        "oracle_matrix": ys.tolist(),
        "coordinates": xs.reshape(4, 4, 2).tolist(),
        "random_init_mixed_residual_max": {k: max(abs(v) for v in vals) for k, vals in residuals.items()},
        "random_init_mixed_residual_max_raw": {k: max(vals) for k, vals in residuals.items()},
        "fit_demonstration": {
            "TypedPairMLP_linear_rho": additive_fit,
            "G8SetMLP_linear_rho": g8_fit,
            "TypedPairMLP_nonlinear_rho": repaired_fit,
        },
        "fusion": fusion,
        "not_implicated": ["TypedTriMLP", "TypedDiskMLP"],
        "class_objects_instantiated": [
            TypedPairMLP.__module__ + ".TypedPairMLP",
            G8SetMLP.__module__ + ".G8SetMLP",
            TypedTriMLP.__module__ + ".TypedTriMLP",
            TypedDiskMLP.__module__ + ".TypedDiskMLP",
        ],
        "limitations": [
            "Four-case success of nonlinear rho is expressivity, not bank generalization.",
            "Identity does not prove every historical miss came from this restriction.",
            "TypedTri/TypedDisk concatenate roles before a nonlinearity; this witness does not apply.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "out": str(OUT),
        "residual_max": report["random_init_mixed_residual_max"],
        "additive_acc": additive_fit["best_threshold_accuracy"],
        "g8_acc": g8_fit["best_threshold_accuracy"],
        "repaired_acc": repaired_fit["best_threshold_accuracy"],
        "repaired_residual": repaired_fit["mixed_residual"],
    }, indent=2))


if __name__ == "__main__":
    main()
