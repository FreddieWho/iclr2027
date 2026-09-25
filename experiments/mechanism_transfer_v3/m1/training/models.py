"""Model family for the M1.2 source matrix.

Two shared implementations are used, not re-derived:

* ``CoordMLP`` from ``experiments/discovery_campaign/coord_mlp.py`` is the same
  MLP that produced the frozen ``raw``/``rich8_*``/``sixdist`` rows in
  ``reports/e832_focus/STRUCTURE_DECISION.md``. Importing it keeps the
  comparison parameter- and depth-matched to that report instead of copying an
  approximation.
* ``RepairedSegmentRho`` from ``common/source_typed.py`` is the vendored
  segment-preserving head whose lineage is
  ``experiments/f095_campaign/u01_models.py::TypedPairMLP`` plus
  ``experiments/e832_focus/structure/a_source_compare.py::RepairedRho``.

``SegmentMomentInteraction`` is new here: it consumes the continuous segment
descriptor and fuses the two segments through a shared nonlinear ``q`` applied
symmetrically. It has no sort anywhere, so endpoint role cannot be destroyed by
a bag order.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

ROOT = HERE.parents[3]
DISCOVERY_CAMPAIGN = ROOT / "experiments" / "discovery_campaign"
if str(DISCOVERY_CAMPAIGN) not in sys.path:
    sys.path.insert(0, str(DISCOVERY_CAMPAIGN))

from _v3common import source_typed  # noqa: E402
from coord_mlp import CoordMLP  # noqa: E402

from features import ARMS, in_dim  # noqa: E402

COORD_MLP_SOURCE = ROOT / "experiments" / "discovery_campaign" / "coord_mlp.py"


class SegmentMomentInteraction(nn.Module):
    """Symmetric nonlinear relation over two continuous segment descriptors.

    ``q`` is shared between the two orderings and the outputs are averaged, so
    segment swap is a symmetry by construction. The descriptor is already
    endpoint-swap invariant, so this model is exactly invariant to the source
    eight-element group without sorting any distance bag.
    """

    def __init__(self, in_dim_value: int = 10, hidden: int = 64, feat: int = 32):
        super().__init__()
        if in_dim_value % 2:
            raise ValueError("segment-moment input must split into two equal blocks")
        self.in_dim = in_dim_value
        self.half = in_dim_value // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim_value, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.feat_head = nn.Linear(hidden, feat)
        self.cls = nn.Linear(feat, 1)

    def forward(self, x, return_feat: bool = False):
        first, second = x[..., : self.half], x[..., self.half :]
        forward_order = torch.cat([first, second], dim=-1)
        backward_order = torch.cat([second, first], dim=-1)
        h = 0.5 * (self.net(forward_order) + self.net(backward_order))
        z = self.feat_head(h)
        logit = self.cls(z).squeeze(-1)
        return (logit, z) if return_feat else logit


def build_model(arm: str) -> tuple[nn.Module, int]:
    """Return (model, input dimension) for one frozen arm."""
    if arm not in ARMS:
        raise KeyError(arm)
    if arm == "segment_moment":
        return SegmentMomentInteraction(in_dim(arm)), in_dim(arm)
    if arm == "repaired_segment_rho":
        return source_typed.RepairedSegmentRho(width=64, k=source_typed.RHO_K), 8
    return CoordMLP(64, 32, in_dim=in_dim(arm)), in_dim(arm)


def count_parameters(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def macs_per_example(model: nn.Module, input_dim: int) -> int:
    """Counted Linear MACs per example, including repeated calls (segment symmetry)."""
    counted = [0]

    def hook(module, inputs, _output):
        rows = inputs[0].reshape(-1, inputs[0].shape[-1]).shape[0]
        counted[0] += int(rows) * module.in_features * module.out_features

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, nn.Linear)
    ]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(2, input_dim))
    for handle in handles:
        handle.remove()
    return counted[0] // 2


def state_digest(state: dict) -> str:
    """Content hash of a state dict in a fixed key order."""
    blob = b"".join(
        key.encode() + state[key].detach().cpu().numpy().tobytes() for key in sorted(state)
    )
    return hashlib.sha256(blob).hexdigest()


def model_card(arm: str, model: nn.Module, input_dim: int) -> dict:
    card = {
        "arm": arm,
        "class": type(model).__name__,
        "in_dim": int(input_dim),
        "n_parameters": count_parameters(model),
        "macs_per_example": macs_per_example(model, input_dim),
    }
    if arm == "repaired_segment_rho":
        card.update(
            {
                "architecture_digest": source_typed.architecture_digest(model),
                "lineage": dict(source_typed.SOURCE_LINEAGE),
                "symmetry": "segment_preserving_psi_plus_nonlinear_rho",
            }
        )
    elif arm == "segment_moment":
        card.update(
            {
                "architecture_digest": state_digest(model.state_dict()),
                "lineage": {
                    "descriptor": "midpoint and direction outer product (m, Q)",
                    "fusion": "shared nonlinear q, symmetrized over the two orders",
                },
                "symmetry": "endpoint_swap_and_segment_swap_exact",
            }
        )
    else:
        card.update(
            {
                "architecture_digest": state_digest(model.state_dict()),
                "lineage": {
                    "backbone": "experiments/discovery_campaign/coord_mlp.py::CoordMLP",
                    "source_sha256": hashlib.sha256(COORD_MLP_SOURCE.read_bytes()).hexdigest(),
                },
                "symmetry": "none_by_construction",
            }
        )
    return card
