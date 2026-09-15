"""AMR R2 (v5cap): CAP control on the decoupled dual-tower (N4, spec M1).

Matched-capacity control for v5: SAME AMRModelV5 towers, SAME schedule
(warm-start + ramp, mechanism-slot weight 2.0), SAME Slepian intervention
distribution, SAME triplet/VICReg/context. The ONLY variable vs v5 is the
mechanism objective: transformation prediction

    L_CAP = || B (z_mode(x+delta) - z_mode(x)) - delta ||^2 / eps^2

with B a restricted linear head (64->40), INSTEAD of Slepian band routing.

Faithfulness notes (spec section 8 M0):
- No stop-grad on either branch: CAP predicts from the difference, both
  sides need grads. If CAP collapses too, that is itself evidence
  (predicted by Track B symmetry analysis) and will be reported, not hidden.
- mu is still logged (sampling distribution identical to v5 route).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from amr_model_v5 import AMRModelV5  # noqa: E402
from amr_model_v4 import (  # noqa: E402, F401  # re-exported for runners
    BAND_EDGES,
    active_triplet_fraction,
    slepian_intervention,
    vicreg_var_cov,
)

CAP_OUT_DIM = 40  # flattened (20, 2) displacement


class AMRModelV5CAP(AMRModelV5):
    """v5 towers + restricted linear CAP readout (no other changes)."""

    def __init__(self, n_bands: int = 3) -> None:
        super().__init__(n_bands=n_bands)
        self.cap_head = nn.Linear(64, CAP_OUT_DIM)
