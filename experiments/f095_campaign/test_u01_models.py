#!/usr/bin/env python3
"""Tests for U01 models: exact G8 invariance (typed) + param accounting."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
from symmetry import GROUP, apply  # noqa: E402
from coord_mlp import CoordMLP  # noqa: E402
from u01_models import TypedPairMLP, SixDistMLP, ConcatMLP, count_params  # noqa: E402


class TestU01(unittest.TestCase):
    def test_typed_exact_invariance(self):
        torch.manual_seed(0)
        m = TypedPairMLP(64).eval()
        rng = np.random.default_rng(3)
        for _ in range(5):
            x = torch.from_numpy(rng.normal(size=(7, 8)).astype(np.float32))
            ref = m(x).detach()
            for g in GROUP:
                xg = torch.from_numpy(apply(x.numpy(), g).astype(np.float32))
                np.testing.assert_allclose(m(xg).detach().numpy(), ref.numpy(),
                                           rtol=1e-5, atol=1e-6)

    def test_typed_keeps_pairing_info(self):
        # Changing AB/CD pairing must be able to change the representation.
        torch.manual_seed(1)
        m = TypedPairMLP(64).eval()
        rng = np.random.default_rng(5)
        diffs = 0
        for _ in range(20):
            x = rng.normal(size=(8,))
            y = x.copy()
            y[0:2], y[4:6] = x[4:6].copy(), x[0:2].copy()  # A<->C: outside G8
            with torch.no_grad():
                if abs(float(m(torch.from_numpy(x[None].astype(np.float32)))) -
                       float(m(torch.from_numpy(y[None].astype(np.float32))))) > 1e-4:
                    diffs += 1
        self.assertGreater(diffs, 0)

    def test_param_counts_near_raw_cells(self):
        for w, f in ((64, 32), (256, 64), (512, 64)):
            raw = count_params(CoordMLP(w, f, in_dim=8))
            typed = count_params(TypedPairMLP(w))
            sixd = count_params(SixDistMLP(w, f))
            conc = count_params(ConcatMLP(w, f))
            for name, v in (("typed", typed), ("sixdist", sixd), ("concat", conc)):
                rel = abs(v - raw) / raw
                print(f"w{w}: raw={raw} {name}={v} rel={rel:.2%}")
                self.assertLess(rel, 0.25, (w, name, v, raw))


if __name__ == "__main__":
    unittest.main(verbosity=2)
