#!/usr/bin/env python3
"""Unit tests for U10 task models: exact task-group invariance + capacity
accounting + role counterexample (Q-vertex / center-endpoint swaps change output)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
from u10_models import TypedTriMLP, TypedDiskMLP, count_params  # noqa: E402
from coord_mlp import CoordMLP  # noqa: E402


class TestTypedT(unittest.TestCase):
    def test_tri_vertex_invariance(self):
        torch.manual_seed(0)
        m = TypedTriMLP().eval()
        rng = np.random.default_rng(3)
        x = torch.from_numpy(rng.normal(size=(5, 8)).astype(np.float32))
        ref = m(x).detach()
        for perm in ([1, 0, 2, 3], [0, 2, 1, 3], [2, 1, 0, 3], [1, 2, 0, 3]):
            xg = x.reshape(5, 4, 2)[:, perm, :].reshape(5, 8)
            np.testing.assert_allclose(m(xg).detach().numpy(), ref.numpy(),
                                       rtol=1e-5, atol=1e-6)

    def test_tri_query_swap_changes_output(self):
        torch.manual_seed(1)
        m = TypedTriMLP().eval()
        rng = np.random.default_rng(4)
        diffs = 0
        for _ in range(20):
            x = rng.normal(size=(8,))
            y = x.copy()
            y[0:2], y[6:8] = x[6:8].copy(), x[0:2].copy()  # Q<->V1: outside group
            with torch.no_grad():
                if abs(float(m(torch.from_numpy(x[None].astype(np.float32)))) -
                       float(m(torch.from_numpy(y[None].astype(np.float32))))) > 1e-4:
                    diffs += 1
        self.assertGreater(diffs, 0)

    def test_disk_ab_invariance(self):
        torch.manual_seed(2)
        m = TypedDiskMLP().eval()
        rng = np.random.default_rng(5)
        x = torch.from_numpy(rng.normal(size=(7, 6)).astype(np.float32))
        ref = m(x).detach()
        xg = x.reshape(7, 3, 2)[:, [1, 0, 2], :].reshape(7, 6)
        np.testing.assert_allclose(m(xg).detach().numpy(), ref.numpy(),
                                   rtol=1e-5, atol=1e-6)

    def test_param_counts_recorded(self):
        for name, m, raw_in in (("tri", TypedTriMLP(), 8), ("disk", TypedDiskMLP(), 6)):
            raw = count_params(CoordMLP(64, 32, in_dim=raw_in))
            typed = count_params(m)
            print(f"{name}: raw={raw} typed={typed} rel={abs(typed-raw)/raw:.2%}")
            self.assertLess(abs(typed - raw) / raw, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
