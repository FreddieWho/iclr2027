#!/usr/bin/env python3
"""Unit tests for U10 oracles: correctness on hand cases, task-group
invariance, and role counterexamples (Q-vertex swap CAN change T1 label;
full-S4-style pooling would lose role info)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
from u10_oracles import tri_oracle, disk_oracle, T2_RADIUS  # noqa: E402


class TestTri(unittest.TestCase):
    def test_inside_outside(self):
        v = np.array([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]])
        self.assertEqual(tri_oracle(np.vstack([v, [[0.0, -0.1]]]), 0.0)["label"], 1)
        self.assertEqual(tri_oracle(np.vstack([v, [[0.0, 0.7]]]), 0.0)["label"], 0)

    def test_vertex_perm_invariance(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            x = rng.uniform(-0.8, 0.8, size=(4, 2))
            try:
                base = tri_oracle(x, 0.03)
            except ValueError:
                continue
            if base["ambiguous"]:
                continue
            for perm in ([1, 0, 2, 3], [0, 2, 1, 3], [2, 1, 0, 3], [1, 2, 0, 3]):
                got = tri_oracle(x[perm], 0.03)
                self.assertEqual(got["label"], base["label"])
                self.assertAlmostEqual(got["margin"], base["margin"], places=9)

    def test_query_vertex_swap_changes_task(self):
        # swapping Q with a vertex is outside the task group and changes input roles
        v = np.array([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]])
        x = np.vstack([v, [[0.0, -0.1]]])  # Q inside
        self.assertEqual(tri_oracle(x, 0.0)["label"], 1)
        y = x[[3, 1, 2, 0]]  # Q<->V1: new query is old V1 (a vertex, on boundary/corner)
        got = tri_oracle(y, 0.0)
        # new Q is a triangle corner of the new triangle -> on boundary or outside
        self.assertFalse(got["label"] == 1 and not got["ambiguous"])


class TestDisk(unittest.TestCase):
    def test_hit_miss(self):
        a = np.array([[-0.6, 0.0], [0.6, 0.0], [0.0, 0.1]])
        self.assertEqual(disk_oracle(a, 0.0)["label"], 1)
        b = np.array([[-0.6, 0.0], [0.6, 0.0], [0.0, 0.6]])
        self.assertEqual(disk_oracle(b, 0.0)["label"], 0)

    def test_ab_swap_invariance(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            x = rng.uniform(-0.8, 0.8, size=(3, 2))
            base = disk_oracle(x, 0.03)
            if base["ambiguous"]:
                continue
            got = disk_oracle(x[[1, 0, 2]], 0.03)
            self.assertEqual(got["label"], base["label"])
            self.assertAlmostEqual(got["margin"], base["margin"], places=9)

    def test_margin_is_distance_minus_radius(self):
        x = np.array([[-0.6, 0.0], [0.6, 0.0], [0.0, 0.6]])
        got = disk_oracle(x, 0.0)
        self.assertAlmostEqual(got["margin"], abs(0.6 - T2_RADIUS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
