#!/usr/bin/env python3
"""Unit tests for the D06 channel oracle (geometry only, no tracking yet)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
from d06_oracle import channel_label, channel_margin, point_segment_dist  # noqa: E402

R, E = 0.5, 1.0


class TestOracle(unittest.TestCase):
    def test_open_without_defenders(self):
        self.assertEqual(channel_label([0, 0], [10, 0], [], R, E)["label"], 1)

    def test_midpoint_block(self):
        out = channel_label([0, 0], [10, 0], [[5, 0.1]], R, E)
        self.assertEqual(out["label"], 0)
        self.assertAlmostEqual(out["margin"], 0.1 - R)

    def test_far_defender_open(self):
        out = channel_label([0, 0], [10, 0], [[5, 6]], R, E)
        self.assertEqual(out["label"], 1)
        self.assertAlmostEqual(out["margin"], 6 - R)

    def test_endpoint_exclusion(self):
        # A defender standing on the receiver is excluded, channel stays open.
        out = channel_label([0, 0], [10, 0], [[10, 0]], R, E)
        self.assertEqual(out["label"], 1)

    def test_defender_order_invariance(self):
        ds = [[5, 0.1], [2, 3], [8, -4]]
        a = channel_margin([0, 0], [10, 0], ds, R, E)
        b = channel_margin([0, 0], [10, 0], ds[::-1], R, E)
        self.assertAlmostEqual(a, b)

    def test_segment_endpoints_projection(self):
        # Defender beyond r projects to the endpoint, not the infinite line.
        d = point_segment_dist([14, 0], [0, 0], [10, 0])
        self.assertAlmostEqual(d, 4.0)

    def test_degenerate_segment(self):
        d = point_segment_dist([1, 1], [2, 2], [2, 2])
        self.assertAlmostEqual(d, float(np.sqrt(2)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
