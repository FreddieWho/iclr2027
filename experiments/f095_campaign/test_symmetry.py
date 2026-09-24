#!/usr/bin/env python3
"""Unit tests for the D02 task symmetry group (design-level, no training).

Covers the route's required function-level checks: group closure, inverses,
oracle-label invariance, same-quartet edit consistency, and the S4
counterexample (untyped four-point pooling would lose pairing info).
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
sys.path.insert(0, str(ROOT / "docs" / "iclr2027_discovery_campaign_20260917" / "core"))
from symmetry import GROUP, IDENTITY, apply, group_average_logits, invert  # noqa: E402
from relations import segment_relation  # noqa: E402


def compose(p, q):
    return tuple(p[q[i]] for i in range(4))


class TestGroup(unittest.TestCase):
    def test_eight_elements_closed_with_inverses(self):
        self.assertEqual(len(GROUP), 8)
        self.assertIn(IDENTITY, GROUP)
        s = set(GROUP)
        for g in GROUP:
            self.assertIn(invert(g), s)
            for h in GROUP:
                self.assertIn(compose(g, h), s)

    def test_generators_present_pairing_swap_absent(self):
        self.assertIn((1, 0, 2, 3), GROUP)  # A<->B
        self.assertIn((0, 1, 3, 2), GROUP)  # C<->D
        self.assertIn((2, 3, 0, 1), GROUP)  # AB<->CD
        self.assertNotIn((2, 1, 0, 3), GROUP)  # A<->C breaks pairing

    def test_oracle_invariant_under_group(self):
        rng = np.random.default_rng(7)
        n_checked = 0
        for _ in range(400):
            x = rng.uniform(-0.8, 0.8, size=(4, 2))
            base = segment_relation(x)
            if base["ambiguous"]:
                continue
            for g in GROUP:
                got = segment_relation(apply(x, g))
                self.assertEqual(got["label"], base["label"])
                self.assertAlmostEqual(got["margin"], base["margin"], places=9)
            n_checked += 1
        self.assertGreater(n_checked, 200)

    def test_s4_counterexample_pairing_matters(self):
        # An A<->C swap (outside G8) can flip the oracle label: pairing info
        # is load-bearing, so untyped four-point pooling is an invalid baseline.
        rng = np.random.default_rng(11)
        found = None
        for _ in range(20000):
            x = rng.uniform(-0.8, 0.8, size=(4, 2))
            base = segment_relation(x)
            if base["ambiguous"]:
                continue
            swapped = segment_relation(apply(x, (2, 1, 0, 3)))
            if not swapped["ambiguous"] and swapped["label"] != base["label"]:
                found = (x, base["label"], swapped["label"])
                break
        self.assertIsNotNone(found, "no A<->C label flip found in 20k draws")

    def test_same_quartet_edit_consistency(self):
        # perm(x + e) == perm(x) + perm(e): one group element for all states.
        rng = np.random.default_rng(13)
        x = rng.uniform(-0.8, 0.8, size=(4, 2))
        e = rng.uniform(-0.1, 0.1, size=(4, 2))
        for g in GROUP:
            np.testing.assert_allclose(apply(x + e, g), apply(x, g) + apply(e, g))

    def test_apply_inverse_roundtrip_and_shapes(self):
        rng = np.random.default_rng(17)
        x = rng.uniform(-0.8, 0.8, size=(5, 4, 2))
        for g in GROUP:
            np.testing.assert_allclose(apply(apply(x, g), invert(g)), x)
        flat = rng.uniform(-0.8, 0.8, size=(5, 8))
        for g in GROUP:
            back = apply(apply(flat, g), invert(g))
            np.testing.assert_allclose(back, flat)
            self.assertEqual(back.shape, flat.shape)

    def test_group_average_is_labeling_invariant(self):
        rng = np.random.default_rng(19)
        f = lambda v: np.sin(v[..., 0] * 3.1) + np.cos(v[..., 1] * 1.7)  # noqa: E731
        x = rng.uniform(-0.8, 0.8, size=(4, 2))
        ref = group_average_logits(lambda v: f(v).sum(-1), x)
        for g in GROUP:
            got = group_average_logits(lambda v: f(v).sum(-1), apply(x, g))
            np.testing.assert_allclose(got, ref, rtol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
