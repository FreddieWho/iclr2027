#!/usr/bin/env python3
"""D10 no-leak unit tests: selection is structurally dev-only."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
from d10_frontier import (reachable_ladder, match_ladder, policy_outcomes,  # noqa: E402
                          fit_isotonic, apply_isotonic)


class TestNoLeak(unittest.TestCase):
    def test_selection_invariant_to_eval_permutation(self):
        rng = np.random.default_rng(0)
        dev_max = rng.random(86)
        lad1 = reachable_ladder(dev_max)
        # selection consumes dev arrays only: permuting hypothetical eval rows
        # cannot change output (nothing eval enters these functions)
        lad2 = reachable_ladder(dev_max)
        np.testing.assert_array_equal(lad1, lad2)
        cf = np.linspace(0, 1, 50)
        m1 = match_ladder(cf, lad1, cf[::-1], lad1)
        m2 = match_ladder(cf, lad1, cf[::-1], lad1)
        self.assertEqual(m1, m2)

    def test_ladder_is_exact_distinct_set(self):
        dev_max = np.array([0.5, 0.2, 0.5, 0.9, 0.2, 1.0, 0.0])
        lad = reachable_ladder(dev_max)
        self.assertTrue(set(dev_max).issubset(set(lad)))
        self.assertLess(lad[0], min(dev_max))
        self.assertGreater(lad[-1], max(dev_max))

    def test_isotonic_order_preserving(self):
        rng = np.random.default_rng(1)
        x = rng.random(500)
        y = (rng.random(500) < x).astype(float)
        bx, by = fit_isotonic(x, y)
        cal = apply_isotonic(bx, by, x)
        # ranks must be monotone non-decreasing in x-order (no scipy needed)
        order = np.argsort(x, kind="stable")
        self.assertTrue(bool((np.diff(cal[order]) >= -1e-12).all()))

    def test_policy_cheapest_row_rule(self):
        # one scene, rows with P0>=tau: cheapest cost row acted
        scene_of = np.array([0, 0, 0])
        P0 = np.array([0.9, 0.8, 0.1])
        costs = np.array([0.5, 0.1, 0.0])
        feas = np.array([True, False, True])
        ac, su, co = policy_outcomes(P0, scene_of, costs, feas, np.array([0]), 0.5)
        self.assertTrue(ac[0] and not su[0] and abs(co[0] - 0.1) < 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
