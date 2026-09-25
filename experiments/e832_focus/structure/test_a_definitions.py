"""Definition checks for the source-task contrast. No bank, no sealed pool, no training."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "e832_focus" / "structure"))
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "e1a933_review"))

from a_source_compare import (  # noqa: E402
    G8,
    RepairedRho,
    SymmetricQ,
    TypedPairMLP,
    apply_perm,
    is_official_quartet,
    parent_ci,
    rich8_sorted,
    rich8_unsorted,
)
from leads_l014_l015_l006 import g8_features  # noqa: E402


class DefinitionTests(unittest.TestCase):
    def test_official_quartet_is_two_preserves_whose_sum_flips(self):
        self.assertTrue(is_official_quartet(0, 0, 0, 1))
        self.assertTrue(is_official_quartet(1, 1, 1, 0))
        self.assertFalse(is_official_quartet(0, 0, 0, 0))
        self.assertFalse(is_official_quartet(0, 1, 0, 1))
        self.assertFalse(is_official_quartet(1, 1, 0, 0))

    def test_sorted_rich8_is_existing_g8_sort_of_the_same_quantities(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(24, 4, 2)).astype(np.float32)
        unsorted = rich8_unsorted(x)
        manual = np.concatenate(
            [np.sort(unsorted[:, 0:2], axis=1), np.sort(unsorted[:, 2:6], axis=1), unsorted[:, 6:8]],
            axis=1,
        )
        self.assertTrue(np.allclose(manual, rich8_sorted(x)))
        self.assertTrue(np.allclose(rich8_sorted(x), g8_features(x)))
        swing_sorted = 0.0
        swing_unsorted = 0.0
        for perm in G8:
            xp = apply_perm(x, perm)
            swing_sorted = max(swing_sorted, float(np.max(np.abs(rich8_sorted(xp) - rich8_sorted(x)))))
            swing_unsorted = max(swing_unsorted, float(np.max(np.abs(rich8_unsorted(xp) - unsorted))))
        self.assertEqual(swing_sorted, 0.0)
        self.assertGreater(swing_unsorted, 0.0)

    def test_additive_mixed_residual_is_numerical_zero_and_repair_is_not_the_same_class(self):
        s = [np.array([[-0.7, 0.4], [0.7, 0.4]]), np.array([[-0.7, -0.4], [0.7, -0.4]])]
        t = [np.array([[0.1, 0.2], [0.1, 0.6]]), np.array([[0.1, -0.6], [0.1, -0.2]])]
        xs = torch.tensor(np.stack([np.concatenate([a, b]) for a in s for b in t]), dtype=torch.float64)
        torch.manual_seed(0)
        additive = TypedPairMLP(64).double().eval()
        with torch.no_grad():
            z = additive(xs).reshape(2, 2).numpy()
        residual = float(z[0, 0] + z[1, 1] - z[0, 1] - z[1, 0])
        self.assertLess(abs(residual), 1e-8)
        torch.manual_seed(1)
        repaired = RepairedRho(64, 16).double().eval()
        with torch.no_grad():
            zr = repaired(xs).reshape(2, 2).numpy()
        self.assertGreater(abs(float(zr[0, 0] + zr[1, 1] - zr[0, 1] - zr[1, 0])), 1e-6)

    def test_segment_swap_symmetry_of_repair_classes(self):
        rng = np.random.default_rng(2)
        x = torch.tensor(rng.normal(size=(16, 8)), dtype=torch.float32)
        swapped = apply_perm(x.numpy(), (2, 3, 0, 1))
        swapped = torch.tensor(swapped)
        for model in (TypedPairMLP(64).eval(), RepairedRho(64, 16).eval(), SymmetricQ(64, 48).eval()):
            with torch.no_grad():
                gap = torch.max(torch.abs(model(x) - model(swapped))).item()
            self.assertLess(gap, 1e-5)

    def test_parent_interval_uses_parent_groups_not_group_elements(self):
        values = np.array([1.0, 0.0, 1.0, 1.0])
        parents = np.array([7, 7, 9, 9])
        out = parent_ci(values, parents, seed=1, n_boot=20)
        self.assertEqual(out["n"], 4)
        self.assertEqual(out["n_parents"], 2)
        self.assertEqual(out["estimate"], 0.75)
        self.assertLessEqual(out["ci95"][0], out["ci95"][1])


if __name__ == "__main__":
    unittest.main()
