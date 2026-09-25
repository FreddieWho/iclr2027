#!/usr/bin/env python3
"""CPU-only contracts for the M2 GPU transfer runner.

No CUDA, no training of record, no sealed pools. The remote run revalidates
everything (bank SHA, head SHA, CUDA gate) before spending GPU time.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_transfer as gt

REPO = Path(__file__).resolve().parents[2]
BANK = REPO / "artifacts" / "mechanism_transfer_v3" / "m2" / "bank"


class TransferContractTest(unittest.TestCase):
    def test_orbit_targets_have_six_columns(self):
        coords = np.zeros((3, 4, 2))
        coords[:, 1, 0] = 0.5
        coords[:, 2, 1] = 0.5
        coords[:, 3] = (0.5, 0.5)
        out = gt.orbit_targets(coords)
        self.assertEqual(out.shape, (3, 6))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_bank_sha_gate_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "data_manifest.json").write_text('{"archive": {"sha256": "dead"}}')
            (tmp / "data.npz").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                gt.load_bank(tmp)

    def test_frozen_head_rebuild_is_deterministic(self):
        torch.manual_seed(7)
        a = gt.CoordMLP(64, 32, 6)
        torch.manual_seed(7)
        b = gt.CoordMLP(64, 32, 6)
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(pa, pb))

    def test_evaluate_states_uses_j3_not_atomic(self):
        logits = np.array(
            [
                [-1.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
                [2.0, -2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        )
        labels = np.ones((4, 4))
        table = gt.evaluate_states(logits, labels, np.array([0, 0, 1, 1]), 4, seed=11)
        self.assertEqual(table["J3"], 0.75)
        self.assertEqual(table["J4"], 0.5)
        self.assertEqual(table["J3_row_weighted_cluster_ci"]["estimate"], 0.75)

    def test_row_weighted_and_equal_parent_cluster_estimands_stay_separate(self):
        row_ci = gt.metrics.row_weighted_cluster_ci(
            np.array([1.0, 0.0, 0.0]), np.array([0, 0, 1]), seed=13
        )
        parent_ci = gt.metrics.parent_cluster_ci(
            np.array([1.0, 0.0, 0.0]), np.array([0, 0, 1]), seed=13
        )
        self.assertAlmostEqual(row_ci["estimate"], 1 / 3)
        self.assertAlmostEqual(parent_ci["estimate"], 1 / 4)

    def test_epoch_size_records_repeated_clean_exposure(self):
        class ScalarLogit(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x):
                return self.linear(x).squeeze(-1)

        train_images = np.array([[0.0], [1.0]], dtype=np.float32)
        train_labels = np.array([0.0, 1.0], dtype=np.float32)
        info = gt.train_loop(
            ScalarLogit(), train_images, train_labels,
            train_images, train_labels,
            seed=19, epochs=2, device=torch.device("cpu"), loss_kind="bce",
            epoch_size=5,
        )
        self.assertEqual(info["train_unique_rows"], 2)
        self.assertEqual(info["train_row_exposures_per_epoch"], 5)
        self.assertEqual(info["train_row_exposures_total"], 10)
        self.assertEqual(info["train_sampling"], "with replacement")

    def test_standardize_matches_recorded_head_stats(self):
        raw = np.zeros((2, 6))
        out = gt.standardize_orbit(raw)
        np.testing.assert_allclose(out[0], -gt.HEAD_STATS_MU / gt.HEAD_STATS_SD)

    def test_geom_front_freezes_bn_or_trains_backbone_as_configured(self):
        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 512, kernel_size=1)
                self.bn = torch.nn.BatchNorm2d(512)

            def forward(self, x):
                return self.bn(self.conv(x))

        torch.manual_seed(37)
        images = torch.randn(4, 3, 8, 8)

        frozen_backbone = TinyBackbone()
        frozen = gt.GeomFront(frozen_backbone, freeze_backbone=True)
        frozen.prep = torch.nn.Identity()
        frozen_running_mean = frozen_backbone.bn.running_mean.clone()
        frozen.train()
        frozen(images).square().mean().backward()
        self.assertFalse(frozen_backbone.training)
        self.assertTrue(torch.equal(frozen_running_mean, frozen_backbone.bn.running_mean))
        self.assertTrue(all(p.grad is None for p in frozen_backbone.parameters()))
        self.assertTrue(any(p.grad is not None for p in frozen.probe.parameters()))

        tuned_backbone = TinyBackbone()
        tuned = gt.GeomFront(tuned_backbone, freeze_backbone=False)
        tuned.prep = torch.nn.Identity()
        before_weight = tuned_backbone.conv.weight.detach().clone()
        before_running_mean = tuned_backbone.bn.running_mean.clone()
        tuned.train()
        prediction = tuned(images)
        loss = torch.nn.functional.mse_loss(prediction, torch.randn_like(prediction))
        loss.backward()
        self.assertTrue(tuned_backbone.training)
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) > 0
                            for p in tuned_backbone.parameters()))
        optimizer = torch.optim.SGD(tuned.parameters(), lr=1e-3)
        optimizer.step()
        self.assertFalse(torch.equal(before_weight, tuned_backbone.conv.weight))
        self.assertFalse(torch.equal(before_running_mean, tuned_backbone.bn.running_mean))


if __name__ == "__main__":
    unittest.main(verbosity=2)
