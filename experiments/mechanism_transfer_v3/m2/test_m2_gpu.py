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

    def test_standardize_matches_recorded_head_stats(self):
        raw = np.zeros((2, 6))
        out = gt.standardize_orbit(raw)
        np.testing.assert_allclose(out[0], -gt.HEAD_STATS_MU / gt.HEAD_STATS_SD)


if __name__ == "__main__":
    unittest.main(verbosity=2)
