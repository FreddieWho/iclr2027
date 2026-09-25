import json
import unittest
from pathlib import Path

import numpy as np

import data_generator
import gpu_run

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "artifacts/e832_focus/route2/data"


class DataAndRunnerTests(unittest.TestCase):
    def test_fresh_seed_freeze_and_no_ab_training_arrays(self):
        self.assertEqual(data_generator.SEEDS["train_parent"], 832701)
        self.assertEqual(data_generator.SEEDS["dev_parent"], 832702)
        self.assertEqual(data_generator.SEEDS["test_parent"], 832703)
        self.assertNotIn("train_AB_images", data_generator.generate.__code__.co_names)
        manifest = data_generator.validate(DATA)
        self.assertEqual(manifest["counts"]["ab_training_images"], 0)
        self.assertTrue(manifest["provenance"]["fresh_generator_only"])
        self.assertFalse(manifest["provenance"]["old_exposed_159_bank_used"])
        with np.load(DATA / "data.npz", allow_pickle=False) as data:
            self.assertNotIn("train_AB_labels", data.files)
            self.assertEqual(data["quartet_images"].shape[1], 4)

    def test_metric_denominators(self):
        labels = np.asarray([
            [0, 1, 1, 0],  # atomics correct, AB wrong
            [0, 1, 1, 1],  # full joint correct
            [0, 1, 1, 1],  # B wrong, AB correct
            [0, 1, 1, 0],  # atomics correct, AB wrong
        ], np.float32)
        logits = np.asarray([
            [-2, 2, 2, 2], [-2, 2, 2, 2], [-2, 2, -2, 2], [-2, 2, 2, 2]
        ], np.float32)
        result = gpu_run.metrics(logits, labels)
        self.assertEqual(result["n_quartets"], 4)
        self.assertAlmostEqual(result["atomic_joint"], 3 / 4)
        self.assertAlmostEqual(result["J3"], 1 / 4)
        self.assertAlmostEqual(result["J4"], 1 / 4)
        self.assertEqual(result["atomic_denominator"], 3)


if __name__ == "__main__":
    unittest.main()
