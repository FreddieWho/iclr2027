import json
import unittest
from pathlib import Path

import numpy as np

import data_generator
import gpu_run
import recompute_from_predictions
import static_baseline

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
            [1, 1, 1, 1],  # all four states correct
            [1, 1, 1, 1],  # P wrong, A/B/AB correct
            [1, 1, 1, 0],  # atomics correct, AB wrong
            [0, 1, 1, 1],  # B wrong, AB correct
        ], np.float32)
        logits = np.asarray([
            [2, 2, 2, 2], [-2, 2, 2, 2], [2, 2, 2, 2], [-2, 2, -2, 2]
        ], np.float32)
        for result in (gpu_run.metrics(logits, labels),
                       static_baseline.metrics(logits, labels),
                       recompute_from_predictions.correct_metrics(logits, labels)):
            self.assertEqual(result["n_quartets"], 4)
            self.assertAlmostEqual(result["atomic_joint"], 3 / 4)
            self.assertAlmostEqual(result["J3"], 2 / 4)
            self.assertAlmostEqual(result["J4"], 1 / 4)
            self.assertEqual(result["atomic_denominator"], 3)

    def test_static_baseline_matches_train_exposure_and_dev_selection(self):
        data = {
            "train_clean": np.asarray([True, False, True, False, True]),
            "train_labels": np.zeros(5, np.float32),
            "dev_clean": np.asarray([True, False, False, True, False]),
            "dev_labels": np.zeros(5, np.float32),
        }
        clean, exposure = static_baseline.matched_clean_exposure(data)
        self.assertEqual(clean.tolist(), [0, 2, 4])
        self.assertEqual(len(exposure), len(data["train_labels"]))
        self.assertEqual(set(exposure.tolist()), set(clean.tolist()))
        self.assertEqual(len(static_baseline.selection_indices(data, "all-singleton")), 5)
        self.assertEqual(static_baseline.selection_indices(data, "clean-only").tolist(), [0, 3])


if __name__ == "__main__":
    unittest.main()
