"""Focused fail-closed checks for the M1.2 source training matrix.

These tests guard the contracts the matrix depends on, not the matrix's
numbers: feature schemas, the batched orbit representative, exact group
invariance of the role-preserving references, functional equivalence between
the vendored segment head and the legacy composition, checkpoint contents, the
dev-BCE selection rule, and the repair-flow identity.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
for path in (
    str(ROOT / "experiments" / "f095_campaign"),
    str(ROOT / "experiments" / "discovery_campaign"),
    str(HERE),
):
    if path not in sys.path:
        sys.path.insert(0, path)

import evaluate  # noqa: E402
import features  # noqa: E402
import models  # noqa: E402
import recipe  # noqa: E402
from _v3common import geom_features, source_typed  # noqa: E402
from u01_models import TypedPairMLP  # noqa: E402

torch.set_num_threads(2)


class FeatureSchemaTest(unittest.TestCase):
    def test_columns_match_declared_dimension(self):
        for arm in features.ARMS:
            schema = features.feature_schema(arm)
            self.assertEqual(len(schema["columns"]), schema["in_dim"], arm)
            self.assertEqual(features.in_dim(arm), schema["in_dim"])

    def test_rich8_sorted_is_the_legacy_bag_sort(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(64, 4, 2)).astype(np.float32)
        unsorted = features.rich8_unsorted_features(x)
        rebuilt = np.concatenate(
            [
                np.sort(unsorted[:, 0:2], axis=1),
                np.sort(unsorted[:, 2:6], axis=1),
                unsorted[:, 6:8],
            ],
            axis=1,
        )
        np.testing.assert_allclose(rebuilt, features.rich8_sorted_features(x), atol=0)

    def test_sorted_bag_is_group_invariant_but_unsorted_is_not(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(32, 4, 2))
        sorted_base = features.rich8_sorted_features(x)
        unsorted_base = features.rich8_unsorted_features(x)
        max_sorted = 0.0
        max_unsorted = 0.0
        for perm in geom_features.SOURCE_PERMS:
            moved = x[:, list(perm), :]
            max_sorted = max(
                max_sorted,
                float(np.max(np.abs(features.rich8_sorted_features(moved) - sorted_base))),
            )
            max_unsorted = max(
                max_unsorted,
                float(np.max(np.abs(features.rich8_unsorted_features(moved) - unsorted_base))),
            )
        self.assertLess(max_sorted, 1e-5)
        self.assertGreater(max_unsorted, 1e-3)


class OrbitRepresentativeTest(unittest.TestCase):
    def test_batched_matches_scalar_reference(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(17, 4, 2))
        batched = features.batch_orbit_representative(x)
        for i in range(len(x)):
            scalar = geom_features.orbit_representative(x[i], geom_features.SOURCE_PERMS)
            np.testing.assert_allclose(batched[i], scalar, atol=1e-12)

    def test_group_invariant_and_spares_regroup_witness(self):
        x, regrouped = geom_features.source_regroup_witness()
        base = features.batch_orbit_representative(x[None])[0]
        moved = features.batch_orbit_representative(regrouped[None])[0]
        self.assertGreater(float(np.max(np.abs(base - moved))), 0.1)
        for perm in geom_features.SOURCE_PERMS:
            permuted = features.batch_orbit_representative(x[None, list(perm), :])[0]
            np.testing.assert_allclose(permuted, base, atol=1e-12)

    def test_tie_fraction_is_reported(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(8, 4, 2))
        value = features.orbit_tie_fraction(x)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_tie_fraction_counts_duplicate_lexicographic_minima(self):
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        self.assertEqual(features.orbit_tie_fraction(square[None]), 1.0)


class SegmentMomentTest(unittest.TestCase):
    def test_descriptor_is_endpoint_and_translation_invariant(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(16, 4, 2))
        base = features.segment_moment_descriptors(x, scale=1.0)
        translated = features.segment_moment_descriptors(x + np.array([3.0, -2.0]), scale=1.0)
        np.testing.assert_allclose(base, translated, atol=1e-12)
        block_swapped = np.concatenate([base[:, 5:], base[:, :5]], axis=1)
        for perm in geom_features.SOURCE_PERMS:
            moved = features.segment_moment_descriptors(x[:, list(perm), :], scale=1.0)
            swaps_segments = set(perm[:2]) == {2, 3}
            # Endpoint swaps leave the descriptor alone; a segment swap only
            # reorders the two descriptor blocks, it never mixes their columns.
            np.testing.assert_allclose(
                moved, block_swapped if swaps_segments else base, atol=1e-12
            )

    def test_model_output_is_group_invariant(self):
        torch.manual_seed(5)
        model = models.SegmentMomentInteraction(10).double().eval()
        rng = np.random.default_rng(5)
        x = torch.from_numpy(rng.normal(size=(12, 4, 2)))
        descriptors = features.segment_moment_descriptors(x.numpy(), scale=1.0)
        with torch.no_grad():
            base = model(torch.from_numpy(descriptors))
            for perm in geom_features.SOURCE_PERMS:
                moved = features.segment_moment_descriptors(
                    x.numpy()[:, list(perm), :], scale=1.0
                )
                output = model(torch.from_numpy(moved))
                torch.testing.assert_close(output, base, atol=1e-10, rtol=0)

    def test_fitted_statistics_are_block_shared(self):
        """Block-shared statistics are what keeps standardization group-covariant."""
        rng = np.random.default_rng(6)
        train = rng.normal(size=(48, 4, 2)).astype(np.float32)
        stats = features.fit_stats("segment_moment", train)
        self.assertEqual(stats["kind"], "train_feature_block_shared")
        np.testing.assert_allclose(stats["mu"][:5], stats["mu"][5:], atol=0)
        np.testing.assert_allclose(stats["sd"][:5], stats["sd"][5:], atol=0)

    def test_standardized_features_are_exactly_group_invariant(self):
        rng = np.random.default_rng(7)
        train = rng.normal(size=(64, 4, 2)).astype(np.float32)
        stats = features.fit_stats("segment_moment", train)
        base = features.featurize("segment_moment", train, stats)
        base_raw = features.segment_moment_descriptors(train, stats["scale"])
        block_swapped_raw = np.concatenate([base_raw[:, 5:], base_raw[:, :5]], axis=1)
        block_swapped = ((block_swapped_raw - stats["mu"]) / stats["sd"]).astype(np.float32)
        for perm in geom_features.SOURCE_PERMS:
            moved = features.featurize(
                "segment_moment", train[:, list(perm), :], stats
            )
            swaps_segments = set(perm[:2]) == {2, 3}
            np.testing.assert_allclose(
                moved, block_swapped if swaps_segments else base, atol=0
            )

    def test_descriptor_spares_illegal_regroup(self):
        x, regrouped = geom_features.source_regroup_witness()
        base = features.segment_moment_descriptors(x[None], scale=1.0)
        moved = features.segment_moment_descriptors(regrouped[None], scale=1.0)
        self.assertGreater(float(np.max(np.abs(base - moved))), 0.1)


class SegmentHeadEquivalenceTest(unittest.TestCase):
    def test_vendored_head_matches_legacy_composition(self):
        torch.manual_seed(7)
        vendored = source_typed.RepairedSegmentRho(width=64, k=source_typed.RHO_K)
        legacy = TypedPairMLP(64)
        hidden = legacy.rho.in_features
        legacy.rho = nn.Sequential(
            nn.Linear(hidden, source_typed.RHO_K), nn.GELU(), nn.Linear(source_typed.RHO_K, 1)
        )
        mapped = {}
        for key, value in legacy.state_dict().items():
            if key.startswith(("phi.", "psi.")):
                mapped[f"backbone.{key}"] = value
            elif key.startswith("rho."):
                mapped[key] = value
            else:
                raise AssertionError(f"unexpected legacy key {key}")
        vendored.load_state_dict(mapped)
        vendored.eval()
        legacy.eval()
        rng = np.random.default_rng(7)
        x = torch.from_numpy(rng.normal(size=(9, 8)).astype(np.float32))
        with torch.no_grad():
            torch.testing.assert_close(vendored(x), legacy(x), atol=1e-6, rtol=0)

    def test_parameter_counts_match_frozen_report(self):
        vendored = source_typed.RepairedSegmentRho(width=64, k=source_typed.RHO_K)
        self.assertEqual(models.count_parameters(vendored), 8001)


class TrainingCellTest(unittest.TestCase):
    def test_checkpoint_and_receipt_contents(self):
        data = recipe.load_data()
        with tempfile.TemporaryDirectory(dir=recipe.ART.parent) as tmp:
            with mock.patch.object(recipe, "RUNS", Path(tmp) / "runs"), mock.patch.object(
                recipe, "EPOCHS", 3
            ):
                receipt = recipe.train_cell("orbit_distance", 11, "clean", 0.001, data)
                checkpoint_path = Path(receipt["checkpoint"])
                self.assertTrue(checkpoint_path.exists(), checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for key in (
            "state",
            "optimizer",
            "arm",
            "seed",
            "supervision",
            "lr",
            "epochs",
            "in_dim",
            "stats",
            "feature_schema",
            "batch_contract",
            "init_state_digest",
            "model_card",
            "train_curve",
            "dev_curve",
            "selection",
        ):
            self.assertIn(key, checkpoint, key)
        self.assertEqual(len(checkpoint["train_curve"]), 3)
        self.assertIn("param_groups", checkpoint["optimizer"])
        for key in (
            "checkpoint_sha256",
            "init_state_digest",
            "macs_per_example",
            "train_state_forwards",
            "batch_contract",
            "feature_schema",
            "n_train_clean",
            "n_dev_clean",
            "n_dev_flip",
        ):
            self.assertIn(key, receipt, key)
        self.assertEqual(receipt["batch_contract"]["regime"], "full_batch")
        self.assertFalse(receipt["batch_contract"]["shuffle"])
        self.assertEqual(
            receipt["batch_contract"]["train_row_order_sha256"][:8],
            recipe.batch_contract(data)["train_row_order_sha256"][:8],
        )
        self.assertNotIn("J3", receipt)


class SelectionRuleTest(unittest.TestCase):
    def test_min_dev_bce_wins_and_smaller_lr_breaks_ties(self):
        with tempfile.TemporaryDirectory(dir=recipe.ART.parent) as tmp:
            runs = Path(tmp) / "runs"
            for lr, dev_bce in ((0.01, 0.5), (0.003, 0.5), (0.001, 0.4)):
                cell = runs / f"raw_s11_clean_lr{lr}"
                cell.mkdir(parents=True)
                (cell / "receipt.json").write_text(
                    json.dumps(
                        {
                            "arm": "raw",
                            "seed": 11,
                            "supervision": "clean",
                            "lr": lr,
                            "dev_bce": dev_bce,
                            "checkpoint": "unused",
                        }
                    )
                )
            with mock.patch.object(recipe, "RUNS", runs), mock.patch.object(
                recipe, "ART", Path(tmp)
            ), mock.patch.object(recipe, "LRS", (0.01, 0.003, 0.001)), mock.patch.object(
                recipe, "SUPERVISIONS", ("clean",)
            ):
                chosen = recipe.select({}, arms=["raw"], seeds=[11])
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["lr"], 0.001)
        self.assertEqual(chosen[0]["dev_bce"], 0.4)


class GroupSwingTest(unittest.TestCase):
    """Structural smoke test: the reference arms must be exactly invariant."""

    def test_reference_arms_have_zero_swing_and_raw_does_not(self):
        data = recipe.load_data()
        bank = data["bank"]
        swings = {}
        for arm in (
            "orbit_distance",
            "segment_moment",
            "repaired_segment_rho",
            "rich8_sorted",
            "raw",
            "rich8_unsorted",
        ):
            torch.manual_seed(11)
            model, _ = models.build_model(arm)
            stats = features.fit_stats(arm, data["train_positions"])
            swings[arm] = evaluate.g8_swing(model, arm, stats, bank)["max_abs_logit_swing"]
        for arm in (
            "orbit_distance",
            "segment_moment",
            "repaired_segment_rho",
            "rich8_sorted",
        ):
            self.assertLessEqual(swings[arm], 1e-4, swings)
        for arm in ("raw", "rich8_unsorted"):
            self.assertGreater(swings[arm], 1e-3, swings)


class MetricsAndFlowTest(unittest.TestCase):
    def test_j3_excludes_p_and_j4_includes_it(self):
        correct = np.zeros((6, 4), dtype=bool)
        correct[:, 1:4] = True
        correct[:, 0] = False
        parents = np.array([0, 0, 1, 1, 2, 2])
        margins = np.tile(np.array([0.5, 0.5, 0.5, 0.5]), (6, 1))
        table, j3 = evaluate.arm_metrics(correct, parents, margins)
        self.assertEqual(table["J3"]["row_mean"], 1.0)
        self.assertEqual(table["J4"]["row_mean"], 0.0)
        self.assertEqual(table["atomic_joint"]["row_mean"], 1.0)
        self.assertTrue(j3.all())

    def test_flow_identity_holds(self):
        base = np.zeros((20, 3), dtype=bool)
        base[:5, 0] = True
        base[:5, 1] = True  # 110 baseline cells
        base[5:8] = True  # 111 baseline cells
        candidate = np.zeros((20, 3), dtype=bool)
        candidate[:3, 0] = True
        candidate[:3, 1] = True
        candidate[:3, 2] = True  # full repairs
        candidate[3:5] = True  # migration: AB only
        parents = np.repeat(np.arange(10), 2)
        flow = evaluate.flow_from(base, candidate, parents)
        self.assertEqual(flow["baseline110_n"], 5)
        self.assertEqual(flow["baseline111_n"], 3)
        self.assertAlmostEqual(
            flow["R_endpoint"]["row_mean"]
            - flow["R_full"]["row_mean"]
            - flow["M"]["row_mean"],
            flow["identity_residual"],
            places=12,
        )


class PreprocessingLeakageTest(unittest.TestCase):
    def test_statistics_are_fit_on_train_only(self):
        train = np.zeros((8, 4, 2), dtype=np.float32)
        train[:, :, 0] = 0.5
        bank = np.full((4, 4, 2), 5.0, dtype=np.float32)
        for arm in ("raw", "repaired_segment_rho"):
            stats = features.fit_stats(arm, train)
            np.testing.assert_allclose(stats["mu"], [0.5, 0.0], atol=1e-6)
            features.featurize(arm, bank, stats)
        for arm in ("rich8_unsorted", "rich8_sorted", "orbit_distance", "segment_moment"):
            stats = features.fit_stats(arm, train)
            self.assertTrue(np.all(np.isfinite(stats["mu"])))
            features.featurize(arm, bank, stats)

    def test_unknown_arm_fails_closed(self):
        with self.assertRaises(KeyError):
            features.feature_schema("sorted_by_vibes")
        with self.assertRaises(KeyError):
            models.build_model("group_mean")


if __name__ == "__main__":
    unittest.main(verbosity=2)
