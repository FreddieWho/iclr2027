"""Focused fail-closed checks for the M2 prep artefacts.

These guard the contracts the M2 bank, mask audit and analytic baseline depend
on: the frozen config, both mining constructions and their yield identities,
uint8 storage, split/geometry separation, delete-and-regenerate determinism,
mask mass accounting, and the analytic endpoint fit. They do not assert any
scientific result.

Sizes are tiny on purpose; the frozen production config is asserted against the
declared constants rather than regenerated here.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analytic_image_baseline as baseline  # noqa: E402
import bank_contract  # noqa: E402
import generate_bank as generate  # noqa: E402
import mask_audit  # noqa: E402
import scene_primitives as primitives  # noqa: E402
from _v3common import metrics, visual_pool  # noqa: E402
from miners import mine_quartets, mine_singletons  # noqa: E402

TINY = generate.TINY


class FrozenContractTest(unittest.TestCase):
    def test_declared_constants_match_the_frozen_config(self):
        self.assertEqual(bank_contract.FROZEN.seeds, bank_contract.FROZEN_SEEDS)
        self.assertEqual(bank_contract.FROZEN.split_sizes, bank_contract.FROZEN_SPLIT_SIZES)
        self.assertEqual(bank_contract.FROZEN.radii, bank_contract.FROZEN_RADII)
        self.assertEqual(
            bank_contract.FROZEN.oracle_margin_floor, bank_contract.FROZEN_ORACLE_MARGIN_FLOOR
        )
        self.assertEqual(
            bank_contract.FROZEN.min_changed_pixels, bank_contract.FROZEN_MIN_CHANGED_PIXELS
        )
        self.assertEqual(bank_contract.FROZEN.dedup_linf, bank_contract.FROZEN_DEDUP_LINF)
        self.assertEqual(bank_contract.FROZEN.image_dtype, "uint8")

    def test_seeds_are_independent_of_the_archived_pilot(self):
        for value in bank_contract.FROZEN.seeds.values():
            self.assertNotIn(value // 100, (8327,), "v3 seeds must not reuse the pilot seed block")

    def test_validation_rejects_wrong_dtype_and_empty_radii(self):
        with self.assertRaises(ValueError):
            bank_contract.BankConfig(image_dtype="float32").validate()
        with self.assertRaises(ValueError):
            bank_contract.BankConfig(radii=()).validate()
        with self.assertRaises(ValueError):
            bank_contract.BankConfig(seeds={"train_parent": 1}).validate()

    def test_render_contract_is_fixed_at_native_64(self):
        self.assertEqual(primitives.IMAGE_SIZE, 64)
        with self.assertRaises(ValueError):
            bank_contract.BankConfig(image_size=224).validate()


class MinerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parents, _, _ = primitives.make_parents(24, 832901, bank_contract.FROZEN.oracle_margin_floor)
        cls.parents = parents[:24]

    def test_singleton_edits_flip_the_label(self):
        arrays, log = mine_singletons(
            self.parents, bank_contract.BankConfig(split_sizes={"train": 6, "dev": 2, "test": 2},
                                                   singleton_draws=4), 832911
        )
        clean = arrays["clean"]
        self.assertGreater(int((~clean).sum()), 0)
        parent_labels = arrays["labels"][clean]
        for index in np.flatnonzero(~clean):
            self.assertNotEqual(
                int(arrays["labels"][index]), int(parent_labels[arrays["parents"][index]])
            )
        self.assertGreaterEqual(arrays["oracle_margins"][~clean].min(), bank_contract.FROZEN.oracle_margin_floor)
        self.assertGreaterEqual(arrays["changed_pixels"][~clean].min(), bank_contract.FROZEN.min_changed_pixels)

    def test_singleton_yield_identity(self):
        _, log = mine_singletons(
            self.parents, bank_contract.BankConfig(split_sizes={"train": 6, "dev": 2, "test": 2},
                                                   singleton_draws=4), 832911
        )
        parts = (
            log["oracle_valueerror"]
            + log["ambiguous"]
            + log["label_not_flipped"]
            + log["below_margin"]
            + log["below_pixel_threshold"]
            + log["single_kept"]
        )
        self.assertEqual(parts, log["drawn"])
        self.assertEqual(log["single_kept"], log["accepted"])

    def test_quartets_follow_the_emergent_contract(self):
        arrays, log = mine_quartets(
            self.parents,
            bank_contract.BankConfig(
                split_sizes={"train": 2, "dev": 2, "test": 6},
                quartet_draws=8,
                max_quartets_total=8,
            ),
            832921,
            832922,
        )
        self.assertGreater(arrays["labels"].shape[0], 0)
        labels = arrays["labels"]
        self.assertTrue((labels[:, 0] == labels[:, 1]).all())
        self.assertTrue((labels[:, 0] == labels[:, 2]).all())
        self.assertTrue((labels[:, 3] != labels[:, 0]).all())
        self.assertGreaterEqual(arrays["margins"].min(), bank_contract.FROZEN.oracle_margin_floor)
        self.assertGreaterEqual(
            arrays["changed_pixels"].min(), bank_contract.FROZEN.min_changed_pixels
        )
        self.assertEqual(log["pair_drawn"], sum(log[k] for k in (
            "pair_oracle_valueerror", "pair_combo_preserved", "pair_below_margin",
            "pair_below_pixel_threshold")) + log["accepted"])

    def test_changed_pixel_filter_is_zero_for_identical_renders(self):
        scene = self.parents[0]
        seed = primitives.render_seed(832999, 3)
        first = primitives.render_scene(scene, seed)
        second = primitives.render_scene(scene, seed)
        self.assertEqual(primitives.changed_pixels(first, second), 0)
        moved = scene.copy()
        moved[2, 0] += 0.4
        self.assertGreater(
            primitives.changed_pixels(first, primitives.render_scene(moved, seed)),
            bank_contract.FROZEN.min_changed_pixels,
        )

    def test_uint8_quantisation_preserves_the_frozen_masks(self):
        scene = self.parents[1]
        image = primitives.render_scene(scene, primitives.render_seed(832999, 4))
        quantised = primitives.to_uint8_checked(image)
        float_from_uint8 = primitives.images_as_float(quantised[None])
        masks_quantised = visual_pool.color_masks(float_from_uint8)
        masks_float = visual_pool.color_masks(image[None])
        for color in ("red", "blue"):
            self.assertTrue(
                bool((masks_quantised[color] == masks_float[color]).all()),
                f"{color} mask changed under uint8 storage",
            )


class BankBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temp.name)
        cls.out = cls.root / "bank"
        cls.manifest = generate.run(cls.out, config=TINY, determinism=True, log=lambda *_: None)
        cls.data = np.load(cls.out / "data.npz", allow_pickle=False)

    @classmethod
    def tearDownClass(cls):
        cls.data.close()
        cls._temp.cleanup()

    def test_manifest_declares_no_sealed_source_and_uint8_storage(self):
        provenance = self.manifest["provenance"]
        self.assertFalse(provenance["sealed_or_dev512_pool_read"])
        self.assertFalse(provenance["loaded_existing_scene_or_image_artifacts"])
        self.assertEqual(self.manifest["status"], "DATA_READY")
        self.assertEqual(self.manifest["scientific_result"], "NOT_RUN")
        self.assertTrue(self.manifest["no_neural_training"])
        self.assertEqual(self.manifest["counts"]["ab_training_images"], 0)
        self.assertEqual(self.manifest["config"]["image_dtype"], "uint8")

    def test_split_disjointness_and_separation_are_reported(self):
        disjoint = self.manifest["split_disjointness"]
        self.assertTrue(disjoint["parent_sets_pairwise_disjoint"])
        self.assertTrue(disjoint["singleton_image_sets_pairwise_disjoint"])
        self.assertGreater(self.manifest["separations"]["min_test_to_train_parent_linf"], 0.0)
        self.assertGreater(self.manifest["separations"]["dedup_linf"], 0.0)

    def test_array_metadata_matches_the_manifest(self):
        for name, meta in self.manifest["arrays"].items():
            self.assertIn(name, self.data, name)
            self.assertEqual(list(self.data[name].shape), meta["shape"], name)
            self.assertEqual(str(self.data[name].dtype), meta["dtype"], name)

    def test_quartet_arrays_are_internally_consistent(self):
        parents = self.data["quartet_parents"]
        self.assertGreater(len(parents), 0)
        self.assertLess(int(parents.max()), len(self.data["test_parent_coords"]))
        self.assertGreaterEqual(int(parents.min()), 0)
        self.assertEqual(self.data["quartet_images"].shape[1:2], (4,))
        self.assertEqual(self.data["quartet_images"].dtype, np.uint8)
        self.assertTrue((self.data["quartet_labels"][:, 1] == self.data["quartet_labels"][:, 0]).all())
        self.assertTrue((self.data["quartet_labels"][:, 3] != self.data["quartet_labels"][:, 0]).all())

    def test_delete_and_regenerate_is_deterministic(self):
        receipt = self.manifest["determinism"]
        self.assertEqual(receipt["determinism_check"], "passed")
        self.assertTrue(receipt["array_sha256_all_equal"])
        self.assertTrue(receipt["archive_sha256_equal"])

    def test_validate_bank_accepts_fresh_and_rejects_tampering(self):
        generate.validate_bank(self.out)
        tampered = self.root / "tampered"
        tampered.mkdir()
        (tampered / "data_manifest.json").write_text((self.out / "data_manifest.json").read_text())
        with np.load(self.out / "data.npz", allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        arrays["test_labels"] = arrays["test_labels"].copy()
        arrays["test_labels"][0] = 1 - arrays["test_labels"][0]
        np.savez_compressed(tampered / "data.npz", **arrays)
        with self.assertRaises(ValueError):
            generate.validate_bank(tampered)


class MaskAuditTest(unittest.TestCase):
    def test_thin_visible_line_is_erased_by_nearest_but_not_by_area(self):
        import torch
        import torch.nn.functional as F

        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 4:9, 8:57] = 1
        self.assertEqual(int(mask.sum()), 245)
        self.assertEqual(int(F.interpolate(mask, size=(7, 7), mode="nearest").sum()), 0)
        self.assertGreater(float(F.interpolate(mask, size=(7, 7), mode="area").sum()), 0)

    def test_mask_mass_stats_separate_missing_colour_from_pooling_loss(self):
        import torch

        empty = torch.zeros(2, 64, 64, dtype=torch.bool)
        empty[0, 4:9, 8:57] = True
        stats = visual_pool.mask_mass_stats(empty, (7, 7), mode="nearest")
        self.assertEqual(int(stats["native_pixels"][1]), 0)
        self.assertFalse(bool(stats["native_visible_but_pooled_zero"][1]))
        self.assertTrue(bool(stats["native_visible_but_pooled_zero"][0]))

    def test_as_float01_accepts_both_storage_dtypes(self):
        uint8 = np.zeros((2, 3, 4, 4), dtype=np.uint8)
        uint8[...] = 255
        self.assertAlmostEqual(float(mask_audit.as_float01(uint8).max()), 1.0, places=6)
        floating = np.full((2, 3, 4, 4), 0.5, dtype=np.float32)
        self.assertAlmostEqual(float(mask_audit.as_float01(floating).mean()), 0.5, places=6)


class AnalyticBaselineTest(unittest.TestCase):
    def test_line_fit_recovers_known_endpoints_and_label(self):
        scene = np.array([[-0.6, -0.5], [0.5, 0.4], [-0.55, 0.45], [0.45, -0.4]])
        self.assertEqual(primitives.oracle(scene)[0], 1)
        image = primitives.render_scene(scene, primitives.render_seed(833001, 7))
        record = baseline.predict_image(primitives.to_uint8(image))
        self.assertTrue(record["valid"])
        self.assertEqual(int(record["label"]), 1)
        self.assertLess(baseline.endpoint_error(record["endpoints"], scene), 0.06)

    def test_non_crossing_scene_is_predicted_negative(self):
        scene = np.array([[-0.6, -0.5], [-0.4, -0.55], [0.2, 0.5], [0.5, 0.45]])
        self.assertEqual(primitives.oracle(scene)[0], 0)
        image = primitives.render_scene(scene, primitives.render_seed(833002, 9))
        record = baseline.predict_image(primitives.to_uint8(image))
        self.assertTrue(record["valid"])
        self.assertEqual(int(record["label"]), 0)

    def test_flow_counts_match_the_J3_J4_contract(self):
        correct = np.array(
            [
                [True, True, True, True],
                [True, True, True, False],
                [False, True, True, False],
            ]
        )
        flow = baseline.flow_counts(correct)
        self.assertEqual(flow["n_A_B_correct"], 3)
        self.assertEqual(flow["n_A_B_correct_AB_wrong"], 2)
        self.assertEqual(flow["n_full_repair_J3"], 1)
        self.assertEqual(flow["n_P_correct"], 2)
        logits = np.where(correct, 1.0, -1.0)
        labels = np.ones((3, 4))
        result = metrics.joint_metrics(logits, labels)
        self.assertAlmostEqual(result["J3"], 1 / 3, places=6)
        self.assertAlmostEqual(result["J4"], 1 / 3, places=6)

    def test_failure_decomposition_counts_degenerate_segments(self):
        blank = np.zeros((3, 64, 64), dtype=np.uint8)
        record = baseline.predict_image(blank)
        self.assertFalse(record["valid"])
        payload = baseline.evaluate_singletons(np.stack([blank, blank]), np.array([0, 1]))
        self.assertEqual(payload["n_fitted"], 0)
        self.assertEqual(payload["failure_decomposition"]["red_degenerate"], 2)
        self.assertEqual(payload["failure_decomposition"]["blue_degenerate"], 2)
        self.assertEqual(payload["accuracy_all_images"], 0.0)
        self.assertIsNone(payload["accuracy_on_fitted"])
        scene = np.array([[-0.6, -0.5], [0.5, 0.4], [-0.55, 0.45], [0.45, -0.4]])
        image = primitives.to_uint8(
            primitives.render_scene(scene, primitives.render_seed(833003, 5))
        )
        mixed = baseline.evaluate_singletons(np.stack([blank, image]), np.array([0, 1]))
        self.assertEqual(mixed["n_fitted"], 1)
        self.assertEqual(mixed["accuracy_all_images"], 0.5)
        self.assertEqual(mixed["accuracy_on_fitted"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
