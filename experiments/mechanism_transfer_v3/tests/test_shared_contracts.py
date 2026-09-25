"""Fail-closed contracts for v3 shared primitives.

Legacy defective formulas are transcribed here as negative controls. They prove
the old behavior fails; they are not reusable definitions.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import bank
import geom_features
import metrics
import source_typed
import visual_pool


class LegacySourceSum(nn.Module):
    """Transcription of the legacy source defect: rho sees a four-point sum."""

    def __init__(self, backbone: source_typed.SegmentTypedBackbone):
        super().__init__()
        self.phi = backbone.phi
        self.rho = nn.Sequential(nn.Linear(48, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x):
        h = self.phi(x.reshape(-1, 4, 2)).reshape(-1, 4, 48)
        return self.rho(h[:, :2].sum(1) + h[:, 2:].sum(1)).reshape(-1)


def legacy_route2_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    correct = (np.asarray(logits) > 0) == (np.asarray(labels) > 0.5)
    joint = correct[:, 1] & correct[:, 2] & correct[:, 3]
    return {"J3": float(joint.mean()), "J4": float(joint.mean())}


class SharedSourceTest(unittest.TestCase):
    def test_four_point_sum_loses_segment_roles(self):
        torch.manual_seed(11)
        backbone = source_typed.SegmentTypedBackbone().double().eval()
        legacy = LegacySourceSum(backbone).double().eval()
        x, regrouped = geom_features.source_regroup_witness()
        xa = torch.tensor(x, dtype=torch.float64).reshape(-1, 8)
        xb = torch.tensor(regrouped, dtype=torch.float64).reshape(-1, 8)
        with torch.no_grad():
            gap = float((legacy(xa) - legacy(xb)).abs().max())
        self.assertLess(gap, 1e-12)

    def test_repaired_head_preserves_segment_roles(self):
        torch.manual_seed(11)
        model = source_typed.RepairedSegmentRho().double().eval()
        x, regrouped = geom_features.source_regroup_witness()
        xa = torch.tensor(x, dtype=torch.float64).reshape(-1, 8)
        xb = torch.tensor(regrouped, dtype=torch.float64).reshape(-1, 8)
        with torch.no_grad():
            gap = float((model(xa) - model(xb)).abs().max())
            swapped = float(
                (
                    model(xa)
                    - model(torch.tensor(x[[2, 3, 0, 1]], dtype=torch.float64).reshape(-1, 8))
                ).abs().max()
            )
        self.assertGreater(gap, 1e-6)
        self.assertLess(swapped, 1e-12)

    def test_model_card_carries_lineage_and_stable_digest(self):
        torch.manual_seed(11)
        first = source_typed.RepairedSegmentRho()
        torch.manual_seed(23)
        second = source_typed.RepairedSegmentRho()
        self.assertEqual(
            source_typed.architecture_digest(first), source_typed.architecture_digest(second)
        )
        card = source_typed.model_card(first)
        self.assertEqual(card["class"], "RepairedSegmentRho")
        self.assertIn("TypedPairMLP", card["lineage"]["backbone"])


class GeometryContractTest(unittest.TestCase):
    def test_t1_sorted_legacy_aliases_opposite_labels(self):
        a, b = geom_features.t1_collision_pair()
        for dtype in (np.float64, np.float32):
            gap = np.max(
                np.abs(
                    geom_features.legacy_t1_features(a.astype(dtype))
                    - geom_features.legacy_t1_features(b.astype(dtype))
                )
            )
            self.assertEqual(float(gap), 0.0)
        unsorted_gap = np.max(
            np.abs(
                geom_features.legacy_t1_features(a, sorted_features=False)
                - geom_features.legacy_t1_features(b, sorted_features=False)
            )
        )
        self.assertGreater(float(unsorted_gap), 0.1)

    def test_t1_orbit_reference_separates_and_respects_group(self):
        a, b = geom_features.t1_collision_pair()
        ref = geom_features.t1_orbit_representative(a)
        other = geom_features.t1_orbit_representative(b)
        self.assertGreater(float(np.max(np.abs(ref - other))), 0.1)
        for perm in geom_features.T1_PERMS:
            moved = np.vstack([a[list(perm[:3])], a[3]])
            np.testing.assert_allclose(
                geom_features.t1_orbit_representative(moved), ref, atol=1e-12, rtol=0
            )
        with self.assertRaises(ValueError):
            geom_features.orbit_representative(a, [(0, 1, 2, 2)])

    def test_t2_wrong_axis_breaks_rotation_but_roles_do_not(self):
        x = np.array([[-0.6, 0.1], [0.5, 0.2], [0.1, 0.7]])
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        rotated = x @ rotation.T
        legacy_gap = np.max(
            np.abs(
                geom_features.legacy_t2_features(x, sorted_features=True)
                - geom_features.legacy_t2_features(rotated, sorted_features=True)
            )
        )
        self.assertGreater(float(legacy_gap), 1e-3)
        for moved in (x[[1, 0, 2]], rotated):
            np.testing.assert_allclose(
                geom_features.t2_role_distances(x),
                geom_features.t2_role_distances(moved),
                atol=1e-12,
                rtol=0,
            )
        self.assertEqual(geom_features.t2_role_distances(x, radius=0.5).shape, (1, 4))


class BankContractTest(unittest.TestCase):
    def test_whole_scene_dedup_and_reference_distance(self):
        base = np.array([[0.0, 0.0], [1.0, 1.0]])
        near_all = base + 5e-5
        single_near = base.copy()
        single_near[0, 0] += 1e-3
        keep, dropped = bank.greedy_dedup([base, near_all, single_near], 1e-4)
        np.testing.assert_array_equal(keep, [0, 2])
        self.assertEqual(dropped, 1)
        distance = bank.min_linf_to_reference(
            [near_all, base + 10.0], [base], q_block=1, r_block=1
        )
        self.assertLess(float(distance[0]), 1e-4)
        self.assertGreater(float(distance[1]), 1.0)

    def test_parent_split_and_stratified_sampling(self):
        parents = np.array([0, 0, 1, 1, 2, 2])
        train, test = bank.split_parents_disjoint(parents, seed=7)
        self.assertTrue(set(parents[train]).isdisjoint(set(parents[test])))
        self.assertEqual(len(train) + len(test), len(parents))
        first = bank.stratified_sample_by_parent(parents, seed=11, per_parent_cap=1)
        second = bank.stratified_sample_by_parent(parents, seed=11, per_parent_cap=1)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 3)
        report = bank.coverage_report(
            np.zeros((3, 2, 2)), [0, 1, 1], families=["a", "b", "b"], radii=[0.1, 0.2, 0.3]
        )
        self.assertEqual((report["n_scenes"], report["n_parents"]), (3, 2))
        self.assertEqual(report["families"], {"a": 1, "b": 2})


class VisualPoolContractTest(unittest.TestCase):
    def test_nearest_pool_erases_but_area_pool_preserves(self):
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 4:9, 8:57] = 1
        self.assertEqual(float(mask.sum()), 245.0)
        self.assertEqual(
            float(F.interpolate(mask, size=(7, 7), mode="nearest").sum()), 0.0
        )
        weights = visual_pool.pooled_mask_weights(mask, (7, 7))
        self.assertGreater(float(weights.sum()), 0.0)
        legacy_stats = visual_pool.mask_mass_stats(mask, (7, 7), mode="nearest")
        self.assertTrue(bool(legacy_stats["native_visible_but_pooled_zero"].any()))
        fixed_stats = visual_pool.mask_mass_stats(mask, (7, 7), mode="area")
        self.assertFalse(bool(fixed_stats["native_visible_but_pooled_zero"].any()))

    def test_area_means_and_empty_mask_failure(self):
        mask = torch.zeros(1, 64, 64, dtype=torch.bool)
        mask[:, 4:9, 8:57] = True
        features = torch.ones(1, 2, 7, 7)
        means = visual_pool.pooled_means(features, {"red": mask}, (7, 7))
        torch.testing.assert_close(means["red"], torch.ones(1, 2))
        with self.assertRaises(visual_pool.EmptyMaskError):
            visual_pool.pooled_means(
                features, {"missing": torch.zeros(1, 64, 64, dtype=torch.bool)}, (7, 7)
            )

    def test_layout_and_size_contracts(self):
        image = torch.full((1, 64, 64, 3), 0.5)
        image[:, 4:9, 8:57, 0] = 0.9
        image[:, 4:9, 8:57, 1:] = 0.1
        nchw = visual_pool.as_nchw(image)
        self.assertEqual(tuple(nchw.shape), (1, 3, 64, 64))
        masks = visual_pool.color_masks(nchw)
        self.assertEqual(int(masks["red"].sum()), 245)
        small = torch.full((1, 8, 8, 3), 0.5)
        prepared = visual_pool.prepare_for_backbone(small, 16)
        self.assertEqual(tuple(prepared.shape), (1, 3, 16, 16))
        with self.assertRaises(ValueError):
            visual_pool.as_nchw(torch.zeros(1, 2, 4, 4))


class MetricsContractTest(unittest.TestCase):
    def test_legacy_j4_ignores_p_but_fixed_contract_does_not(self):
        logits = np.array([[-1.0, -1.0, -1.0, 1.0]])
        labels = np.array([[1.0, 0.0, 0.0, 1.0]])
        legacy = legacy_route2_metrics(logits, labels)
        self.assertEqual((legacy["J3"], legacy["J4"]), (1.0, 1.0))
        fixed = metrics.joint_metrics(logits, labels)
        self.assertEqual((fixed["J3"], fixed["J4"]), (1.0, 0.0))
        all_correct = metrics.joint_metrics(
            np.array([[1.0, -1.0, -1.0, 1.0]]), np.array([[1.0, 0.0, 0.0, 1.0]])
        )
        self.assertEqual((all_correct["J3"], all_correct["J4"]), (1.0, 1.0))
        missing_p = metrics.joint_metrics(
            np.array([[-1.0, -1.0, 1.0]]), np.array([[0.0, 0.0, 1.0]])
        )
        self.assertEqual(missing_p["J3"], 1.0)
        self.assertIsNone(missing_p["J4"])

    def test_parent_ci_and_table_schema(self):
        values = np.array([1.0, 0.0, 1.0, 1.0])
        parents = np.array([0, 0, 1, 1])
        first = metrics.parent_cluster_ci(values, parents, seed=7, n_boot=200)
        second = metrics.parent_cluster_ci(values, parents, seed=7, n_boot=200)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate"], 0.75)
        self.assertEqual(first["unit"], metrics.UNIT)
        checked = metrics.validate_table_record(
            {
                "n_quartets": 28,
                "n_eligible_parents": 19,
                "n_sampled_parents": 19,
                "row_mean": 0.571,
                "parent_mean": 0.571,
            }
        )
        self.assertEqual(checked["n_quartets"], 28)
        with self.assertRaises(ValueError):
            metrics.validate_table_record({"n_quartets": 28})


if __name__ == "__main__":
    unittest.main(verbosity=2)
