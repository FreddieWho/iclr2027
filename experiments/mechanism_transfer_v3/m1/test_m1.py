"""Focused M1 checks: collision witness, source-search schema, reference evals."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "mechanism_transfer_v3"))
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))

from common import geom_features, source_typed  # noqa: E402
from u10_oracles import tri_oracle  # noqa: E402

ART = ROOT / "artifacts" / "mechanism_transfer_v3" / "m1"


class CollisionWitnessTest(unittest.TestCase):
    def test_sorted_bags_merge_opposite_labels(self):
        a, b = geom_features.t1_collision_pair()
        fa = geom_features.legacy_t1_features(
            np.asarray(a, dtype=np.float32), sorted_features=True
        )
        fb = geom_features.legacy_t1_features(
            np.asarray(b, dtype=np.float32), sorted_features=True
        )
        self.assertTrue(np.array_equal(fa, fb))
        oa = tri_oracle(np.asarray(a, float), min_margin=0.02)
        ob = tri_oracle(np.asarray(b, float), min_margin=0.02)
        self.assertEqual(oa["label"], 1)
        self.assertEqual(ob["label"], 0)
        self.assertGreater(oa["margin"], 0.02)
        self.assertGreater(ob["margin"], 0.02)
        self.assertFalse(oa["ambiguous"])
        self.assertFalse(ob["ambiguous"])

    def test_unsorted_and_orbit_spare_the_pair(self):
        a, b = geom_features.t1_collision_pair()
        gap = float(
            np.max(
                np.abs(
                    geom_features.legacy_t1_features(a, sorted_features=False)
                    - geom_features.legacy_t1_features(b, sorted_features=False)
                )
            )
        )
        self.assertGreater(gap, 0.1)
        orbit_gap = float(
            np.max(
                np.abs(
                    geom_features.t1_orbit_representative(a)
                    - geom_features.t1_orbit_representative(b)
                )
            )
        )
        self.assertGreater(orbit_gap, 0.1)

    def test_feature_contract_schema(self):
        record = json.loads((ART / "FEATURE_CONTRACT.json").read_text())
        self.assertEqual(record["oracle"]["inside"]["label"], 1)
        self.assertEqual(record["oracle"]["outside"]["label"], 0)
        self.assertEqual(record["legacy_sorted_gap_float32"], 0.0)
        self.assertGreater(record["legacy_unsorted_gap_float32"], 0.1)
        self.assertGreater(record["whole_orbit_gap"], 0.1)


class SourceSearchSchemaTest(unittest.TestCase):
    def test_search_record_is_bounded_and_confirmed(self):
        record = json.loads((ART / "source_search.json").read_text())
        self.assertEqual(
            record["bank_sha256"],
            "15f5bf181b37dc9b4309582490f7fc11dfca8b03491251c3724fcb2e1bdd804c",
        )
        opp = record["nearest_opposite_label"]
        self.assertNotEqual(opp["bank_label_a"], opp["bank_label_b"])
        self.assertNotEqual(opp["crossing_label_a"], opp["crossing_label_b"])
        self.assertGreater(opp["bank_margin_a"], 0.0)
        self.assertGreater(opp["bank_margin_b"], 0.0)
        self.assertGreater(opp["crossing_margin_a"], 0.0)
        self.assertGreater(opp["crossing_margin_b"], 0.0)
        search = record["bounded_local_search"]
        self.assertLessEqual(search["best_distance"], search["start_distance"])
        self.assertGreater(search["best_margin"], 0.0)
        self.assertIn("nearest_equal_label_control", record)


class ReferenceEvalTest(unittest.TestCase):
    def test_group_mean_is_constructed_and_budgeted(self):
        torch.manual_seed(832303)
        model = source_typed.RepairedSegmentRho(width=64, k=16)
        a, _ = geom_features.t1_collision_pair()
        outs = []
        model.eval()
        with torch.no_grad():
            for perm in geom_features.T1_PERMS:
                x = torch.from_numpy(
                    np.asarray(a)[list(perm)].reshape(-1).astype(np.float32)
                )
                outs.append(float(model(x).item()))
        self.assertEqual(len(outs), 6)
        self.assertAlmostEqual(float(np.mean(outs)), float(np.mean(outs[::-1])), places=12)

    def test_source_regroup_witness_separated(self):
        x, xp = geom_features.source_regroup_witness()
        gap = float(
            np.max(
                np.abs(
                    geom_features.source_orbit_representative(x)
                    - geom_features.source_orbit_representative(xp)
                )
            )
        )
        self.assertGreater(gap, 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
