"""Lane B (R3/R4/R5/R9.7) acceptance tests for the v2 evidence artifacts.

These tests call the production artifacts produced by the v2 scripts and
verify the schema/estimand invariants the rectification plan requires.
They do NOT retrain or re-evaluate models; heavy regeneration is done by
running the v2 scripts themselves.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(rel):
    p = ROOT / rel
    if not p.exists():
        pytest.skip("missing artifact: %s (run the v2 script first)" % rel)
    return json.loads(p.read_text())


# ---------- R3: relflip confirm reproduce entry ----------


class TestRelflipConfirmV2:
    def test_provenance_recorded(self):
        d = _load("artifacts/next_novelty/relflip_v2/RELFLIP_CONFIRM.json")
        assert d["bank"] == "confirm1007"
        assert len(d["bank_sha256"]) == 64
        assert d["bank_manifest"]["tag"] == "confirm1007"
        prov = d["provenance"]
        assert "model.pt" in prov["raw_prediction_source"]
        assert "not oracle-free" in prov["feature_boundary"]

    def test_four_arms_same_bank_paired(self):
        d = _load("artifacts/next_novelty/relflip_v2/RELFLIP_CONFIRM.json")
        arms = d["arms"]
        for arm in ("relflip", "raw_clean", "raw_flipmine", "relfeat"):
            assert arm in arms, "missing arm %s" % arm
            for seed in ("s11", "s23", "s47"):
                m = arms[arm][seed]
                assert m is not None, "%s/%s reported missing" % (arm, seed)
                assert "error" not in m
                assert m["n_q"] == 509  # same bank -> same quartet count
                assert len(m["model_sha256"]) == 64

    def test_migration_same_bank_identity(self):
        d = _load("artifacts/next_novelty/relflip_v2/RELFLIP_CONFIRM.json")
        mig = d["migration"]
        for seed in ("s11", "s23", "s47"):
            m = mig[seed]
            assert m is not None, "migration %s must not be stitched" % seed
            rf, re_, mm = m["R_full"][0], m["R_endpoint"][0], m["M"][0]
            assert abs(rf + mm - re_) < 2e-3  # identity on rounded values
            assert 0.0 <= m["M_over_R_endpoint"] <= 1.0

    def test_reproduce_entry_parameterized(self):
        out = subprocess.run(
            [sys.executable, "experiments/ccm_audit/relflip_confirm_eval.py", "--help"],
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0
        for flag in ("--bank", "--model-map", "--out"):
            assert flag in out.stdout


# ---------- R4: P3 v2 (8x8 matrix, identity, ratios, paired dJ) ----------


class TestP3FinalV2:
    def test_full_8x8_transition_matrix(self):
        d = _load("artifacts/next_novelty/p3_v2/P3_FINAL.json")
        for seed in ("s11", "s23", "s47"):
            mat = d[seed]["transition_matrix_full8x8"]
            assert len(mat) == 8, "8 source states"
            for src, row in mat.items():
                assert len(row) == 8, "8 target states for %s" % src
                assert set(row.keys()) == set(mat.keys())

    def test_identity_and_ratio(self):
        d = _load("artifacts/next_novelty/p3_v2/P3_FINAL.json")
        for seed in ("s11", "s23", "s47"):
            m = d[seed]["metrics"]
            assert m["identity_R_endpoint_eq_R_full_plus_M"] is True, seed
            assert abs(m["identity_residual"]) < 1e-9
            # M/R_endpoint ~79-94%, no denominator mixing (2dp rounding)
            assert 0.79 <= round(m["M_over_R_endpoint"], 2) <= 0.94

    def test_paired_delta_j_ci_same_resample(self):
        d = _load("artifacts/next_novelty/p3_v2/P3_FINAL.json")
        for seed in ("s11", "s23", "s47"):
            m = d[seed]["metrics"]
            dj = m["delta_J_repair_minus_clean"]
            assert len(dj) == 2 and len(dj[1]) == 2
            lo, hi = dj[1]
            assert lo <= dj[0] <= hi

    def test_rank_sort_diagnostic_present(self):
        d = _load("artifacts/next_novelty/p3_v2/P3_FINAL.json")
        for seed in ("s11", "s23", "s47"):
            diag = d[seed]["metrics"]["rank_diag"]
            assert 0.0 <= diag["spearman_ABscore_clean_vs_flipmine"] <= 1.0
            assert 0.0 <= diag["same_AB_sign_frac"] <= 1.0
            assert diag["n_base110"] > 0

    def test_reproduces_frozen_point_estimates(self):
        """v2 recomputation must reproduce the frozen v1 point estimates."""
        new = _load("artifacts/next_novelty/p3_v2/P3_FINAL.json")
        old = json.loads((ROOT / "artifacts/next_novelty/p3/P3_FINAL.json").read_text())
        for seed in ("s11", "s23", "s47"):
            for k in ("R_full", "R_endpoint", "M_migrate", "J_clean", "J_repair"):
                assert new[seed]["metrics"][k][0] == old[seed]["metrics"][k][0], (seed, k)


# ---------- R5: P2 v2 (dev-mask fix, split estimands, threshold enum) ----------


class TestP2OperatingV2:
    def test_estimands_split_and_counts(self):
        d = _load("artifacts/next_novelty/p2_v2/P2_OPERATING_V2.json")
        assert d["n_dev_scenes"] > 0 and d["n_eval_scenes"] > 0
        assert d["n_dev_rows"] > 0
        for seed in ("s11", "s23", "s47"):
            s = d[seed]
            assert s["equal_cov_eval"], "no equal-coverage eval levels"
            for lv in s["equal_cov_eval"]:
                assert "policy_quality_difference" in lv
                assert "common_act_difference_ci" in lv
                assert "eval_cov_gap" in lv  # dev-matched coverage != eval-matched
            ds = s["dev_stats"]
            assert "repair_coverage_lexico_tau0.5" in ds
            assert "repair_prevalence_devrows" in ds

    def test_threshold_enum_and_edge(self):
        d = _load("artifacts/next_novelty/p2_v2/P2_OPERATING_V2.json")
        for seed in ("s11", "s23", "s47"):
            s = d[seed]
            te = s["threshold_enum"]
            assert te["n_thresholds_tried"] > 0
            assert "dev_lexico_success_at_tau" in te
            if seed == "s11":
                # the frozen wide-grid affine must be flagged at its grid edge
                assert s["affine_fit_grid"]["at_grid_edge"] is True
            # score-shape control vs best-dev baseline must be separate policies
            assert s["lex_affine_grid"]["control_kind"] != s["lex_thresh_enum"]["control_kind"]
            assert "paired_ci" in s


# ---------- R3 entry: relfeat_eval parameterized ----------


class TestRelfeatEvalEntry:
    def test_parameterized_help(self):
        out = subprocess.run(
            [sys.executable, "experiments/ccm_audit/relfeat_eval.py", "--help"],
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0
        for flag in ("--bank", "--model-map", "--out"):
            assert flag in out.stdout
