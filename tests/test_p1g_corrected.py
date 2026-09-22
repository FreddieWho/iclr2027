"""Unit tests for the R1 threshold loader and corrected P1G flow (R1/R9.7)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "next_novelty"))
import p1g_repair as P1G  # noqa: E402

REAL = json.load(open(ROOT / "artifacts/next_novelty/p1_freeze.json"))


def test_loader_accepts_real_schema_and_returns_frozen_values():
    assert P1G.load_threshold(REAL, "s11_clean", "t25") == 20.7555
    assert P1G.load_threshold(REAL, "s23_clean", "t25") == 22.1445
    assert P1G.load_threshold(REAL, "s47_clean", "t25") == 20.7557


def test_loader_hard_fails_missing_key():
    with pytest.raises(KeyError):
        P1G.load_threshold(REAL, "s11_nope", "t25")
    with pytest.raises(KeyError):
        P1G.load_threshold(REAL, "s11", "t25")  # legacy wrong schema must fail
    with pytest.raises(KeyError):
        P1G.load_threshold(REAL, "s11_clean", "t99")


def test_loader_rejects_non_mapping_and_bad_types():
    with pytest.raises(ValueError):
        P1G.load_threshold([1, 2], "s11_clean", "t25")
    with pytest.raises(ValueError):
        P1G.load_threshold({"s11_clean": {"t25": "oops"}}, "s11_clean", "t25")


def test_corrected_output_thresholds_byte_equal_freeze():
    d = json.load(open(ROOT / "artifacts/next_novelty/p1g_confirm/P1G_CORRECTED.json"))
    for s in ("s11", "s23", "s47"):
        want = REAL[f"{s}_clean"]["t25"]
        assert d["results"][s]["thr"] == want, (s, d["results"][s]["thr"], want)


def test_main_comparison_is_full_baseline_fixed_cohort():
    """Main cohort must NOT screen on repair-start-correct; only the renamed
    secondary may. n_high (main) must exceed common-start-correct n."""
    d = json.load(open(ROOT / "artifacts/next_novelty/p1g_confirm/P1G_CORRECTED.json"))
    for s, r in d["results"].items():
        assert "main" in r and "common_start_correct_secondary" in r
        assert r["n_high"] > r["common_correct_n"], s
        m = r["main"]
        assert 0 < m["repair_start_err"] < 1  # repair start errors visible in main
        assert m["n_repair_start_wrong"] >= 1
        assert m["paired_delta_ci"][0] <= m["paired_delta"] <= m["paired_delta_ci"][1]


def test_output_never_overwrites_legacy_file():
    assert (ROOT / "artifacts/next_novelty/p1g_confirm/P1G.json").exists()
    d = json.load(open(ROOT / "artifacts/next_novelty/p1g_confirm/P1G_CORRECTED.json"))
    assert d["mode"] == "confirm"
    assert d["manifest"]["bank"] == "confirm1007"
