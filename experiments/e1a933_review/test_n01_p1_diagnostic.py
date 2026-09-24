"""Regression checks for the N01 P1 diagnostic (R04 contract, vision quartets).

Mirrors `test_data_repairs.py::test_high_confidence_uses_start_and_is_parent_order_invariant`
for the vision lane: retention is decided by the start score only, and the two
P1 denominators (unconditional vs start-correct) must stay separate.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "e1a933_review"))

from n01_p1_diagnostic import KEEP, FLIP, p1_metrics, start_confidence_retention  # noqa: E402


def test_retention_uses_start_score_and_is_order_invariant():
    # R04 counterexample shape: start=[10,1], end=[.1,10], threshold 5 selects the first.
    start = {11: 10.0, 23: 1.0}
    assert start_confidence_retention(start, np.array([11, 23]), 5.0).tolist() == [True, False]
    assert start_confidence_retention(start, np.array([23, 11]), 5.0).tolist() == [False, True]
    assert start_confidence_retention(start, np.array([23, 11]), 0.5).tolist() == [True, True]


def test_vision_quartet_columns_are_keep_keep_flip():
    assert KEEP.tolist() == [False, True, True, False]
    assert FLIP.tolist() == [True, False, False, True]


def test_p1_denominators_are_not_pooled():
    # Two quartets, both retained.  q0 start-correct, q1 start-wrong.
    logits = np.array([[2.0, -1.0, -1.0, -1.0],
                       [-2.0, 1.0, 1.0, 1.0]])
    labels = np.array([[1, 1, 1, 0],
                       [1, 1, 1, 0]])
    parent = np.array([0, 1])
    retained = np.array([True, True])
    out, *_ = p1_metrics(logits, labels, parent, retained)

    assert out["n_quartets"] == 2 and out["n_parents"] == 2
    assert out["keep_n_retained"] == 4 and out["flip_n_retained"] == 2
    # D2 keeps the start-wrong quartet; D3 must drop it
    assert out["n_quartets_retained_startok"] == 1
    assert out["keep_n_retained_startok"] == 2 and out["flip_n_retained_startok"] == 1
    assert out["keep_err_retained"] == 0.5 and out["keep_err_retained_startok"] == 1.0
    assert out["flip_err_retained"] == 0.5 and out["flip_err_retained_startok"] == 0.0
    # D1 is the unfiltered bank and must also differ from D2 here
    assert out["keep_err_all"] == 0.5 and out["flip_err_all"] == 0.5


def test_p1_metrics_reports_realised_coverage_not_a_nominal_fraction():
    logits = np.zeros((4, 4))
    labels = np.array([[1, 1, 1, 0]] * 4)
    parent = np.array([0, 0, 1, 1])
    retained = np.array([True, False, False, False])
    out, *_ = p1_metrics(logits, labels, parent, retained)
    assert out["coverage_quartet"] == 0.25
    assert out["coverage_parent"] == 0.5
