"""L007 follow-up primitives: crossing width on a logit grid."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "last15h"))
from l007_grazing import crossing_width  # noqa: E402


def test_crossing_width_symmetric():
    t = np.linspace(0, 1, 129)
    lg = (t - 0.5) * 10
    w = crossing_width(t, lg, 0.5, k=1.0)
    assert w is not None and abs(w - 0.2) < 0.02


def test_crossing_width_unreached_band():
    t = np.linspace(0, 1, 129)
    lg = (t - 0.5) * 2
    assert crossing_width(t, lg, 0.5, k=2.0) is None


def test_crossing_width_narrower_band_smaller():
    t = np.linspace(0, 1, 129)
    lg = (t - 0.5) * 10
    w1 = crossing_width(t, lg, 0.5, k=1.0)
    w2 = crossing_width(t, lg, 0.5, k=2.0)
    assert w1 < w2


def test_greedy_assignment_matches_diagonal():
    from l007_connectivity import linear_sum_assignment
    rng = np.random.default_rng(0)
    C = np.eye(8) + 0.01 * rng.normal(size=(8, 8))
    perm = linear_sum_assignment(-C)
    assert sorted(perm.tolist()) == list(range(8))
    assert (perm == np.arange(8)).all()


def test_greedy_assignment_beats_identity():
    from l007_connectivity import linear_sum_assignment, match_corr
    rng = np.random.default_rng(2)
    A = rng.normal(size=(200, 16))
    P0 = rng.permutation(16)
    B = A[:, P0] + 0.05 * rng.normal(size=(200, 16))
    perm = match_corr(A, B)
    # B[:, j] ~= A[:, P0[j]] so perm[i] must equal inverse-permutation
    assert (perm == np.argsort(P0)).all()
