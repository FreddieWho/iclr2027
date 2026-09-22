"""U2 ordering primitives: G-separability, monotone invariance, H bound."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "ccm_audit"))
from ordering import (  # noqa: E402
    G_of,
    H_of,
    boundary_mask,
    fixed_threshold_J,
    separable_mask,
)


def test_separable_iff_positive_G_both_directions():
    # yAB=1 needs fAB above both; yAB=0 needs fAB below both.
    assert separable_mask([0.0], [0.1], [0.5], [1]).tolist() == [True]
    assert separable_mask([0.0], [0.1], [-0.5], [1]).tolist() == [False]
    assert separable_mask([0.0], [0.1], [-0.5], [0]).tolist() == [True]
    assert separable_mask([0.0], [0.1], [0.5], [0]).tolist() == [False]


def test_boundary_equality_reported_not_separable():
    assert separable_mask([0.0], [0.0], [0.0], [1]).tolist() == [False]
    assert boundary_mask([0.0], [0.0], [0.0], [1]).tolist() == [True]


def test_positive_affine_preserves_sign_G():
    fA = np.array([-1.0, 0.2, 3.0])
    fB = np.array([0.5, -0.4, 1.0])
    fAB = np.array([2.0, -2.0, 0.0])
    yAB = np.array([1, 0, 1])
    g = G_of(fA, fB, fAB, yAB)
    for alpha, b in [(2.0, -5.0), (0.1, 7.0), (1.0, 0.0)]:
        g2 = G_of(alpha * fA + b, alpha * fB + b, alpha * fAB + b, yAB)
        assert np.sign(g2).tolist() == np.sign(g).tolist()


def test_strictly_monotone_preserves_separability():
    fA = np.array([-1.0, 0.2])
    fB = np.array([0.5, -0.4])
    fAB = np.array([2.0, -2.0])
    yAB = np.array([1, 0])
    before = separable_mask(fA, fB, fAB, yAB)
    mono = np.tanh
    after = separable_mask(mono(fA), mono(fB), mono(fAB), yAB)
    assert after.tolist() == before.tolist()


def test_H_is_upper_bound_for_any_fixed_threshold():
    rng = np.random.default_rng(7)
    fA, fB, fAB = rng.normal(size=(3, 60))
    yAB = rng.integers(0, 2, size=60)
    yS = 1 - yAB
    H, _ = H_of(fA, fB, fAB, yAB)
    for t in (-1.0, 0.0, 1.0):
        assert fixed_threshold_J(fA, fB, fAB, yS, yS, yAB, t) <= H + 1e-9


def test_dev_threshold_transfer_never_exceeds_eval_H():
    # Property mirrored by u2_ordering_eval: a dev-selected global threshold
    # evaluated elsewhere cannot beat that pool's oracle separability H.
    rng = np.random.default_rng(11)
    fA, fB, fAB = rng.normal(size=(3, 80))
    yAB = rng.integers(0, 2, size=80)
    yS = 1 - yAB
    sel, ev = np.arange(40), np.arange(40, 80)
    cands = np.unique(np.concatenate([fA[sel], fB[sel], fAB[sel]]))
    best = max(fixed_threshold_J(fA[sel], fB[sel], fAB[sel], yS[sel],
                                 yS[sel], yAB[sel], float(t)) for t in cands)
    He, _ = H_of(fA[ev], fB[ev], fAB[ev], yAB[ev])
    assert best <= 1.0 and He >= 0.0
    # transfer J at the selected threshold is bounded by eval H
    t_best = cands[np.argmax([fixed_threshold_J(
        fA[sel], fB[sel], fAB[sel], yS[sel], yS[sel], yAB[sel], float(t))
        for t in cands])]
    assert fixed_threshold_J(fA[ev], fB[ev], fAB[ev], yS[ev], yS[ev],
                             yAB[ev], float(t_best)) <= He + 1e-9
