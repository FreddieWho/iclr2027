"""R05/R06 production-path contract regressions.

Ported from `vision_regression.py` so that pytest collects them. Every check calls the
same production functions used by the vision pipeline, so restoring the pre-fix
behaviour makes the corresponding test fail:

- R06: a head-only probe must freeze BatchNorm buffers, not just parameters.
  Mutation guard: bare `net.train()` changes the backbone hash.
- R05: the auxiliary target must be matched over the *hidden endpoint group* H, with one
  consistent group element per structure. Mutation guards: the old ordered (index-fixed)
  objective, and per-coordinate sorting.

Scope: these are contract regressions on synthetic structures. The heavier 512-parent
observational audit (pixel-orbit ambiguity, label constancy, irreducible-variance bound)
is an audit, not a unit test, and remains in `vision_regression.py`.

Correction carried over from `vision_regression.py`: its four
`normalization_commutes_<p>` checks were **non-discriminating** and were removed here.
Permuting a structure only row-permutes its orbit, and the check compared
`sort(normalize(orbit))` against `sort(normalize(orbit[p], stats))`; sorting along the group
axis makes any per-dimension affine normalization agree, so the check passed for the pooled
scheme *and* for the buggy per-index scheme. It is replaced by two checks that can fail:
`test_orbit_equivariant_under_declared_group` and
`test_colour_swap_is_outside_the_declared_group`.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/f095_campaign"))

from d04_vision_train import backbone_hash, make_net, train_mode  # noqa: E402
from u06_relation_distill import (  # noqa: E402
    H_ENDPOINTS,
    matched_mse,
    normalize_orbit,
    shuffled_indices,
    target_orbit,
)

torch.set_num_threads(2)


def _net(mode, seed=803):
    """Build through the production factory; skip only if pretrained weights are absent."""
    try:
        return make_net(mode, seed)
    except Exception as exc:  # pragma: no cover - offline/uncached environment
        pytest.skip(f"pretrained weights unavailable: {exc}")


def _coords(n=32, seed=7):
    """Synthetic (n, 4, 2) structures. Data-independent: H-invariance is a property of the code."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, 4, 2))


def _orbit_tensors(coords=None):
    orbit = target_orbit(_coords() if coords is None else coords)
    normalized, stats = normalize_orbit(orbit)
    return (torch.tensor(normalized, dtype=torch.float32), stats)


# --------------------------------------------------------------------------- R06


def test_headonly_probe_freezes_parameters_and_buffers():
    """A frozen probe must not move parameters *or* BN running statistics."""
    net = _net("headonly")
    before = backbone_hash(net)
    train_mode(net, "headonly")
    opt = torch.optim.SGD(net.fc.parameters(), lr=0.1)
    net(torch.rand(4, 3, 64, 64)).mean().backward()
    opt.step()
    assert backbone_hash(net) == before


def test_bare_train_mode_bn_bug_is_detected():
    """Mutation guard: the pre-fix behaviour (`net.train()`) must change BN buffers.

    If a future refactor made `backbone_hash` ignore buffers, or made BN stateless, this
    guard would stop firing and the R06 fix could silently regress.
    """
    net = _net("headonly")
    before = backbone_hash(net)
    net.train()
    net(torch.rand(4, 3, 64, 64))
    assert backbone_hash(net) != before


def test_bn_adaptation_is_a_distinct_arm():
    """BN adaptation is a separate, explicitly allowed arm - not an accident of `train()`."""
    net = _net("bnadapt")
    before = backbone_hash(net)
    train_mode(net, "bnadapt")
    net(torch.rand(4, 3, 64, 64))
    assert backbone_hash(net) != before


# --------------------------------------------------------------------------- R05


def test_one_consistent_group_element_gives_zero_matched_mse():
    """Matching over H must be able to reach exactly zero for a consistent target."""
    targets, _ = _orbit_tensors()
    pred = targets[:, 1]
    assert float(matched_mse(pred, targets)) < 1e-12


def test_matched_mse_is_invariant_to_the_hidden_group_element():
    """Every g in H must give the same matched loss: H is the declared invariance group."""
    targets, _ = _orbit_tensors()
    baseline = float(matched_mse(targets[:, 0], targets))
    for index in range(len(H_ENDPOINTS)):
        assert float(matched_mse(targets[:, index], targets)) == pytest.approx(baseline, abs=1e-6)


def test_ordered_target_mse_conflicts_with_matching():
    """Mutation guard: the old index-fixed (ordered) objective is inconsistent with H."""
    targets, _ = _orbit_tensors()
    pred = targets[:, 1]
    assert float(((pred - targets[:, 0]) ** 2).mean()) > 1e-4


def test_orbit_equivariant_under_declared_group():
    """H must be an actual symmetry group of the target map, not just a declared list.

    For every g in H the orbit of the permuted structure must be a row-permutation of the
    original orbit (the same set of relation vectors). This fails if H stops being closed
    under composition with `rel10`, e.g. if a non-involution is added to H_ENDPOINTS.
    """
    coords = _coords()
    orbit = target_orbit(coords)
    for permutation in H_ENDPOINTS:
        permuted = target_orbit(coords[:, permutation])
        assert np.allclose(np.sort(permuted, axis=1), np.sort(orbit, axis=1))


def test_colour_swap_is_outside_the_declared_group():
    """Boundary guard: H must be exactly the red/blue-internal swaps, not the colour swap.

    Without this, the equivariance check above would be satisfied by a trivially larger
    group and would stop discriminating.
    """
    coords = _coords()
    orbit = target_orbit(coords)
    colour_swap = target_orbit(coords[:, [2, 3, 0, 1]])
    assert not np.allclose(np.sort(colour_swap, axis=1), np.sort(orbit, axis=1))
    assert np.abs(np.sort(colour_swap, axis=1) - np.sort(orbit, axis=1)).max() > 1e-3


def test_independent_coordinate_sort_is_not_a_valid_matching():
    """Mutation guard: per-coordinate sorting is not one consistent group element."""
    targets, _ = _orbit_tensors()
    chimeras = targets[:, 0].clone()
    chimeras[:, :3] = targets[:, 1, :3]
    assert float(matched_mse(chimeras, targets)) > 1e-4


def test_shuffled_indices_independent_of_global_rng():
    """The shuffle must own its Generator, not consume the ambient RNG stream."""
    first = shuffled_indices(64, 803)
    torch.rand(1000)
    assert torch.equal(first, shuffled_indices(64, 803))
