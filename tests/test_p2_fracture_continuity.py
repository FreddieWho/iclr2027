import numpy as np
import pytest

from scripts.p2_fracture_controls import (
    boundary_valid, deterministic_tie_break, enumerate_target_supports,
    enumerate_vector_bijections, make_fracture_anchor, reassigned_vector_field,
    scale_displacement_field, spectral_topology_features_2d, stable_seed,
    unit_base_displacement_field, validate_fracture_supports,
)


def test_deterministic_seed_anchor_and_tie_break():
    positions = np.array([[-.8, -.8], [.8, -.8], [-.8, .8], [.8, .8], [0., 0.]])
    kwargs = dict(positions=positions, team_slots=[1, 1, 1, 1, 1], source_support=[0, 1],
                  target_support=[2, 3], epsilon=.2, seed_parts=("sample", 7))
    left, right = make_fracture_anchor(**kwargs), make_fracture_anchor(**kwargs)
    assert stable_seed("sample", 7) == left.seed
    assert np.array_equal(left.displacement, right.displacement)
    alternate = make_fracture_anchor(**{**kwargs, "target_support": [3, 4]})
    assert np.array_equal(left.base_field, alternate.base_field)
    assert deterministic_tie_break(["b", "a"], 9) == deterministic_tie_break(["b", "a"], 9)


def test_energy_and_path_scaling():
    base = np.array([[3., 4.], [0., 0.]]) / 5
    assert np.isclose(np.linalg.norm(scale_displacement_field(base, .3)), .3)
    assert np.allclose(scale_displacement_field(base, .6), 2 * scale_displacement_field(base, .3))


def test_exact_vector_multiset_round_trip_and_bijections():
    base = unit_base_displacement_field(4, [0, 1], ("unit",))
    moved = reassigned_vector_field(base, [0, 1], [2, 3], {0: 3, 1: 2}, .25)
    scaled = scale_displacement_field(base, .25)
    assert sorted(map(tuple, scaled[scaled.any(axis=1)])) == sorted(map(tuple, moved[moved.any(axis=1)]))
    assert len(enumerate_vector_bijections([0, 1], [2, 3])) == 2


def test_support_team_size_disjoint_and_boundary():
    teams = [1, 1, 1, 1, 2]
    assert validate_fracture_supports(teams, [0, 1], [2, 3]) == ((0, 1), (2, 3))
    assert len(enumerate_target_supports(teams, [0, 1])) == 1
    assert boundary_valid(np.array([[.8, 0.], [0., 0.]]), np.array([[.1, 0.], [0., 0.]]))
    assert not boundary_valid(np.array([[.8, 0.]]), np.array([[.3, 0.]]))
    with pytest.raises(ValueError): validate_fracture_supports(teams, [0, 1], [1, 2])


def test_2d_spectral_frequency_rq_topology_features():
    eigenvectors = np.eye(4)
    features = spectral_topology_features_2d(np.arange(4.), eigenvectors,
        np.array([[1., 2.], [0., 0.], [2., 0.], [0., 0.]]), [0, 2],
        np.array([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0.]]), 2)
    assert features.power_x == (1., 0., 4., 0.)
    assert features.power_y == (4., 0., 0., 0.)
    assert np.isclose(sum(features.band_power_total), 1.)
    assert features.rq_total >= 0 and features.component_count == 1
    with pytest.raises(ValueError): spectral_topology_features_2d(np.arange(4.), eigenvectors, np.zeros((4, 2)), [0, 2], np.eye(4), 2)
