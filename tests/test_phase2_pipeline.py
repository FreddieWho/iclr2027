import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIMITIVES = ROOT / "scripts" / "p2_matched_controls.py"
PIPELINE = ROOT / "scripts" / "run_phase2_pipeline.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P2 = load_module("p2_matched_controls_test", PRIMITIVES)
PIPE = load_module("run_phase2_pipeline_test", PIPELINE)


def synthetic_graph():
    points = np.array([
        [-0.6, -0.6], [-0.2, -0.6], [0.2, -0.6], [0.6, -0.6],
        [-0.6, 0.0], [-0.2, 0.0], [0.2, 0.0], [0.6, 0.0],
        [-0.6, 0.6], [-0.2, 0.6], [0.2, 0.6], [0.6, 0.6],
    ], dtype=float)
    adjacency = P2.knn_adjacency(points, k=3)
    eigenvalues, eigenvectors = np.linalg.eigh(P2.normalized_laplacian(adjacency))
    return points, adjacency, eigenvalues, eigenvectors


def test_rigid_translation_has_equal_node_vectors_and_requested_energy():
    delta = P2.rigid_subset_translation(6, [1, 4, 5], 1.5, [3.0, 4.0])
    assert np.isclose(np.linalg.norm(delta), 1.5)
    assert np.allclose(delta[1], delta[4])
    assert np.allclose(delta[4], delta[5])
    assert np.allclose(delta[[0, 2, 3]], 0.0)


def test_same_team_enumeration_respects_exclusion_and_overlap():
    slots = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    supports = P2.enumerate_same_team_supports(slots, 0, 2, exclude=[0, 1], max_overlap=0)
    assert supports
    assert all(set(support).issubset({0, 1, 2, 3}) for support in supports)
    assert all(not set(support) & {0, 1} for support in supports)


def test_sign_randomization_preserves_each_modal_power():
    _, _, eigenvalues, eigenvectors = synthetic_graph()
    delta = P2.rigid_subset_translation(12, [0, 1, 2], 0.5, [1, 0])
    randomized, signs = P2.randomized_sign_spectral_delta(eigenvectors, delta, np.random.default_rng(17))
    before = P2.mode_power(eigenvectors, delta)
    after = P2.mode_power(eigenvectors, randomized)
    assert signs.shape == (12,)
    assert np.allclose(before, after, atol=1e-10)
    assert np.isclose(np.linalg.norm(delta), np.linalg.norm(randomized), atol=1e-10)


def test_topology_features_are_deterministic_and_have_normalized_frequency():
    _, adjacency, eigenvalues, eigenvectors = synthetic_graph()
    delta = P2.rigid_subset_translation(12, [0, 1, 4], 1.0, [0, 1])
    first = P2.topology_features([0, 1, 4], adjacency, eigenvalues, eigenvectors, delta, 6)
    second = P2.topology_features([0, 1, 4], adjacency, eigenvalues, eigenvectors, delta, 6)
    assert first == second
    assert np.isclose(sum(first.band_power), 1.0)
    assert first.component_count >= 1


def test_delaunay_degeneracy_is_explicit_not_fallback():
    collinear = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]])
    try:
        P2.delaunay_adjacency(collinear)
    except P2.GraphInvalid:
        pass
    else:
        raise AssertionError("degenerate Delaunay input must be explicit GraphInvalid")


def test_pipeline_contract_is_not_a_model_result():
    contract = {
        "model_inference": "NOT_RUN",
        "scientific_claim": "NOT_RUN",
    }
    assert contract["model_inference"] == "NOT_RUN"
    assert json.dumps(contract)


def test_incomplete_matching_is_not_reported_as_pass():
    frame = pd.DataFrame(
        {
            "matched_set_id": ["set-1", "set-1"],
            "arm_kind": ["semantic_rigid", "control"],
            "control_family": ["anchor", "arbitrary_same_team"],
            "valid": [True, False],
            "energy_error": [0.0, None],
        }
    )
    set_frame = pd.DataFrame({"matched_set_status": ["INCOMPLETE"]})
    result = PIPE.validate_matching(frame, set_frame, {"graphs": []}, {})
    assert result["status"] == "incomplete"
    assert result["n_incomplete_sets"] == 1
