import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P2 = load_module("p2_matched_controls_optimization_test", "scripts/p2_matched_controls.py")
PIPE = load_module("run_phase2_pipeline_optimization_test", "scripts/run_phase2_pipeline.py")
DIAG = load_module("p2_matching_diagnostics_test", "scripts/p2_matching_diagnostics.py")


def test_boundary_slack_is_positive_at_margin_and_negative_outside():
    positions = np.zeros((3, 2), dtype=float)
    inside = np.array([[0.5, 0.0], [0.0, -0.25], [0.0, 0.0]])
    outside = np.array([[1.1, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert np.isclose(P2.boundary_slack(positions, inside), 0.5)
    assert P2.boundary_slack(positions, outside) < 0


def test_exact_spectrum_retry_is_deterministic_and_preserves_mode_power():
    eigenvectors = np.eye(4)
    positions = np.zeros((4, 2), dtype=float)
    delta = np.array([[0.25, 0.0], [0.0, 0.25], [0.0, 0.0], [0.0, 0.0]])
    first = P2.randomized_sign_spectral_delta_with_retries(
        eigenvectors, delta, positions, seed=17, max_attempts=4
    )
    second = P2.randomized_sign_spectral_delta_with_retries(
        eigenvectors, delta, positions, seed=17, max_attempts=4
    )
    assert first.success is True
    assert first.attempt_count == 1
    assert np.array_equal(first.signs, second.signs)
    assert np.allclose(first.delta, second.delta)
    assert np.allclose(P2.mode_power(eigenvectors, delta), P2.mode_power(eigenvectors, first.delta))


def test_exact_spectrum_retry_exhaustion_is_explicit():
    eigenvectors = np.eye(3)
    positions = np.array([[0.99, 0.0], [0.0, 0.0], [0.0, 0.0]])
    delta = np.array([[0.25, 0.0], [0.0, 0.0], [0.0, 0.0]])
    result = P2.randomized_sign_spectral_delta_with_retries(
        eigenvectors, delta, positions, seed=7, max_attempts=3, preserve_dc=True
    )
    assert result.success is False
    assert result.delta is None
    assert result.signs is None
    assert result.attempt_count == 3
    assert len(result.boundary_slacks) == 3
    assert max(result.boundary_slacks) < 0


def test_candidate_id_and_frequency_ratio_are_deterministic():
    first = P2.candidate_id("sample-1", "knn4", "team0_defender", [0, 1], [2, 3])
    second = P2.candidate_id("sample-1", "knn4", "team0_defender", [1, 0], [3, 2])
    assert first == second
    assert P2.frequency_ratio(0.04, 0.10, rq_reference=0.05, band_reference=0.20) == pytest.approx(0.8)


def test_stratified_selector_is_deterministic_and_covers_each_match():
    samples = [
        SimpleNamespace(match_id=match_id, sample_id=f"{match_id}_{index}", frame=index)
        for match_id in ["m1", "m2", "m3"]
        for index in range(5)
    ]
    first = PIPE.select_samples_per_match(samples, samples_per_match=2)
    second = PIPE.select_samples_per_match(samples, samples_per_match=2)
    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert [sample.sample_id for sample in first] == [
        "m1_0", "m1_4", "m2_0", "m2_4", "m3_0", "m3_4"
    ]


def test_response_blind_schema_rejects_model_outputs():
    safe = pd.DataFrame({"matched_set_id": ["s1"], "frequency_ratio": [0.5], "eligible": [True]})
    DIAG.assert_response_blind_schema(safe)
    with pytest.raises(ValueError, match="response-blind"):
        DIAG.assert_response_blind_schema(safe.assign(response_distance=0.1))
    with pytest.raises(ValueError, match="response-blind"):
        DIAG.assert_response_blind_schema(safe.assign(model_id="deepsets"))


def test_coverage_curve_is_nested_and_keeps_unmatched_sets_in_denominator():
    candidates = pd.DataFrame(
        {
            "matched_set_id": ["s1", "s1", "s2", "s3"],
            "candidate_id": ["c1", "c2", "c3", "c4"],
            "frequency_ratio": [0.5, 1.5, 1.2, 0.2],
            "eligible": [True, True, True, False],
        }
    )
    curve = DIAG.build_coverage_curve(candidates, expected_set_ids=["s1", "s2", "s3"], levels=[0.5, 1.0, 1.5])
    assert curve["n_expected_sets"].tolist() == [3, 3, 3]
    assert curve["n_matched_sets"].tolist() == [1, 1, 2]
    assert curve["coverage"].is_monotonic_increasing


def test_candidate_geometry_is_factorized_and_feasibility_is_response_blind():
    positions = np.array(
        [
            [-0.75, -0.25],
            [-0.45, 0.20],
            [-0.10, -0.10],
            [0.20, 0.25],
            [0.50, -0.20],
            [0.75, 0.15],
        ],
        dtype=float,
    )
    adjacency = np.zeros((6, 6), dtype=float)
    for left, right in zip(range(5), range(1, 6)):
        adjacency[left, right] = adjacency[right, left] = 1.0
    eigenvalues, eigenvectors = np.linalg.eigh(P2.normalized_laplacian(adjacency))
    sample = PIPE.Sample(
        sample_index=0,
        sample_id="sample-1",
        match_id="match-1",
        split="train",
        frame=10,
        positions=positions,
        adjacency=adjacency,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        team_slots=np.zeros(6, dtype=int),
        role_groups=("defender",) * 6,
    )
    graph = PIPE.GraphView("knn4", adjacency, eigenvalues, eigenvectors)
    anchor = np.array([0, 1], dtype=int)
    candidates = P2.enumerate_same_team_supports(
        sample.team_slots, 0, len(anchor), exclude=anchor, max_overlap=1
    )
    calipers = {
        "rq_normalized_caliper": 0.05,
        "band_power_l1_caliper": 0.20,
        "density_scale": 0.10,
        "cut_scale": 1.0,
        "require_same_component_count": True,
    }

    geometry = PIPE.build_candidate_geometry(
        sample,
        graph,
        coalition_id="team0_defender",
        team=0,
        role="defender",
        anchor_support=anchor,
        candidates=candidates,
        n_bands=3,
        calipers=calipers,
    )
    geometry_frame = pd.DataFrame(geometry)
    assert len(geometry_frame) == len(candidates)
    assert geometry_frame["candidate_id"].is_unique
    assert {"epsilon", "direction"}.isdisjoint(geometry_frame.columns)
    assert (geometry_frame["frequency_ratio"] >= 0).all()
    DIAG.assert_response_blind_schema(geometry_frame)

    feasibility = PIPE.build_candidate_feasibility(
        sample,
        geometry,
        matched_set_id="set-1",
        epsilon=0.25,
        direction=np.array([1.0, 0.0]),
    )
    feasibility_frame = pd.DataFrame(feasibility)
    assert len(feasibility_frame) == len(candidates)
    assert feasibility_frame["matched_set_id"].eq("set-1").all()
    assert np.isfinite(feasibility_frame["boundary_slack"]).all()
    assert (
        feasibility_frame["eligible"].astype(bool)
        <= feasibility_frame["boundary_valid"].astype(bool)
    ).all()
    DIAG.assert_response_blind_schema(feasibility_frame)


def test_topology_control_ranks_frequency_before_topology_score():
    sample = SimpleNamespace(positions=np.zeros((4, 2), dtype=float))
    geometry = [
        {
            "candidate_id": "frequency-first",
            "candidate_support": "[0,1]",
            "component_match": True,
            "component_match_required": True,
            "rq_difference": 0.01,
            "band_power_l1_difference": 0.04,
            "frequency_ratio": 0.2,
            "topology_match_score": 9.0,
        },
        {
            "candidate_id": "topology-first",
            "candidate_support": "[2,3]",
            "component_match": True,
            "component_match_required": True,
            "rq_difference": 0.04,
            "band_power_l1_difference": 0.16,
            "frequency_ratio": 0.8,
            "topology_match_score": 1.0,
        },
    ]
    feasibility = [
        {"candidate_id": item["candidate_id"], "eligible": True}
        for item in geometry
    ]
    support, _, _, _, selected_id, _ = PIPE.choose_topology_frequency_from_diagnostics(
        sample,
        geometry,
        feasibility,
        epsilon=0.25,
        direction=np.array([1.0, 0.0]),
        calipers={"rq_normalized_caliper": 0.05, "band_power_l1_caliper": 0.20},
    )
    assert selected_id == "frequency-first"
    assert support.tolist() == [0, 1]
