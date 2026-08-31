import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_phase1_pipeline.py"
SPEC = importlib.util.spec_from_file_location("phase1_pipeline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_sample():
    rng = np.random.default_rng(7)
    positions = rng.uniform(-0.35, 0.35, size=(20, 2))
    adjacency = MODULE.P0.knn_adjacency(positions, k=4)
    eigenvalues, eigenvectors = np.linalg.eigh(MODULE.P0.normalized_laplacian(adjacency))
    return MODULE.Sample(
        sample_id="sc_test_0000",
        match_id="test-match",
        split="train",
        frame=0,
        timestamp=None,
        positions=positions,
        raw_positions=positions.copy(),
        centered_positions=positions - positions.mean(axis=0, keepdims=True),
        adjacency=adjacency,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        team_slots=np.array([0] * 10 + [1] * 10),
        roles=["defender"] * 6 + ["midfielder"] * 8 + ["forward"] * 6,
        role_groups=["defender"] * 6 + ["midfielder"] * 8 + ["forward"] * 6,
        phase_label="low_block",
        phase_status="unique",
        pitch_length=105.0,
        pitch_width=68.0,
    )


def test_phase_split_is_fixed_and_disjoint():
    rows = MODULE.split_rows()
    assert len(rows) == 10
    by_split = {}
    for row in rows:
        by_split.setdefault(row["split"], set()).add(row["match_id"])
    assert {len(by_split["train"]), len(by_split["dev"]), len(by_split["heldout"])} == {6, 2}
    assert not (by_split["train"] & by_split["dev"])
    assert not (by_split["train"] & by_split["heldout"])
    assert not (by_split["dev"] & by_split["heldout"])


def test_point_only_contract_defers_vision_explicitly():
    contract = MODULE.build_execution_contract(
        n_samples_per_match=25,
        visual_stride=10,
        skip_vision=False,
        point_only=True,
    )
    assert contract["mainline_modality"] == "coordinate_point_set"
    assert contract["vision_policy"] == "deferred_auxiliary"
    assert contract["point_only"] is True
    assert contract["skip_vision"] is True


def test_interventions_have_unique_pairs_and_exact_modes():
    sample = synthetic_sample()
    interventions = MODULE.generate_interventions([sample])
    assert interventions["intervention_id"].is_unique
    assert interventions["pair_id"].is_unique
    exact = interventions[interventions["kind"].astype(str).str.startswith("exact_mode_")]
    assert set(exact["mode_rank"].dropna().astype(int)) == set(range(20))
    assert (interventions["kind"] == "identity").sum() == 1
    validation = MODULE.validate_interventions([sample], interventions)
    assert validation["status"] == "pass", validation
    valid = interventions[interventions["valid"] & (interventions["epsilon"] > 0)]
    assert float(valid["energy_error"].max()) <= 1e-7


def test_renderer_is_fixed_size_and_deterministic():
    sample = synthetic_sample()
    first = np.asarray(MODULE.render_minimap(sample.positions, sample.team_slots))
    second = np.asarray(MODULE.render_minimap(sample.positions, sample.team_slots))
    assert first.shape == (224, 224, 3)
    assert np.array_equal(first, second)


def test_coordinate_models_expose_expected_embedding_shapes():
    import torch

    sample = synthetic_sample()
    deep_sets, gat_ae, phase_gat = MODULE.build_torch_models()
    positions = torch.from_numpy(sample.positions.astype(np.float32))[None]
    teams = torch.from_numpy(sample.team_slots.astype(np.int64))[None]
    adjacency = torch.from_numpy(sample.adjacency.astype(np.float32))[None]
    assert deep_sets(20)(positions, teams)[0].shape == (1, 128)
    assert gat_ae(20)(positions, teams, adjacency)[0].shape == (1, 128)
    assert phase_gat(1)(positions, teams, adjacency)[1].shape == (1, 1)
