import importlib.util
import sys
from pathlib import Path

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


ADAPTER = load_module("p2_point_model_adapter_test", "scripts/p2_point_model_adapter.py")
SMOKE = load_module("run_phase2_response_smoke_test", "scripts/run_phase2_response_smoke.py")
RESOURCE = load_module("run_phase2_resource_gate_test", "scripts/run_phase2_resource_gate.py")


def test_hash_verification_fails_closed(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-the-expected-checkpoint")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ADAPTER.verify_file_hash(checkpoint, "0" * 64, label="checkpoint")


def test_deepsets_adapter_matches_saved_p1_baseline():
    adapter, receipt = ADAPTER.load_deepsets_adapter(ROOT, model_id="deepsets_ae_seed11")
    with np.load(ROOT / "artifacts/phase1/canonical_samples.npz") as archive:
        positions = np.asarray(archive["positions"][:2], dtype=np.float32)
        team_slots = np.asarray(archive["team_slots"][:2], dtype=np.int64)
    expected = np.load(ROOT / "artifacts/phase1/embeddings/deepsets_ae_seed11_baseline.npy")[:2]
    first = adapter.encode(positions, team_slots)
    second = adapter.encode(positions, team_slots)
    assert first.shape == (2, 128)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert np.allclose(first, expected, rtol=1e-6, atol=5e-7)
    assert receipt["device"] == "cpu"
    assert receipt["weights_only"] is True


def test_gat_adapter_requires_fixed_baseline_adjacency():
    adapter, receipt = ADAPTER.load_point_adapter(ROOT, model_id="gat_ae_seed11", batch_size=2)
    with np.load(ROOT / "artifacts/phase1/canonical_samples.npz") as archive:
        positions = np.asarray(archive["positions"][:2], dtype=np.float32)
        team_slots = np.asarray(archive["team_slots"][:2], dtype=np.int64)
        adjacency = np.asarray(archive["adjacency"][:2], dtype=np.float32)
    with pytest.raises(ValueError, match="adjacency"):
        adapter.encode(positions, team_slots)
    encoded = adapter.encode(positions, team_slots, adjacency=adjacency)
    assert encoded.shape == (2, 128)
    assert np.isfinite(encoded).all()
    assert receipt["graph"] == "baseline_fixed_weighted_knn4"


def four_arm_set(set_id, epsilon=0.25, status="COMPLETE"):
    families = [
        ("semantic_rigid", "anchor"),
        ("control", "arbitrary_same_team"),
        ("control", "topology_frequency_matched"),
        ("control", "spectrum_exact_sign_randomized"),
    ]
    return [
        {
            "matched_set_id": set_id,
            "matched_set_status": status,
            "epsilon": epsilon,
            "valid": status == "COMPLETE",
            "arm_kind": arm_kind,
            "control_family": family,
        }
        for arm_kind, family in families
    ]


def test_response_selection_keeps_only_complete_low_energy_four_arm_sets():
    frame = pd.DataFrame(
        four_arm_set("low", 0.25)
        + four_arm_set("mid", 1.0)
        + four_arm_set("incomplete", 0.5, status="INCOMPLETE")
    )
    selected = SMOKE.select_complete_low_energy_sets(frame, energies=(0.25, 0.5))
    assert selected["matched_set_id"].unique().tolist() == ["low"]
    assert len(selected) == 4


def test_response_selection_rejects_malformed_complete_set():
    malformed = pd.DataFrame(four_arm_set("bad")[:-1])
    with pytest.raises(ValueError, match="four-arm"):
        SMOKE.select_complete_low_energy_sets(malformed, energies=(0.25, 0.5))


def test_normalized_cosine_distance_is_finite_and_scale_invariant():
    before = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    after = np.array([[0.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    distance = SMOKE.normalized_cosine_distance(before, after)
    assert np.allclose(distance, [1.0, 0.0], atol=1e-7)
    assert np.allclose(distance, SMOKE.normalized_cosine_distance(before * 3, after * 7))


def test_success_summary_remains_wiring_only():
    summary = SMOKE.build_pipeline_summary({"status": "pass", "n_response_rows": 4})
    assert summary["status"] == "WIRING_ONLY_NOT_SCIENTIFIC"
    assert summary["scientific_interpretation"] == "NOT_RUN"


def test_resource_projection_scales_rows_by_samples_models_and_four_arms():
    projection = RESOURCE.project_formal_resources(
        benchmarks=[
            {"architecture": "DeepSets-AE", "rows_per_second": 100.0, "load_seconds": 1.0},
            {"architecture": "GAT-small-AE", "rows_per_second": 50.0, "load_seconds": 2.0},
            {"architecture": "Phase-GAT", "rows_per_second": 25.0, "load_seconds": 3.0},
        ],
        source_samples=20,
        source_sets=1260,
        target_samples=250,
        seeds_per_architecture=3,
        source_response_rows=5040,
        source_response_bytes=100_800,
        observed_peak_rss_mib=1000.0,
    )
    assert projection["projected_sets"] == 15750
    assert projection["projected_rows_per_model"] == 63000
    assert projection["projected_total_response_rows"] == 567000
    assert projection["projected_response_bytes"] == 11_340_000
    assert projection["formal_run"] == "NOT_RUN"
