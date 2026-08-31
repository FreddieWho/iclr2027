import hashlib
import importlib.util
import json
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


FORMAL = load_module("run_phase2_rigid_formal_test", "scripts/run_phase2_rigid_formal.py")


def test_formal_config_locks_selected_matcher_and_full_design():
    formal, matching = FORMAL.load_formal_config(
        ROOT, ROOT / "configs/phase2_rigid_formal_v2.yaml"
    )
    assert formal["run_id"] == "p2_rigid_formal_v2"
    assert formal["expected"]["samples"] == 250
    assert formal["expected"]["matches"] == 10
    assert formal["expected"]["matched_sets"] == 45600
    assert matching["energies"] == [0.25, 0.5, 1.0, 2.0]
    assert matching["graphs"] == ["knn4", "delaunay"]
    assert len(formal["models"]) == 9


def test_formal_config_fails_closed_on_parent_hash_mismatch(tmp_path):
    parent = tmp_path / "parent.yaml"
    parent.write_text("energies: [0.25]\n", encoding="utf-8")
    config = tmp_path / "formal.yaml"
    config.write_text(
        "run_id: test\n"
        "parent_matching_config:\n"
        f"  path: {parent}\n"
        f"  sha256: {'0' * 64}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parent matching config SHA256 mismatch"):
        FORMAL.load_formal_config(ROOT, config)


def test_group_samples_by_match_is_sorted_and_complete():
    samples = [
        type("S", (), {"match_id": "b", "frame": 3, "sample_id": "b3"})(),
        type("S", (), {"match_id": "a", "frame": 2, "sample_id": "a2"})(),
        type("S", (), {"match_id": "a", "frame": 1, "sample_id": "a1"})(),
    ]
    grouped = FORMAL.group_samples_by_match(samples)
    assert list(grouped) == ["a", "b"]
    assert [sample.sample_id for sample in grouped["a"]] == ["a1", "a2"]


def test_resume_receipt_requires_exact_inputs_and_output_hash(tmp_path):
    output = tmp_path / "part.parquet"
    output.write_bytes(b"first")
    expected = {"config_sha256": "abc", "shard_id": "match=a"}
    receipt = {
        **expected,
        "status": "completed",
        "output_path": str(output),
        "output_sha256": hashlib.sha256(b"first").hexdigest(),
    }
    assert FORMAL.receipt_can_resume(receipt, expected)
    output.write_bytes(b"corrupt")
    assert not FORMAL.receipt_can_resume(receipt, expected)
    output.write_bytes(b"first")
    assert not FORMAL.receipt_can_resume(receipt, {**expected, "config_sha256": "changed"})


def test_receipt_output_validation_fails_closed_on_upstream_tamper(tmp_path):
    output = tmp_path / "upstream.parquet"
    output.write_bytes(b"frozen")
    receipt = {
        "status": "completed",
        "outputs": {
            "arms": {
                "path": str(output),
                "sha256": hashlib.sha256(b"frozen").hexdigest(),
                "bytes": len(b"frozen"),
            }
        },
    }
    FORMAL.validate_receipt_outputs(receipt)
    output.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash|size"):
        FORMAL.validate_receipt_outputs(receipt)


def test_locked_source_files_reject_code_drift(tmp_path):
    source = tmp_path / "runner.py"
    source.write_text("value = 1\n", encoding="utf-8")
    locked = {"runner": {"path": "runner.py", "sha256": FORMAL.file_sha256(source)}}
    observed = FORMAL.validate_locked_files(tmp_path, locked, label="source code")
    assert observed == locked
    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source code SHA256 mismatch"):
        FORMAL.validate_locked_files(tmp_path, locked, label="source code")


class RecordingAdapter:
    batch_size = 8
    uses_adjacency = True
    architecture = "GAT-small-AE"

    def __init__(self):
        self.adjacency_seen = None

    def encode(self, positions, team_slots, adjacency=None):
        assert adjacency is not None
        self.adjacency_seen = np.asarray(adjacency)
        flat = np.asarray(positions).reshape(len(positions), -1)
        return np.pad(flat, ((0, 0), (0, 128 - flat.shape[1])))[:, :128]


def test_response_builder_uses_canonical_adjacency_and_only_valid_arms():
    matching = pd.DataFrame(
        [
            {
                "matched_set_id": "s1",
                "matched_set_status": "INCOMPLETE",
                "arm_kind": "semantic_rigid",
                "control_family": "anchor",
                "source_sample_id": "sample0",
                "sample_index": 0,
                "source_match_id": "m0",
                "split": "train",
                "frame": 1,
                "graph_id": "knn4",
                "coalition_id": "team0_defender",
                "team_slot": 0,
                "role_group": "defender",
                "epsilon": 0.5,
                "direction": "[1.0,0.0]",
                "valid": True,
                "delta": "[[0.1,0.0],[0.0,0.0]]",
                "band_power": "[0.5,0.5]",
                "rayleigh_quotient_normalized": 0.2,
            },
            {
                "matched_set_id": "s1",
                "matched_set_status": "INCOMPLETE",
                "arm_kind": "control",
                "control_family": "arbitrary_same_team",
                "source_sample_id": "sample0",
                "sample_index": 0,
                "source_match_id": "m0",
                "split": "train",
                "frame": 1,
                "graph_id": "knn4",
                "coalition_id": "team0_defender",
                "team_slot": 0,
                "role_group": "defender",
                "epsilon": 0.5,
                "direction": "[1.0,0.0]",
                "valid": False,
                "delta": None,
                "band_power": None,
                "rayleigh_quotient_normalized": None,
            },
        ]
    )
    positions = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    teams = np.array([[0, 1]], dtype=np.int64)
    adjacency = np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
    baseline = np.pad(positions.reshape(1, -1), ((0, 0), (0, 124)))
    adapter = RecordingAdapter()
    response = FORMAL.build_model_response_rows(
        matching,
        model_id="gat_ae_seed11",
        architecture="GAT-small-AE",
        model_seed=11,
        baseline=baseline,
        positions=positions,
        team_slots=teams,
        adjacency=adjacency,
        adapter=adapter,
    )
    assert len(response) == 1
    assert response.iloc[0]["control_family"] == "anchor"
    assert response.iloc[0]["model_seed"] == 11
    assert adapter.adjacency_seen.shape == (1, 2, 2)
    assert np.isfinite(response[["response_distance", "normalized_response"]]).all().all()


def test_response_builder_preserves_shuffled_sample_to_adjacency_mapping():
    matching = pd.DataFrame(
        [
            {
                "matched_set_id": f"s{sample_index}",
                "matched_set_status": "INCOMPLETE",
                "arm_kind": "semantic_rigid",
                "control_family": "anchor",
                "source_sample_id": f"sample{sample_index}",
                "sample_index": sample_index,
                "source_match_id": "m0",
                "split": "train",
                "frame": sample_index,
                "graph_id": "delaunay",
                "coalition_id": "team0_defender",
                "team_slot": 0,
                "role_group": "defender",
                "epsilon": 0.5,
                "direction": "[1.0,0.0]",
                "valid": True,
                "delta": "[[0.1,0.0],[0.0,0.0]]",
                "band_power": "[0.5,0.5]",
                "rayleigh_quotient_normalized": 0.2,
            }
            for sample_index in [1, 0]
        ]
    )
    positions = np.array(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]], dtype=np.float32
    )
    teams = np.array([[0, 1], [0, 1]], dtype=np.int64)
    adjacency = np.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 2.0], [2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    baseline = np.pad(positions.reshape(2, -1), ((0, 0), (0, 124)))
    adapter = RecordingAdapter()
    FORMAL.build_model_response_rows(
        matching,
        model_id="gat_ae_seed11",
        architecture="GAT-small-AE",
        model_seed=11,
        baseline=baseline,
        positions=positions,
        team_slots=teams,
        adjacency=adjacency,
        adapter=adapter,
    )
    assert np.array_equal(adapter.adjacency_seen, adjacency[[1, 0]])


def test_response_validation_rejects_missing_valid_arm():
    matching = pd.DataFrame(
        {
            "matched_set_id": ["a", "a"],
            "arm_kind": ["semantic_rigid", "control"],
            "control_family": ["anchor", "arbitrary_same_team"],
            "valid": [True, True],
        }
    )
    response = pd.DataFrame(
        {
            "matched_set_id": ["a"],
            "arm_kind": ["semantic_rigid"],
            "control_family": ["anchor"],
            "response_distance": [0.1],
            "normalized_response": [0.2],
        }
    )
    validation = FORMAL.validate_response_shard(response, matching)
    assert validation["status"] == "fail"
    assert "row count" in " ".join(validation["errors"])


def test_response_stage_rejects_invalid_matching_chain_before_inference(tmp_path, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("upstream matching hash mismatch")

    monkeypatch.setattr(FORMAL, "validate_matching_chain", reject)
    formal = {"status_contract": {"matching_complete": "complete"}}
    with pytest.raises(ValueError, match="upstream matching hash mismatch"):
        FORMAL.run_response_stage(
            tmp_path,
            tmp_path / "config.yaml",
            formal,
            tmp_path / "run",
            resume=False,
        )


def test_statistics_stage_rejects_invalid_response_chain_before_pairing(tmp_path, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("upstream response hash mismatch")

    monkeypatch.setattr(FORMAL, "validate_response_chain", reject)
    formal = {"statistics": {"strict_band_power_l1": 0.2}}
    with pytest.raises(ValueError, match="upstream response hash mismatch"):
        FORMAL.run_statistics_stage(
            tmp_path,
            tmp_path / "config.yaml",
            formal,
            tmp_path / "run",
            resume=False,
        )


def test_report_stage_rejects_invalid_statistics_chain_before_writing(tmp_path, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("upstream statistics hash mismatch")

    monkeypatch.setattr(FORMAL, "validate_statistics_chain", reject)
    with pytest.raises(ValueError, match="upstream statistics hash mismatch"):
        FORMAL.run_report_stage(
            tmp_path,
            tmp_path / "config.yaml",
            {},
            tmp_path / "run",
            resume=False,
        )
