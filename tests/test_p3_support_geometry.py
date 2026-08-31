from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p3_support_geometry as p3  # noqa: E402


class ToyGeometry:
    uses_adjacency = False
    available_layers = ("pooled_embedding",)

    def layer_values(self, positions, team_slots, adjacency=None):
        del team_slots, adjacency
        node = torch.cat([2.0 * positions, positions.square()], dim=-1)
        return {"pooled_embedding": node.sum(dim=1)}


def test_jvp_matches_finite_difference_for_normalized_output():
    model = ToyGeometry()
    positions = np.array([[0.2, -0.4], [0.7, 0.3]], dtype=np.float32)
    teams = np.array([0, 1], dtype=np.int64)
    direction = np.array([[0.3, -0.2], [0.1, 0.7]], dtype=np.float32)
    direction /= np.linalg.norm(direction)
    value, jacobian = p3.jacobian_for_layer(model, positions, teams, None, "pooled_embedding")
    _, jvp = p3.jvp_for_layer(model, positions, teams, None, "pooled_embedding", direction)
    finite = p3.finite_difference(model, positions, teams, None, "pooled_embedding", direction, 1e-4)
    np.testing.assert_allclose(jvp, jacobian @ direction.reshape(-1), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(finite, jvp, rtol=3e-3, atol=3e-4)
    assert value.shape == (4,)


def test_block_decomposition_has_additive_identities():
    rng = np.random.default_rng(4)
    jacobian = rng.normal(size=(5, 8)).astype(np.float32)
    delta = rng.normal(size=(4, 2)).astype(np.float32)
    teams = np.array([0, 0, 1, 1], dtype=np.int64)
    adjacency = np.array(
        [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]],
        dtype=np.float32,
    )
    parts = p3.block_decomposition(jacobian, delta, teams, adjacency, [0, 1])
    assert abs(parts["identity_full_minus_diag_off"]) < 1e-10
    assert abs(parts["identity_off_minus_team"]) < 1e-10
    assert abs(parts["identity_off_minus_graph"]) < 1e-10
    assert abs(parts["identity_full_minus_support"]) < 1e-10
    np.testing.assert_allclose(
        parts["q_full"],
        parts["q_diagonal"] + parts["q_off_diagonal"],
        rtol=1e-12,
        atol=1e-12,
    )


def test_team_pool_is_permutation_consistent():
    h = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    teams = torch.tensor([[0, 1, 0, 1]])
    permutation = torch.tensor([2, 0, 3, 1])
    original = p3._team_pool(h, teams)
    permuted = p3._team_pool(h[:, permutation], teams[:, permutation])
    torch.testing.assert_close(original, permuted)


def test_relational_pool_is_permutation_consistent_and_shape_preserving():
    h = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6).requires_grad_(True)
    teams = torch.tensor([[0, 1, 0, 1]])
    permutation = torch.tensor([2, 0, 3, 1])
    original = p3._relational_pool(h, teams)
    permuted = p3._relational_pool(h[:, permutation], teams[:, permutation])
    assert original.shape == (1, 12)
    torch.testing.assert_close(original, permuted)
    original.sum().backward()
    assert h.grad is not None


def test_grouped_cv_keeps_held_out_matches_out_of_training():
    rows = []
    for match in ("m1", "m2", "m3"):
        for idx in range(3):
            rows.append(
                {
                    "model_id": "toy",
                    "matched_set_id": f"{match}-{idx}",
                    "source_match_id": match,
                    "observed_pair_effect": float(idx + (match == "m3")),
                    "delta_q_full": float(idx + 1),
                }
            )
    frame = pd.DataFrame(rows)
    predictions = p3.grouped_cv_predictions(frame, "full_geometry")
    assert set(predictions["source_match_id"]) == {"m1", "m2", "m3"}
    assert len(predictions) == len(frame)
    for match in ("m1", "m2", "m3"):
        assert predictions.loc[predictions["source_match_id"] == match, "predictor_family"].eq("full_geometry").all()


def test_frozen_file_hash_is_stable(tmp_path):
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(b"frozen")
    before = p3.file_sha256(path)
    assert before == p3.file_sha256(path)
    assert path.read_bytes() == b"frozen"


def test_json_hash_is_deterministic():
    payload = {"b": [2, 1], "a": {"z": True}}
    assert p3.json_sha256(payload) == p3.json_sha256({"a": {"z": True}, "b": [2, 1]})


def test_prospective_manifest_is_response_blind_and_summary_matches_tables():
    root = ROOT / "artifacts" / "phase3" / "support_geometry_prospective_v1"
    manifest = json.loads((root / "intervention_manifest.json").read_text())
    summary = json.loads((root / "prospective_summary.json").read_text())
    assert manifest["response_blind"] is True
    assert manifest["response_paths_read"] == []
    assert summary["response_blind_generation"] is True
    assert summary["layer_or_formula_changed_after_response"] is False
    assert summary["n_arm_rows"] == len(pd.read_parquet(root / "prospective_arm_geometry.parquet"))
    assert summary["n_pair_rows"] == len(pd.read_parquet(root / "prospective_pair_geometry.parquet"))


def test_node_localization_is_response_blind_and_geometry_parity_is_small():
    root = ROOT / "artifacts" / "phase3" / "selected_causal_switch_v1"
    columns = pd.read_parquet(root / "node_sensitivity_localization_v2.parquet", engine="pyarrow").columns
    assert "response" not in columns
    assert "raw_response" not in columns
    summary = json.loads((root / "node_sensitivity_localization_v2_summary.json").read_text())
    assert summary["response_blind"] is True
    assert summary["response_columns_read"] == []
    assert summary["max_q_full_parity_abs_error"] < 1e-7
