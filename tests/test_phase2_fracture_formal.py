import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.p2_fracture_statistics import build_pair_effects, aggregate_match_effects, aggregate_architecture_matches
from scripts.run_phase2_fracture_formal import array_json, read_config, write_pipeline_summary


def _response_frame() -> pd.DataFrame:
    common = {
        "matched_set_id": "set-1",
        "source_sample_id": "sample-1",
        "source_match_id": "match-1",
        "split": "train",
        "graph_id": "knn4",
        "coalition_id": "team0_defender",
        "team_slot": 0,
        "role_group": "defender",
        "epsilon": 0.25,
        "fracture_draw": 11,
        "vector_multiset_id": "vectors-1",
        "frequency_residual": 0.01,
        "topology_residual": 0.02,
        "model_id": "deepsets_ae_seed11",
        "architecture": "DeepSets-AE",
        "model_seed": 11,
        "valid": True,
        "matched_set_status": "COMPLETE",
    }
    anchor = {**common, "arm_kind": "anchor", "control_family": "anchor", "response_distance": 0.20, "normalized_response": 0.80}
    control = {**common, "arm_kind": "control", "control_family": "topology_frequency_same_vector_reassignment", "response_distance": 0.10, "normalized_response": 0.40, "frequency_residual": 0.03, "topology_residual": 0.04}
    return pd.DataFrame([anchor, control])


def test_fracture_pair_and_match_aggregation_preserve_two_arm_estimand():
    pairs = build_pair_effects(_response_frame())
    assert len(pairs) == 1
    assert np.isclose(float(pairs.iloc[0]["raw_scg"]), 0.10)
    assert np.isclose(float(pairs.iloc[0]["frequency_residual"]), 0.03)
    assert np.isclose(float(pairs.iloc[0]["topology_residual"]), 0.04)
    _, model_match = aggregate_match_effects(pairs)
    architecture_match = aggregate_architecture_matches(model_match)
    assert len(model_match) == 1
    assert len(architecture_match) == 1
    assert np.isclose(float(architecture_match.iloc[0]["raw_scg"]), 0.10)


def test_full_precision_delta_json_round_trip():
    value = np.array([[0.12345678901234567, -0.9876543210987654], [0.0, 0.0]], dtype=np.float64)
    restored = np.asarray(json.loads(array_json(value)), dtype=np.float64)
    assert np.array_equal(value, restored)


def test_completed_pipeline_summary_is_monotonic(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {
        "status": "P2_FRACTURE_C2_COMPLETE",
        "checkpoint": "P2-FC-E",
        "last_completed_stage": "report",
        "stage_rank": 4,
    }
    (run_dir / "pipeline_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_pipeline_summary(Path("."), run_dir, {"run_id": "test"}, "matching", {"status": "old"}, False)
    observed = json.loads((run_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert observed == summary
