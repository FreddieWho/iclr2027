import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STATS = load_module("p2_statistics_test", "scripts/p2_statistics.py")


def response_set(set_id, match_id, model_id="m_seed11", seed=11, complete=True):
    families = [
        ("semantic_rigid", "anchor", 0.8, "[0.5,0.5]"),
        ("control", "arbitrary_same_team", 0.3, "[0.7,0.3]"),
        ("control", "topology_frequency_matched", 0.4, "[0.55,0.45]"),
        ("control", "spectrum_exact_sign_randomized", 0.5, "[0.5,0.5]"),
    ]
    if not complete:
        families = families[:-1]
    rows = []
    for arm_kind, family, normalized, bands in families:
        rows.append(
            {
                "model_id": model_id,
                "architecture": "DeepSets-AE",
                "model_seed": seed,
                "matched_set_id": set_id,
                "matched_set_status": "COMPLETE" if complete else "INCOMPLETE",
                "arm_kind": arm_kind,
                "control_family": family,
                "source_sample_id": f"sample-{set_id}",
                "source_match_id": match_id,
                "split": "train",
                "frame": 1,
                "graph_id": "knn4",
                "coalition_id": "team0_defender",
                "team_slot": 0,
                "role_group": "defender",
                "epsilon": 0.5,
                "direction": "[1.0,0.0]",
                "response_distance": normalized * 0.5,
                "normalized_response": normalized,
                "band_power": bands,
            }
        )
    return rows


def test_pairing_keeps_controls_separate_and_uses_semantic_minus_control():
    response = pd.DataFrame(response_set("s1", "match1"))
    pairs = STATS.build_paired_effects(response)
    common = pairs[pairs["support_mode"].eq("common_four")]
    assert set(common["control_family"]) == {
        "arbitrary_same_team",
        "topology_frequency_matched",
        "spectrum_exact_sign_randomized",
    }
    observed = common.set_index("control_family")["normalized_scg"].to_dict()
    assert np.isclose(observed["arbitrary_same_team"], 0.5)
    assert np.isclose(observed["topology_frequency_matched"], 0.4)
    assert np.isclose(observed["spectrum_exact_sign_randomized"], 0.3)


def test_incomplete_set_is_pairwise_sensitivity_not_common_support():
    response = pd.DataFrame(response_set("s1", "match1", complete=False))
    pairs = STATS.build_paired_effects(response)
    assert not pairs["support_mode"].eq("common_four").any()
    assert pairs[pairs["support_mode"].eq("pairwise")]["control_family"].nunique() == 2


def test_control_without_valid_semantic_anchor_is_not_made_into_a_pair():
    valid = response_set("valid", "match1")
    orphan = response_set("orphan", "match1", complete=False)[1:]
    pairs = STATS.build_paired_effects(pd.DataFrame(valid + orphan))
    assert set(pairs["matched_set_id"]) == {"valid"}


def test_overall_role_stratum_weights_roles_equally_after_team_aggregation():
    pairs = []
    for team_slot, value in [(0, 0.0), (1, 2.0)]:
        pairs.append(
            {
                "model_id": "m11",
                "architecture": "DeepSets-AE",
                "model_seed": 11,
                "control_family": "arbitrary_same_team",
                "support_mode": "common_four",
                "source_match_id": "match1",
                "split": "train",
                "source_sample_id": "sample1",
                "frame": 1,
                "graph_id": "knn4",
                "coalition_id": f"team{team_slot}_defender",
                "team_slot": team_slot,
                "role_group": "defender",
                "epsilon": 0.25,
                "direction": "[1,0]",
                "normalized_scg": value,
                "raw_scg": value * 0.25,
            }
        )
    pairs.append(
        {
            **pairs[0],
            "coalition_id": "team0_forward",
            "role_group": "forward",
            "normalized_scg": 5.0,
            "raw_scg": 1.25,
        }
    )
    _, architecture = STATS.aggregate_match_effects(pd.DataFrame(pairs))
    overall = architecture[architecture["role_stratum"].eq("__overall__")]
    assert len(overall) == 1
    # Defender is first averaged across its two teams: (0 + 2) / 2 = 1;
    # then defender and forward are equally weighted: (1 + 5) / 2 = 3.
    assert np.isclose(overall.iloc[0]["normalized_scg"], 3.0)


def test_match_and_seed_aggregation_prevents_pseudoreplication():
    rows = []
    for index in range(20):
        rows.extend(response_set(f"m1-{index}", "match1", model_id="m_seed11", seed=11))
        rows.extend(response_set(f"m1b-{index}", "match1", model_id="m_seed23", seed=23))
    match2_a = response_set("m2", "match2", model_id="m_seed11", seed=11)
    match2_b = response_set("m2b", "match2", model_id="m_seed23", seed=23)
    for row in match2_a + match2_b:
        if row["control_family"] == "anchor":
            row["normalized_response"] = 2.8
            row["response_distance"] = 1.4
        elif row["control_family"] == "arbitrary_same_team":
            row["normalized_response"] = -0.2
            row["response_distance"] = -0.1
    rows.extend(match2_a + match2_b)
    pairs = STATS.build_paired_effects(pd.DataFrame(rows))
    model_match, architecture_match = STATS.aggregate_match_effects(pairs)
    selected = architecture_match[
        architecture_match["support_mode"].eq("common_four")
        & architecture_match["control_family"].eq("arbitrary_same_team")
        & architecture_match["role_stratum"].eq("__overall__")
    ]
    assert len(selected) == 2
    assert set(selected["source_match_id"]) == {"match1", "match2"}
    summary = STATS.summarize_match_effects(selected, n_bootstrap=200, seed=7)
    assert summary.iloc[0]["n_matches"] == 2
    assert np.isclose(summary.iloc[0]["mean_normalized_scg"], 1.75)
    assert model_match["model_seed"].nunique() == 2


def test_bootstrap_and_leave_one_match_out_are_deterministic():
    frame = pd.DataFrame(
        {
            "architecture": ["DeepSets-AE"] * 3,
            "control_family": ["arbitrary_same_team"] * 3,
            "support_mode": ["common_four"] * 3,
            "source_match_id": ["a", "b", "c"],
            "split": ["train"] * 3,
            "graph_id": ["knn4"] * 3,
            "epsilon": [0.25] * 3,
            "role_stratum": ["__overall__"] * 3,
            "normalized_scg": [1.0, 2.0, 3.0],
            "raw_scg": [0.25, 0.5, 0.75],
            "seed_spread_normalized": [0.0] * 3,
            "seed_spread_raw": [0.0] * 3,
        }
    )
    first = STATS.summarize_match_effects(frame, n_bootstrap=500, seed=19)
    second = STATS.summarize_match_effects(frame, n_bootstrap=500, seed=19)
    pd.testing.assert_frame_equal(first, second)
    loo = STATS.leave_one_match_out(frame)
    assert len(loo) == 3
    assert set(loo["omitted_match_id"]) == {"a", "b", "c"}


def test_full_aggregation_order_equal_weights_every_level():
    def row(match_id, sample_id, role, team, coalition, direction, value):
        return {
            "model_id": "m11",
            "architecture": "DeepSets-AE",
            "model_seed": 11,
            "control_family": "arbitrary_same_team",
            "support_mode": "common_four",
            "source_match_id": match_id,
            "split": "train",
            "source_sample_id": sample_id,
            "frame": 1,
            "graph_id": "knn4",
            "coalition_id": coalition,
            "team_slot": team,
            "role_group": role,
            "epsilon": 0.25,
            "direction": direction,
            "normalized_scg": value,
            "raw_scg": value * 0.25,
        }

    pairs = pd.DataFrame(
        [
            # sample A defender: direction mean 1, coalition mean 2,
            # then team mean (2 + 6) / 2 = 4.
            row("match1", "A", "defender", 0, "d0", "x", 0.0),
            row("match1", "A", "defender", 0, "d0", "y", 2.0),
            row("match1", "A", "defender", 0, "d1", "x", 3.0),
            row("match1", "A", "defender", 1, "d2", "x", 6.0),
            # sample A overall role mean: (defender 4 + forward 10) / 2 = 7.
            row("match1", "A", "forward", 0, "f0", "x", 10.0),
            # match1 sample mean: (A 7 + B 1) / 2 = 4.
            row("match1", "B", "defender", 0, "d0", "x", 1.0),
            # match2 mean = 8.
            row("match2", "C", "defender", 0, "d0", "x", 8.0),
        ]
    )
    _, architecture = STATS.aggregate_match_effects(pairs)
    overall = architecture[architecture["role_stratum"].eq("__overall__")]
    observed = overall.set_index("source_match_id")["normalized_scg"].to_dict()
    assert np.isclose(observed["match1"], 4.0)
    assert np.isclose(observed["match2"], 8.0)
    summary = STATS.summarize_match_effects(overall, n_bootstrap=100, seed=3)
    assert np.isclose(summary.iloc[0]["mean_normalized_scg"], 6.0)
